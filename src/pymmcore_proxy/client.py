"""RemoteMMCore — drop-in network client for pymmcore-plus.

Usage:
    from pymmcore_proxy import connect
    core = connect("http://127.0.0.1:5600")

    # Use exactly like CMMCorePlus:
    core.snapImage()
    img = core.getImage()           # real numpy array
    core.setXYPosition(100, 200)
    core.events.propertyChanged.connect(my_callback)

    # MDA runs on the server, signals stream back via WebSocket:
    from useq import MDASequence
    core.mda.events.frameReady.connect(on_frame)
    core.mda.run(MDASequence(time_plan={"interval": 1, "loops": 10}))

    core.close()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import threading
import time
import warnings
from typing import Any

import httpx
import pymmcore
from psygnal import Signal
from pymmcore_plus import CMMCorePlus

from ._serialize import decode, encode

logger = logging.getLogger("pymmcore_proxy.client")


# ------------------------------------------------------------------
# Nested proxy — transparent dotted attribute access
# ------------------------------------------------------------------

class _NestedProxy:
    """Proxy for nested attribute access on the remote core.

    Accessing ``core.mda.engine._z_correction`` produces a chain of
    _NestedProxy objects.  Calling, bool-testing, iterating, etc.
    resolve the value via RPC.
    """

    def __init__(self, client: RemoteMMCore, path: str):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _NestedProxy(self._client, f"{self._path}.{name}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            self._client._rpc_setattr(f"{self._path}.{name}", value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._client._rpc(self._path, *args, **kwargs)

    def _resolve(self) -> Any:
        return self._client._rpc_getattr(self._path)

    def __bool__(self) -> bool:
        return bool(self._resolve())

    def __eq__(self, other: object) -> bool:
        return self._resolve() == other

    def __repr__(self) -> str:
        return repr(self._resolve())

    def __int__(self) -> int:
        return int(self._resolve())

    def __float__(self) -> float:
        return float(self._resolve())

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self) -> int:
        return len(self._resolve())

    def __getitem__(self, key: Any) -> Any:
        return self._resolve()[key]

    def __contains__(self, item: Any) -> bool:
        return item in self._resolve()

    def __hash__(self) -> int:
        return hash(self._resolve())


# ------------------------------------------------------------------
# Signal groups — mirror CMMCorePlus signal signatures
# ------------------------------------------------------------------

class _CoreSignals:
    """Signals emitted by the microscope core (mirrors CMMCorePlus.events)."""

    propertiesChanged = Signal()
    propertyChanged = Signal(object, object, object)
    channelGroupChanged = Signal(object)
    configGroupChanged = Signal(object, object)
    systemConfigurationLoaded = Signal()
    pixelSizeChanged = Signal(object)
    pixelSizeAffineChanged = Signal(object, object, object, object, object, object)
    stagePositionChanged = Signal(object, object)
    XYStagePositionChanged = Signal(object, object, object)
    exposureChanged = Signal(object, object)
    SLMExposureChanged = Signal(object, object)
    configSet = Signal(object, object)
    imageSnapped = Signal(object)
    mdaEngineRegistered = Signal(object, object)
    continuousSequenceAcquisitionStarting = Signal()
    continuousSequenceAcquisitionStarted = Signal()
    sequenceAcquisitionStarting = Signal(object)
    sequenceAcquisitionStarted = Signal(object)
    sequenceAcquisitionStopped = Signal(object)
    autoShutterSet = Signal(object)
    shutterOpenChanged = Signal(object, object)
    configGroupDeleted = Signal(object)
    configDeleted = Signal(object, object)
    configDefined = Signal(object, object, object, object, object)
    roiSet = Signal(object, object, object, object, object)


class _DevicePropertySignal:
    """Callable signal filter for device property changes.

    Mirrors CMMCorePlus's ``devicePropertyChanged`` — a callable that
    returns filtered signals:

        core.events.devicePropertyChanged("Camera", "Gain").connect(cb)
        core.events.devicePropertyChanged("Camera").connect(cb)

    Internally listens to ``propertyChanged`` and routes matching events
    to the appropriate filtered signal.
    """

    def __init__(self, core_signals: _CoreSignals):
        self._cache: dict[tuple, Any] = {}
        core_signals.propertyChanged.connect(self._on_property_changed)

    def __call__(self, device: str, prop: str | None = None) -> Any:
        key = (device, prop)
        if key not in self._cache:
            if prop is not None:
                # device + property filter → emits (value,)
                holder = type("_S", (), {"sig": Signal(object)})()
            else:
                # device-only filter → emits (property, value)
                holder = type("_S", (), {"sig": Signal(object, object)})()
            self._cache[key] = holder.sig
        return self._cache[key]

    def _on_property_changed(self, device: str, prop: str, value: str) -> None:
        # Route to device + property specific signals
        key = (device, prop)
        if key in self._cache:
            self._cache[key].emit(value)
        # Route to device-only signals
        key = (device, None)
        if key in self._cache:
            self._cache[key].emit(prop, value)


class _MDASignals:
    """Signals emitted during MDA acquisition (mirrors MDARunner.events)."""

    frameReady = Signal(object, object, object)
    sequenceStarted = Signal(object, object)
    sequenceFinished = Signal(object)
    sequenceCanceled = Signal(object)
    sequencePauseToggled = Signal(object)
    awaitingEvent = Signal(object, object)
    eventStarted = Signal(object)


# ------------------------------------------------------------------
# MDA controller — server-driven, signals forwarded via WebSocket
# ------------------------------------------------------------------

class _MDAController:
    """Server-driven MDA controller.

    Delegates run/cancel/pause to the server's CMMCorePlus.mda.
    Signals are forwarded over WebSocket and emitted locally.
    Private attributes (``_running``, ``_paused``, etc.) are fetched
    from the server via RPC getattr.
    """

    def __init__(self, client: RemoteMMCore):
        self._client = client
        self.events = _MDASignals()

    # -- signals alias (pymmcore-plus tests access core.mda._signals) --

    @property
    def _signals(self) -> _MDASignals:
        return self.events

    # -- delegation to server --

    def run(
        self,
        sequence: Any,
        *,
        output: Any = None,
        overwrite: bool = False,
        dimension_overrides: Any = None,
    ) -> None:
        """Run an MDA sequence on the server (blocking).

        Accepts an ``MDASequence``, a ``dict``, or any iterable of
        ``MDAEvent`` objects (including generators and ``Queue``-backed
        iterators).

        The *output*, *overwrite*, and *dimension_overrides* parameters
        are accepted for signature compatibility with the upstream
        ``MDARunner.run`` (see pymmcore-plus 0.18.1) but ignored —
        output handlers run on the server side.

        For serializable sequences (``MDASequence``/``dict``), the whole
        sequence is sent via RPC.  For arbitrary iterables, events are
        streamed one-by-one over a dedicated WebSocket so that
        lazily-produced events (e.g. from a ``Queue``) work correctly.
        """
        from useq import MDASequence

        if isinstance(sequence, dict):
            sequence = MDASequence(**sequence)

        # Reset stale cancel flag (a previous cancel() may have arrived
        # after the MDA finished, leaving _canceled=True on the server).
        try:
            self._client._rpc_setattr("mda._canceled", False)
        except Exception:
            pass

        if isinstance(sequence, MDASequence):
            self._run_rpc(sequence)
        else:
            self._run_streaming(sequence)

    # -- RPC-based run (full sequence sent at once) --

    def _run_rpc(self, sequence: Any) -> None:
        done = threading.Event()

        def _on_done(*args: Any) -> None:
            done.set()

        self.events.sequenceFinished.connect(_on_done)
        self.events.sequenceCanceled.connect(_on_done)
        try:
            self._client._rpc("mda.run", sequence, _timeout=None)
            done.wait()
        finally:
            self.events.sequenceFinished.disconnect(_on_done)
            self.events.sequenceCanceled.disconnect(_on_done)

    # -- Streaming run (events sent one-by-one via WebSocket) --

    def _run_streaming(self, events: Any) -> None:
        from websockets.sync.client import connect as ws_connect

        ws_url = (
            self._client._url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/mda/stream"
        )

        done = threading.Event()

        def _on_done(*args: Any) -> None:
            done.set()

        self.events.sequenceFinished.connect(_on_done)
        self.events.sequenceCanceled.connect(_on_done)
        try:
            with ws_connect(ws_url) as ws:
                ws.send(json.dumps({"action": "start"}))

                # Stream events — runs in this thread, which is fine
                # because the signal listener runs in its own thread.
                for event in events:
                    if done.is_set():
                        break
                    ws.send(json.dumps({"event": encode(event)}))

                ws.send(json.dumps({"action": "stop"}))

                # Wait for server "done" acknowledgement
                try:
                    raw = ws.recv(timeout=30.0)
                    msg = json.loads(raw)
                    if msg.get("error"):
                        raise RuntimeError(msg["error"])
                except TimeoutError:
                    pass

            # Ensure sequenceFinished/Canceled signal has arrived.
            # No timeout: mda.run() is non-blocking, so the server ack arrives
            # before the MDA completes. We must wait indefinitely here.
            done.wait()
        finally:
            self.events.sequenceFinished.disconnect(_on_done)
            self.events.sequenceCanceled.disconnect(_on_done)

    def cancel(self) -> None:
        self._client._rpc("mda.cancel")

    def set_paused(self, paused: bool) -> None:
        self._client._rpc("mda.set_paused", paused)

    def toggle_pause(self) -> None:
        self.set_paused(not self.is_paused())

    def is_running(self) -> bool:
        return self._client._rpc("mda.is_running")

    def is_paused(self) -> bool:
        return self._client._rpc("mda.is_paused")

    def set_engine(self, engine: Any) -> None:
        self._client._rpc("mda.set_engine", engine)

    @property
    def engine(self) -> _NestedProxy:
        return _NestedProxy(self._client, "mda.engine")

    @property
    def _engine(self) -> _NestedProxy:
        return _NestedProxy(self._client, "mda._engine")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Proxy private attrs like _running, _paused, _canceled to server
        return self._client._rpc_getattr(f"mda.{name}")


# ------------------------------------------------------------------
# Signal listener (WebSocket, background thread)
# ------------------------------------------------------------------

# Maps "group.signal" → (signal_group_attr, signal_name)
_SIGNAL_MAP: dict[str, tuple[str, str]] = {}
for _name in [
    "propertiesChanged", "propertyChanged",
    "channelGroupChanged", "configGroupChanged",
    "systemConfigurationLoaded",
    "pixelSizeChanged", "pixelSizeAffineChanged",
    "stagePositionChanged", "XYStagePositionChanged",
    "exposureChanged", "SLMExposureChanged",
    "configSet", "imageSnapped", "mdaEngineRegistered",
    "continuousSequenceAcquisitionStarting",
    "continuousSequenceAcquisitionStarted",
    "sequenceAcquisitionStarting", "sequenceAcquisitionStarted",
    "sequenceAcquisitionStopped",
    "autoShutterSet", "shutterOpenChanged", "configGroupDeleted",
    "configDeleted", "configDefined", "roiSet",
]:
    _SIGNAL_MAP[f"events.{_name}"] = ("events", _name)
for _name in [
    "frameReady", "sequenceStarted", "sequenceFinished",
    "sequenceCanceled", "sequencePauseToggled",
    "awaitingEvent", "eventStarted",
]:
    _SIGNAL_MAP[f"mda.events.{_name}"] = ("mda.events", _name)

# Warning category lookup for forwarded server-side warnings
_WARNING_CATEGORIES: dict[str, type[Warning]] = {
    "UserWarning": UserWarning,
    "DeprecationWarning": DeprecationWarning,
    "RuntimeWarning": RuntimeWarning,
    "FutureWarning": FutureWarning,
    "PendingDeprecationWarning": PendingDeprecationWarning,
    "SyntaxWarning": SyntaxWarning,
    "ResourceWarning": ResourceWarning,
}


# ------------------------------------------------------------------
# pymmcore.CMMCore public method names (for RPC interception)
# ------------------------------------------------------------------
_CMMCORE_METHODS = frozenset(
    name for name in dir(pymmcore.CMMCore)
    if not name.startswith("_") and callable(getattr(pymmcore.CMMCore, name, None))
)

# CMMCorePlus methods that wrap a SWIG call with a mutable out-parameter
# (e.g. ``getLastImageMD(md)`` fills ``md`` in-place).  Such mutations are
# lost across the RPC boundary, so we forward the whole CMMCorePlus method
# to the server and let it execute locally there.
_CMMCORE_PLUS_FORWARD = frozenset({
    "getLastImageAndMD",
    "popNextImageAndMD",
    "getNBeforeLastImageAndMD",
})


# ------------------------------------------------------------------
# Qt cross-thread signal emission
# ------------------------------------------------------------------

class _EmitEvent:
    """Custom QEvent that emits a psygnal signal on the main thread.

    ``QApplication.postEvent`` is safe to call from *any* thread and
    delivers the event to the receiver's thread event loop — no QObject
    creation on the background thread required.
    """

    _EVENT_TYPE: int | None = None
    _receiver: Any = None  # singleton QObject on main thread

    @classmethod
    def post(cls, sig: Any, args: list) -> None:
        from qtpy.QtCore import QEvent
        from qtpy.QtWidgets import QApplication

        if cls._EVENT_TYPE is None:
            cls._EVENT_TYPE = QEvent.registerEventType()
            cls._install_receiver()

        app = QApplication.instance()
        if app is None:
            return
        event = QEvent(QEvent.Type(cls._EVENT_TYPE))
        event.sig = sig  # type: ignore[attr-defined]
        event.args = args  # type: ignore[attr-defined]
        app.postEvent(cls._receiver, event)

    @classmethod
    def _install_receiver(cls) -> None:
        """Create a QObject and move it to the main thread to receive events."""
        from qtpy.QtCore import QObject
        from qtpy.QtWidgets import QApplication

        class _Receiver(QObject):
            def event(self, ev: Any) -> bool:
                if ev.type() == _EmitEvent._EVENT_TYPE:
                    try:
                        ev.sig.emit(*ev.args)
                    except Exception as e:
                        logger.warning("Error emitting signal: %s", e)
                    return True
                return super().event(ev)

        cls._receiver = _Receiver()
        # Ensure event delivery happens on the main thread even if
        # _install_receiver is called from a background thread.
        app = QApplication.instance()
        if app is not None:
            cls._receiver.moveToThread(app.thread())


# ------------------------------------------------------------------
# The client
# ------------------------------------------------------------------

class RemoteMMCore(CMMCorePlus):
    """Drop-in replacement for CMMCorePlus that proxies calls over HTTP.

    All method calls are forwarded to the server via POST /rpc.
    Signals are received via a WebSocket listener and emitted locally.
    MDA sequences run on the server — signals stream back via WebSocket.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:5600",
        *,
        timeout: float = 30.0,
        connect_signals: bool = True,
    ):
        # Initialize SWIG C++ base for safety, but skip CMMCorePlus.__init__
        pymmcore.CMMCore.__init__(self)

        self._url = url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.Client(base_url=self._url, timeout=timeout)

        # Local signal groups (use _events/_mda_runner so CMMCorePlus
        # properties .events and .mda work transparently)
        self._events = _CoreSignals()
        self._events.devicePropertyChanged = _DevicePropertySignal(self._events)
        self._mda_runner = _MDAController(self)

        # Private attrs that inherited CMMCorePlus methods rely on
        # (normally set by CMMCorePlus.__init__ which we skip)
        self._objective_regex = re.compile(
            r"(.+)?(nosepiece|obj(ective)?)(turret)?s?", re.IGNORECASE
        )
        self._channel_group_regex = re.compile(
            r"(chan{1,2}(el)?|filt(er)?)s?", re.IGNORECASE
        )
        self._last_sys_config: str | None = None
        self._last_config: tuple[str, str] = ("", "")
        self._last_xy_position: dict[str | None, tuple[float, float]] = {}

        # WebSocket signal listener
        self._signal_stop = threading.Event()
        self._signal_ws: Any = None  # active websocket, for clean shutdown
        self._signal_thread: threading.Thread | None = None
        self._flush_events: dict[str, threading.Event] = {}
        self._flush_counter = 0
        self._flush_lock = threading.Lock()
        if connect_signals:
            self._start_signal_listener()

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    # Mapping of server exception type names to Python exception classes.
    _EXCEPTION_TYPES: dict[str, type[Exception]] = {
        "FileNotFoundError": FileNotFoundError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "OSError": OSError,
        "IOError": IOError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "OverflowError": OverflowError,
        "ZeroDivisionError": ZeroDivisionError,
        "NotImplementedError": NotImplementedError,
        "PermissionError": PermissionError,
        "TimeoutError": TimeoutError,
    }

    @staticmethod
    def _raise_remote_error(data: dict) -> None:
        """Raise the appropriate exception from an RPC error response."""
        error_type = data.get("error_type", "RuntimeError")
        error_msg = data.get("error", "Unknown error")
        exc_cls = RemoteMMCore._EXCEPTION_TYPES.get(error_type, RuntimeError)
        raise exc_cls(error_msg)

    def _rpc(self, method: str, *args: Any, _timeout: Any = ..., **kwargs: Any) -> Any:
        """Call a method on the remote core (supports dotted paths).

        Pass ``_timeout=None`` to disable the per-request timeout (e.g. for
        long-running calls like ``mda.run`` whose duration is unbounded).
        """
        payload = {
            "method": method,
            "args": [encode(a) for a in args],
            "kwargs": {k: encode(v) for k, v in kwargs.items()},
        }
        timeout = self._timeout if _timeout is ... else _timeout
        resp = self._http.post("/rpc", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return decode(data.get("value"))
        self._raise_remote_error(data)

    def _rpc_getattr(self, attr: str) -> Any:
        """Read an attribute from the remote core."""
        resp = self._http.post("/rpc", json={"action": "getattr", "attr": attr})
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return decode(data.get("value"))
        self._raise_remote_error(data)

    def _rpc_setattr(self, attr: str, value: Any) -> None:
        """Set an attribute on the remote core."""
        resp = self._http.post(
            "/rpc", json={"action": "setattr", "attr": attr, "value": encode(value)}
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            self._raise_remote_error(data)

    def __getattribute__(self, name: str) -> Any:
        # Intercept pymmcore.CMMCore methods and route them through RPC
        # instead of calling the local C++ core.
        if name in _RPC_FORWARD_METHODS:
            _rpc = object.__getattribute__(self, "_rpc")

            def _proxy(*args: Any, **kwargs: Any) -> Any:
                return _rpc(name, *args, **kwargs)

            return _proxy
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        # Python internals — never proxy
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Single-underscore private attrs — fetch from remote
        if name.startswith("_"):
            return self._rpc_getattr(name)

        # Public attrs — return a proxy callable
        def _proxy(*args: Any, **kwargs: Any) -> Any:
            return self._rpc(name, *args, **kwargs)

        return _proxy

    # ------------------------------------------------------------------
    # Signal listener
    # ------------------------------------------------------------------

    def _start_signal_listener(self) -> None:
        """Start the background WebSocket signal listener."""
        self._signal_thread = threading.Thread(
            target=self._signal_listener_loop,
            name="pymmcore-proxy-signals",
            daemon=True,
        )
        self._signal_thread.start()

    def _signal_listener_loop(self) -> None:
        """Background loop: connect to /signals WS, dispatch to local signals."""
        ws_url = self._url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/signals"

        while not self._signal_stop.is_set():
            try:
                self._listen_ws(ws_url)
            except Exception as e:
                if self._signal_stop.is_set():
                    return
                logger.debug("Signal connection lost (%s), reconnecting...", e)
                # Back off before reconnecting
                self._signal_stop.wait(timeout=1.0)

    def _listen_ws(self, ws_url: str) -> None:
        """Connect and listen until disconnected."""
        from websockets.sync.client import connect

        with connect(ws_url, ping_interval=None) as ws:
            self._signal_ws = ws
            logger.debug("Signal listener connected to %s", ws_url)
            try:
                while not self._signal_stop.is_set():
                    try:
                        raw = ws.recv()
                    except Exception:
                        break
                    try:
                        msg = json.loads(raw)
                        self._dispatch_signal(msg)
                    except Exception as e:
                        logger.warning("Failed to dispatch signal: %s", e)
            finally:
                self._signal_ws = None

    def _dispatch_signal(self, msg: dict) -> None:
        """Route an incoming signal message to the correct local signal."""
        group = msg.get("group", "")
        signal_name = msg.get("signal", "")
        raw_args = msg.get("args", [])

        # Handle internal markers
        if group == "_internal":
            if signal_name == "_flush":
                flush_id = raw_args[0] if raw_args else ""
                ev = self._flush_events.get(flush_id)
                if ev is not None:
                    ev.set()
                return
            if signal_name == "_warning":
                category_name = raw_args[0] if raw_args else "UserWarning"
                message = raw_args[1] if len(raw_args) > 1 else ""
                category = _WARNING_CATEGORIES.get(category_name, UserWarning)
                warnings.warn(message, category, stacklevel=2)
                return
            if signal_name == "_log":
                levelno = raw_args[0] if raw_args else logging.INFO
                message = raw_args[1] if len(raw_args) > 1 else ""
                filename = raw_args[2] if len(raw_args) > 2 else ""
                lineno = raw_args[3] if len(raw_args) > 3 else 0
                record = logging.LogRecord(
                    name="pymmcore-plus",
                    level=levelno,
                    pathname=filename,
                    lineno=lineno,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
                record._proxy_forwarded = True  # prevent re-forwarding loop
                pmm_logger = logging.getLogger("pymmcore-plus")
                for h in pmm_logger.handlers:
                    if not isinstance(h, logging.handlers.RotatingFileHandler):
                        h.handle(record)
                return
            return

        key = f"{group}.{signal_name}"
        mapping = _SIGNAL_MAP.get(key)
        if mapping is None:
            logger.debug("Unknown signal: %s", key)
            return

        group_attr, sig_name = mapping
        # Navigate to the signal group
        obj = self
        for part in group_attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return

        sig = getattr(obj, sig_name, None)
        if sig is None:
            return

        decoded_args = [decode(a) for a in raw_args]
        self._emit_signal(sig, decoded_args, key)

    @staticmethod
    def _emit_signal(sig: Any, args: list, key: str) -> None:
        """Emit a signal, marshaling to the Qt main thread if available.

        Signals arrive on the WebSocket background thread.  Qt widgets
        (timers, UI updates) require callbacks to run on the main thread.
        We use a Qt signal bridge to reliably post the emission across
        threads (QTimer.singleShot doesn't work from non-Qt threads).
        """
        try:
            from qtpy.QtCore import QThread
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None and QThread.currentThread() is not app.thread():
                _EmitEvent.post(sig, args)
                return
        except ImportError:
            pass

        try:
            sig.emit(*args)
        except Exception as e:
            logger.warning("Error emitting %s: %s", key, e)

    def iterProperties(
        self,
        property_type: Any = None,
        property_name_pattern: Any = None,
        *,
        device_type: Any = None,
        device_label: Any = None,
        has_limits: bool | None = None,
        is_read_only: bool | None = None,
        is_sequenceable: bool | None = None,
        as_object: bool = True,
    ) -> Any:
        """Iterate over currently loaded (device_label, property_name) pairs.

        Mirrors ``CMMCorePlus.iterProperties``.
        """
        if property_name_pattern:
            if isinstance(property_name_pattern, str):
                ptrn = re.compile(property_name_pattern, re.IGNORECASE)
            else:
                ptrn = property_name_pattern
        else:
            ptrn = None

        if property_type is None:
            property_types: set[int] = set()
        elif isinstance(property_type, int):
            property_types = {property_type}
        else:
            property_types = set(property_type)

        for dev in self.iterDevices(device_type, device_label, as_object=False):
            for prop in self._rpc("getDevicePropertyNames", dev):
                if ptrn and not ptrn.search(prop):
                    continue
                if (
                    property_type is not None
                    and self._rpc("getPropertyType", dev, prop) not in property_types
                ):
                    continue
                if (
                    has_limits is not None
                    and self._rpc("hasPropertyLimits", dev, prop) != has_limits
                ):
                    continue
                if (
                    is_read_only is not None
                    and self._rpc("isPropertyReadOnly", dev, prop) != is_read_only
                ):
                    continue
                if (
                    is_sequenceable is not None
                    and self._rpc("isPropertySequenceable", dev, prop) != is_sequenceable
                ):
                    continue

                if as_object:
                    yield self.getPropertyObject(dev, prop)
                else:
                    yield (dev, prop)

    def iterDeviceAdapters(
        self,
        adapter_pattern: Any = None,
        *,
        as_object: bool = True,
    ) -> Any:
        """Iterate over all available device adapters.

        Mirrors ``CMMCorePlus.iterDeviceAdapters``.
        """
        adapters: list[str] = list(self._rpc("getDeviceAdapterNames"))

        if adapter_pattern:
            if isinstance(adapter_pattern, str):
                ptrn = re.compile(adapter_pattern, re.IGNORECASE)
            else:
                ptrn = adapter_pattern
            adapters = [d for d in adapters if ptrn.search(d)]

        for adapter in adapters:
            if as_object:
                yield self.getAdapterObject(adapter)
            else:
                yield adapter

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush_signals(self, timeout: float = 5.0) -> None:
        """Block until all pending signals have been delivered.

        Sends a flush request to the server, which responds with a marker
        over the signal WebSocket.  Since WS messages are ordered, the
        marker arrives after all previously queued signals.

        The flush_id is generated client-side and registered BEFORE the
        HTTP request to avoid a race where the WebSocket marker arrives
        before the Event is registered.
        """
        with self._flush_lock:
            self._flush_counter += 1
            flush_id = f"f{self._flush_counter}"
        ev = threading.Event()
        self._flush_events[flush_id] = ev
        try:
            resp = self._http.get(f"/signals/flush?id={flush_id}")
            resp.raise_for_status()
        except Exception:
            self._flush_events.pop(flush_id, None)
            return
        ev.wait(timeout=timeout)
        self._flush_events.pop(flush_id, None)

    def health(self) -> dict:
        """Check server health."""
        resp = self._http.get("/health")
        return resp.json()

    def close(self) -> None:
        """Disconnect from the server."""
        self._signal_stop.set()
        # Close the WS to unblock recv() immediately
        ws = self._signal_ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._signal_thread is not None:
            self._signal_thread.join(timeout=3.0)
        self._http.close()

    def __enter__(self) -> RemoteMMCore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RemoteMMCore({self._url!r})"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Compute CMMCore methods that need RPC forwarding
# ------------------------------------------------------------------
# CMMCorePlus inherits ~200 C++ methods from pymmcore.CMMCore.  Without
# interception these would call the *local* C++ core.  __getattribute__
# checks this set and routes matching calls through RPC instead.
# Methods explicitly defined on RemoteMMCore are excluded.
_RPC_FORWARD_METHODS = (
    _CMMCORE_METHODS | _CMMCORE_PLUS_FORWARD
) - frozenset(RemoteMMCore.__dict__)
