"""ProxyServer — expose a pymmcore-plus core over HTTP + WebSocket.

Routes:
    POST /rpc              — call any method on the core
    POST /mda/exec_event   — execute a single MDA event (configure + snap)
    GET  /health           — liveness check
    WS   /signals          — stream core signals to connected clients
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ._serialize import decode, encode

logger = logging.getLogger("pymmcore_proxy.server")

# Signals to forward from core.events
_CORE_SIGNALS = [
    "propertiesChanged",
    "propertyChanged",
    "channelGroupChanged",
    "configGroupChanged",
    "systemConfigurationLoaded",
    "pixelSizeChanged",
    "pixelSizeAffineChanged",
    "stagePositionChanged",
    "XYStagePositionChanged",
    "exposureChanged",
    "SLMExposureChanged",
    "configSet",
    "imageSnapped",
    "mdaEngineRegistered",
    "continuousSequenceAcquisitionStarting",
    "continuousSequenceAcquisitionStarted",
    "sequenceAcquisitionStarting",
    "sequenceAcquisitionStarted",
    "sequenceAcquisitionStopped",
    "autoShutterSet",
    "shutterOpenChanged",
    "configGroupDeleted",
    "configDeleted",
    "configDefined",
    "roiSet",
]

# Signals to forward from core.mda.events
_MDA_SIGNALS = [
    "frameReady",
    "sequenceStarted",
    "sequenceFinished",
    "sequenceCanceled",
    "sequencePauseToggled",
    "awaitingEvent",
    "eventStarted",
]


class _LogForwarder(logging.Handler):
    """Logging handler that broadcasts log records to WebSocket clients."""

    def __init__(self, broadcast_fn):
        super().__init__(level=logging.INFO)
        self._broadcast = broadcast_fn

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_proxy_forwarded", False):
            return  # already re-emitted by a client in the same process; skip
        try:
            self._broadcast("_internal", "_log", (
                record.levelno,
                record.getMessage(),
                record.filename,
                record.lineno,
            ))
        except Exception:
            pass


class _WSEventIterator:
    """Iterator that yields MDAEvents from a queue, fed by a WebSocket.

    Used by the /mda/stream endpoint.  ``__next__`` blocks until an event
    is available.  Cancellation is handled by putting ``None`` (sentinel)
    into the queue from the RPC handler when ``mda.cancel`` is called.
    """

    def __init__(self, event_queue: queue.Queue):
        self._queue = event_queue

    def __iter__(self) -> _WSEventIterator:
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item


class ProxyServer:
    """Wraps a CMMCorePlus/UniMMCore and serves it over HTTP + WebSocket."""

    def __init__(self, core, host: str = "127.0.0.1", port: int = 5600,
                 extra_routes: list | None = None):
        self.core = core
        self.host = host
        self.port = port
        self._ws_clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._signals_connected = False
        self._stream_queue: queue.Queue | None = None
        self._stream_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._start_time = time.time()
        self._command_count = 0
        self._recent_commands: list[dict] = []
        # Queue-based signal sender: guarantees ordered, reliable delivery.
        self._signal_queue: asyncio.Queue | None = None
        self._sender_task: asyncio.Task | None = None

        @asynccontextmanager
        async def _lifespan(app):
            self._on_startup()
            yield

        self.app = Starlette(
            routes=[
                Route("/rpc", self._handle_rpc, methods=["POST"]),
                Route("/mda/exec_event", self._handle_mda_event, methods=["POST"]),
                Route("/health", self._handle_health, methods=["GET"]),
                Route("/info", self._handle_info, methods=["GET"]),
                Route("/signals/flush", self._handle_flush, methods=["GET"]),
                Route("/stats", self._handle_stats, methods=["GET"]),
                WebSocketRoute("/signals", self._handle_signals),
                WebSocketRoute("/mda/stream", self._handle_mda_stream),
                *(extra_routes or []),
            ],
            lifespan=_lifespan,
        )

    def _on_startup(self):
        self._loop = asyncio.get_running_loop()
        self._signal_queue = asyncio.Queue()
        self._sender_task = asyncio.create_task(self._run_signal_sender())
        self._connect_signal_forwarding()
        self._install_warning_hook()
        self._install_log_forwarding()
        self._signals_connected = True

    def _install_warning_hook(self):
        """Hook warnings.showwarning to broadcast warnings to WS clients.

        The original showwarning is still called, so same-process consumers
        (like pytest.warns) continue to work.  Remote clients receive the
        warning as a ``_internal._warning`` WebSocket signal.
        """
        original = warnings.showwarning

        def _hook(message, category, filename, lineno, file=None, line=None):
            # Forward warnings from pymmcore code, not from third-party libraries
            if "pymmcore" in str(filename):
                self._broadcast_signal(
                    "_internal", "_warning",
                    (category.__name__, str(message)),
                )
            # Call original so same-process consumers still see it
            original(message, category, filename, lineno, file, line)

        warnings.showwarning = _hook

    def _install_log_forwarding(self):
        """Forward pymmcore-plus log records to WebSocket clients.

        The MDA runner logs each event at INFO level.  By attaching a
        handler to the ``"pymmcore-plus"`` logger we relay those records
        to the client, where they are re-emitted through the local logger
        so the user sees the same output as with a local core.
        """
        pmm_logger = logging.getLogger("pymmcore-plus")
        pmm_logger.addHandler(_LogForwarder(self._broadcast_signal))

    def _warn_if_qt_signals(self):
        """Detect Qt signals on the core and warn about silent cross-thread drops.

        When the host process has a ``QApplication`` running (e.g. napari),
        ``CMMCorePlus`` auto-selects ``QCoreSignaler``.  We connect to those
        signals from the asyncio event-loop thread, and PyQt silently drops
        cross-thread connections — the server runs but never broadcasts
        hardware signals.  Warn loudly with the fix.
        """
        events = getattr(self.core, "events", None)
        if events is None:
            return
        if type(events).__name__ in ("QCoreSignaler", "QMDASignaler"):
            logger.warning(
                "ProxyServer: core uses Qt signals (%s). Cross-thread connect "
                "from asyncio drops silently — clients will not receive any "
                "hardware events. Construct the core with "
                "PYMM_SIGNALS_BACKEND=psygnal in the environment, or set "
                "os.environ['PYMM_SIGNALS_BACKEND']='psygnal' before "
                "CMMCorePlus().",
                type(events).__name__,
            )

    def _connect_signal_forwarding(self):
        """Connect to core signals and forward them over WebSocket."""
        self._warn_if_qt_signals()
        events = getattr(self.core, "events", None)
        if events:
            for name in _CORE_SIGNALS:
                sig = getattr(events, name, None)
                if sig is not None:
                    sig.connect(self._make_signal_handler("events", name))

        mda = getattr(self.core, "mda", None)
        if mda:
            mda_events = getattr(mda, "events", None)
            if mda_events:
                for name in _MDA_SIGNALS:
                    sig = getattr(mda_events, name, None)
                    if sig is not None:
                        sig.connect(self._make_signal_handler("mda.events", name))

    def _make_signal_handler(self, group: str, name: str):
        def handler(*args):
            self._broadcast_signal(group, name, args)
        return handler

    async def _run_signal_sender(self) -> None:
        """Drain the signal queue and send each message to all WS clients.

        Using a single persistent task instead of per-signal
        run_coroutine_threadsafe guarantees:
        - Messages are sent in emission order (FIFO queue).
        - Each send is properly awaited (no silent Future discard).
        - No concurrent sends on the same WebSocket.
        """
        while True:
            try:
                msg = await self._signal_queue.get()
                if msg is None:       # shutdown sentinel
                    return
                for ws in list(self._ws_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        self._ws_clients.discard(ws)
            except Exception as e:
                logger.warning("Signal sender error: %s", e)

    def _broadcast_signal(self, group: str, name: str, args: tuple):
        if not self._ws_clients or self._loop is None:
            return
        try:
            msg = json.dumps({
                "group": group,
                "signal": name,
                "args": [encode(a) for a in args],
            })
        except Exception as e:
            logger.warning(f"Failed to serialize signal {group}.{name}: {e}")
            return
        if self._signal_queue is not None:
            # Thread-safe enqueue: call_soon_threadsafe schedules put_nowait
            # in the event loop, preserving emission order from any thread.
            self._loop.call_soon_threadsafe(self._signal_queue.put_nowait, msg)
        else:
            # Fallback before startup (should not normally occur).
            for ws in list(self._ws_clients):
                try:
                    asyncio.run_coroutine_threadsafe(ws.send_text(msg), self._loop)
                except Exception:
                    self._ws_clients.discard(ws)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _handle_info(self, request: Request) -> JSONResponse:
        return JSONResponse({"core_type": type(self.core).__name__})

    async def _handle_flush(self, request: Request) -> JSONResponse:
        """Flush pending signals to all connected WebSocket clients.

        Enqueues a ``_flush`` marker through the same signal queue so
        that it is guaranteed to arrive after all previously enqueued
        signals.  The client knows all prior signals have been delivered
        when it receives the marker.
        """
        flush_id = request.query_params.get("id", str(time.monotonic()))
        msg = json.dumps({
            "group": "_internal",
            "signal": "_flush",
            "args": [flush_id],
        })
        if self._signal_queue is not None:
            self._signal_queue.put_nowait(msg)
        else:
            await asyncio.sleep(0)
            for ws in list(self._ws_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self._ws_clients.discard(ws)
        return JSONResponse({"ok": True, "id": flush_id})

    async def _handle_stats(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "uptime_s": time.time() - self._start_time,
            "command_count": self._command_count,
            "recent_commands": self._recent_commands[-20:],
        })

    def _record_command(self, method: str):
        with self._stats_lock:
            self._command_count += 1
            self._recent_commands.append({"t": time.time(), "method": method})
            if len(self._recent_commands) > 50:
                self._recent_commands = self._recent_commands[-50:]

    def _resolve_dotted(self, path: str) -> Any:
        """Resolve a dotted path like 'mda.engine.setup_event' on the core."""
        obj = self.core
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    async def _handle_rpc(self, request: Request) -> JSONResponse:
        body = await request.json()
        action = body.get("action", "call")

        try:
            if action == "getattr":
                attr = body.get("attr", "")
                self._record_command(f"getattr:{attr}")
                result = self._resolve_dotted(attr)
                return JSONResponse({"ok": True, "value": encode(result)})

            if action == "setattr":
                attr = body.get("attr", "")
                value = decode(body.get("value"))
                self._record_command(f"setattr:{attr}")
                # Split into parent path + final attr name
                parts = attr.rsplit(".", 1)
                if len(parts) == 2:
                    parent = self._resolve_dotted(parts[0])
                    setattr(parent, parts[1], value)
                else:
                    setattr(self.core, attr, value)
                return JSONResponse({"ok": True, "value": None})

            # Default: method call
            method = body.get("method", "")
            args = [decode(a) for a in body.get("args", [])]
            kwargs = {k: decode(v) for k, v in body.get("kwargs", {}).items()}
            self._record_command(method)

            func = self._resolve_dotted(method)
            result = await asyncio.to_thread(func, *args, **kwargs)

            # Unblock streaming iterator immediately on cancel.
            # Snapshot _stream_queue without the lock to avoid deadlock
            # (the streaming handler holds _stream_lock while waiting).
            if method == "mda.cancel":
                q = self._stream_queue
                if q is not None:
                    q.put(None)

            return JSONResponse({"ok": True, "value": encode(result)})
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": str(e), "error_type": type(e).__name__},
                status_code=200,  # keep 200 — error is in the payload
            )

    async def _handle_mda_event(self, request: Request) -> JSONResponse:
        """Execute a single MDA event: configure hardware + snap."""
        body = await request.json()
        event_data = body.get("event", {})
        self._record_command("mda.exec_event")

        try:
            result = await asyncio.to_thread(self._execute_event, event_data)
            return JSONResponse({"ok": True, "value": encode(result)})
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": str(e), "error_type": type(e).__name__},
            )

    def _execute_event(self, event_data: dict) -> dict[str, Any]:
        """Configure hardware per MDAEvent spec, snap, return image + metadata."""
        core = self.core

        # Channel
        channel = event_data.get("channel")
        if channel and isinstance(channel, dict):
            group = channel.get("group", "")
            config = channel.get("config", "")
            if group and config:
                core.setConfig(group, config)

        # Exposure
        exposure = event_data.get("exposure")
        if exposure is not None:
            core.setExposure(float(exposure))

        # XY position
        x = event_data.get("x_pos")
        y = event_data.get("y_pos")
        if x is not None and y is not None:
            core.setXYPosition(float(x), float(y))
            core.waitForDevice(core.getXYStageDevice())

        # Z position
        z = event_data.get("z_pos")
        if z is not None:
            try:
                focus_dev = core.getFocusDevice()
                core.setPosition(focus_dev, float(z))
                core.waitForDevice(focus_dev)
            except Exception:
                pass

        # Snap
        core.snapImage()
        image = core.getImage()

        return {
            "image": image,
            "metadata": {
                "time_utc": time.time(),
                "index": event_data.get("index", {}),
            },
        }

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    async def _handle_mda_stream(self, websocket: WebSocket) -> None:
        """Stream MDA events from a client iterable.

        Protocol:
            1. Client sends ``{"action": "start"}``
            2. Client sends ``{"event": <encoded MDAEvent>}`` for each event
            3. Client sends ``{"action": "stop"}`` when the iterable is exhausted
            4. Server sends ``{"action": "done"}`` (with optional ``"error"``)
        """
        await websocket.accept()
        self._record_command("mda.stream")

        raw = await websocket.receive_text()
        msg = json.loads(raw)
        if msg.get("action") != "start":
            await websocket.close()
            return

        # Only one streaming MDA at a time
        acquired = self._stream_lock.acquire(blocking=False)
        if not acquired:
            await websocket.send_text(json.dumps({
                "action": "done",
                "error": "Another streaming MDA is already running",
            }))
            await websocket.close()
            return

        event_queue: queue.Queue = queue.Queue()
        iterator = _WSEventIterator(event_queue)
        self._stream_queue = event_queue
        mda_error: list[Exception | None] = [None]
        mda_done = asyncio.Event()

        async def _run_mda() -> None:
            try:
                await asyncio.to_thread(self.core.mda.run, iterator)
            except Exception as e:
                mda_error[0] = e
            finally:
                mda_done.set()

        mda_task = asyncio.create_task(_run_mda())

        # Receive events from the client, stop when MDA finishes or client
        # sends "stop" (iterator exhausted).
        try:
            while not mda_done.is_set():
                # Wait for either a WS message or MDA completion — whichever
                # comes first — so cancel unblocks immediately.
                recv_task = asyncio.ensure_future(websocket.receive_text())
                done_task = asyncio.ensure_future(mda_done.wait())
                finished, _ = await asyncio.wait(
                    {recv_task, done_task}, return_when=asyncio.FIRST_COMPLETED,
                )
                if done_task in finished:
                    recv_task.cancel()
                    break
                done_task.cancel()
                raw = recv_task.result()
                msg = json.loads(raw)
                if msg.get("action") == "stop":
                    event_queue.put(None)
                    break
                event_data = msg.get("event")
                if event_data is not None:
                    event_queue.put(decode(event_data))
        except WebSocketDisconnect:
            event_queue.put(None)

        await mda_task
        self._stream_queue = None
        self._stream_lock.release()

        result: dict[str, Any] = {"action": "done"}
        if mda_error[0]:
            result["error"] = str(mda_error[0])
        try:
            await websocket.send_text(json.dumps(result))
        except Exception:
            pass

    async def _handle_signals(self, websocket: WebSocket):
        # Lazy init: if lifespan didn't run (uvicorn lifespan="off" or startup
        # exception), set up signals here on first WS connection instead.
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        if self._signal_queue is None:
            self._signal_queue = asyncio.Queue()
            self._sender_task = asyncio.create_task(self._run_signal_sender())
        if not self._signals_connected:
            self._connect_signal_forwarding()
            self._install_warning_hook()
            self._install_log_forwarding()
            self._signals_connected = True
        await websocket.accept()
        self._ws_clients.add(websocket)
        try:
            while True:
                # Keep alive — client doesn't send messages (for now)
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self._ws_clients.discard(websocket)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Start the server (blocking)."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port, ws="wsproto")


def serve(core, host: str = "127.0.0.1", port: int = 5600):
    """Convenience: create and run a ProxyServer."""
    server = ProxyServer(core, host=host, port=port)
    server.run()
