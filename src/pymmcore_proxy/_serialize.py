"""JSON-safe serialization for pymmcore-proxy wire format.

Handles: numpy arrays, tuples, bytes, MDAEvent, enums, and basic types.
Tagged dicts with "__type__" key distinguish special types from plain dicts.
"""

from __future__ import annotations

import base64
import enum
from typing import Any

import numpy as np


def encode(obj: Any) -> Any:
    """Encode a Python object into a JSON-safe representation."""
    # Enums before basic types — IntEnum is also int, would be caught below
    if isinstance(obj, enum.Enum):
        return {
            "__type__": "enum",
            "class": type(obj).__qualname__,
            "module": type(obj).__module__,
            "value": obj.value,
        }

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": base64.b64encode(obj.tobytes()).decode("ascii"),
        }

    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [encode(x) for x in obj]}

    if isinstance(obj, list):
        return [encode(x) for x in obj]

    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}

    if isinstance(obj, bytes):
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}

    # useq MDAEvent and other pydantic models
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
        # MDASequence.uid is excluded from model_dump by default, but we
        # need it so the client can match frameReady events to the sequence
        # that was announced via sequenceStarted.
        if hasattr(obj, "uid") and "uid" not in data:
            data["uid"] = str(obj.uid)
        # Also preserve uid in the nested sequence inside MDAEvent.
        seq = getattr(obj, "sequence", None)
        if seq is not None and hasattr(seq, "uid"):
            nested = data.get("sequence")
            if isinstance(nested, dict) and "uid" not in nested:
                nested["uid"] = str(seq.uid)
        return {
            "__type__": "model",
            "class": type(obj).__qualname__,
            "module": type(obj).__module__,
            "data": data,
        }

    # pymmcore-plus Metadata → tagged dict of key-value pairs
    cls_name = type(obj).__name__
    cls_module = getattr(type(obj), "__module__", "")
    if cls_name == "Metadata" and hasattr(obj, "GetKeys"):
        try:
            return {"__type__": "metadata", "data": dict(obj)}
        except Exception:
            pass

    # pymmcore-plus Configuration → tagged list of (device, prop, value) triples
    if cls_name == "Configuration" and hasattr(obj, "getSetting"):
        try:
            items = []
            for i in range(obj.size()):
                ps = obj.getSetting(i)
                items.append(
                    [ps.getDeviceLabel(), ps.getPropertyName(), ps.getPropertyValue()]
                )
            return {
                "__type__": "configuration",
                "plus": "pymmcore_plus" in cls_module,
                "data": items,
            }
        except Exception:
            pass

    # Generic iterable fallback
    if not isinstance(obj, (str, bytes)) and hasattr(obj, "__iter__"):
        try:
            return [encode(x) for x in obj]
        except Exception:
            pass

    # Fallback: try to convert to string
    return str(obj)


def decode(obj: Any) -> Any:
    """Decode a JSON representation back into Python objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, list):
        return [decode(x) for x in obj]

    if isinstance(obj, dict):
        t = obj.get("__type__")

        if t == "ndarray":
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()

        if t == "tuple":
            return tuple(decode(x) for x in obj["items"])

        if t == "bytes":
            return base64.b64decode(obj["data"])

        if t == "model":
            return _reconstruct_model(obj)

        if t == "enum":
            return _reconstruct_enum(obj)

        if t == "metadata":
            return _reconstruct_metadata(obj)

        if t == "configuration":
            return _reconstruct_configuration(obj)

        # Plain dict
        return {k: decode(v) for k, v in obj.items()}

    return obj


def _reconstruct_enum(obj: dict) -> Any:
    """Reconstruct an enum from serialized form, falling back to raw value."""
    value = obj["value"]
    module = obj.get("module", "")
    cls_name = obj.get("class", "")
    if module and cls_name:
        try:
            import importlib

            mod = importlib.import_module(module)
            cls = getattr(mod, cls_name)
            return cls(value)
        except Exception:
            pass
    return value


def _reconstruct_metadata(obj: dict) -> Any:
    """Reconstruct a pymmcore-plus Metadata from serialized key-value pairs."""
    data = obj.get("data", {})
    try:
        from pymmcore_plus.core._metadata import Metadata

        return Metadata(data)
    except Exception:
        return data


def _reconstruct_configuration(obj: dict) -> Any:
    """Reconstruct a pymmcore Configuration from serialized settings."""
    data = obj.get("data", [])
    is_plus = obj.get("plus", True)
    try:
        import pymmcore as _pymmcore

        if is_plus:
            from pymmcore_plus.core._config import Configuration

            cfg = Configuration()
        else:
            cfg = _pymmcore.Configuration()
        for device, prop, value in data:
            cfg.addSetting(_pymmcore.PropertySetting(device, prop, value))
        return cfg
    except Exception:
        return data


def _reconstruct_model(obj: dict) -> Any:
    """Reconstruct a pydantic model from serialized form."""
    module = obj.get("module", "")
    cls_name = obj.get("class", "")
    data = obj.get("data", {})

    # Try to import and reconstruct (check MDASequence before MDAEvent)
    # GeneratorMDASequence is pymmcore-plus's internal wrapper for generator-based
    # MDA runs; it carries the same fields as MDASequence so reconstruct as one.
    if cls_name in ("MDASequence", "GeneratorMDASequence"):
        try:
            from useq import MDASequence

            return _make_sequence(MDASequence, data)
        except Exception:
            pass
    if "useq" in module or cls_name == "MDAEvent":
        try:
            from useq import MDAEvent, MDASequence

            event = MDAEvent(**data)
            # Restore uid on the nested sequence if present
            seq_data = data.get("sequence")
            if event.sequence is not None and isinstance(seq_data, dict):
                _restore_sequence_uid(event.sequence, seq_data)
            return event
        except Exception:
            pass

    # Fallback: return the raw dict
    return data


def _restore_sequence_uid(seq: Any, data: dict) -> None:
    """Set _uid on an MDASequence from serialized data."""
    uid_str = data.get("uid")
    if uid_str:
        from uuid import UUID

        seq._uid = UUID(uid_str) if isinstance(uid_str, str) else uid_str


def _make_sequence(cls: type, data: dict) -> Any:
    """Create an MDASequence and restore its uid from serialized data."""
    seq = cls(**data)
    _restore_sequence_uid(seq, data)
    return seq
