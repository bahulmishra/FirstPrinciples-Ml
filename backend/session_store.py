"""
Simple in-memory session store keyed by UUID.
Stores raw DataFrame, processed arrays, trained model weights, and results.
"""
import uuid
from typing import Any, Dict, Optional


_STORE: Dict[str, Dict[str, Any]] = {}


def create_session() -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())
    _STORE[sid] = {}
    return sid


def get_session(sid: str) -> Optional[Dict[str, Any]]:
    return _STORE.get(sid)


def set_key(sid: str, key: str, value: Any) -> None:
    if sid not in _STORE:
        _STORE[sid] = {}
    _STORE[sid][key] = value


def get_key(sid: str, key: str, default: Any = None) -> Any:
    return _STORE.get(sid, {}).get(key, default)


def delete_session(sid: str) -> None:
    _STORE.pop(sid, None)
