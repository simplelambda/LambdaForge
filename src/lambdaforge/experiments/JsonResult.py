"""Base object for immutable mapping-shaped JSON results."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4


class JsonResult(dict[str, Any], ABC):
    """Expose an immutable typed result through dict and JSON contracts."""

    def _freeze_mapping(self, payload: dict[str, Any]) -> None:
        """Initialize the native dict storage and reject later mutations."""
        dict.__init__(self, payload)
        object.__setattr__(self, "_result_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow construction-time attributes and freeze them with the mapping."""
        if getattr(self, "_result_frozen", False):
            raise AttributeError(f"{type(self).__name__} is immutable.")
        object.__setattr__(self, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """Reject item assignment on immutable results."""
        del key, value
        raise TypeError(f"{type(self).__name__} does not support item assignment.")

    def __delitem__(self, key: str) -> None:
        """Reject item deletion on immutable results."""
        del key
        raise TypeError(f"{type(self).__name__} does not support item deletion.")

    def clear(self) -> None:
        """Reject destructive mapping operations."""
        raise TypeError(f"{type(self).__name__} is immutable.")

    def pop(self, key: str, default: Any = None) -> Any:
        """Reject destructive mapping operations."""
        del key, default
        raise TypeError(f"{type(self).__name__} is immutable.")

    def popitem(self) -> tuple[str, Any]:
        """Reject destructive mapping operations."""
        raise TypeError(f"{type(self).__name__} is immutable.")

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Reject mutating mapping operations."""
        del key, default
        raise TypeError(f"{type(self).__name__} is immutable.")

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Reject mutating mapping operations."""
        del args, kwargs
        raise TypeError(f"{type(self).__name__} is immutable.")

    def __ior__(self, other: Any) -> JsonResult:  # type: ignore[override,misc]
        """Reject in-place mapping union."""
        del other
        raise TypeError(f"{type(self).__name__} is immutable.")

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""

    def write_json(self, path: str | Path) -> Path:
        """Atomically write UTF-8 JSON and return its artifact path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
