"""Small immutable JSON containers shared by durable infrastructure models."""

from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, SupportsIndex
from uuid import uuid4


class FrozenJsonMapping(dict[str, Any]):
    """Retain JSON object encoding while rejecting ordinary mutation."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        dict.__init__(
            self,
            {str(key): self.freeze_value(value) for key, value in (values or {}).items()},
        )

    @classmethod
    def freeze_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return FrozenJsonList(value)
        return copy.deepcopy(value)

    def __setitem__(self, key: str, value: Any) -> None:
        del key, value
        self._reject()

    def __delitem__(self, key: str) -> None:
        del key
        self._reject()

    def clear(self) -> None:
        self._reject()

    def pop(self, key: str, default: Any = None) -> Any:
        del key, default
        self._reject()

    def popitem(self) -> tuple[str, Any]:
        self._reject()

    def setdefault(self, key: str, default: Any = None) -> Any:
        del key, default
        self._reject()

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._reject()

    def __ior__(self, other: Any) -> FrozenJsonMapping:  # type: ignore[override,misc]
        del other
        self._reject()

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {key: copy.deepcopy(value, memo) for key, value in dict.items(self)}

    def __reduce__(self) -> tuple[type[FrozenJsonMapping], tuple[dict[str, Any]]]:
        return (type(self), (dict(self),))

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("FrozenJsonMapping is immutable.")


class FrozenJsonList(list[Any]):
    """Retain JSON list encoding while rejecting ordinary mutation."""

    def __init__(self, values: Iterable[Any] = ()) -> None:
        list.__init__(self, (FrozenJsonMapping.freeze_value(value) for value in values))

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        del key, value
        self._reject()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        del key
        self._reject()

    def append(self, value: Any) -> None:
        del value
        self._reject()

    def clear(self) -> None:
        self._reject()

    def extend(self, values: Iterable[Any]) -> None:
        del values
        self._reject()

    def insert(self, index: SupportsIndex, value: Any) -> None:
        del index, value
        self._reject()

    def pop(self, index: SupportsIndex = -1) -> Any:
        del index
        self._reject()

    def remove(self, value: Any) -> None:
        del value
        self._reject()

    def reverse(self) -> None:
        self._reject()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._reject()

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(value, memo) for value in list.__iter__(self)]

    def __reduce__(self) -> tuple[type[FrozenJsonList], tuple[list[Any]]]:
        """Rebuild through the constructor instead of pickle's mutating list protocol."""
        return (type(self), (list(self),))

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("FrozenJsonList is immutable.")


class JsonResult(dict[str, Any], ABC):
    """Base object for immutable mapping-shaped persisted values."""

    def _freeze_mapping(self, payload: dict[str, Any]) -> None:
        dict.__init__(self, payload)
        object.__setattr__(self, "_result_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_result_frozen", False):
            raise AttributeError(f"{type(self).__name__} is immutable.")
        object.__setattr__(self, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        del key, value
        raise TypeError(f"{type(self).__name__} is immutable.")

    def __delitem__(self, key: str) -> None:
        del key
        raise TypeError(f"{type(self).__name__} is immutable.")

    clear = FrozenJsonMapping.clear
    pop = FrozenJsonMapping.pop
    popitem = FrozenJsonMapping.popitem
    setdefault = FrozenJsonMapping.setdefault
    update = FrozenJsonMapping.update

    def __ior__(self, other: Any) -> JsonResult:  # type: ignore[override,misc]
        del other
        self._reject()

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("JsonResult is immutable.")

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
