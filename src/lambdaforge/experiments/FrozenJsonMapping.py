"""Read-only dictionary container for nested JSON result values."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn


class FrozenJsonMapping(dict[str, Any]):
    """Retain JSON object encoding while rejecting every ordinary mutation."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        dict.__init__(
            self,
            {str(key): self.freeze_value(value) for key, value in (values or {}).items()},
        )

    @classmethod
    def freeze_value(cls, value: Any) -> Any:
        """Recursively freeze mapping and sequence containers."""
        from lambdaforge.experiments.FrozenJsonList import FrozenJsonList

        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
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
        """Return an independent ordinary mapping for defensive exports."""
        return {key: copy.deepcopy(value, memo) for key, value in dict.items(self)}

    def __reduce__(self) -> tuple[type[FrozenJsonMapping], tuple[dict[str, Any]]]:
        """Rebuild through the validating constructor when crossing a process boundary."""
        return (type(self), (dict(self),))

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("FrozenJsonMapping is immutable.")
