"""Read-only list container for nested JSON result values."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, NoReturn, SupportsIndex


class FrozenJsonList(list[Any]):
    """Retain JSON list encoding while rejecting every ordinary mutation."""

    def __init__(self, values: Iterable[Any] = ()) -> None:
        from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping

        list.__init__(self, (FrozenJsonMapping.freeze_value(value) for value in values))

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        del key, value
        self._reject()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        del key
        self._reject()

    def __iadd__(  # type: ignore[misc]
        self,
        values: Iterable[Any],
    ) -> FrozenJsonList:
        del values
        self._reject()

    def __imul__(
        self,
        count: SupportsIndex,
    ) -> FrozenJsonList:
        del count
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
        """Return an independent ordinary list for defensive exports."""
        return [copy.deepcopy(value, memo) for value in list.__iter__(self)]

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("FrozenJsonList is immutable.")
