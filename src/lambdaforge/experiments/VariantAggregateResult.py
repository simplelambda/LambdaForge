"""Typed aggregate statistics for one sweep variant."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from lambdaforge.experiments.JsonResult import JsonResult


class VariantAggregateResult(JsonResult):
    """Provide typed accessors over a forward-compatible aggregate payload."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload.get("metrics", {}), Mapping):
            raise TypeError("Variant aggregate 'metrics' must be a mapping.")
        self._payload = copy.deepcopy(dict(payload))
        self._freeze_mapping(copy.deepcopy(self._payload))

    @property
    def variant(self) -> str:
        """Return the stable sweep variant identifier."""
        return str(self._payload.get("variant", "base"))

    @property
    def complete(self) -> bool:
        """Return whether every expected run completed successfully."""
        return bool(self._payload.get("complete", False))

    @property
    def terminal(self) -> bool:
        """Return whether every expected run reached a terminal state."""
        return bool(self._payload.get("terminal", False))

    @property
    def expected_runs(self) -> int:
        """Return the number of materialized runs expected for the variant."""
        return int(self._payload.get("expected_n", 0))

    @property
    def completed_runs(self) -> int:
        """Return the number of runs included in metric aggregation."""
        return int(self._payload.get("n_seeds", 0))

    @property
    def metrics(self) -> Mapping[str, Mapping[str, Any]]:
        """Return a defensive metric-to-statistics mapping."""
        return copy.deepcopy(dict(self._payload.get("metrics", {})))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VariantAggregateResult:
        """Create an aggregate result from current or persisted JSON."""
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible payload."""
        return copy.deepcopy(dict(self))
