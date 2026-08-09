"""Semantic configuration comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigurationDiff:
    """Compare nested configuration values by semantic path, ignoring key order."""

    def compare(
        self, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Return only added, removed or changed leaf paths."""
        left_flat = self._flatten(left)
        right_flat = self._flatten(right)
        missing = object()
        changes: dict[str, dict[str, Any]] = {}
        for path in sorted(set(left_flat) | set(right_flat)):
            before = left_flat.get(path, missing)
            after = right_flat.get(path, missing)
            if before != after:
                changes[path] = {
                    "before": None if before is missing else before,
                    "after": None if after is missing else after,
                    "change": "added"
                    if before is missing
                    else "removed"
                    if after is missing
                    else "changed",
                }
        return changes

    @classmethod
    def _flatten(cls, value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            if not value and prefix:
                output[prefix] = {}
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                output.update(cls._flatten(item, path))
            return output
        return {prefix: value}
