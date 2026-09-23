"""Deterministic prefix-stable candidate generation for uncapped adaptive studies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.hpo.SobolSearch import SobolSearch


class DeterministicCandidateGenerator:
    """Expose a reproducible Sobol sequence whose existing prefix never changes."""

    version = "scrambled-sobol-prefix-v1"

    def __init__(self, space: Mapping[str, Any]) -> None:
        normalized: dict[str, dict[str, Any]] = {}
        for raw_name, raw_rule in space.items():
            name = str(raw_name)
            rule = dict(raw_rule) if isinstance(raw_rule, Mapping) else {"values": raw_rule}
            if "values" in rule:
                descriptor: dict[str, Any] = {"type": "choice", "values": list(rule["values"])}
            else:
                kind = str(rule.get("type", "uniform"))
                if rule.get("scale") == "log":
                    kind = "loguniform"
                bounds = rule.get("range")
                low = rule.get("low") if bounds is None else bounds[0]
                high = rule.get("high") if bounds is None else bounds[1]
                descriptor = {
                    "type": kind,
                    "low": low,
                    "high": high,
                }
            if "when" in rule:
                descriptor["when"] = dict(rule["when"])
            normalized[name] = descriptor
        self.space = normalized

    def prefix(self, count: int) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value.parameters) for value in SobolSearch(self.space).trials(count))

    def extend(self, current: int, count: int) -> tuple[dict[str, Any], ...]:
        values = self.prefix(current + count)
        return values[current:]


__all__ = ["DeterministicCandidateGenerator"]
