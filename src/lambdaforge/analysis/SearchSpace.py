"""Condition-preserving scientific search-space representation."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.hpo.ParameterSpace import ParameterDescriptor, ParameterSpace

INACTIVE = "<inactive>"


def build_space(
    candidates: Sequence[Mapping[str, Any]], authored: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Build an ordered mixed domain while retaining exact authored activation conditions."""
    authored = authored or {}
    for name, configured in authored.items():
        if isinstance(configured, Mapping) and "when" in configured:
            condition = configured.get("when")
            if not isinstance(condition, Mapping) or not condition:
                raise ValueError(f"Search parameter {name!r} has an invalid activation condition.")
    points = tuple(
        candidate.get("parameters", {})
        for candidate in candidates
        if isinstance(candidate.get("parameters", {}), Mapping)
    )
    geometry = ParameterSpace.from_schema(authored, points)
    output: dict[str, dict[str, Any]] = {}
    for descriptor in geometry.descriptors:
        common = {"when": descriptor.when, "conditional": descriptor.conditional}
        if descriptor.kind in {"continuous", "integer"}:
            output[descriptor.name] = {
                "kind": "numeric",
                "low": descriptor.low,
                "high": descriptor.high,
                "scale": descriptor.scale,
                "integer": descriptor.kind == "integer",
                "values": list(descriptor.values) if descriptor.values else None,
                **common,
            }
        else:
            output[descriptor.name] = {
                "kind": "categorical",
                "values": list(descriptor.values),
                **common,
            }
    return output


def condition_active(rule: Mapping[str, Any], point: Mapping[str, Any]) -> bool:
    """Return whether a parameter must exist for this parent configuration."""
    condition = rule.get("when")
    return not isinstance(condition, Mapping) or all(
        point.get(str(name), INACTIVE) == expected for name, expected in condition.items()
    )


def canonicalize(
    point: Mapping[str, Any], space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Remove inactive descendants and return one valid ordered projection."""
    return ParameterSpace.from_schema(space).canonicalize(point)


def valid_point(
    point: Mapping[str, Any],
    space: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> bool:
    """Validate values and exact active/inactive membership for a scientific point."""
    return ParameterSpace.from_schema(space).valid(point, require_complete=require_complete)


def sample_point(space: Mapping[str, Mapping[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Sample a valid point, resolving parent conditions before descendants."""
    return ParameterSpace.from_schema(space).sample_point(rng)


def assign_values(
    base: Mapping[str, Any],
    assignments: Mapping[str, Any],
    space: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Apply requested values and reject combinations inconsistent with ``when``."""
    return ParameterSpace.from_schema(space).assign(base, assignments)


def grid(rule: Mapping[str, Any], count: int) -> list[Any]:
    """Return deterministic candidate values for one dimension, excluding inactivity."""
    # A standalone marginal grid deliberately ignores the rule's parent condition; activity is
    # enforced when the value is assigned to a complete point through the same ParameterSpace.
    marginal = {key: value for key, value in rule.items() if key != "when"}
    return list(ParameterSpace.from_schema({"value": marginal}).grid("value", count))


__all__ = [
    "INACTIVE",
    "ParameterDescriptor",
    "ParameterSpace",
    "assign_values",
    "build_space",
    "canonicalize",
    "condition_active",
    "grid",
    "sample_point",
    "valid_point",
]
