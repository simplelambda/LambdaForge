"""Condition-preserving scientific search-space representation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

INACTIVE = "<inactive>"


def build_space(
    candidates: Sequence[Mapping[str, Any]], authored: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Build an ordered mixed domain while retaining exact authored activation conditions."""
    authored = authored or {}
    observed_names = {
        str(name) for candidate in candidates for name in candidate.get("parameters", {})
    }
    names = [str(name) for name in authored]
    names.extend(sorted(observed_names - set(names)))
    output: dict[str, dict[str, Any]] = {}
    for name in names:
        configured = authored.get(name)
        configured = configured if isinstance(configured, Mapping) else {}
        condition = configured.get("when")
        if condition is not None and (not isinstance(condition, Mapping) or not condition):
            raise ValueError(f"Search parameter {name!r} has an invalid activation condition.")
        present = [candidate.get("parameters", {}).get(name, INACTIVE) for candidate in candidates]
        active = [value for value in present if value != INACTIVE]
        authored_values = configured.get("values")
        declared_values = (
            list(authored_values)
            if isinstance(authored_values, Sequence)
            and not isinstance(authored_values, str | bytes)
            else None
        )
        numeric = bool(active) and all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in active
        )
        if declared_values is not None:
            numeric = bool(declared_values) and all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in declared_values
            )
        raw_range = configured.get("range")
        if isinstance(raw_range, Sequence) and not isinstance(raw_range, str | bytes):
            numeric = True
        common = {
            "when": {str(key): value for key, value in condition.items()}
            if isinstance(condition, Mapping)
            else None,
            "conditional": condition is not None,
        }
        if numeric:
            if (
                isinstance(raw_range, Sequence)
                and not isinstance(raw_range, str | bytes)
                and len(raw_range) == 2
            ):
                low, high = float(raw_range[0]), float(raw_range[1])
            elif active:
                domain_values = declared_values if declared_values is not None else active
                low, high = min(map(float, domain_values)), max(map(float, domain_values))
            else:
                raise ValueError(f"Numeric parameter {name!r} has no bounds or observations.")
            output[name] = {
                "kind": "numeric",
                "low": low,
                "high": high,
                "scale": str(configured.get("scale", "linear")),
                "integer": str(configured.get("type", "")) == "int"
                or bool(declared_values if declared_values is not None else active)
                and all(
                    isinstance(value, int)
                    for value in (declared_values if declared_values is not None else active)
                ),
                "values": declared_values,
                **common,
            }
        else:
            values = (
                declared_values
                if declared_values is not None
                else sorted({value for value in active}, key=str)
            )
            output[name] = {"kind": "categorical", "values": values, **common}
    _validate_dependencies(output)
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
    output: dict[str, Any] = {}
    for name, rule in space.items():
        if condition_active(rule, output) and name in point:
            output[name] = point[name]
    return output


def valid_point(
    point: Mapping[str, Any],
    space: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> bool:
    """Validate values and exact active/inactive membership for a scientific point."""
    if set(point) - set(space):
        return False
    resolved: dict[str, Any] = {}
    for name, rule in space.items():
        active = condition_active(rule, resolved)
        present = name in point
        if active and require_complete and not present:
            return False
        if not active and present:
            return False
        if not present:
            continue
        value = point[name]
        if rule["kind"] == "numeric":
            if not isinstance(value, int | float) or isinstance(value, bool):
                return False
            numeric = float(value)
            if not math.isfinite(numeric) or not (
                float(rule["low"]) <= numeric <= float(rule["high"])
            ):
                return False
            if rule.get("integer") and not numeric.is_integer():
                return False
            if rule.get("values") is not None and value not in rule["values"]:
                return False
        elif value not in rule.get("values", ()):
            return False
        resolved[name] = value
    return True


def sample_point(space: Mapping[str, Mapping[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Sample a valid point, resolving parent conditions before descendants."""
    point: dict[str, Any] = {}
    for name, rule in space.items():
        if not condition_active(rule, point):
            continue
        if rule["kind"] == "numeric":
            if rule.get("values") is not None:
                point[name] = rng.choice(list(rule["values"]))
                continue
            low, high = float(rule["low"]), float(rule["high"])
            if rule.get("scale") == "log" and low > 0:
                value = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                value = rng.uniform(low, high)
            point[name] = int(round(value)) if rule.get("integer") else value
        else:
            values = list(rule.get("values", ()))
            if not values:
                raise ValueError(f"Categorical parameter {name!r} has no levels.")
            point[name] = rng.choice(values)
    assert valid_point(point, space)
    return point


def assign_values(
    base: Mapping[str, Any],
    assignments: Mapping[str, Any],
    space: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Apply requested values and reject combinations inconsistent with ``when``."""
    requested = dict(base)
    for name, value in assignments.items():
        if value == INACTIVE:
            requested.pop(name, None)
        else:
            requested[name] = value
    projected = canonicalize(requested, space)
    for name, value in assignments.items():
        if value == INACTIVE:
            if name in projected:
                return None
        elif projected.get(name, INACTIVE) != value:
            return None
    return projected if valid_point(projected, space) else None


def grid(rule: Mapping[str, Any], count: int) -> list[Any]:
    """Return deterministic candidate values for one dimension, excluding inactivity."""
    if rule["kind"] == "categorical":
        return list(rule.get("values", ()))
    if rule.get("values") is not None:
        return list(rule["values"])
    low, high = float(rule["low"]), float(rule["high"])
    if count <= 1 or low == high:
        return [int(low) if rule.get("integer") else low]
    if rule.get("scale") == "log" and low > 0:
        values = [
            math.exp(math.log(low) + index * (math.log(high) - math.log(low)) / (count - 1))
            for index in range(count)
        ]
    else:
        values = [low + index * (high - low) / (count - 1) for index in range(count)]
    output = [int(round(value)) for value in values] if rule.get("integer") else values
    return list(dict.fromkeys(output))


def _validate_dependencies(space: Mapping[str, Mapping[str, Any]]) -> None:
    known: set[str] = set()
    for name, rule in space.items():
        condition = rule.get("when")
        dependencies = set(map(str, condition)) if isinstance(condition, Mapping) else set()
        missing = dependencies - known
        if missing:
            raise ValueError(
                f"Conditional parameter {name!r} must follow its parent(s): {sorted(missing)}."
            )
        known.add(name)


__all__ = [
    "INACTIVE",
    "assign_values",
    "build_space",
    "canonicalize",
    "condition_active",
    "grid",
    "sample_point",
    "valid_point",
]
