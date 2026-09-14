"""Canonical authored-space geometry shared by every HPO subsystem.

Candidate dictionaries remain the persisted scientific identity.  This module only owns their
mathematical representation, including logarithmic topology and conditional inactivity.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ParameterKind = Literal["continuous", "integer", "categorical", "boolean"]
INACTIVE = "<inactive>"


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    """One authored parameter domain and its transform."""

    name: str
    kind: ParameterKind
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] = ()
    scale: str = "linear"
    when: Mapping[str, Any] | None = None

    @property
    def conditional(self) -> bool:
        return self.when is not None

    def active(self, point: Mapping[str, Any]) -> bool:
        return self.when is None or all(
            point.get(str(name), INACTIVE) == expected for name, expected in self.when.items()
        )

    def normalize(self, value: Any) -> float:
        if self.kind in {"continuous", "integer"}:
            assert self.low is not None and self.high is not None
            numeric = float(value)
            low, high = self.low, self.high
            if self.scale == "log":
                if low <= 0 or high <= 0 or numeric <= 0:
                    raise ValueError(f"Log-scale parameter {self.name!r} requires positive values.")
                numeric, low, high = math.log(numeric), math.log(low), math.log(high)
            return 0.0 if high == low else min(1.0, max(0.0, (numeric - low) / (high - low)))
        key = _value_key(value)
        keys = tuple(_value_key(item) for item in self.values)
        if key not in keys:
            raise ValueError(f"Value {value!r} is outside parameter {self.name!r}.")
        return float(keys.index(key))

    def denormalize(self, coordinate: float) -> Any:
        if self.kind in {"continuous", "integer"}:
            assert self.low is not None and self.high is not None
            bounded = min(1.0, max(0.0, float(coordinate)))
            if self.scale == "log":
                value = math.exp(
                    math.log(self.low)
                    + bounded * (math.log(self.high) - math.log(self.low))
                )
            else:
                value = self.low + bounded * (self.high - self.low)
            return int(round(value)) if self.kind == "integer" else value
        if not self.values:
            raise ValueError(f"Categorical parameter {self.name!r} has no values.")
        return self.values[min(len(self.values) - 1, max(0, int(round(coordinate))))]

    def sample(self, coordinate: float) -> Any:
        """Decode a unit-cube design coordinate without leaving an explicit finite domain."""
        bounded = min(1.0, max(0.0, float(coordinate)))
        if self.values:
            return self.values[min(len(self.values) - 1, int(bounded * len(self.values)))]
        if self.kind == "integer":
            assert self.low is not None and self.high is not None
            lower, upper = math.ceil(self.low), math.floor(self.high)
            if upper < lower:
                raise ValueError(f"Integer parameter {self.name!r} contains no integer.")
            return min(upper, lower + int(bounded * (upper - lower + 1)))
        return self.denormalize(bounded)


class ParameterSpace:
    """Single mixed/conditional geometry for generation, modelling and analysis."""

    def __init__(self, descriptors: Sequence[ParameterDescriptor]) -> None:
        pending = list(descriptors)
        ordered: list[ParameterDescriptor] = []
        known_names = {value.name for value in pending}
        while pending:
            ready = next(
                (
                    value
                    for value in pending
                    if set(map(str, value.when or {})).issubset(
                        {item.name for item in ordered}
                    )
                ),
                None,
            )
            if ready is None:
                unresolved = {
                    value.name: sorted(set(map(str, value.when or {})) - known_names)
                    for value in pending
                }
                raise ValueError(f"Cyclic or missing conditional dependencies: {unresolved}.")
            ordered.append(ready)
            pending.remove(ready)
        self.descriptors = tuple(ordered)
        self._by_name = {value.name: value for value in self.descriptors}

    @classmethod
    def from_schema(
        cls,
        authored: Mapping[str, Any] | None,
        candidates: Sequence[Mapping[str, Any]] = (),
    ) -> ParameterSpace:
        authored = authored or {}
        observed_names = {str(name) for point in candidates for name in point}
        names = [str(name) for name in authored]
        names.extend(sorted(observed_names - set(names)))
        descriptors: list[ParameterDescriptor] = []
        for name in names:
            raw = authored.get(name, {})
            rule = raw if isinstance(raw, Mapping) else {"values": raw}
            present = [point[name] for point in candidates if name in point]
            raw_values = rule.get("values")
            values = (
                tuple(raw_values)
                if isinstance(raw_values, Sequence) and not isinstance(raw_values, str | bytes)
                else tuple(dict.fromkeys(present))
            )
            raw_range = rule.get("range")
            if not (
                isinstance(raw_range, Sequence)
                and not isinstance(raw_range, str | bytes)
                and len(raw_range) == 2
            ) and "low" in rule and "high" in rule:
                raw_range = (rule["low"], rule["high"])
            distribution = str(rule.get("type", ""))
            scale = "log" if distribution == "loguniform" else str(rule.get("scale", "linear"))
            numeric_values = bool(values) and all(_numeric(value) for value in values)
            low: float | None
            high: float | None
            if raw_range is not None:
                assert isinstance(raw_range, Sequence)
                low, high = float(raw_range[0]), float(raw_range[1])
                kind: ParameterKind = (
                    "integer"
                    if distribution == "int" or bool(rule.get("integer"))
                    else "continuous"
                )
            elif numeric_values:
                low, high = min(map(float, values)), max(map(float, values))
                explicit_integer = distribution == "int" or str(rule.get("type", "")) == "int"
                kind = (
                    "integer"
                    if explicit_integer or all(isinstance(value, int) for value in values)
                    else "continuous"
                )
            elif distribution in {"uniform", "loguniform", "int", "float"}:
                raise ValueError(f"Numeric parameter {name!r} has no authored range or values.")
            else:
                low, high = None, None
                kind = (
                    "boolean"
                    if values and all(isinstance(value, bool) for value in values)
                    else "categorical"
                )
            raw_when = rule.get("when")
            when = (
                {str(key): value for key, value in raw_when.items()}
                if isinstance(raw_when, Mapping)
                else None
            )
            descriptors.append(ParameterDescriptor(name, kind, low, high, values, scale, when))
        return cls(descriptors)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.descriptors)

    def descriptor(self, name: str) -> ParameterDescriptor:
        """Return immutable authored metadata for one named dimension."""
        return self._by_name[name]

    def to_schema(self) -> dict[str, dict[str, Any]]:
        """Return a portable schema that reconstructs the same mathematical geometry."""
        output: dict[str, dict[str, Any]] = {}
        for descriptor in self.descriptors:
            rule: dict[str, Any] = {}
            if descriptor.values:
                rule["values"] = list(descriptor.values)
            elif descriptor.low is not None and descriptor.high is not None:
                rule["range"] = [descriptor.low, descriptor.high]
            if descriptor.kind == "integer":
                rule["type"] = "int"
            if descriptor.scale != "linear":
                rule["scale"] = descriptor.scale
            if descriptor.when is not None:
                rule["when"] = dict(descriptor.when)
            output[descriptor.name] = rule
        return output

    @property
    def categorical_dimensions(self) -> tuple[int, ...]:
        dimensions: list[int] = []
        offset = 0
        for descriptor in self.descriptors:
            if descriptor.kind in {"categorical", "boolean"}:
                dimensions.append(offset)
            offset += 1
            if descriptor.conditional:
                dimensions.append(offset)
                offset += 1
        return tuple(dimensions)

    def is_active(self, name: str, point: Mapping[str, Any]) -> bool:
        return self._by_name[name].active(point)

    def active_mask(self, point: Mapping[str, Any]) -> tuple[bool, ...]:
        return tuple(value.active(point) and value.name in point for value in self.descriptors)

    def normalize(self, name: str, value: Any) -> float:
        return self._by_name[name].normalize(value)

    def denormalize(self, name: str, value: float) -> Any:
        return self._by_name[name].denormalize(value)

    def encode(self, point: Mapping[str, Any]) -> tuple[float, ...]:
        vector: list[float] = []
        for descriptor in self.descriptors:
            active = descriptor.active(point) and descriptor.name in point
            vector.append(descriptor.normalize(point[descriptor.name]) if active else 0.0)
            if descriptor.conditional:
                vector.append(1.0 if active else 0.0)
        return tuple(vector)

    def decode(self, vector: Sequence[float]) -> dict[str, Any]:
        point: dict[str, Any] = {}
        offset = 0
        for descriptor in self.descriptors:
            coordinate = vector[offset]
            offset += 1
            encoded_active = True
            if descriptor.conditional:
                encoded_active = vector[offset] >= 0.5
                offset += 1
            if encoded_active and descriptor.active(point):
                point[descriptor.name] = descriptor.denormalize(coordinate)
        if offset != len(vector):
            raise ValueError("Encoded parameter vector has an incompatible dimension.")
        return point

    def distance(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        ignore: Sequence[str] = (),
    ) -> float:
        ignored = set(ignore)
        distances: list[float] = []
        for descriptor in self.descriptors:
            if descriptor.name in ignored:
                continue
            left_active = descriptor.active(left) and descriptor.name in left
            right_active = descriptor.active(right) and descriptor.name in right
            if not left_active and not right_active:
                continue
            if left_active != right_active:
                distances.append(1.0)
            elif descriptor.kind in {"continuous", "integer"}:
                distances.append(
                    abs(
                        descriptor.normalize(left[descriptor.name])
                        - descriptor.normalize(right[descriptor.name])
                    )
                )
            else:
                distances.append(0.0 if left[descriptor.name] == right[descriptor.name] else 1.0)
        return (
            math.sqrt(sum(value * value for value in distances) / len(distances))
            if distances
            else 0.0
        )

    def value_distance(self, name: str, left: Any, right: Any) -> float:
        """Measure one dimension while retaining explicit inactivity semantics."""
        if left == INACTIVE or right == INACTIVE:
            return 0.0 if left == right else 1.0
        descriptor = self._by_name[name]
        if descriptor.kind in {"continuous", "integer"}:
            return abs(descriptor.normalize(left) - descriptor.normalize(right))
        return 0.0 if left == right else 1.0

    def transformed_bounds(self) -> tuple[tuple[float, float], ...]:
        bounds: list[tuple[float, float]] = []
        for descriptor in self.descriptors:
            if descriptor.kind in {"continuous", "integer"}:
                bounds.append((0.0, 1.0))
            else:
                bounds.append((0.0, float(max(0, len(descriptor.values) - 1))))
            if descriptor.conditional:
                bounds.append((0.0, 1.0))
        return tuple(bounds)

    def canonicalize(self, point: Mapping[str, Any]) -> dict[str, Any]:
        """Remove inactive descendants while preserving canonical dependency order."""
        output: dict[str, Any] = {}
        for descriptor in self.descriptors:
            if descriptor.active(output) and descriptor.name in point:
                output[descriptor.name] = point[descriptor.name]
        return output

    def valid(self, point: Mapping[str, Any], *, require_complete: bool = True) -> bool:
        """Validate exact active membership and authored value domains."""
        if set(point) - set(self.names):
            return False
        resolved: dict[str, Any] = {}
        for descriptor in self.descriptors:
            active = descriptor.active(resolved)
            present = descriptor.name in point
            if active and require_complete and not present:
                return False
            if not active and present:
                return False
            if not present:
                continue
            value = point[descriptor.name]
            if descriptor.kind in {"continuous", "integer"}:
                if not _numeric(value):
                    return False
                numeric = float(value)
                assert descriptor.low is not None and descriptor.high is not None
                if not descriptor.low <= numeric <= descriptor.high:
                    return False
                if descriptor.kind == "integer" and not numeric.is_integer():
                    return False
                if descriptor.values and value not in descriptor.values:
                    return False
            elif value not in descriptor.values:
                return False
            resolved[descriptor.name] = value
        return True

    def sample_point(self, rng: random.Random) -> dict[str, Any]:
        """Sample one valid authored point using the same transforms as Sobol and models."""
        point: dict[str, Any] = {}
        for descriptor in self.descriptors:
            if descriptor.active(point):
                point[descriptor.name] = descriptor.sample(rng.random())
        return point

    def assign(
        self, base: Mapping[str, Any], assignments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Apply values and reject requests that conflict with conditional activity."""
        requested = dict(base)
        for name, value in assignments.items():
            if value == INACTIVE:
                requested.pop(name, None)
            else:
                requested[name] = value
        projected = self.canonicalize(requested)
        for name, value in assignments.items():
            if (value == INACTIVE and name in projected) or (
                value != INACTIVE and projected.get(name, INACTIVE) != value
            ):
                return None
        return projected if self.valid(projected) else None

    def grid(self, name: str, count: int) -> tuple[Any, ...]:
        """Return a deterministic transformed-space grid for one active dimension."""
        descriptor = self._by_name[name]
        if descriptor.values:
            return descriptor.values
        if count <= 1:
            return (descriptor.denormalize(0.0),)
        return tuple(
            dict.fromkeys(
                descriptor.denormalize(index / (count - 1)) for index in range(count)
            )
        )

    def support(self, name: str) -> Mapping[str, Any]:
        """Expose bounded authored-domain metadata without leaking sampler internals."""
        descriptor = self._by_name[name]
        return {
            "kind": descriptor.kind,
            "bounds": (
                (descriptor.low, descriptor.high)
                if descriptor.low is not None and descriptor.high is not None
                else None
            ),
            "values": descriptor.values,
            "scale": descriptor.scale,
            "when": descriptor.when,
        }

    def boundary_position(self, name: str, value: Any) -> float:
        """Return distance to the nearest authored boundary in normalized coordinates."""
        normalized = self._by_name[name].normalize(value)
        if self._by_name[name].kind in {"categorical", "boolean"}:
            return 0.0
        return min(normalized, 1.0 - normalized)


def _numeric(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


__all__ = ["INACTIVE", "ParameterDescriptor", "ParameterKind", "ParameterSpace"]
