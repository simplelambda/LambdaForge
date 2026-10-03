"""Small immutable activation grammar shared by generation and scientific analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _key(value: Any) -> str:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("Activation values must be JSON scalars; use {in: [...]} for membership.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Activation values must be finite.")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ActivationCondition:
    """An AND of equality or explicit finite-membership predicates.

    Equality serializes in its historical scalar form. Membership ordering is immaterial to
    identity, while parameter domain ordering is owned separately by ParameterSpace.
    """

    predicates: tuple[tuple[str, str, tuple[Any, ...]], ...] = ()

    @classmethod
    def parse(cls, value: Any) -> ActivationCondition:
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or not value:
            raise ValueError("when must be a non-empty mapping of parent parameters.")
        predicates: list[tuple[str, str, tuple[Any, ...]]] = []
        for name, raw in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("when parent names must be non-empty strings.")
            operator, values = "eq", (raw,)
            if isinstance(raw, Mapping):
                if len(raw) != 1 or next(iter(raw)) not in {"eq", "in"}:
                    raise ValueError(f"when.{name} accepts only eq or in.")
                operator = str(next(iter(raw)))
                if operator == "eq":
                    values = (raw[operator],)
                else:
                    members = raw[operator]
                    if (
                        not isinstance(members, Sequence)
                        or isinstance(members, str | bytes)
                        or not members
                    ):
                        raise ValueError(f"when.{name}.in must be a non-empty list.")
                    values = tuple(members)
            keys = [_key(item) for item in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"when.{name}.in contains duplicate values.")
            values = tuple(item for _, item in sorted(zip(keys, values, strict=True)))
            predicates.append((name, operator, values))
        return cls(tuple(sorted(predicates)))

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset(name for name, _, _ in self.predicates)

    def matches(self, point: Mapping[str, Any]) -> bool:
        return all(name in point and point[name] in values for name, _, values in self.predicates)

    def to_mapping(self) -> dict[str, Any] | None:
        if not self.predicates:
            return None
        return {
            name: values[0] if operator == "eq" else {"in": list(values)}
            for name, operator, values in self.predicates
        }

    def validate_domains(self, domains: Mapping[str, Sequence[Any]]) -> None:
        for name, operator, values in self.predicates:
            if name not in domains:
                raise ValueError(f"when references missing parent parameter {name!r}.")
            domain = domains[name]
            if not domain:
                if operator == "eq":
                    continue  # Preserve historical equality on a numeric parent range.
                raise ValueError(f"when parent {name!r} must have an explicit finite domain.")
            invalid = [value for value in values if value not in domain]
            if invalid:
                raise ValueError(f"when.{name} contains values outside its domain: {invalid!r}.")

    def describe(self) -> str:
        return " and ".join(
            f"{name} = {values[0]!r}" if operator == "eq" else f"{name} ∈ {list(values)!r}"
            for name, operator, values in self.predicates
        )


__all__ = ["ActivationCondition"]
