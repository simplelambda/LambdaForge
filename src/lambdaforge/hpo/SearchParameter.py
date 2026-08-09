"""One typed adaptive-search dimension."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchParameter:
    """Encode/decode one conditional value without inventing categorical geometry.

    ``decode`` owns the unit-coordinate sampling representation. ``model_encode`` owns the
    surrogate representation: unordered values are integer labels consumed as categorical
    dimensions, ordered values retain their declared order, and an inactive conditional has its
    own state plus an explicit activity mask.
    """

    path: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    scale: str = "linear"
    when: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.path or any(not part for part in self.path.split(".")):
            raise ValueError("Search parameter paths must be non-empty dotted paths.")
        if self.kind not in {"float", "int", "ordinal", "categorical", "bool"}:
            raise ValueError(f"Unsupported search parameter type: {self.kind!r}.")
        if self.scale not in {"linear", "log"}:
            raise ValueError("Search parameter scale must be 'linear' or 'log'.")
        if self.kind in {"float", "int"}:
            if self.low is None or self.high is None:
                raise ValueError(f"Numeric search parameter {self.path!r} requires low/high.")
            if not math.isfinite(self.low) or not math.isfinite(self.high) or self.high <= self.low:
                raise ValueError(f"Search bounds for {self.path!r} must be finite with high > low.")
            if self.scale == "log" and self.low <= 0:
                raise ValueError(f"Log search parameter {self.path!r} requires low > 0.")
        elif self.scale != "linear":
            raise ValueError("Only numeric search parameters accept a non-linear scale.")
        if self.kind in {"categorical", "ordinal"} and not self.choices:
            raise ValueError(
                f"{self.kind.title()} search parameter {self.path!r} requires choices."
            )
        if self.kind == "bool" and self.choices:
            raise ValueError("Boolean search parameters do not accept choices.")
        if len(self.choices) != len({self._choice_key(choice) for choice in self.choices}):
            raise ValueError(f"Search choices for {self.path!r} must be unique.")
        if self.kind == "categorical":
            object.__setattr__(
                self,
                "choices",
                tuple(sorted(self.choices, key=self._choice_key)),
            )

    @classmethod
    def from_mapping(cls, path: str, value: Mapping[str, Any]) -> SearchParameter:
        """Validate one YAML search-space entry."""
        kind = str(value.get("type", "categorical"))
        raw_choices = value.get("choices", value.get("values", ()))
        if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
            raise TypeError(f"Search choices for {path!r} must be a sequence.")
        when = value.get("when")
        if when is not None and not isinstance(when, Mapping):
            raise TypeError(f"Search condition for {path!r} must be a mapping.")
        return cls(
            path=str(path),
            kind=kind,
            low=float(value["low"]) if "low" in value else None,
            high=float(value["high"]) if "high" in value else None,
            choices=tuple(raw_choices),
            scale=str(value.get("scale", "linear")),
            when=dict(when) if when is not None else None,
        )

    def active(self, parameters: Mapping[str, Any]) -> bool:
        """Return whether all parent-value conditions are satisfied."""
        return self.when is None or all(
            parameters.get(key) == value for key, value in self.when.items()
        )

    def decode(self, unit_value: float) -> Any:
        """Map one clamped unit coordinate to its configured value domain."""
        value = min(max(float(unit_value), 0.0), math.nextafter(1.0, 0.0))
        if self.kind == "bool":
            return value >= 0.5
        if self.kind in {"categorical", "ordinal"}:
            return self.choices[min(int(value * len(self.choices)), len(self.choices) - 1)]
        assert self.low is not None and self.high is not None
        if self.scale == "log":
            decoded = math.exp(
                math.log(self.low) + value * (math.log(self.high) - math.log(self.low))
            )
        else:
            decoded = self.low + value * (self.high - self.low)
        if self.kind == "int":
            return min(max(int(round(decoded)), math.ceil(self.low)), math.floor(self.high))
        return decoded

    def encode(self, value: Any) -> float:
        """Map a configured value back to its unit-coordinate sampling representation."""
        if self.kind == "bool":
            return 1.0 if bool(value) else 0.0
        if self.kind in {"categorical", "ordinal"}:
            try:
                index = self.choices.index(value)
            except ValueError as error:
                raise ValueError(f"Unknown choice {value!r} for {self.path!r}.") from error
            return (index + 0.5) / len(self.choices)
        assert self.low is not None and self.high is not None
        numeric = float(value)
        if self.scale == "log":
            return (math.log(numeric) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (numeric - self.low) / (self.high - self.low)

    @property
    def conditional(self) -> bool:
        """Return whether this dimension can be structurally absent."""
        return self.when is not None

    @property
    def model_width(self) -> int:
        """Return the fixed number of surrogate features for this parameter."""
        return 2 if self.conditional else 1

    @property
    def model_value_is_categorical(self) -> bool:
        """Return whether the first surrogate feature uses Hamming geometry."""
        return self.kind in {"categorical", "bool"}

    def model_encode(self, value: Any = None, *, active: bool) -> tuple[float, ...]:
        """Return model features with a distinct inactive state and activity mask."""
        if not active:
            inactive = float(self.category_count) if self.model_value_is_categorical else 0.0
            return (inactive, 0.0) if self.conditional else (inactive,)
        if self.kind == "bool":
            encoded = 1.0 if bool(value) else 0.0
        elif self.kind == "categorical":
            try:
                encoded = float(self.choices.index(value))
            except ValueError as error:
                raise ValueError(f"Unknown choice {value!r} for {self.path!r}.") from error
        elif self.kind == "ordinal":
            try:
                index = self.choices.index(value)
            except ValueError as error:
                raise ValueError(f"Unknown choice {value!r} for {self.path!r}.") from error
            encoded = index / max(1, len(self.choices) - 1)
        else:
            encoded = self.encode(value)
        return (encoded, 1.0) if self.conditional else (encoded,)

    @property
    def category_count(self) -> int:
        """Return active categorical cardinality, excluding the inactive sentinel."""
        if self.kind == "bool":
            return 2
        return len(self.choices) if self.kind == "categorical" else 0

    @staticmethod
    def _choice_key(value: Any) -> str:
        """Return an order-independent canonical key for an unordered category."""
        import json

        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("Adaptive search choices must be JSON-compatible values.") from error
