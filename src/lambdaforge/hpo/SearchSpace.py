"""Typed, reproducible adaptive hyperparameter search space."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.hpo.SearchParameter import SearchParameter


class SearchSpace:
    """Own sampling dimensions and a mixed surrogate representation."""

    def __init__(self, parameters: Sequence[SearchParameter]) -> None:
        self.parameters = tuple(parameters)
        paths = [parameter.path for parameter in self.parameters]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("Adaptive search space paths must be non-empty and unique.")
        positions = {path: index for index, path in enumerate(paths)}
        for parameter in self.parameters:
            for parent in parameter.when or {}:
                if parent not in positions or positions[parent] >= positions[parameter.path]:
                    raise ValueError(
                        f"Conditional parameter {parameter.path!r} must reference an earlier "
                        "dimension."
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Mapping[str, Any]]) -> SearchSpace:
        """Create a search space while retaining YAML declaration order."""
        return cls(tuple(SearchParameter.from_mapping(path, spec) for path, spec in value.items()))

    @property
    def dimension(self) -> int:
        """Return unit-coordinate sampling dimensionality."""
        return len(self.parameters)

    @property
    def model_dimension(self) -> int:
        """Return mixed surrogate dimensionality, including activity masks."""
        return sum(parameter.model_width for parameter in self.parameters)

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        """Return surrogate dimensions that require permutation-invariant Hamming geometry."""
        output: list[int] = []
        offset = 0
        for parameter in self.parameters:
            if parameter.model_value_is_categorical:
                output.append(offset)
            offset += parameter.model_width
        return tuple(output)

    def decode(self, vector: Sequence[float]) -> dict[str, Any]:
        """Decode a unit vector, omitting inactive conditional values."""
        if len(vector) != self.dimension:
            raise ValueError(f"Expected {self.dimension} search coordinates, got {len(vector)}.")
        output: dict[str, Any] = {}
        for parameter, coordinate in zip(self.parameters, vector, strict=True):
            if parameter.active(output):
                output[parameter.path] = parameter.decode(float(coordinate))
        return output

    def encode(self, values: Mapping[str, Any]) -> tuple[float, ...]:
        """Encode mixed surrogate features without treating inactivity as a midpoint."""
        output: list[float] = []
        for parameter in self.parameters:
            is_active = parameter.active(values)
            if is_active and parameter.path not in values:
                raise KeyError(f"Active search parameter {parameter.path!r} is missing.")
            output.extend(
                parameter.model_encode(
                    values.get(parameter.path),
                    active=is_active and parameter.path in values,
                )
            )
        return tuple(output)

    def decode_model(self, vector: Sequence[float]) -> dict[str, Any]:
        """Decode a mixed model vector while enforcing conditional hierarchy."""
        if len(vector) != self.model_dimension:
            raise ValueError(
                f"Expected {self.model_dimension} surrogate coordinates, got {len(vector)}."
            )
        sampling: list[float] = []
        offset = 0
        for parameter in self.parameters:
            raw = float(vector[offset])
            if parameter.kind in {"categorical", "bool"}:
                if parameter.kind == "bool":
                    coordinate = 0.75 if round(raw) == 1 else 0.25
                else:
                    coordinate = (round(raw) + 0.5) / len(parameter.choices)
            else:
                coordinate = raw
            sampling.append(coordinate)
            offset += parameter.model_width
        return self.decode(sampling)

    def model_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return lower/upper bounds for the mixed surrogate representation."""
        lower: list[float] = []
        upper: list[float] = []
        for parameter in self.parameters:
            lower.append(0.0)
            if parameter.model_value_is_categorical:
                upper.append(float(parameter.category_count - 1))
            else:
                upper.append(1.0)
            if parameter.conditional:
                lower.append(0.0)
                upper.append(1.0)
        return tuple(lower), tuple(upper)

    def categorical_assignments(self) -> tuple[dict[int, float], ...]:
        """Enumerate hierarchy-consistent mixed fixed features.

        Conditions whose parents are discrete are fixed exactly. A condition on a continuous
        parent leaves its activity mask free because equality cannot be inferred while enumerating
        categorical branches.
        """
        branches: list[dict[str, Any]] = [{}]
        for parameter in self.parameters:
            if not parameter.model_value_is_categorical:
                continue
            values = (False, True) if parameter.kind == "bool" else parameter.choices
            branches = [
                {**branch, parameter.path: value} for branch in branches for value in values
            ]
        output: list[dict[int, float]] = []
        for branch in branches or [{}]:
            assignment: dict[int, float] = {}
            offset = 0
            for parameter in self.parameters:
                condition_known = parameter.when is None or all(
                    parent in branch for parent in parameter.when
                )
                active = parameter.active(branch) if condition_known else True
                if parameter.model_value_is_categorical:
                    if active:
                        encoded = parameter.model_encode(branch[parameter.path], active=True)
                    elif condition_known:
                        encoded = parameter.model_encode(active=False)
                    else:
                        encoded = parameter.model_encode(branch[parameter.path], active=True)
                    assignment[offset] = encoded[0]
                if parameter.conditional and condition_known:
                    assignment[offset + 1] = 1.0 if active else 0.0
                offset += parameter.model_width
            if assignment not in output:
                output.append(assignment)
        return tuple(output)

    def materialize(self, base: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
        """Apply dotted values to an independent experiment mapping."""
        output = copy.deepcopy(dict(base))
        for path, value in values.items():
            ExperimentConfig.set_value(output, path, copy.deepcopy(value))
        return output

    def identifier(self, values: Mapping[str, Any]) -> str:
        """Return a short stable scientific configuration identifier."""
        encoded = json.dumps(dict(values), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "config-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return a stable JSON-compatible definition."""
        output: dict[str, dict[str, Any]] = {}
        for parameter in self.parameters:
            item: dict[str, Any] = {"type": parameter.kind, "scale": parameter.scale}
            if parameter.low is not None:
                item.update(low=parameter.low, high=parameter.high)
            if parameter.choices:
                item["choices"] = list(parameter.choices)
            if parameter.when is not None:
                item["when"] = dict(parameter.when)
            output[parameter.path] = item
        return output
