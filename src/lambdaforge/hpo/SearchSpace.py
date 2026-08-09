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
    """Own ordered dimensions and transform between vectors and experiment mappings."""

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
        """Return fixed encoded dimensionality, including conditional dimensions."""
        return len(self.parameters)

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
        """Encode a parameter mapping; inactive dimensions use a neutral midpoint."""
        output: list[float] = []
        for parameter in self.parameters:
            output.append(
                parameter.encode(values[parameter.path])
                if parameter.path in values and parameter.active(values)
                else 0.5
            )
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
