"""Reproducible low-discrepancy candidate generation for adaptive studies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from lambdaforge.hpo.Trial import Trial


class SobolSearch:
    """Generate unique mixed candidates from a scrambled Sobol sequence."""

    def __init__(self, space: Mapping[str, Mapping[str, Any]], *, seed: int = 0) -> None:
        self.space = {str(name): dict(specification) for name, specification in space.items()}
        self.seed = int(seed)

    def trials(self, count: int) -> tuple[Trial, ...]:
        """Return exactly ``count`` deterministic unique candidates when the space permits it."""
        if count < 1:
            raise ValueError("Sobol search trial count must be positive.")
        if not self.space:
            raise ValueError("Sobol search requires at least one parameter.")
        engine = torch.quasirandom.SobolEngine(
            dimension=len(self.space), scramble=True, seed=self.seed
        )
        output: list[Trial] = []
        seen: set[str] = set()
        attempts = 0
        while len(output) < count and attempts < count * 100:
            attempts += 1
            point = engine.draw(1, dtype=torch.float64).reshape(-1).tolist()
            parameters: dict[str, Any] = {}
            for (name, specification), coordinate in zip(self.space.items(), point, strict=True):
                condition = specification.get("when")
                if condition is not None:
                    if not isinstance(condition, Mapping) or any(
                        parameters.get(str(key)) != expected for key, expected in condition.items()
                    ):
                        continue
                parameters[name] = self._decode(specification, float(coordinate))
            encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
            fingerprint = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                output.append(Trial(len(output), parameters, self.seed, fingerprint))
        if len(output) != count:
            raise ValueError("Search space cannot produce the requested number of unique trials.")
        return tuple(output)

    @staticmethod
    def _decode(specification: Mapping[str, Any], coordinate: float) -> Any:
        kind = str(specification.get("type", "choice"))
        if kind == "choice":
            values = specification.get("values")
            if not isinstance(values, Sequence) or isinstance(values, str | bytes) or not values:
                raise ValueError("Choice search parameters require a non-empty values sequence.")
            index = min(len(values) - 1, int(coordinate * len(values)))
            return values[index]
        low, high = float(specification["low"]), float(specification["high"])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Search bounds must be finite with high > low.")
        if kind == "loguniform":
            if low <= 0:
                raise ValueError("Log-uniform search requires low > 0.")
            return math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low)))
        if kind == "int":
            lower, upper = math.ceil(low), math.floor(high)
            if upper < lower:
                raise ValueError("Integer search range contains no integer.")
            return min(upper, lower + int(coordinate * (upper - lower + 1)))
        if kind == "uniform":
            return low + coordinate * (high - low)
        raise ValueError(f"Unsupported Sobol search distribution: {kind!r}.")


__all__ = ["SobolSearch"]
