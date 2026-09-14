"""Reproducible low-discrepancy candidate generation for adaptive studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.hpo.Trial import Trial


class SobolSearch:
    """Generate unique mixed candidates from a scrambled Sobol sequence."""

    def __init__(self, space: Mapping[str, Mapping[str, Any]], *, seed: int = 0) -> None:
        self.space = {str(name): dict(specification) for name, specification in space.items()}
        self.parameter_space = ParameterSpace.from_schema(self.space)
        self.seed = int(seed)

    def trials(self, count: int) -> tuple[Trial, ...]:
        """Return exactly ``count`` deterministic unique candidates when the space permits it."""
        if count < 1:
            raise ValueError("Sobol search trial count must be positive.")
        if not self.space:
            raise ValueError("Sobol search requires at least one parameter.")
        engine = torch.quasirandom.SobolEngine(
            dimension=len(self.parameter_space.descriptors), scramble=True, seed=self.seed
        )
        output: list[Trial] = []
        seen: set[str] = set()
        attempts = 0
        while len(output) < count and attempts < count * 100:
            attempts += 1
            point = engine.draw(1, dtype=torch.float64).reshape(-1).tolist()
            parameters: dict[str, Any] = {}
            for descriptor, coordinate in zip(
                self.parameter_space.descriptors, point, strict=True
            ):
                if descriptor.active(parameters):
                    parameters[descriptor.name] = descriptor.sample(float(coordinate))
            encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
            fingerprint = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                output.append(Trial(len(output), parameters, self.seed, fingerprint))
        if len(output) != count:
            raise ValueError("Search space cannot produce the requested number of unique trials.")
        return tuple(output)

__all__ = ["SobolSearch"]
