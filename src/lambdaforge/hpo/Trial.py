"""Immutable HPO trial definition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True, init=False)
class Trial:
    """Record trial number, parameters, seed and reproducible identity."""

    number: int
    parameters: Mapping[str, Any]
    seed: int
    fingerprint: str

    def __init__(
        self, number: int, parameters: Mapping[str, Any], seed: int, fingerprint: str
    ) -> None:
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "parameters", FrozenJsonMapping(parameters))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fingerprint", fingerprint)
