"""Named collection of reproducible architecture conformance cases."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from lambdaforge.nn.conformance.ArchitectureConformanceCase import (
    ArchitectureConformanceCase,
)
from lambdaforge.nn.conformance.ArchitectureConformanceResult import (
    ArchitectureConformanceResult,
)


class ArchitectureConformancePack:
    """Run related paper/provenance-linked references as one assertion boundary."""

    def __init__(self, name: str, cases: Sequence[ArchitectureConformanceCase]) -> None:
        if not name:
            raise ValueError("Pack name cannot be empty.")
        if not cases or any(not isinstance(case, ArchitectureConformanceCase) for case in cases):
            raise TypeError("cases must be a non-empty sequence of conformance cases.")
        names = [case.name for case in cases]
        if len(names) != len(set(names)):
            raise ValueError("Conformance case names must be unique inside a pack.")
        self.name = name
        self.cases = tuple(cases)

    def run(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[ArchitectureConformanceResult, ...]:
        """Run every case in stable declaration order."""
        return tuple(case.run(device) for case in self.cases)

    def assert_conformant(self, device: torch.device | str = "cpu") -> None:
        """Raise one actionable error listing every failed reference case."""
        failures = [result for result in self.run(device) if not result.passed]
        if failures:
            details = ", ".join(
                f"{result.name}: {result.message or 'unknown mismatch'}" for result in failures
            )
            raise AssertionError(f"Architecture conformance pack {self.name!r} failed: {details}")
