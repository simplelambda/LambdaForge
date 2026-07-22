"""Object contract for paired statistical tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult


class PairedTest(ABC):
    """Test a finite vector of already oriented paired improvements."""

    @abstractmethod
    def compute(self, improvements: Sequence[float]) -> PairedTestResult:
        """Return inference and diagnostics for paired improvements."""
