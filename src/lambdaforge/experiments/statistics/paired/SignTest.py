"""Exact paired sign test retained as the compatibility strategy."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from lambdaforge.experiments.statistics.paired.PairedAlternative import PairedAlternative
from lambdaforge.experiments.statistics.paired.PairedTest import PairedTest
from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult


class SignTest(PairedTest):
    """Run an exact binomial test over non-zero improvement signs."""

    def __init__(
        self,
        alternative: PairedAlternative = PairedAlternative.OBSERVED_DIRECTION,
        *,
        zero_tolerance: float = 1e-12,
    ) -> None:
        if not math.isfinite(float(zero_tolerance)) or float(zero_tolerance) < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        self.alternative = PairedAlternative(alternative)
        self.zero_tolerance = float(zero_tolerance)

    def compute(self, improvements: Sequence[float]) -> PairedTestResult:
        """Return exact two-sided and one-sided sign probabilities."""
        numeric = [float(value) for value in improvements]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Sign-test improvements must all be finite.")
        wins = sum(value > self.zero_tolerance for value in numeric)
        losses = sum(value < -self.zero_tolerance for value in numeric)
        ties = len(numeric) - wins - losses
        n_effective = wins + losses
        observed_mean = statistics.fmean(numeric) if numeric else 0.0
        if n_effective == 0:
            return PairedTestResult(
                method=PairedTestMethod.SIGN.value,
                alternative=self.alternative.value,
                calculation_requested="exact",
                calculation_used=None,
                statistic=None,
                positive_statistic=float(wins),
                negative_statistic=float(losses),
                p_value=None,
                p_value_two_sided=None,
                p_value_better=None,
                p_value_worse=None,
                z_statistic=None,
                n_pairs=len(numeric),
                n_effective=0,
                n_zero=ties,
                wins=wins,
                losses=losses,
                ties=ties,
                has_rank_ties=False,
                status="unavailable",
                reason="no_nonzero_differences",
                zero_tolerance=self.zero_tolerance,
            )

        p_better = self._upper_tail(n_effective, wins)
        p_worse = self._lower_tail(n_effective, wins)
        p_two_sided = min(1.0, 2.0 * min(p_better, p_worse))
        p_value = self.alternative.select(
            two_sided=p_two_sided,
            better=p_better,
            worse=p_worse,
            observed_mean=observed_mean,
        )
        statistic = (
            float(min(wins, losses))
            if self.alternative
            in {PairedAlternative.TWO_SIDED, PairedAlternative.OBSERVED_DIRECTION}
            else float(wins)
        )
        return PairedTestResult(
            method=PairedTestMethod.SIGN.value,
            alternative=self.alternative.value,
            calculation_requested="exact",
            calculation_used="exact",
            statistic=statistic,
            positive_statistic=float(wins),
            negative_statistic=float(losses),
            p_value=p_value,
            p_value_two_sided=p_two_sided,
            p_value_better=p_better,
            p_value_worse=p_worse,
            z_statistic=None,
            n_pairs=len(numeric),
            n_effective=n_effective,
            n_zero=ties,
            wins=wins,
            losses=losses,
            ties=ties,
            has_rank_ties=False,
            status="ok",
            zero_tolerance=self.zero_tolerance,
        )

    def _upper_tail(self, n: int, observed: int) -> float:
        return sum(math.comb(n, value) for value in range(observed, n + 1)) / (2**n)

    def _lower_tail(self, n: int, observed: int) -> float:
        return sum(math.comb(n, value) for value in range(0, observed + 1)) / (2**n)
