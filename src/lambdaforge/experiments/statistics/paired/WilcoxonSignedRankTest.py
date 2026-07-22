"""Native deterministic Wilcoxon signed-rank test for paired improvements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from lambdaforge.experiments.statistics.paired.PairedAlternative import PairedAlternative
from lambdaforge.experiments.statistics.paired.PairedTest import PairedTest
from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult
from lambdaforge.experiments.statistics.paired.WilcoxonCalculation import (
    WilcoxonCalculation,
)
from lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod import WilcoxonZeroMethod


class WilcoxonSignedRankTest(PairedTest):
    """Test signed ranks using exact sign enumeration or a normal approximation.

    Exact mode conditions on the observed absolute ranks. Average ranks make
    ties deterministic, while the three documented zero conventions control
    which ranks enter the random sign distribution.
    """

    def __init__(
        self,
        alternative: PairedAlternative = PairedAlternative.TWO_SIDED,
        *,
        calculation: WilcoxonCalculation = WilcoxonCalculation.AUTO,
        zero_method: WilcoxonZeroMethod = WilcoxonZeroMethod.WILCOX,
        continuity_correction: bool = False,
        exact_max_pairs: int = 50,
        zero_tolerance: float = 1e-12,
        round_decimals: int | None = 12,
    ) -> None:
        if int(exact_max_pairs) < 0:
            raise ValueError("exact_max_pairs must be non-negative.")
        if not math.isfinite(float(zero_tolerance)) or float(zero_tolerance) < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        if round_decimals is not None and not 0 <= int(round_decimals) <= 15:
            raise ValueError("round_decimals must be null or between 0 and 15.")
        self.alternative = PairedAlternative(alternative)
        self.calculation = WilcoxonCalculation(calculation)
        self.zero_method = WilcoxonZeroMethod(zero_method)
        self.continuity_correction = bool(continuity_correction)
        self.exact_max_pairs = int(exact_max_pairs)
        self.zero_tolerance = float(zero_tolerance)
        self.round_decimals = None if round_decimals is None else int(round_decimals)

    def compute(self, improvements: Sequence[float]) -> PairedTestResult:
        """Return Wilcoxon inference for an already paired finite sample."""
        numeric = [float(value) for value in improvements]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Wilcoxon improvements must all be finite.")
        if self.round_decimals is not None:
            numeric = [round(value, self.round_decimals) for value in numeric]

        zeros = [abs(value) <= self.zero_tolerance for value in numeric]
        wins = sum(value > self.zero_tolerance for value in numeric)
        losses = sum(value < -self.zero_tolerance for value in numeric)
        ties = len(numeric) - wins - losses
        ranks, has_rank_ties = self._rank_absolute_values(numeric, zeros)
        nonzero_ranks = [rank for rank, is_zero in zip(ranks, zeros, strict=True) if not is_zero]
        positive_nonzero = sum(
            rank
            for value, rank, is_zero in zip(numeric, ranks, zeros, strict=True)
            if not is_zero and value > 0.0
        )
        negative_nonzero = sum(nonzero_ranks) - positive_nonzero
        zero_rank_sum = sum(rank for rank, is_zero in zip(ranks, zeros, strict=True) if is_zero)
        offset = zero_rank_sum / 2.0 if self.zero_method is WilcoxonZeroMethod.ZSPLIT else 0.0
        positive_statistic = positive_nonzero + offset
        negative_statistic = negative_nonzero + offset
        n_effective = len(nonzero_ranks)
        observed_mean = statistics.fmean(numeric) if numeric else 0.0

        if n_effective == 0:
            return self._unavailable(
                numeric,
                wins,
                losses,
                ties,
                has_rank_ties,
                positive_statistic,
                negative_statistic,
                "no_nonzero_differences",
            )

        calculation_used = self._calculation_for(n_effective)
        if calculation_used is None:
            return self._unavailable(
                numeric,
                wins,
                losses,
                ties,
                has_rank_ties,
                positive_statistic,
                negative_statistic,
                "exact_pair_limit_exceeded",
            )

        if calculation_used is WilcoxonCalculation.EXACT:
            p_better, p_worse = self._exact_tails(nonzero_ranks, positive_nonzero)
            z_statistic = None
        else:
            asymptotic_ranks = (
                ranks if self.zero_method is WilcoxonZeroMethod.ZSPLIT else nonzero_ranks
            )
            asymptotic_positive = (
                positive_statistic
                if self.zero_method is WilcoxonZeroMethod.ZSPLIT
                else positive_nonzero
            )
            p_better, p_worse, z_statistic = self._asymptotic_tails(
                asymptotic_ranks,
                asymptotic_positive,
            )
        p_two_sided = min(1.0, 2.0 * min(p_better, p_worse))
        p_value = self.alternative.select(
            two_sided=p_two_sided,
            better=p_better,
            worse=p_worse,
            observed_mean=observed_mean,
        )
        statistic = (
            min(positive_statistic, negative_statistic)
            if self.alternative
            in {PairedAlternative.TWO_SIDED, PairedAlternative.OBSERVED_DIRECTION}
            else positive_statistic
        )
        return PairedTestResult(
            method=PairedTestMethod.WILCOXON.value,
            alternative=self.alternative.value,
            calculation_requested=self.calculation.value,
            calculation_used=calculation_used.value,
            statistic=float(statistic),
            positive_statistic=float(positive_statistic),
            negative_statistic=float(negative_statistic),
            p_value=float(p_value),
            p_value_two_sided=float(p_two_sided),
            p_value_better=float(p_better),
            p_value_worse=float(p_worse),
            z_statistic=z_statistic,
            n_pairs=len(numeric),
            n_effective=n_effective,
            n_zero=ties,
            wins=wins,
            losses=losses,
            ties=ties,
            has_rank_ties=has_rank_ties,
            status="ok",
            zero_method=self.zero_method.value,
            continuity_correction=self.continuity_correction,
            exact_max_pairs=self.exact_max_pairs,
            zero_tolerance=self.zero_tolerance,
            round_decimals=self.round_decimals,
        )

    def _rank_absolute_values(
        self,
        numeric: Sequence[float],
        zeros: Sequence[bool],
    ) -> tuple[list[float], bool]:
        ranks = [0.0] * len(numeric)
        included = [
            index
            for index, is_zero in enumerate(zeros)
            if self.zero_method is not WilcoxonZeroMethod.WILCOX or not is_zero
        ]
        included.sort(key=lambda index: abs(numeric[index]))
        has_rank_ties = False
        start = 0
        while start < len(included):
            end = start + 1
            magnitude = abs(numeric[included[start]])
            while end < len(included) and abs(numeric[included[end]]) == magnitude:
                end += 1
            average_rank = ((start + 1) + end) / 2.0
            for position in range(start, end):
                ranks[included[position]] = average_rank
            if end - start > 1 and magnitude > self.zero_tolerance:
                has_rank_ties = True
            start = end
        return ranks, has_rank_ties

    def _calculation_for(self, n_effective: int) -> WilcoxonCalculation | None:
        if self.calculation is WilcoxonCalculation.AUTO:
            return (
                WilcoxonCalculation.EXACT
                if n_effective <= self.exact_max_pairs
                else WilcoxonCalculation.ASYMPTOTIC
            )
        if self.calculation is WilcoxonCalculation.EXACT and n_effective > self.exact_max_pairs:
            return None
        return self.calculation

    def _exact_tails(
        self,
        ranks: Sequence[float],
        positive_statistic: float,
    ) -> tuple[float, float]:
        scaled_ranks = [int(round(rank * 2.0)) for rank in ranks]
        observed = int(round(positive_statistic * 2.0))
        maximum = sum(scaled_ranks)
        counts = [0] * (maximum + 1)
        counts[0] = 1
        reachable = 0
        for rank in scaled_ranks:
            for total in range(reachable, -1, -1):
                if counts[total]:
                    counts[total + rank] += counts[total]
            reachable += rank
        outcomes = 2 ** len(scaled_ranks)
        p_worse = sum(counts[: observed + 1]) / outcomes
        p_better = sum(counts[observed:]) / outcomes
        return min(1.0, p_better), min(1.0, p_worse)

    def _asymptotic_tails(
        self,
        ranks: Sequence[float],
        positive_statistic: float,
    ) -> tuple[float, float, float]:
        expected = sum(ranks) / 2.0
        standard_deviation = math.sqrt(sum(rank * rank for rank in ranks) / 4.0)
        delta = positive_statistic - expected
        correction = 0.0
        if self.continuity_correction:
            correction = 0.5 if delta > 0.0 else -0.5 if delta < 0.0 else 0.0
        z_statistic = (delta - correction) / standard_deviation
        distribution = statistics.NormalDist()
        p_worse = distribution.cdf(z_statistic)
        p_better = distribution.cdf(-z_statistic)
        return min(1.0, p_better), min(1.0, p_worse), z_statistic

    def _unavailable(
        self,
        numeric: Sequence[float],
        wins: int,
        losses: int,
        ties: int,
        has_rank_ties: bool,
        positive_statistic: float,
        negative_statistic: float,
        reason: str,
    ) -> PairedTestResult:
        return PairedTestResult(
            method=PairedTestMethod.WILCOXON.value,
            alternative=self.alternative.value,
            calculation_requested=self.calculation.value,
            calculation_used=None,
            statistic=None,
            positive_statistic=float(positive_statistic),
            negative_statistic=float(negative_statistic),
            p_value=None,
            p_value_two_sided=None,
            p_value_better=None,
            p_value_worse=None,
            z_statistic=None,
            n_pairs=len(numeric),
            n_effective=wins + losses,
            n_zero=ties,
            wins=wins,
            losses=losses,
            ties=ties,
            has_rank_ties=has_rank_ties,
            status="unavailable",
            reason=reason,
            zero_method=self.zero_method.value,
            continuity_correction=self.continuity_correction,
            exact_max_pairs=self.exact_max_pairs,
            zero_tolerance=self.zero_tolerance,
            round_decimals=self.round_decimals,
        )
