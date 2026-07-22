"""Composition root for configurable interval and paired-test strategies."""

from __future__ import annotations

from collections.abc import Sequence

from lambdaforge.experiments.statistics.intervals.BootstrapConfidenceInterval import (
    BootstrapConfidenceInterval,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalEstimator import (
    ConfidenceIntervalEstimator,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod import (
    ConfidenceIntervalMethod,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult import (
    ConfidenceIntervalResult,
)
from lambdaforge.experiments.statistics.intervals.NormalConfidenceInterval import (
    NormalConfidenceInterval,
)
from lambdaforge.experiments.statistics.paired.PairedAlternative import PairedAlternative
from lambdaforge.experiments.statistics.paired.PairedTest import PairedTest
from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult
from lambdaforge.experiments.statistics.paired.SignTest import SignTest
from lambdaforge.experiments.statistics.paired.WilcoxonSignedRankTest import (
    WilcoxonSignedRankTest,
)
from lambdaforge.experiments.statistics.StatisticalComparisonConfig import (
    StatisticalComparisonConfig,
)


class StatisticalComparisonEngine:
    """Build immutable statistical collaborators from one validated protocol."""

    def __init__(self, config: StatisticalComparisonConfig) -> None:
        self.config = config
        self._interval = self._build_interval()
        self._legacy_interval = NormalConfidenceInterval(0.95)
        self._selected_test = self._build_test()
        self._legacy_sign_test = SignTest(
            PairedAlternative.OBSERVED_DIRECTION,
            zero_tolerance=config.zero_tolerance,
        )

    def confidence_interval(
        self,
        improvements: Sequence[float],
        *,
        identity: Sequence[str],
    ) -> ConfidenceIntervalResult:
        """Estimate the configured interval for paired improvements."""
        return self._interval.compute(improvements, identity=identity)

    def legacy_ci95(self, improvements: Sequence[float]) -> ConfidenceIntervalResult:
        """Retain the historical normal 95-percent columns."""
        return self._legacy_interval.compute(improvements)

    def paired_test(self, improvements: Sequence[float]) -> PairedTestResult:
        """Run the selected paired test."""
        return self._selected_test.compute(improvements)

    def legacy_sign_test(self, improvements: Sequence[float]) -> PairedTestResult:
        """Always produce the historical sign-test diagnostics."""
        return self._legacy_sign_test.compute(improvements)

    def _build_interval(self) -> ConfidenceIntervalEstimator:
        if self.config.confidence_interval_method is ConfidenceIntervalMethod.NORMAL:
            return NormalConfidenceInterval(self.config.confidence_level)
        return BootstrapConfidenceInterval(
            self.config.confidence_level,
            resamples=self.config.bootstrap_resamples,
            seed=self.config.bootstrap_seed,
            batch_size=self.config.bootstrap_batch_size,
            max_batch_elements=self.config.bootstrap_max_batch_elements,
        )

    def _build_test(self) -> PairedTest:
        if self.config.paired_test_method is PairedTestMethod.SIGN:
            return SignTest(
                self.config.paired_alternative,
                zero_tolerance=self.config.zero_tolerance,
            )
        return WilcoxonSignedRankTest(
            self.config.paired_alternative,
            calculation=self.config.wilcoxon_calculation,
            zero_method=self.config.wilcoxon_zero_method,
            continuity_correction=self.config.wilcoxon_continuity_correction,
            exact_max_pairs=self.config.wilcoxon_exact_max_pairs,
            zero_tolerance=self.config.zero_tolerance,
            round_decimals=self.config.round_decimals,
        )
