"""Configurable statistical comparison objects for experiment aggregation."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.statistics.intervals.BootstrapConfidenceInterval import (
        BootstrapConfidenceInterval,
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
    from lambdaforge.experiments.statistics.paired.PairedAlternative import (
        PairedAlternative,
    )
    from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
    from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult
    from lambdaforge.experiments.statistics.paired.SignTest import SignTest
    from lambdaforge.experiments.statistics.paired.WilcoxonCalculation import (
        WilcoxonCalculation,
    )
    from lambdaforge.experiments.statistics.paired.WilcoxonSignedRankTest import (
        WilcoxonSignedRankTest,
    )
    from lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod import (
        WilcoxonZeroMethod,
    )
    from lambdaforge.experiments.statistics.StatisticalComparisonConfig import (
        StatisticalComparisonConfig,
    )
    from lambdaforge.experiments.statistics.StatisticalComparisonEngine import (
        StatisticalComparisonEngine,
    )

LazyExports.install(
    __name__,
    {
        name: (module, name)
        for name, module in {
            "StatisticalComparisonConfig": (
                "lambdaforge.experiments.statistics.StatisticalComparisonConfig"
            ),
            "StatisticalComparisonEngine": (
                "lambdaforge.experiments.statistics.StatisticalComparisonEngine"
            ),
            "BootstrapConfidenceInterval": (
                "lambdaforge.experiments.statistics.intervals.BootstrapConfidenceInterval"
            ),
            "ConfidenceIntervalMethod": (
                "lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod"
            ),
            "ConfidenceIntervalResult": (
                "lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult"
            ),
            "NormalConfidenceInterval": (
                "lambdaforge.experiments.statistics.intervals.NormalConfidenceInterval"
            ),
            "PairedAlternative": ("lambdaforge.experiments.statistics.paired.PairedAlternative"),
            "PairedTestMethod": ("lambdaforge.experiments.statistics.paired.PairedTestMethod"),
            "PairedTestResult": ("lambdaforge.experiments.statistics.paired.PairedTestResult"),
            "SignTest": "lambdaforge.experiments.statistics.paired.SignTest",
            "WilcoxonCalculation": (
                "lambdaforge.experiments.statistics.paired.WilcoxonCalculation"
            ),
            "WilcoxonSignedRankTest": (
                "lambdaforge.experiments.statistics.paired.WilcoxonSignedRankTest"
            ),
            "WilcoxonZeroMethod": ("lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod"),
        }.items()
    },
)

__all__ = [
    "BootstrapConfidenceInterval",
    "ConfidenceIntervalMethod",
    "ConfidenceIntervalResult",
    "NormalConfidenceInterval",
    "PairedAlternative",
    "PairedTestMethod",
    "PairedTestResult",
    "SignTest",
    "StatisticalComparisonConfig",
    "StatisticalComparisonEngine",
    "WilcoxonCalculation",
    "WilcoxonSignedRankTest",
    "WilcoxonZeroMethod",
]
