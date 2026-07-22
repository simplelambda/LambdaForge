"""Paired statistical tests used by experiment comparisons."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.statistics.paired.PairedAlternative import (
        PairedAlternative,
    )
    from lambdaforge.experiments.statistics.paired.PairedTest import PairedTest
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

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.experiments.statistics.paired.{name}", name)
        for name in (
            "PairedAlternative",
            "PairedTest",
            "PairedTestMethod",
            "PairedTestResult",
            "SignTest",
            "WilcoxonCalculation",
            "WilcoxonSignedRankTest",
            "WilcoxonZeroMethod",
        )
    },
)

__all__ = [
    "PairedAlternative",
    "PairedTest",
    "PairedTestMethod",
    "PairedTestResult",
    "SignTest",
    "WilcoxonCalculation",
    "WilcoxonSignedRankTest",
    "WilcoxonZeroMethod",
]
