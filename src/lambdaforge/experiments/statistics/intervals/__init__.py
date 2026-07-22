"""Confidence-interval strategies used by experiment comparisons."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
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

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.experiments.statistics.intervals.{name}", name)
        for name in (
            "BootstrapConfidenceInterval",
            "ConfidenceIntervalEstimator",
            "ConfidenceIntervalMethod",
            "ConfidenceIntervalResult",
            "NormalConfidenceInterval",
        )
    },
)

__all__ = [
    "BootstrapConfidenceInterval",
    "ConfidenceIntervalEstimator",
    "ConfidenceIntervalMethod",
    "ConfidenceIntervalResult",
    "NormalConfidenceInterval",
]
