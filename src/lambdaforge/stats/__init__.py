"""Generic statistical evidence; importing this namespace does not import numerical backends."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .bootstrap import bootstrap_quantile, bootstrap_statistic
    from .BootstrapResult import BootstrapResult
    from .CalibrationResult import CalibrationResult
    from .EmpiricalNullCalibrator import EmpiricalNullCalibrator

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.stats.{name}", name)
        for name in ("EmpiricalNullCalibrator", "CalibrationResult", "BootstrapResult")
    }
    | {
        name: ("lambdaforge.stats.bootstrap", name)
        for name in ("bootstrap_statistic", "bootstrap_quantile")
    },
)

__all__ = [
    "EmpiricalNullCalibrator",
    "CalibrationResult",
    "BootstrapResult",
    "bootstrap_statistic",
    "bootstrap_quantile",
]
