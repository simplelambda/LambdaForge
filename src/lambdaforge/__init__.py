"""LambdaForge public researcher API."""

from typing import TYPE_CHECKING

from lambdaforge import clustering
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.work.Work import Work

LazyExports.install(__name__, {"Work": ("lambdaforge.work.Work", "Work")})

__version__ = LambdaForgeVersion.CURRENT
__all__ = ["Work", "__version__", "clustering"]
