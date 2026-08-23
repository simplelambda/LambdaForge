"""LambdaForge public researcher API."""

from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.work import Work

__version__ = LambdaForgeVersion.CURRENT
__all__ = ["Work", "__version__"]
