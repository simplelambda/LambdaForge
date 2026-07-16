"""LambdaForge: object-oriented infrastructure for reproducible ML training."""

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.LambdaForge import LambdaForge

__version__ = LambdaForge.VERSION
__all__ = ["Experiment", "LambdaForge", "__version__"]
