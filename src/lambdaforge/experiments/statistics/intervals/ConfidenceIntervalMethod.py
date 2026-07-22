"""Supported confidence-interval strategies for paired comparisons."""

from enum import Enum


class ConfidenceIntervalMethod(str, Enum):
    """Select how LambdaForge estimates uncertainty around a paired mean."""

    NORMAL = "normal"
    BOOTSTRAP_PERCENTILE = "bootstrap_percentile"
