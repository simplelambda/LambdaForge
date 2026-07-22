"""P-value calculation modes for the Wilcoxon signed-rank test."""

from enum import Enum


class WilcoxonCalculation(str, Enum):
    """Choose exact enumeration, a normal approximation or bounded auto mode."""

    AUTO = "auto"
    EXACT = "exact"
    ASYMPTOTIC = "asymptotic"
