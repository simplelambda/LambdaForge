"""Reduction policies for streaming multiclass curve metrics."""

from enum import Enum


class MulticlassCurveAverage(str, Enum):
    """Name the supported scalar reductions over one-vs-rest classes."""

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"
