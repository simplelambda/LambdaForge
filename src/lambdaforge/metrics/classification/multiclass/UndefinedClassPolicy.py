"""Policies for classes with undefined one-vs-rest curves."""

from enum import Enum


class UndefinedClassPolicy(str, Enum):
    """Control how macro and weighted reductions handle undefined classes."""

    IGNORE = "ignore"
    NAN = "nan"
    ZERO = "zero"
