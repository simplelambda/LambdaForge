"""Supported statistical tests for paired experiment differences."""

from enum import Enum


class PairedTestMethod(str, Enum):
    """Select the inferential test applied to paired improvements."""

    SIGN = "sign"
    WILCOXON = "wilcoxon"
