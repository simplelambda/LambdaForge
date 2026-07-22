"""Zero-difference conventions for Wilcoxon signed ranks."""

from enum import Enum


class WilcoxonZeroMethod(str, Enum):
    """Control how zero paired differences contribute to ranks."""

    WILCOX = "wilcox"
    PRATT = "pratt"
    ZSPLIT = "zsplit"
