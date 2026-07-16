"""Built-in logger modes accepted by the training API."""

from enum import Enum


class LoggerMode(str, Enum):
    """Name the built-in metric logger strategies."""

    NONE = "none"
    CSV = "csv"
    LIGHTNING_CSV = "lightning_csv"
