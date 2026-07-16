"""Optimization directions for monitored Trainer values."""

from enum import Enum


class MonitorMode(str, Enum):
    """Name the supported checkpoint and early-stopping directions."""

    MIN = "min"
    MAX = "max"
