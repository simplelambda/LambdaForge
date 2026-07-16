"""Checkpoint policy values accepted by the training API."""

from enum import Enum


class CheckpointPolicy(str, Enum):
    """Name the supported checkpoint retention strategies."""

    NONE = "none"
    LAST = "last"
    BEST = "best"
    LAST_AND_BEST = "last_and_best"
    ALL = "all"
