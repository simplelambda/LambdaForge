"""Named checkpoint-selection policies for loading trained runs."""

from enum import Enum


class CheckpointChoice(str, Enum):
    """Checkpoint aliases understood by :class:`RunLoader`."""

    AUTO = "auto"
    BEST = "best"
    LAST = "last"
