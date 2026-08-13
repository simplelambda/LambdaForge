"""Checkpoint roles available to post-run actions."""

from enum import Enum


class PostRunCheckpoint(str, Enum):
    """Select a stable checkpoint without retaining live training objects."""

    BEST = "best"
    LAST = "last"
    CURRENT = "current"
    NONE = "none"
