"""Lifecycle of an externally detached experiment launcher."""

from enum import Enum


class DetachedStatus(str, Enum):
    """States persisted for a detached launcher process."""

    LAUNCHED = "launched"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
