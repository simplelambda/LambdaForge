"""Checkpoint roles that can survive post-aggregation retention."""

from enum import Enum


class CheckpointRetention(str, Enum):
    """Select which LambdaForge checkpoint roles must be retained."""

    ALL = "all"
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"

    @property
    def roles(self) -> tuple[str, ...]:
        """Return the concrete roles that need unambiguous resolution."""
        if self is CheckpointRetention.BEST:
            return ("best",)
        if self is CheckpointRetention.LAST:
            return ("last",)
        if self is CheckpointRetention.LAST_AND_BEST:
            return ("best", "last")
        return ()
