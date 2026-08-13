"""Portable lifecycle states for local and scheduled work."""

from enum import Enum


class JobState(str, Enum):
    """Normalize provider-specific scheduler states."""

    CREATED = "created"
    STAGING = "staging"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        """Return whether no further state transition is expected."""
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED, self.TIMEOUT}
