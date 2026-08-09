"""Terminal states for generic LambdaForge task executions."""

from enum import Enum


class TaskStatus(str, Enum):
    """Stable machine-readable states written by the generic task runner."""

    OK = "ok"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DRY_RUN = "dry_run"
    UNKNOWN = "unknown"
