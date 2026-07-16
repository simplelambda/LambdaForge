"""Lifecycle states written to experiment result files."""

from enum import Enum


class RunStatus(str, Enum):
    """Stable machine-readable statuses for a materialized experiment run."""

    OK = "ok"
    FAILED = "failed"
    DRY_RUN = "dry_run"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"
