"""Stable statuses returned by the retention lifecycle."""

from enum import Enum


class ArtifactRetentionStatus(str, Enum):
    """Represent a safe terminal or preview retention outcome."""

    DISABLED = "disabled"
    NOT_READY = "not_ready"
    PREVIEW = "preview"
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    ROLLED_BACK = "rolled_back"
    PARTIAL = "partial"
    CONFLICT = "conflict"
