"""Durable phases of an artifact-retention transaction."""

from enum import Enum


class ArtifactRetentionPhase(str, Enum):
    """Distinguish rollback-safe work from forward-only commit recovery."""

    PREPARED = "prepared"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"
    COMMITTING = "committing"
