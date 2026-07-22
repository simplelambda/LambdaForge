"""Execution modes for artifact-retention policies."""

from enum import Enum


class ArtifactRetentionMode(str, Enum):
    """Control whether retention is disabled, previewed or applied automatically."""

    DISABLED = "disabled"
    PREVIEW = "preview"
    APPLY = "apply"
