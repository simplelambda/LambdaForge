"""Actions supported by artifact-retention rules."""

from enum import Enum


class ArtifactRetentionAction(str, Enum):
    """Describe a generic artifact operation."""

    COMPRESS = "compress"
    PRUNE = "prune"
    PRUNE_CHECKPOINT = "prune_checkpoint"
