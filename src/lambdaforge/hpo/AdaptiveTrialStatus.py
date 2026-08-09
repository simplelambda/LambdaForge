"""Lifecycle states for adaptive configuration/seed trials."""

from enum import Enum


class AdaptiveTrialStatus(str, Enum):
    """Distinguish controller pruning from training and infrastructure outcomes."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    EARLY_STOPPED = "early_stopped"
    HPO_PRUNED = "hpo_pruned"
    OOM_GPU = "oom_gpu"
    FAILED = "failed"
    CANCELLED = "cancelled"
