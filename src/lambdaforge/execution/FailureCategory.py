"""Portable terminal failure taxonomy."""

from enum import Enum


class FailureCategory(str, Enum):
    """Classify failures for retry and reporting policy."""

    CANCELLED = "cancelled"
    PREEMPTED = "preempted"
    CPU_OOM = "cpu_oom"
    GPU_OOM = "gpu_oom"
    TRANSIENT = "transient"
    USER_ERROR = "user_error"
    UNKNOWN = "unknown"
