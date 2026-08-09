"""Explicit attempt-state semantics."""

from enum import Enum


class AttemptMode(str, Enum):
    """Distinguish state reuse, clean rerun, failure retry and identity fork."""

    RESUME = "resume"
    RESTART = "restart"
    RETRY = "retry"
    FORK = "fork"
