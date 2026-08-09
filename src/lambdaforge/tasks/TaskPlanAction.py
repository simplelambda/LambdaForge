"""Resolved action selected by a task execution plan."""

from enum import Enum


class TaskPlanAction(str, Enum):
    """Distinguish real execution from reuse of a valid successful result."""

    RUN = "run"
    SKIP = "skip"
