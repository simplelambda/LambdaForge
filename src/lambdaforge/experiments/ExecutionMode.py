"""Experiment execution strategies."""

from enum import Enum


class ExecutionMode(str, Enum):
    """Supported relationships between experiment runs and GPU processes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DDP = "ddp"
