"""Workload hints for bounded preprocessing concurrency."""

from enum import Enum


class PreprocessingWorkload(str, Enum):
    """Describe the dominant work without exposing executor implementation details."""

    AUTO = "auto"
    IO = "io"
    CPU = "cpu"
    GPU = "gpu"
