"""Per-record failure policy for preprocessing pipelines."""

from enum import Enum


class PreprocessingErrorPolicy(str, Enum):
    """Choose whether one failed record aborts or is recorded and skipped."""

    FAIL = "fail"
    SKIP = "skip"
