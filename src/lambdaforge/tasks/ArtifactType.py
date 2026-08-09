"""Closed artifact categories shared by tasks and preprocessing."""

from enum import Enum


class ArtifactType(str, Enum):
    """Describe the scientific role of a task-produced path."""

    FILE = "file"
    DIRECTORY = "directory"
    DATASET = "dataset"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    FIGURE = "figure"
    TABLE = "table"
    REPORT = "report"
    CHECKPOINT = "checkpoint"
    OTHER = "other"
