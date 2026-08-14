"""Kinds of runnable LambdaForge configuration."""

from enum import Enum


class ConfigurationKind(str, Enum):
    """Identify the runner selected for one authoring document."""

    EXPERIMENT = "experiment"
    DATASET = "dataset"
    TASK = "task"
    WORKFLOW = "workflow"
