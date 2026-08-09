"""Base contract for arbitrary reproducible LambdaForge work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskOutput import TaskOutput


class Task(ABC):
    """Execute one non-training unit of work.

    Inheritance is recommended for reusable plugins but is not required for a
    fully qualified YAML ``target``. The task runner also accepts an object
    exposing a compatible ``run(context)`` or zero-argument ``run()`` method.
    """

    @abstractmethod
    def run(self, context: TaskContext) -> TaskOutput | Mapping[str, Any] | None:
        """Perform the work and return structured or mapping-shaped outputs."""
