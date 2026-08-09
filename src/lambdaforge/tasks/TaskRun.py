"""High-level object API for one generic task configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
from lambdaforge.tasks.TaskResult import TaskResult
from lambdaforge.tasks.TaskRunner import TaskRunner
from lambdaforge.tasks.TaskValidationReport import TaskValidationReport
from lambdaforge.tasks.TaskValidator import TaskValidator


class TaskRun:
    """Validate, inspect, execute and audit one non-training task document."""

    def __init__(self, config: TaskConfig | Mapping[str, Any]) -> None:
        self.config = config if isinstance(config, TaskConfig) else TaskConfig(config)
        self._runner: TaskRunner | None = None

    @property
    def runner(self) -> TaskRunner:
        """Create the concrete runner only when planning or execution is requested."""
        if self._runner is None:
            self._runner = TaskRunner()
        return self._runner

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskRun:
        """Load one generic task YAML into the object API."""
        return cls(TaskConfig.from_yaml(path))

    def validate(self, *, check_imports: bool = True) -> TaskValidationReport:
        """Validate Schema, imports and the task constructor without side effects."""
        return TaskValidator().validate(self.config, check_imports=check_imports)

    def inspect(self) -> TaskExecutionPlan:
        """Return the immutable execution plan without constructing user code."""
        report = self.validate(check_imports=False)
        if not report.is_valid:
            raise ValueError(report.summary())
        return self.runner.plan(self.config)

    def run(
        self,
        *,
        dry_run: bool = False,
        stop_event: Any = None,
    ) -> TaskResult | TaskExecutionPlan:
        """Execute the task or return its dry-run plan."""
        return self.runner.run(self.config, dry_run=dry_run, stop_event=stop_event)

    def result_catalog(self) -> ResultCatalog:
        """Create a shared result catalog across task fingerprints and attempts."""
        return ResultCatalog(self.config.suite_dir)
