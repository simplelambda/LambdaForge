"""Immutable, serializable plan for one generic task execution."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.tasks.TaskPlanAction import TaskPlanAction


class TaskExecutionPlan(JsonResult):
    """Describe task identity, paths and action without constructing user code."""

    def __init__(
        self,
        *,
        name: str,
        run_dir: str | Path,
        suite_dir: str | Path,
        config_fingerprint: str,
        task_target: str,
        action: TaskPlanAction | str,
        reason: str,
        required_artifacts: Sequence[str] = (),
        inputs: Sequence[Mapping[str, Any]] = (),
        execution: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = str(name)
        self.run_dir = str(run_dir)
        self.suite_dir = str(suite_dir)
        self.config_fingerprint = str(config_fingerprint)
        self.task_target = str(task_target)
        self.action = TaskPlanAction(action)
        self.reason = str(reason)
        self.required_artifacts = tuple(str(path) for path in required_artifacts)
        self.inputs = tuple(FrozenJsonMapping(value) for value in inputs)
        self.execution = FrozenJsonMapping(execution or {"mode": "sequential"})
        self._freeze_mapping(self.to_dict())

    @property
    def will_run(self) -> bool:
        """Return whether execution would construct and invoke the task."""
        return self.action is TaskPlanAction.RUN

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON plan envelope."""
        return {
            "plan_version": 1,
            "kind": "task",
            "name": self.name,
            "action": self.action.value,
            "reason": self.reason,
            "task_target": self.task_target,
            "config_fingerprint": self.config_fingerprint,
            "suite_dir": self.suite_dir,
            "run_dir": self.run_dir,
            "required_artifacts": list(self.required_artifacts),
            "inputs": [copy.deepcopy(value) for value in self.inputs],
            "execution": copy.deepcopy(self.execution),
        }

    def summary(self) -> str:
        """Render a concise inspect/dry-run description."""
        return (
            f"Task plan: {self.name} action={self.action.value} target={self.task_target} "
            f"run_dir={self.run_dir} ({self.reason})."
        )
