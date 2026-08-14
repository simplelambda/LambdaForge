"""High-level object API for one dataset recipe."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lambdaforge.data.DatasetBuildPlan import DatasetBuildPlan
from lambdaforge.data.DatasetBuildResult import DatasetBuildResult
from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetRecipeConfig import DatasetRecipeConfig
from lambdaforge.data.DatasetRecipeSchemaCatalog import DatasetRecipeSchemaCatalog
from lambdaforge.data.DatasetRecipeValidationReport import DatasetRecipeValidationReport
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskRun import TaskRun


class DatasetRecipe:
    """Validate, inspect and execute a recipe without conflating its build/version."""

    def __init__(self, config: DatasetRecipeConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetRecipe:
        """Load one composed recipe."""
        return cls(DatasetRecipeConfig.from_yaml(path))

    def validate(self, *, check_imports: bool = True) -> DatasetRecipeValidationReport:
        """Validate recipe structure and every referenced task without execution."""
        values = {
            "kind": "dataset",
            "schema_version": "1.0",
            "dataset": self.config.dataset,
            "stages": {
                stage.name: {
                    "task": str(stage.task) if isinstance(stage.task, Path) else dict(stage.task),
                    "needs": list(stage.needs),
                    "bindings": dict(stage.bindings),
                    "required": stage.required,
                    "reuse": stage.reuse,
                }
                for stage in self.config.stages
            },
            "publish": self.config.publish,
        }
        errors = list(DatasetRecipeSchemaCatalog().validation_errors(values))
        for stage in self.config.stages:
            try:
                task = (
                    TaskRun.from_yaml(stage.task)
                    if isinstance(stage.task, Path)
                    else TaskRun(
                        TaskConfig(
                            stage.task,
                            source=(
                                self.config.source_dir
                                / ".lambdaforge-embedded-dataset-stage.yaml"
                            ),
                        )
                    )
                )
                report = task.validate(check_imports=check_imports)
                errors.extend(f"stage {stage.name}: {error}" for error in report.errors)
            except Exception as error:
                errors.append(f"stage {stage.name}: {error.__class__.__name__}: {error}")
        return DatasetRecipeValidationReport(tuple(errors))

    def inspect(
        self,
        *,
        on: str = "local",
        force: bool = False,
        force_stages: Sequence[str] = (),
    ) -> DatasetBuildPlan:
        """Return stage reuse/publication decisions without constructing project code."""
        return DatasetBuildService().plan(
            self.config,
            cluster=on,
            force=force,
            force_stages=force_stages,
        )

    def run(
        self,
        *,
        force: bool = False,
        force_stages: Sequence[str] = (),
    ) -> DatasetBuildResult:
        """Execute a local build and atomically publish its resulting DatasetVersion."""
        return DatasetBuildService().build(
            self.config,
            force=force,
            force_stages=force_stages,
        )
