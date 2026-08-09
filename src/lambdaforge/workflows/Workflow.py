"""High-level workflow DAG API."""

from __future__ import annotations

from pathlib import Path

from lambdaforge.workflows.WorkflowConfig import WorkflowConfig
from lambdaforge.workflows.WorkflowPlan import WorkflowPlan
from lambdaforge.workflows.WorkflowResult import WorkflowResult
from lambdaforge.workflows.WorkflowValidationReport import WorkflowValidationReport


class Workflow:
    """Validate, inspect and execute one workflow document."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Workflow:
        """Load a workflow YAML document."""
        return cls(WorkflowConfig.from_yaml(path))

    def inspect(self) -> WorkflowPlan:
        """Build a side-effect-free topological execution plan."""
        from lambdaforge.workflows.WorkflowRunner import WorkflowRunner

        return WorkflowRunner().plan(self.config)

    def validate(self, *, check_imports: bool = True) -> WorkflowValidationReport:
        """Validate the graph and every composed node without executing user code."""
        from lambdaforge.workflows.WorkflowValidator import WorkflowValidator

        return WorkflowValidator().validate(self.config, check_imports=check_imports)

    def run(self, *, dry_run: bool = False) -> WorkflowResult | WorkflowPlan:
        """Execute the DAG or return its plan."""
        from lambdaforge.workflows.WorkflowRunner import WorkflowRunner

        return WorkflowRunner().run(self.config, dry_run=dry_run)
