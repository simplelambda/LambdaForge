"""Side-effect-free validation for complete workflow DAGs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
from lambdaforge.experiments.ValidationReport import ValidationReport
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskValidationReport import TaskValidationReport
from lambdaforge.tasks.TaskValidator import TaskValidator
from lambdaforge.workflows.WorkflowConfig import WorkflowConfig
from lambdaforge.workflows.WorkflowRunner import WorkflowRunner
from lambdaforge.workflows.WorkflowValidationReport import WorkflowValidationReport


class WorkflowValidator:
    """Validate a workflow graph and every composed node without execution."""

    def validate_file(
        self, path: str | Path, *, check_imports: bool = True
    ) -> WorkflowValidationReport:
        """Load and validate one workflow document without creating artifacts."""
        source = Path(path).resolve()
        try:
            config = WorkflowConfig.from_yaml(source)
        except Exception as error:
            return WorkflowValidationReport(
                source=str(source),
                errors=(f"{error.__class__.__name__}: {error}",),
                imports_checked=False,
            )
        return self.validate(config, check_imports=check_imports)

    def validate(
        self, config: WorkflowConfig, *, check_imports: bool = True
    ) -> WorkflowValidationReport:
        """Return graph and per-node validation facts in declaration order."""
        errors: list[str] = []
        warnings: list[str] = []
        node_reports: list[dict[str, Any]] = []
        task_validator = TaskValidator()
        experiment_validator = ExperimentValidator()
        node_by_name = {node.name: node for node in config.nodes}
        ordered_names = (name for level in WorkflowRunner().plan(config).levels for name in level)
        for node in (node_by_name[name] for name in ordered_names):
            try:
                data, source, resolution = node.materialize()
                report: TaskValidationReport | ValidationReport
                if data.get("kind") == "task":
                    task_config = TaskConfig(data, source=source, resolution=resolution)
                    report = task_validator.validate(task_config, check_imports=check_imports)
                else:
                    if resolution is not None and resolution.contains_secrets:
                        raise ValueError(
                            "Experiment workflow nodes cannot contain composed secrets."
                        )
                    experiment_config = ExperimentConfig(data, source=source)
                    report = experiment_validator.validate(
                        experiment_config, check_imports=check_imports
                    )
                payload = report.to_dict()
                payload["name"] = node.name
                node_reports.append(payload)
                errors.extend(f"node {node.name}: {item}" for item in report.errors)
                warnings.extend(f"node {node.name}: {item}" for item in report.warnings)
            except Exception as error:
                message = f"node {node.name}: {error.__class__.__name__}: {error}"
                errors.append(message)
                node_reports.append({"name": node.name, "valid": False, "errors": [message]})
        return WorkflowValidationReport(
            source=str(config.source) if config.source is not None else None,
            name=config.name,
            errors=tuple(dict.fromkeys(errors)),
            warnings=tuple(dict.fromkeys(warnings)),
            imports_checked=check_imports,
            node_reports=tuple(node_reports),
        )
