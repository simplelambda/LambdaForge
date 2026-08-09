"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import importlib
import inspect as python_inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff
from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import ExperimentSchemaCatalog
from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
    MigrationPreviewFormat,
)
from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.registry.ExperimentRegistry import ExperimentRegistry
from lambdaforge.registry.LocalDashboard import LocalDashboard
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
from lambdaforge.tasks.TaskStatus import TaskStatus
from lambdaforge.tasks.TaskValidator import TaskValidator
from lambdaforge.workflows.Workflow import Workflow
from lambdaforge.workflows.WorkflowPlan import WorkflowPlan
from lambdaforge.workflows.WorkflowSchemaCatalog import WorkflowSchemaCatalog
from lambdaforge.workflows.WorkflowValidator import WorkflowValidator


class CommandLineInterface:
    """Parse CLI arguments and dispatch them to the public object API."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the CLI and return a process exit code."""
        parser = cls._parser()
        arguments = parser.parse_args(argv)
        if arguments.command == "init":
            return cls._initialize(arguments.directory, force=arguments.force)
        if arguments.command == "target":
            try:
                module_name, symbol_name = arguments.path.rsplit(".", 1)
                symbol = getattr(importlib.import_module(module_name), symbol_name)
                print(f"{arguments.path}{python_inspect.signature(symbol)}")
                if python_inspect.getdoc(symbol):
                    print(python_inspect.getdoc(symbol))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "explain":
            try:
                catalogs = {
                    "task": TaskSchemaCatalog,
                    "experiment": ExperimentSchemaCatalog,
                    "workflow": WorkflowSchemaCatalog,
                }
                schema = catalogs[arguments.kind]().schema()
                node = cls._schema_node(schema, arguments.path)
                print(json.dumps(node, indent=2))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "compose":
            try:
                resolved = ConfigurationComposer().resolve(arguments.config)
                print(
                    json.dumps(
                        {
                            "values": resolved.materialized(),
                            "provenance": dict(resolved.provenance),
                            "sources": [str(value) for value in resolved.sources],
                        },
                        indent=2,
                    )
                )
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "diff":
            try:
                composer = ConfigurationComposer()
                left = composer.resolve(arguments.left).materialized()
                right = composer.resolve(arguments.right).materialized()
                print(json.dumps(ConfigurationDiff().compare(left, right), indent=2))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "registry":
            try:
                registry = ExperimentRegistry(arguments.root)
                if arguments.output:
                    print(registry.export(arguments.output))
                else:
                    print(json.dumps(registry.query(), indent=2, default=str))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "dashboard":
            try:
                print(LocalDashboard().build(arguments.root, arguments.output))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "plugins":
            descriptors = PluginRegistry.default().discover(arguments.kind)
            payload = [descriptor.to_dict() for descriptor in descriptors]
            if arguments.json:
                print(json.dumps(payload, indent=2))
            elif not payload:
                print("No LambdaForge plugins found.")
            else:
                for plugin in payload:
                    provider = plugin["distribution"] or "unknown distribution"
                    version = f" {plugin['version']}" if plugin["version"] else ""
                    print(
                        f"{plugin['kind']}:{plugin['name']} -> {plugin['value']} "
                        f"[{provider}{version}]"
                    )
            return 0
        if arguments.command == "validate":
            if cls._is_workflow(arguments.config):
                workflow_report = WorkflowValidator().validate_file(
                    arguments.config, check_imports=not arguments.no_imports
                )
                print(
                    json.dumps(workflow_report.to_dict(), indent=2)
                    if arguments.json
                    else workflow_report.summary()
                )
                return 0 if workflow_report.is_valid else 1
            validator = (
                TaskValidator()
                if TaskConfig.is_task_file(arguments.config)
                else ExperimentValidator()
            )
            report = validator.validate_file(
                arguments.config, check_imports=not arguments.no_imports
            )
            print(json.dumps(report.to_dict(), indent=2) if arguments.json else report.summary())
            return 0 if report.is_valid else 1
        if arguments.command == "migrate":
            if arguments.force and arguments.output is None:
                print("ERROR: --force requires --output.", file=sys.stderr)
                return 1
            if arguments.check and arguments.output is not None:
                print("ERROR: --check cannot be combined with --output.", file=sys.stderr)
                return 1
            try:
                result = ExperimentConfigMigrator.default().preview_file(
                    arguments.config,
                    target_version=arguments.target_version,
                )
                rendered = result.render(arguments.format)
                if rendered:
                    print(rendered, end="" if rendered.endswith("\n") else "\n")
                elif arguments.format == MigrationPreviewFormat.DIFF.value:
                    print(
                        f"Already at experiment Schema {result.target_version}; "
                        "no migration required."
                    )
                if arguments.check:
                    return 1 if result.changed else 0
                if arguments.output is not None:
                    result.write_yaml(arguments.output, overwrite=arguments.force)
                    print(f"Wrote migrated YAML: {arguments.output}", file=sys.stderr)
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "retain":
            try:
                experiment = Experiment.from_yaml(arguments.config)
                retention_result = (
                    experiment.apply_retention()
                    if arguments.apply
                    else experiment.preview_retention()
                )
                print(
                    json.dumps(retention_result.to_dict(), indent=2)
                    if arguments.json
                    else retention_result.summary()
                )
                successful = {"applied", "already_applied"} if arguments.apply else {"preview"}
                return 0 if retention_result.status.value in successful else 1
            except Exception as error:
                if arguments.json:
                    print(
                        json.dumps(
                            {
                                "status": "error",
                                "error_type": error.__class__.__name__,
                                "message": str(error),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(
                        f"ERROR: {error.__class__.__name__}: {error}",
                        file=sys.stderr,
                    )
                return 1
        if arguments.command == "results":
            try:
                source = arguments.source
                catalog = (
                    TaskRun.from_yaml(source).result_catalog()
                    if source.suffix.lower() in {".yaml", ".yml"}
                    and TaskConfig.is_task_file(source)
                    else Experiment.from_yaml(source).result_catalog()
                    if source.suffix.lower() in {".yaml", ".yml"}
                    else ResultCatalog(source)
                )
                records = catalog.records(
                    status=arguments.status,
                    include_archived=not arguments.no_archived,
                )
                if arguments.duplicates:
                    duplicate_ids = {
                        record.attempt_id
                        for group in catalog.duplicate_groups().values()
                        for record in group
                    }
                    records = tuple(
                        record for record in records if record.attempt_id in duplicate_ids
                    )
                index_path = catalog.write_index() if arguments.write_index else None
                if arguments.json:
                    print(
                        json.dumps(
                            {
                                "summary": catalog.summary(),
                                "index_path": str(index_path) if index_path else None,
                                "records": [record.to_dict() for record in records],
                            },
                            indent=2,
                        )
                    )
                else:
                    print(catalog.summary())
                    for record in records:
                        location = "archived" if record.archived else "current"
                        print(
                            f"{record.attempt_id}  {record.status:<11} {location:<8} "
                            f"{record.result_path}"
                        )
                    if index_path is not None:
                        print(f"Wrote result index: {index_path}")
                return 2 if arguments.fail_on_ambiguous and catalog.ambiguous_successes() else 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if cls._is_workflow(arguments.config):
            if arguments.command == "aggregate":
                print("ERROR: aggregate applies only to training experiment YAML.", file=sys.stderr)
                return 1
            workflow = Workflow.from_yaml(arguments.config)
            workflow_outcome = workflow.run(
                dry_run=arguments.command == "inspect" or arguments.dry_run
            )
            print(json.dumps(workflow_outcome.to_dict(), indent=2, default=str))
            return (
                0
                if isinstance(workflow_outcome, WorkflowPlan) or workflow_outcome.status == "ok"
                else 1
            )
        if TaskConfig.is_task_file(arguments.config):
            if arguments.command == "aggregate":
                print("ERROR: aggregate applies only to training experiment YAML.", file=sys.stderr)
                return 1
            task = TaskRun.from_yaml(arguments.config)
            if arguments.command == "inspect":
                print(json.dumps(task.inspect().to_dict(), indent=2))
                return 0
            task_overrides = (
                arguments.mode,
                arguments.gpus,
                arguments.jobs_per_gpu,
                arguments.devices_per_job,
                arguments.grace_seconds,
            )
            if any(value is not None for value in task_overrides):
                print(
                    "ERROR: experiment GPU overrides do not apply to kind: task.",
                    file=sys.stderr,
                )
                return 1
            task_outcome = task.run(dry_run=arguments.dry_run)
            if isinstance(task_outcome, TaskExecutionPlan):
                print(json.dumps(task_outcome.to_dict(), indent=2))
                return 0
            print(
                f"LambdaForge task {task_outcome.name!r} finished "
                f"with status={task_outcome.status.value}; "
                f"artifacts={len(task_outcome.artifacts)}."
            )
            return 0 if task_outcome.status is TaskStatus.OK else 1

        experiment = Experiment.from_yaml(arguments.config)
        if arguments.command == "inspect":
            inspection = experiment.inspect()
            inspection_payload = (
                inspection.to_dict() if hasattr(inspection, "to_dict") else inspection
            )
            print(json.dumps(inspection_payload, indent=2, default=str))
            return 0
        if arguments.command == "aggregate":
            experiment.aggregate(make_plots=not arguments.no_plots)
            return 0
        experiment_overrides = {
            "mode": arguments.mode,
            "gpus": arguments.gpus,
            "jobs_per_gpu": arguments.jobs_per_gpu,
            "devices_per_job": arguments.devices_per_job,
            "grace_seconds": arguments.grace_seconds,
        }
        results = experiment.run(
            dry_run=arguments.dry_run,
            execution_overrides=experiment_overrides,
            aggregate_plots=not arguments.no_plots,
        )
        if hasattr(results, "to_dict"):
            print(json.dumps(results.to_dict(), indent=2, default=str))
            summary = getattr(results, "summary", {})
            return 1 if summary.get("status") == "failed" else 0
        failed = sum(result.get("status") == "failed" for result in results)
        print(f"LambdaForge finished {len(results)} run(s); failed={failed}.")
        return 1 if failed else 0

    @staticmethod
    def _is_workflow(path: str | Path) -> bool:
        """Detect an explicit workflow root without constructing any user target."""
        try:
            return ConfigurationComposer().resolve(path).values.get("kind") == "workflow"
        except Exception:
            return False

    @staticmethod
    def _initialize(directory: Path, *, force: bool) -> int:
        """Create a minimal installable consumer project without overwriting by default."""
        files = {
            "pyproject.toml": """[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["lambdaforge>=0.4,<0.5"]

[tool.setuptools.packages.find]
where = ["src"]
""",
            "src/my_project/__init__.py": '"""Project-local datasets, models and tasks."""\n',
            "src/my_project/tasks.py": '''"""Project task implementations."""

import json

from lambdaforge.tasks import ArtifactDeclaration, Task, TaskContext, TaskOutput


class ExampleTask(Task):
    """Write one small JSON artifact through the public task contract."""

    def run(self, context: TaskContext) -> TaskOutput:
        """Create the configured example output."""
        path = context.output_path("output.json", create_parent=True)
        path.write_text(json.dumps({"status": "ready"}) + "\\n", encoding="utf-8")
        return TaskOutput(
            outputs={"status": "ready"},
            artifacts=[ArtifactDeclaration("output.json", media_type="application/json")],
        )
''',
            "experiments/task.yaml": """kind: task
schema_version: "1.0"
name: example
task:
  target: my_project.tasks.ExampleTask
required_artifacts: [output.json]
""",
            "README.md": """# My AI project

Create an environment, install LambdaForge and this package, then run:

```bash
python -m pip install -e .
lambdaforge validate experiments/task.yaml
lambdaforge inspect experiments/task.yaml
lambdaforge run experiments/task.yaml --dry-run
lambdaforge run experiments/task.yaml
```
""",
            "schemas/lambdaforge-task.schema.json": json.dumps(
                TaskSchemaCatalog().schema(), indent=2
            )
            + "\n",
            ".vscode/settings.json": json.dumps(
                {
                    "yaml.schemas": {
                        "./schemas/lambdaforge-task.schema.json": "experiments/task*.yaml"
                    }
                },
                indent=2,
            )
            + "\n",
            ".gitignore": """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/
.ipynb_checkpoints/
.lambdaforge/
.env
.env.*
!.env.example
runs/
dist/
build/
*.egg-info/
*.whl
lambdaforge-dashboard.html
slurm-*.out
slurm-*.err
""",
        }
        collisions = [directory / relative for relative in files if (directory / relative).exists()]
        if collisions and not force:
            print(f"ERROR: refusing to overwrite {collisions[0]}; use --force.", file=sys.stderr)
            return 1
        for relative, content in files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"Initialized LambdaForge consumer project: {directory.resolve()}")
        return 0

    @staticmethod
    def _schema_node(schema: dict[str, object], path: str) -> object:
        """Resolve a dotted JSON-Schema property path and local references."""
        node: object = schema
        for part in path.split(".") if path else ():
            if not isinstance(node, dict):
                raise KeyError(f"Schema path is not an object at {part!r}.")
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/"):
                node = schema
                for segment in reference[2:].split("/"):
                    node = node[segment]  # type: ignore[index]
            properties = node.get("properties") if isinstance(node, dict) else None
            if not isinstance(properties, dict) or part not in properties:
                raise KeyError(f"Unknown Schema path: {path!r}.")
            node = properties[part]
        return node

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="lambdaforge",
            description="Configure and run reproducible ML experiments and generic tasks.",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        initialize = subparsers.add_parser(
            "init", help="Create a minimal installable consumer project."
        )
        initialize.add_argument("directory", type=Path)
        initialize.add_argument("--force", action="store_true")
        target = subparsers.add_parser(
            "target", help="Print an importable target signature and docstring."
        )
        target.add_argument("path")
        explain = subparsers.add_parser(
            "explain", help="Explain one dotted experiment/task Schema property."
        )
        explain.add_argument("kind", choices=("experiment", "task", "workflow"))
        explain.add_argument("path", nargs="?", default="")
        compose = subparsers.add_parser(
            "compose", help="Resolve includes/interpolation and show redacted provenance."
        )
        compose.add_argument("config", type=Path)
        difference = subparsers.add_parser(
            "diff", help="Semantically compare two composed configurations."
        )
        difference.add_argument("left", type=Path)
        difference.add_argument("right", type=Path)
        registry = subparsers.add_parser("registry", help="Query or export result registry data.")
        registry.add_argument("root", type=Path)
        registry.add_argument("--output", type=Path)
        dashboard = subparsers.add_parser(
            "dashboard", help="Build a read-only local HTML result dashboard."
        )
        dashboard.add_argument("root", type=Path)
        dashboard.add_argument("--output", type=Path, default=Path("lambdaforge-dashboard.html"))
        run = subparsers.add_parser(
            "run", help="Execute an experiment, task or workflow YAML file."
        )
        run.add_argument("config", type=Path)
        run.add_argument("--dry-run", action="store_true")
        run.add_argument("--no-plots", action="store_true")
        run.add_argument("--mode", choices=("sequential", "parallel", "ddp"))
        run.add_argument("--gpus", help="Comma-separated logical GPU indices.")
        run.add_argument("--jobs-per-gpu", type=int)
        run.add_argument("--devices-per-job", type=int)
        run.add_argument("--grace-seconds", type=float)
        inspect = subparsers.add_parser(
            "inspect",
            help="Print expanded experiment runs or an immutable task/workflow plan as JSON.",
        )
        inspect.add_argument("config", type=Path)
        validate = subparsers.add_parser(
            "validate",
            help="Validate Schema, resources and import paths without running.",
        )
        validate.add_argument("config", type=Path)
        validate.add_argument(
            "--no-imports",
            action="store_true",
            help="Skip importing target/ref paths and resolving plugins.",
        )
        validate.add_argument(
            "--json", action="store_true", help="Print a machine-readable report."
        )
        migrate = subparsers.add_parser(
            "migrate",
            help="Preview a versioned YAML configuration migration.",
        )
        migrate.add_argument("config", type=Path)
        migrate.add_argument(
            "--target-version",
            help="Exact MAJOR.MINOR target; defaults to the current Schema.",
        )
        migrate.add_argument(
            "--format",
            choices=tuple(value.value for value in MigrationPreviewFormat),
            default=MigrationPreviewFormat.DIFF.value,
            help="Preview as a unified diff, full YAML or stable JSON.",
        )
        migrate.add_argument(
            "--output",
            type=Path,
            help="Atomically write to a different path; the source is never overwritten.",
        )
        migrate.add_argument(
            "--force",
            action="store_true",
            help="Allow --output to replace an existing destination.",
        )
        migrate.add_argument(
            "--check",
            action="store_true",
            help="Return 1 when a migration is required, otherwise 0; never write.",
        )
        plugins = subparsers.add_parser(
            "plugins",
            help="List installed entry-point plugins without importing their modules.",
        )
        plugins.add_argument(
            "--kind",
            choices=tuple(kind.value for kind in PluginKind),
            help="Restrict discovery to one plugin contract.",
        )
        plugins.add_argument("--json", action="store_true", help="Print metadata as JSON.")
        aggregate = subparsers.add_parser("aggregate", help="Rebuild suite aggregates from disk.")
        aggregate.add_argument("config", type=Path)
        aggregate.add_argument("--no-plots", action="store_true")
        retain = subparsers.add_parser(
            "retain",
            help="Preview or explicitly apply artifact retention after complete aggregation.",
        )
        retain.add_argument("config", type=Path)
        retain.add_argument(
            "--apply",
            action="store_true",
            help="Apply the transaction; omission is always read-only.",
        )
        retain.add_argument("--json", action="store_true", help="Print a machine-readable result.")
        results = subparsers.add_parser(
            "results",
            help="Audit current and archived attempts by scientific configuration identity.",
        )
        results.add_argument("source", type=Path, help="Experiment YAML or result-tree root.")
        results.add_argument("--status", choices=("ok", "failed", "interrupted", "dry_run"))
        results.add_argument(
            "--no-archived",
            action="store_true",
            help="Only show the canonical result.json for each run directory.",
        )
        results.add_argument(
            "--duplicates",
            action="store_true",
            help="Only show identities with more than one attempt.",
        )
        results.add_argument(
            "--write-index",
            action="store_true",
            help="Atomically write .lambdaforge/result-index.json under the suite root.",
        )
        results.add_argument(
            "--fail-on-ambiguous",
            action="store_true",
            help="Return exit code 2 when one identity has multiple successful attempts.",
        )
        results.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
        return parser
