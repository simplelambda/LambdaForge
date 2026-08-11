"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import importlib
import inspect as python_inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.AuthoringSchemaCatalog import AuthoringSchemaCatalog
from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobState import JobState
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataService import DataService
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import ExperimentSchemaCatalog
from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
    MigrationPreviewFormat,
)
from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.registry.ExperimentRegistry import ExperimentRegistry
from lambdaforge.registry.LocalDashboard import LocalDashboard
from lambdaforge.reproducibility.IdentityExplainer import IdentityExplainer
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
            return cls._initialize(
                arguments.directory, force=arguments.force, template=arguments.template
            )
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
                if arguments.kind == "changes":
                    current = cls._scientific_payload(arguments.path)
                    previous = (
                        cls._scientific_payload(arguments.against)
                        if arguments.against is not None
                        else None
                    )
                    explanation = IdentityExplainer().compare(current, previous)
                    print(
                        json.dumps(explanation.to_dict(), indent=2)
                        if arguments.json
                        else explanation.summary()
                    )
                    return 0
                catalogs: dict[str, type[Any]] = {
                    "authoring": AuthoringSchemaCatalog,
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
        if arguments.command == "doctor":
            try:
                doctor_report = Doctor(ClusterCatalog.load(arguments.clusters)).check(arguments.on)
                print(
                    json.dumps(doctor_report.to_dict(), indent=2)
                    if arguments.json
                    else doctor_report.summary()
                )
                return 0 if doctor_report.ok else 1
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "clusters":
            try:
                cluster_catalog = ClusterCatalog.load(arguments.catalog)
                if arguments.cluster_command == "list":
                    cluster_payload = [
                        cluster_catalog.get(name).to_dict() for name in cluster_catalog.names()
                    ]
                    if arguments.json:
                        print(json.dumps(cluster_payload, indent=2))
                    else:
                        for profile_payload in cluster_payload:
                            print(
                                f"{profile_payload['name']:<16} "
                                f"{profile_payload['transport']:<6} "
                                f"{profile_payload['scheduler']:<6} "
                                f"{profile_payload['host'] or '-'}"
                            )
                    return 0
                cluster_profile = cluster_catalog.get(arguments.name)
                if arguments.cluster_command == "show":
                    print(json.dumps(cluster_profile.to_dict(), indent=2))
                    return 0
                if arguments.cluster_command == "test":
                    cluster_report = Doctor(cluster_catalog).check(cluster_profile.name)
                    print(
                        json.dumps(cluster_report.to_dict(), indent=2)
                        if arguments.json
                        else cluster_report.summary()
                    )
                    return 0 if cluster_report.ok else 1
                transport = ControlPlaneFactory().transport(cluster_profile)
                created = transport.run(("mkdir", "-p", cluster_profile.workspace))
                if created.returncode:
                    raise RuntimeError(created.stderr)
                checked = transport.run(
                    (
                        *cluster_profile.command_prefix,
                        cluster_profile.python,
                        "-c",
                        "import lambdaforge",
                    )
                )
                if checked.returncode:
                    raise RuntimeError(
                        "Workspace exists, but LambdaForge is unavailable in the configured "
                        "Python. Install the pinned release in that environment."
                    )
                print(
                    f"Cluster {cluster_profile.name!r} workspace and "
                    "existing environment are ready."
                )
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "jobs":
            try:
                jobs = JobService(ClusterCatalog.load(arguments.clusters))
                if arguments.job_command == "list":
                    job_records = jobs.list(cluster=arguments.cluster, state=arguments.state)
                    if arguments.json:
                        print(json.dumps([record.to_dict() for record in job_records], indent=2))
                    else:
                        for job_record in job_records:
                            print(
                                f"{job_record.job_id:<32} {job_record.state.value:<10} "
                                f"{job_record.cluster:<12} {job_record.scheduler_id or '-'}"
                            )
                    return 0
                if arguments.job_command == "status":
                    job_record = jobs.get(arguments.job_id)
                    print(
                        json.dumps(job_record.to_dict(), indent=2)
                        if arguments.json
                        else (
                            f"{job_record.job_id}: {job_record.state.value} on {job_record.cluster}"
                        )
                    )
                    return 0
                if arguments.job_command == "logs":
                    print(jobs.logs(arguments.job_id, tail=arguments.tail), end="")
                    return 0
                if arguments.job_command == "cancel":
                    cancelled_job = jobs.cancel(arguments.job_id)
                    print(f"{cancelled_job.job_id}: {cancelled_job.state.value}")
                    return 0
                handle = jobs.retry(arguments.job_id, dry_run=arguments.dry_run)
                print(json.dumps(handle.to_dict(), indent=2))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "data":
            try:
                service = DataService(
                    DataCatalog.from_yaml(arguments.catalog),
                    ClusterCatalog.load(arguments.clusters),
                )
                if arguments.data_command == "list":
                    data_payload: object = service.list()
                elif arguments.data_command == "locations":
                    data_payload = service.locations(arguments.dataset)
                else:
                    data_payload = service.replicate(
                        arguments.dataset,
                        source_environment=arguments.source,
                        destination_environment=arguments.destination,
                        dry_run=not arguments.apply,
                    ).to_dict()
                print(json.dumps(data_payload, indent=2))
                if isinstance(data_payload, dict) and data_payload.get("returncode", 0) != 0:
                    return 1
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
            validation_report = validator.validate_file(
                arguments.config, check_imports=not arguments.no_imports
            )
            print(
                json.dumps(validation_report.to_dict(), indent=2)
                if arguments.json
                else validation_report.summary()
            )
            return 0 if validation_report.is_valid else 1
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
                result_catalog = (
                    TaskRun.from_yaml(source).result_catalog()
                    if source.suffix.lower() in {".yaml", ".yml"}
                    and TaskConfig.is_task_file(source)
                    else Experiment.from_yaml(source).result_catalog()
                    if source.suffix.lower() in {".yaml", ".yml"}
                    else ResultCatalog(source)
                )
                result_records = result_catalog.records(
                    status=arguments.status,
                    include_archived=not arguments.no_archived,
                )
                if arguments.duplicates:
                    duplicate_ids = {
                        record.attempt_id
                        for group in result_catalog.duplicate_groups().values()
                        for record in group
                    }
                    result_records = tuple(
                        record for record in result_records if record.attempt_id in duplicate_ids
                    )
                index_path = result_catalog.write_index() if arguments.write_index else None
                if arguments.json:
                    print(
                        json.dumps(
                            {
                                "summary": result_catalog.summary(),
                                "index_path": str(index_path) if index_path else None,
                                "records": [record.to_dict() for record in result_records],
                            },
                            indent=2,
                        )
                    )
                else:
                    print(result_catalog.summary())
                    for result_record in result_records:
                        location = "archived" if result_record.archived else "current"
                        print(
                            f"{result_record.attempt_id}  {result_record.status:<11} "
                            f"{location:<8} {result_record.result_path}"
                        )
                    if index_path is not None:
                        print(f"Wrote result index: {index_path}")
                return (
                    2 if arguments.fail_on_ambiguous and result_catalog.ambiguous_successes() else 0
                )
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "run" and (
            arguments.on is not None or arguments.profile is not None
        ):
            try:
                run_catalog = ClusterCatalog.load(arguments.clusters)
                cluster = arguments.on
                base_resources = None
                if arguments.profile is not None:
                    execution_profile = run_catalog.execution_profile(arguments.profile)
                    cluster = execution_profile.cluster
                    base_resources = execution_profile.resources
                request = cls._resource_request(arguments, base=base_resources)
                handle, bundle = ControlPlane(run_catalog).submit(
                    arguments.config,
                    cluster=cluster,
                    resources=request,
                    dry_run=arguments.dry_run,
                    run_arguments=cls._remote_run_arguments(arguments),
                )
                submission_payload = {"job": handle.to_dict(), "bundle": bundle.to_dict()}
                print(json.dumps(submission_payload, indent=2))
                return 0 if handle.state is not JobState.FAILED else 1
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "inspect" and arguments.resolved:
            try:
                print(
                    json.dumps(
                        AuthoringConfig.from_yaml(arguments.config).materialize().explanation(),
                        indent=2,
                    )
                )
                return 0
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
            task.config = task.config.with_execution_policy(
                force=arguments.force,
                restart=arguments.restart,
                no_resume=arguments.no_resume,
            )
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
        if arguments.force or arguments.restart or arguments.no_resume:
            experiment_values = experiment.config.as_dict()
            if arguments.force or arguments.restart:
                ExperimentConfig.set_value(experiment_values, "experiment.rerun_completed", True)
            if arguments.restart or arguments.no_resume:
                ExperimentConfig.set_value(experiment_values, "experiment.resume", False)
            if arguments.restart:
                ExperimentConfig.set_value(experiment_values, "experiment.ckpt_path", None)
            experiment = Experiment(ExperimentConfig(experiment_values, source=arguments.config))
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
        """Detect an explicit or concise workflow without constructing user targets."""
        try:
            return AuthoringConfig.from_yaml(path).materialize().kind is ConfigurationKind.WORKFLOW
        except Exception:
            return False

    @staticmethod
    def _scientific_payload(path: str | Path) -> dict[str, object]:
        """Return the normalized scientific payload for explain-changes."""
        materialized = AuthoringConfig.from_yaml(path).materialize()
        if materialized.kind is ConfigurationKind.TASK:
            return TaskConfig(materialized.values, source=path).scientific_payload()
        if materialized.kind is ConfigurationKind.EXPERIMENT:
            runs = ExperimentConfig(materialized.values, source=path).expand()
            if len(runs) == 1:
                return RunFingerprint.payload(runs[0])
            return {
                "suite_identity_version": 1,
                "runs": [RunFingerprint.payload(run) for run in runs],
            }
        return {"workflow_identity_version": 1, "config": materialized.to_dict()}

    @staticmethod
    def _resource_request(
        arguments: argparse.Namespace, *, base: ResourceRequest | None = None
    ) -> ResourceRequest:
        """Merge portable YAML resources with explicit CLI overrides."""
        values: dict[str, object] = base.to_dict() if base is not None else {}
        if base is None and TaskConfig.is_task_file(arguments.config):
            values.update(TaskConfig.from_yaml(arguments.config).resources.to_dict())
        overrides = {
            "cpus": arguments.cpus,
            "memory": arguments.memory,
            "gpus": arguments.resource_gpus,
            "gpu_memory": arguments.gpu_memory,
            "time": arguments.resource_time,
            "processes": arguments.processes,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return ResourceRequest.from_mapping(values)

    @staticmethod
    def _remote_run_arguments(arguments: argparse.Namespace) -> tuple[str, ...]:
        """Forward runner controls while keeping control-plane dry-run local."""
        values: list[str] = []
        for enabled, flag in (
            (arguments.no_plots, "--no-plots"),
            (arguments.force, "--force"),
            (arguments.restart, "--restart"),
            (arguments.no_resume, "--no-resume"),
        ):
            if enabled:
                values.append(flag)
        for value, flag in (
            (arguments.mode, "--mode"),
            (arguments.gpus, "--gpus"),
            (arguments.jobs_per_gpu, "--jobs-per-gpu"),
            (arguments.devices_per_job, "--devices-per-job"),
            (arguments.grace_seconds, "--grace-seconds"),
        ):
            if value is not None:
                values.extend((flag, str(value)))
        return tuple(values)

    @staticmethod
    def _initialize(directory: Path, *, force: bool, template: str = "minimal") -> int:
        """Create a minimal installable consumer project without overwriting by default."""
        files = {
            "pyproject.toml": """[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["lambdaforge>=0.5,<0.6"]

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
        preprocessing_files = {
            "src/my_project/preprocessing.py": '''"""Project preprocessing functions."""


def normalize_record(value: object) -> object:
    """Replace this example with the domain transformation."""
    return value
''',
            "data/raw.jsonl": '{"id": "example", "value": 1}\n',
            "experiments/preprocessing.yaml": """name: prepare-data
inputs:
  raw: ../data/raw.jsonl
outputs:
  processed: processed
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  key_field: id
  workers: 2
  workload: io
""",
        }
        training_files = {
            "src/my_project/models.py": '''"""Project model definitions."""

from torch import Tensor, nn


class ProjectModel(nn.Module):
    """Tiny baseline to replace with the research model."""

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return one binary logit per sample."""
        return self.head(x)
''',
            "src/my_project/data.py": '''"""Project dataset definitions."""

import torch
from torch.utils.data import Dataset


class ProjectDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy data to verify the complete training path."""

    def __init__(self, split: str, size: int = 32) -> None:
        generator = torch.Generator().manual_seed(7 if split == "train" else 17)
        self.x = torch.randn(size, 4, generator=generator)
        self.target = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "target": self.target[index]}
''',
            "experiments/training.yaml": """schema_version: "1.1"
experiment:
  name: baseline
  output_root: ../runs/experiments
  seeds: [7]
data:
  train: {target: my_project.data.ProjectDataset, params: {split: train}}
  val: {target: my_project.data.ProjectDataset, params: {split: val}}
  datamodule:
    target: lambdaforge.training.data.LightningDataModule
    params: {batch_size: 8, num_workers: 0}
model: my_project.models.ProjectModel
losses:
  - target: lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss
    params: {output_key: logits, target_key: target}
val_metrics:
  - target: lambdaforge.metrics.BinaryAccuracy
    params: {pred_key: logits, target_key: target}
optimizer: {ref: torch.optim.AdamW, params: {lr: 0.001}}
task:
  target: lambdaforge.training.LightningTask
  params: {model_input_key: x, model_output_key: logits}
trainer: {max_epochs: 2, accelerator: auto, devices: auto}
execution: {mode: sequential}
""",
        }
        if template in {"preprocessing", "full"}:
            files.update(preprocessing_files)
        if template in {"training", "full"}:
            files.update(training_files)
        if template in {"preprocessing", "training"}:
            files.pop("src/my_project/tasks.py")
            files.pop("experiments/task.yaml")
        entry = {
            "minimal": "experiments/task.yaml",
            "preprocessing": "experiments/preprocessing.yaml",
            "training": "experiments/training.yaml",
            "full": "experiments/preprocessing.yaml",
        }[template]
        files["README.md"] = f"""# My AI project

Create an environment and install both packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
lambdaforge doctor
lambdaforge validate {entry}
lambdaforge inspect {entry} --resolved
lambdaforge run {entry} --dry-run
lambdaforge run {entry}
```
"""
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
        initialize.add_argument(
            "--template",
            choices=("minimal", "preprocessing", "training", "full"),
            default="minimal",
            help="Select a focused starter instead of generating unused features.",
        )
        target = subparsers.add_parser(
            "target", help="Print an importable target signature and docstring."
        )
        target.add_argument("path")
        explain = subparsers.add_parser(
            "explain", help="Explain one dotted experiment/task Schema property."
        )
        explain.add_argument(
            "kind", choices=("authoring", "experiment", "task", "workflow", "changes")
        )
        explain.add_argument("path", nargs="?", default="")
        explain.add_argument("--against", type=Path, help="Previous YAML for explain changes.")
        explain.add_argument("--json", action="store_true")
        doctor = subparsers.add_parser(
            "doctor", help="Check local or remote Python, LambdaForge, scheduler and CUDA."
        )
        doctor.add_argument("--on", default="local", help="Cluster profile name.")
        doctor.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        doctor.add_argument("--json", action="store_true")
        clusters = subparsers.add_parser("clusters", help="Inspect and test cluster profiles.")
        clusters.add_argument("--catalog", type=Path, help="Cluster catalogue YAML.")
        cluster_commands = clusters.add_subparsers(dest="cluster_command", required=True)
        cluster_list = cluster_commands.add_parser("list")
        cluster_list.add_argument("--json", action="store_true")
        cluster_show = cluster_commands.add_parser("show")
        cluster_show.add_argument("name")
        cluster_show.add_argument("--json", action="store_true")
        cluster_test = cluster_commands.add_parser("test")
        cluster_test.add_argument("name")
        cluster_test.add_argument("--json", action="store_true")
        cluster_bootstrap = cluster_commands.add_parser("bootstrap")
        cluster_bootstrap.add_argument("name")
        cluster_bootstrap.add_argument("--json", action="store_true")
        jobs = subparsers.add_parser("jobs", help="List and control persistent jobs.")
        jobs.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        job_commands = jobs.add_subparsers(dest="job_command", required=True)
        jobs_list = job_commands.add_parser("list")
        jobs_list.add_argument("--cluster")
        jobs_list.add_argument("--state", choices=tuple(state.value for state in JobState))
        jobs_list.add_argument("--json", action="store_true")
        jobs_status = job_commands.add_parser("status")
        jobs_status.add_argument("job_id")
        jobs_status.add_argument("--json", action="store_true")
        jobs_logs = job_commands.add_parser("logs")
        jobs_logs.add_argument("job_id")
        jobs_logs.add_argument("--tail", type=int)
        jobs_cancel = job_commands.add_parser("cancel")
        jobs_cancel.add_argument("job_id")
        jobs_retry = job_commands.add_parser("retry")
        jobs_retry.add_argument("job_id")
        jobs_retry.add_argument("--dry-run", action="store_true")
        data = subparsers.add_parser("data", help="Inspect or explicitly replicate datasets.")
        data.add_argument("--catalog", type=Path, required=True, help="Data catalogue YAML.")
        data.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        data_commands = data.add_subparsers(dest="data_command", required=True)
        data_commands.add_parser("list")
        data_locations = data_commands.add_parser("locations")
        data_locations.add_argument("dataset")
        data_replicate = data_commands.add_parser("replicate")
        data_replicate.add_argument("dataset")
        data_replicate.add_argument("--from", dest="source", default="local")
        data_replicate.add_argument("--to", dest="destination", required=True)
        data_replicate.add_argument(
            "--apply", action="store_true", help="Transfer bytes; omission is preview-only."
        )
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
        placement = run.add_mutually_exclusive_group()
        placement.add_argument("--on", help="Submit through one configured cluster profile.")
        placement.add_argument("--profile", help="Use a named cluster/resource preset.")
        run.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        lifecycle = run.add_mutually_exclusive_group()
        lifecycle.add_argument("--force", action="store_true", help="Run even after success.")
        lifecycle.add_argument(
            "--restart", action="store_true", help="Run from scratch without partial state."
        )
        run.add_argument("--no-resume", action="store_true", help="Do not resume partial state.")
        run.add_argument("--cpus", type=int, help="Portable CPU-core request for --on.")
        run.add_argument("--memory", help="Portable RAM request, for example 32GiB.")
        run.add_argument("--resource-gpus", type=int, help="Portable GPU count for --on.")
        run.add_argument("--gpu-memory", help="Requested memory per schedulable unit.")
        run.add_argument("--time", dest="resource_time", help="Runtime request, e.g. 4h.")
        run.add_argument("--processes", type=int, help="Distributed process count.")
        inspect = subparsers.add_parser(
            "inspect",
            help="Print expanded experiment runs or an immutable task/workflow plan as JSON.",
        )
        inspect.add_argument("config", type=Path)
        inspect.add_argument(
            "--resolved", action="store_true", help="Show the strict materialized configuration."
        )
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
