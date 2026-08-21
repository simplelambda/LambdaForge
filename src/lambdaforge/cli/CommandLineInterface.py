"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import importlib
import inspect as python_inspect
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.cli.clusters import run_cluster_command
from lambdaforge.cli.common import (
    current_diagnostic_context,
    diagnostic_context,
    guarded,
    print_resources,
    print_storage,
    report_diagnostic,
    report_error,
    validation_diagnostic,
)
from lambdaforge.cli.DatasetCommands import DatasetCommands
from lambdaforge.cli.evidence import run_evidence_command
from lambdaforge.cli.jobs import run_job_command
from lambdaforge.cli.LiveJobMonitor import LiveJobMonitor
from lambdaforge.cli.parser import build_parser
from lambdaforge.cli.scaffold import initialize
from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.AuthoringSchemaCatalog import AuthoringSchemaCatalog
from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.configuration.ProjectConfigService import ProjectConfigService
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.MultiClusterSubmissionService import MultiClusterSubmissionService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataService import DataService
from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetRecipe import DatasetRecipe
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.diagnostics import (
    DiagnosticContext,
    ErrorCategory,
    RetryDisposition,
    diagnostic,
    execution_failure_diagnostic,
    job_failure_diagnostic,
)
from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver
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
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.preprocessing.PreprocessingDebugService import PreprocessingDebugService
from lambdaforge.registry.ExperimentRegistry import ExperimentRegistry
from lambdaforge.registry.LocalDashboard import LocalDashboard
from lambdaforge.reproducibility.IdentityExplainer import IdentityExplainer
from lambdaforge.results.ResultService import ResultService
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
from lambdaforge.tasks.TaskResult import TaskStatus
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
from lambdaforge.tasks.TaskValidator import TaskValidator
from lambdaforge.workflows.models import WorkflowPlan
from lambdaforge.workflows.Workflow import Workflow
from lambdaforge.workflows.WorkflowSchemaCatalog import WorkflowSchemaCatalog
from lambdaforge.workflows.WorkflowValidator import WorkflowValidator


class CommandLineInterface:
    """Parse CLI arguments and dispatch them to the public object API."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the CLI and return a process exit code."""
        supplied = list(argv) if argv is not None else sys.argv[1:]
        context = DiagnosticContext.from_argv(supplied)
        parent = current_diagnostic_context()
        if parent.arguments:
            context = replace(
                context,
                json_output=context.json_output or parent.json_output,
                debug=context.debug or parent.debug,
                verbose=context.verbose or parent.verbose,
            )
        # Operational verbosity and internal tracebacks are global UX concerns. Accept
        # them before or after any subcommand without duplicating argparse options.
        parsed = [value for value in supplied if value not in {"--debug", "--verbose", "--json"}]
        with diagnostic_context(context):
            try:
                return cls._dispatch(parsed)
            except KeyboardInterrupt:
                interrupted_job = next(
                    (value for value in context.arguments if value.startswith("job-")), None
                )
                commands = (
                    [("Inspect remote job", f"lf jobs show {interrupted_job}")]
                    if interrupted_job
                    else [("Inspect jobs", "lf jobs list --all")]
                )
                if context.cluster:
                    commands.append(("Inspect cluster", f"lf doctor --on {context.cluster}"))
                return report_diagnostic(
                    diagnostic(
                        ErrorCategory.CANCELLED,
                        "Operation cancelled by the user.",
                        "The local LambdaForge command received an interrupt.",
                        reason="KeyboardInterrupt was requested from the terminal.",
                        impact=(
                            "A previously submitted remote job may still be running; closing "
                            "the client does not implicitly cancel it.",
                        ),
                        fixes=("Inspect persistent jobs before submitting replacement work.",),
                        commands=commands,
                        context={"cluster": context.cluster, "job": interrupted_job},
                        retryable=RetryDisposition.AFTER_FIX,
                        operation=context.operation,
                    )
                )
            except Exception as error:
                return report_error(error)

    @classmethod
    def _dispatch(cls, argv: Sequence[str] | None = None) -> int:
        """Parse and execute one already-scoped CLI invocation."""
        parser = cls._parser()
        supplied_arguments = list(argv) if argv is not None else sys.argv[1:]
        default_cluster, default_source = cls._default_cluster()
        implicit_default = bool(
            default_cluster
            and "--on" not in supplied_arguments
            and "--profile" not in supplied_arguments
            and cls._uses_default_cluster(supplied_arguments)
        )
        raw_arguments = cls._normalize_arguments(list(supplied_arguments))
        if (
            len(raw_arguments) >= 2
            and raw_arguments[0] == "results"
            and raw_arguments[1] not in {"audit", "list", "show", "compare", "export", "sync"}
        ):
            raw_arguments.insert(1, "audit")
        arguments = parser.parse_args(raw_arguments)
        invocation = current_diagnostic_context()
        arguments.debug = invocation.debug
        arguments.json = getattr(arguments, "json", False) or invocation.json_output
        # The outer context removed the flag so commands need no duplicated parser option.
        arguments.verbose = getattr(arguments, "verbose", False) or invocation.verbose
        arguments.default_cluster_source = default_source if implicit_default else None
        if arguments.command in {"run", "validate", "inspect"} and not arguments.config.exists():
            try:
                arguments.config = ProjectConfigService().resolve(arguments.config)
            except Exception as error:
                return report_error(error)
        if arguments.command == "init":
            return cls._initialize(
                arguments.directory, force=arguments.force, template=arguments.template
            )
        if arguments.command == "completion":
            print(cls._completion(arguments.shell), end="")
            return 0
        if arguments.command == "target":
            try:
                module_name, symbol_name = arguments.path.rsplit(".", 1)
                symbol = getattr(importlib.import_module(module_name), symbol_name)
                print(f"{arguments.path}{python_inspect.signature(symbol)}")
                if python_inspect.getdoc(symbol):
                    print(python_inspect.getdoc(symbol))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "explain":
            try:
                if arguments.subject == "changes":
                    current = cls._scientific_payload(arguments.path)
                    previous = (
                        cls._scientific_payload(arguments.against)
                        if arguments.against is not None
                        else None
                    )
                    identity_explanation = IdentityExplainer().compare(current, previous)
                    print(
                        json.dumps(identity_explanation.to_dict(), indent=2)
                        if arguments.json
                        else identity_explanation.summary()
                    )
                    return 0
                catalogs: dict[str, type[Any]] = {
                    "authoring": AuthoringSchemaCatalog,
                    "task": TaskSchemaCatalog,
                    "experiment": ExperimentSchemaCatalog,
                    "workflow": WorkflowSchemaCatalog,
                }
                if arguments.subject in catalogs:
                    schema = catalogs[arguments.subject]().schema()
                    node = cls._schema_node(schema, arguments.path)
                    print(json.dumps(node, indent=2))
                    return 0
                if arguments.path:
                    raise ValueError(
                        "Configuration explanation accepts one path. Schema explanation uses "
                        "'lf explain KIND DOTTED_PATH'."
                    )
                from lambdaforge.configuration.explanation import explain_configuration

                config_explanation = explain_configuration(arguments.subject)
                if arguments.json:
                    print(json.dumps(config_explanation, indent=2))
                else:
                    cls._print_configuration_explanation(config_explanation)
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "doctor":
            try:
                doctor_report = Doctor(ClusterCatalog.load(arguments.clusters)).check(
                    arguments.on, config_path=arguments.config
                )
                print(
                    json.dumps(doctor_report.to_dict(), indent=2)
                    if arguments.json
                    else doctor_report.summary()
                )
                return doctor_report.exit_code
            except Exception as error:
                return report_error(error)
        if arguments.command in {"overview", "top"}:
            try:
                overview_catalog = ClusterCatalog.load(arguments.clusters)
                overview_service = OverviewService(overview_catalog)
                top_command = arguments.command == "top"
                follow = bool(getattr(arguments, "follow", False))
                once = bool(getattr(arguments, "once", False))
                if (
                    top_command
                    and not arguments.json
                    and not once
                    and sys.stdin.isatty()
                    and sys.stdout.isatty()
                    and os.name == "posix"
                ):
                    return LiveJobMonitor(
                        overview_service,
                        JobService(overview_catalog),
                        interval=arguments.interval,
                        history_seconds=arguments.history,
                    ).run()
                while True:
                    overview_payload = overview_service.snapshot()
                    if arguments.json:
                        print(
                            json.dumps(
                                overview_payload,
                                separators=(",", ":") if follow else None,
                                indent=None if follow else 2,
                            ),
                            flush=follow,
                        )
                    else:
                        cls._print_overview(overview_payload)
                    if not top_command or once or not follow:
                        return 0
                    time.sleep(arguments.interval)
            except Exception as error:
                return report_error(error)
        if arguments.command == "resources":
            try:
                resource_service = ResourceService(ClusterCatalog.load(arguments.clusters))
                if arguments.processes:
                    resource_payload: object = resource_service.processes(arguments.on or "local")
                elif arguments.all:
                    resource_payload = [value.to_dict() for value in resource_service.all()]
                else:
                    resource_payload = resource_service.get(arguments.on or "local").to_dict()
                if arguments.json:
                    print(json.dumps(resource_payload, indent=2))
                else:
                    print_resources(resource_payload)
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "storage":
            try:
                storage_service = StorageService(ClusterCatalog.load(arguments.clusters))
                if arguments.storage_command == "status":
                    reports = (
                        storage_service.all()
                        if arguments.all
                        else (storage_service.status(arguments.on),)
                    )
                    storage_payload = [value.to_dict() for value in reports]
                    offline = tuple(value for value in reports if not value.online)
                    if offline:
                        return report_diagnostic(
                            diagnostic(
                                ErrorCategory.CONNECTION,
                                "Storage status is unavailable for one or more clusters.",
                                "; ".join(
                                    f"{value.cluster}: {value.error or 'unavailable'}"
                                    for value in offline
                                ),
                                reason=(
                                    "LambdaForge could not query those filesystems through their "
                                    "configured transport."
                                ),
                                impact=(
                                    "No storage was changed; successful cluster reports remain "
                                    "valid.",
                                ),
                                fixes=("Diagnose each unavailable cluster before running GC.",),
                                commands=tuple(
                                    ("Diagnose cluster", f"lf doctor --on {value.cluster}")
                                    for value in offline
                                ),
                                context={
                                    "offline_clusters": [value.cluster for value in offline],
                                    "reports": storage_payload,
                                },
                                operation="storage status",
                            )
                        )
                    if arguments.json:
                        print(json.dumps(storage_payload, indent=2))
                    else:
                        print_storage(storage_payload)
                    return 0
                plan = storage_service.gc(arguments.on, apply=arguments.apply)
                if arguments.apply and plan.blocked_reason:
                    return report_diagnostic(
                        diagnostic(
                            ErrorCategory.OPERATION_REFUSED,
                            "LambdaForge intentionally refused cache collection.",
                            plan.blocked_reason,
                            reason=(
                                "Active references protect reconstructible cache entries from "
                                "being removed while they may still be in use."
                            ),
                            impact=("No cache entry or scientific artifact was deleted.",),
                            fixes=("Wait for or inspect the active work holding the reference.",),
                            commands=(
                                ("Inspect jobs", f"lf jobs list --cluster {arguments.on}"),
                                ("Preview again", f"lf storage gc --on {arguments.on}"),
                            ),
                            context={
                                "cluster": arguments.on,
                                "reclaimable_bytes": plan.reclaimable_bytes,
                            },
                            operation="storage gc",
                        )
                    )
                print(json.dumps(plan.to_dict(), indent=2))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "environments":
            try:
                environment_service = StorageService(ClusterCatalog.load(arguments.clusters))
                if arguments.environment_command == "gc":
                    print(
                        json.dumps(
                            environment_service.gc(arguments.on, apply=arguments.apply).to_dict(),
                            indent=2,
                        )
                    )
                    return 0
                values = environment_service.environments(arguments.on)
                if arguments.environment_command == "show":
                    values = tuple(
                        value
                        for value in values
                        if value.get("environment_id") == arguments.environment_id
                    )
                    if not values:
                        raise KeyError(f"Unknown environment {arguments.environment_id!r}.")
                print(json.dumps(values, indent=2))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "project":
            try:
                configs = ProjectConfigService(arguments.root)
                job_values = JobService().list(refresh=False)
                config_records = [record.to_dict() for record in configs.list()]
                active_jobs = [value.to_dict() for value in job_values if not value.state.terminal]
                payload = {
                    "project_root": str(configs.root),
                    "lambdaforge_version": LambdaForgeVersion.CURRENT,
                    "default_cluster": cls._project_default_cluster(),
                    "configs": config_records,
                    "dataset_registry": str(DatasetRegistry.project_path(configs.root)),
                    "active_jobs": active_jobs,
                }
                if arguments.json:
                    print(json.dumps(payload, indent=2))
                else:
                    print(f"Project: {payload['project_root']}")
                    print(f"LambdaForge: {payload['lambdaforge_version']}")
                    print(f"Default cluster: {payload['default_cluster'] or 'none'}")
                    print(f"Known configs: {len(config_records)}")
                    print(f"Active jobs: {len(active_jobs)}")
                    print(f"Dataset registry: {payload['dataset_registry']}")
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "datasets":
            try:
                return DatasetCommands.run(arguments)
            except Exception as error:
                return report_error(error)
        if arguments.command in {"configs", "experiments", "tasks"}:
            try:
                kind = (
                    None if arguments.command == "configs" else arguments.command.removesuffix("s")
                )
                config_service = ProjectConfigService(arguments.root)
                if arguments.entity_command == "list":
                    records = [value.to_dict() for value in config_service.list(kind=kind)]
                    if arguments.json or arguments.command != "experiments":
                        print(json.dumps(records, indent=2))
                    else:
                        cls._print_experiment_list(records)
                    return 0
                config_record = config_service.show(arguments.selector)
                if kind is not None and config_record.kind != kind:
                    raise ValueError(f"{config_record.name!r} is {config_record.kind}, not {kind}.")
                if arguments.entity_command == "show":
                    record_payload = config_record.to_dict()
                    revision = getattr(arguments, "revision", None)
                    if revision is not None:
                        executions = [
                            value
                            for value in record_payload["executions"]
                            if str(value.get("scientific_revision", "")).startswith(revision)
                            or str(value.get("scientific_identity", "")).startswith(revision)
                        ]
                        if not executions and not str(
                            record_payload.get("scientific_revision", "")
                        ).startswith(revision):
                            raise KeyError(
                                f"Unknown revision {revision!r} for experiment "
                                f"{config_record.name!r}."
                            )
                        record_payload["selected_revision"] = revision
                        record_payload["executions"] = executions
                    if arguments.json or arguments.command != "experiments":
                        print(json.dumps(record_payload, indent=2))
                    else:
                        cls._print_experiment(record_payload)
                    return 0
                if arguments.entity_command == "status":
                    status_payload = config_record.to_dict()
                    if arguments.json:
                        print(json.dumps(status_payload, indent=2))
                    else:
                        cls._print_experiment(status_payload)
                    return 0
                if arguments.entity_command == "history":
                    history_payload = config_record.to_dict()
                    history = {
                        "name": history_payload["name"],
                        "current_revision": history_payload["scientific_revision"],
                        "executions": history_payload["executions"],
                    }
                    print(json.dumps(history, indent=2))
                    return 0
                if arguments.entity_command in {"runs", "results"}:
                    results = ResultService().show(config_record.path)
                    if arguments.entity_command == "runs":
                        results = {
                            "experiment": config_record.name,
                            "revision": config_record.scientific_revision,
                            "runs": results["records"],
                        }
                    print(json.dumps(results, indent=2))
                    return 0
                forwarded = (
                    "inspect" if arguments.entity_command == "plan" else arguments.entity_command
                )
                command = [forwarded, str(config_record.path)]
                if forwarded == "run" and arguments.on:
                    for cluster in arguments.on:
                        command.extend(("--on", cluster))
                if getattr(arguments, "dry_run", False):
                    command.append("--dry-run")
                if getattr(arguments, "independent_hpo", False):
                    command.append("--independent-hpo")
                if getattr(arguments, "allow_duplicate", False):
                    command.append("--allow-duplicate")
                return cls.main(command)
            except Exception as error:
                return report_error(error)
        if arguments.command == "clusters":
            return guarded(lambda: run_cluster_command(arguments))
        if arguments.command == "jobs":
            return guarded(lambda: run_job_command(arguments))
        if arguments.command in {"status", "logs", "cancel", "retry"}:
            if (
                arguments.command == "status"
                and arguments.job_id is None
                and arguments.on is None
                and arguments.state is None
                and arguments.name is None
            ):
                forwarded = ["overview"]
                if arguments.clusters is not None:
                    forwarded.extend(("--clusters", str(arguments.clusters)))
                if arguments.json:
                    forwarded.append("--json")
                return cls.main(forwarded)
            forwarded = ["jobs"]
            if arguments.clusters is not None:
                forwarded.extend(("--clusters", str(arguments.clusters)))
            if arguments.command == "status" and arguments.job_id is None:
                forwarded.append("list")
                for value, flag in (
                    (arguments.on, "--cluster"),
                    (arguments.state, "--state"),
                    (arguments.name, "--name"),
                ):
                    if value is not None:
                        forwarded.extend((flag, str(value)))
                if arguments.json:
                    forwarded.append("--json")
            else:
                forwarded.extend((arguments.command, arguments.job_id))
                if arguments.command == "status" and arguments.json:
                    forwarded.append("--json")
                if arguments.command == "logs":
                    if arguments.tail is not None:
                        forwarded.extend(("--tail", str(arguments.tail)))
                    if arguments.follow:
                        forwarded.append("--follow")
                if arguments.command == "retry" and arguments.dry_run:
                    forwarded.append("--dry-run")
            return cls.main(forwarded)
        if arguments.command == "data":
            try:
                data_service = DataService(
                    DataCatalog.from_yaml(arguments.catalog),
                    ClusterCatalog.load(arguments.clusters),
                )
                if arguments.data_command == "list":
                    data_payload: object = data_service.list()
                elif arguments.data_command == "locations":
                    data_payload = data_service.locations(arguments.dataset)
                elif arguments.data_command == "inspect":
                    data_payload = data_service.inspect(arguments.dataset)
                else:
                    data_payload = data_service.replicate(
                        arguments.dataset,
                        source_environment=arguments.source,
                        destination_environment=arguments.destination,
                        dry_run=not arguments.apply,
                    ).to_dict()
                if isinstance(data_payload, dict) and data_payload.get("returncode", 0) != 0:
                    return report_diagnostic(
                        diagnostic(
                            ErrorCategory.DATA,
                            f"Dataset transfer {arguments.dataset!r} did not complete.",
                            str(data_payload.get("message") or "Transfer process failed."),
                            reason=(
                                "The explicit transfer provider returned a non-zero status; "
                                "LambdaForge did not register an unverified copy."
                            ),
                            impact=("No successful destination placement was published.",),
                            fixes=("Inspect source/destination access and retry after fixing it.",),
                            commands=(
                                (
                                    "Retry transfer",
                                    f"lf data --catalog {arguments.catalog} replicate "
                                    f"{arguments.dataset} --from {arguments.source} "
                                    f"--to {arguments.destination} --apply",
                                ),
                            ),
                            context={
                                "dataset": arguments.dataset,
                                "source": arguments.source,
                                "destination": arguments.destination,
                                "returncode": data_payload.get("returncode"),
                            },
                            operation="external dataset replication",
                        )
                    )
                print(json.dumps(data_payload, indent=2))
                return 0
            except Exception as error:
                return report_error(error)
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
                return report_error(error)
        if arguments.command == "diff":
            try:
                composer = ConfigurationComposer()
                left = composer.resolve(arguments.left).materialized()
                right = composer.resolve(arguments.right).materialized()
                print(json.dumps(ConfigurationDiff().compare(left, right), indent=2))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "registry":
            try:
                registry = ExperimentRegistry(arguments.root)
                if arguments.output:
                    print(registry.export(arguments.output))
                else:
                    print(json.dumps(registry.query(), indent=2, default=str))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "dashboard":
            try:
                print(LocalDashboard().build(arguments.root, arguments.output))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "plugins":
            descriptors = PluginRegistry.default().discover(arguments.kind)
            plugin_payload = [descriptor.to_dict() for descriptor in descriptors]
            if arguments.json:
                print(json.dumps(plugin_payload, indent=2))
            elif not plugin_payload:
                print("No LambdaForge plugins found.")
            else:
                for plugin in plugin_payload:
                    provider = plugin["distribution"] or "unknown distribution"
                    version = f" {plugin['version']}" if plugin["version"] else ""
                    print(
                        f"{plugin['kind']}:{plugin['name']} -> {plugin['value']} "
                        f"[{provider}{version}]"
                    )
            return 0
        if arguments.command == "validate":
            materialized_kind = AuthoringConfig.from_yaml(arguments.config).materialize().kind
            if materialized_kind is ConfigurationKind.DATASET:
                report = DatasetRecipe.from_yaml(arguments.config).validate(
                    check_imports=not arguments.no_imports
                )
                if not report.is_valid:
                    return report_diagnostic(
                        validation_diagnostic(
                            arguments.config, report.errors, kind="dataset recipe"
                        )
                    )
                print(
                    json.dumps({"valid": report.is_valid, "errors": list(report.errors)}, indent=2)
                    if arguments.json
                    else report.summary()
                )
                return 0 if report.is_valid else 1
            if cls._is_workflow(arguments.config):
                workflow_report = WorkflowValidator().validate_file(
                    arguments.config, check_imports=not arguments.no_imports
                )
                if not workflow_report.is_valid:
                    return report_diagnostic(
                        validation_diagnostic(
                            arguments.config, workflow_report.errors, kind="workflow"
                        )
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
            if not validation_report.is_valid:
                return report_diagnostic(
                    validation_diagnostic(
                        arguments.config,
                        validation_report.errors,
                        kind="task" if isinstance(validator, TaskValidator) else "experiment",
                    )
                )
            print(
                json.dumps(validation_report.to_dict(), indent=2)
                if arguments.json
                else validation_report.summary()
            )
            return 0 if validation_report.is_valid else 1
        if arguments.command == "migrate":
            if arguments.force and arguments.output is None:
                raise ValueError("--force requires --output; no file was modified.")
            if arguments.check and arguments.output is not None:
                raise ValueError("--check cannot be combined with --output; no file was modified.")
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
                return report_error(error)
        if arguments.command == "retain":
            try:
                experiment = Experiment.from_yaml(arguments.config)
                retention_result = (
                    experiment.apply_retention()
                    if arguments.apply
                    else experiment.preview_retention()
                )
                successful = {"applied", "already_applied"} if arguments.apply else {"preview"}
                if retention_result.status.value not in successful:
                    retention_errors = tuple(getattr(retention_result, "errors", ()))
                    retention_reason = getattr(retention_result, "reason", None)
                    reclaimed_bytes = int(getattr(retention_result, "reclaimed_bytes", 0))
                    category = (
                        ErrorCategory.OPERATION_REFUSED
                        if retention_result.status.value in {"disabled", "not_ready", "conflict"}
                        else ErrorCategory.EXECUTION
                    )
                    return report_diagnostic(
                        diagnostic(
                            category,
                            "Artifact retention did not complete successfully.",
                            "; ".join(retention_errors)
                            or str(retention_reason or "")
                            or f"Retention ended as {retention_result.status.value}.",
                            reason=(
                                "The retention transaction protected artifacts instead of "
                                "assuming an unsafe partial result."
                            ),
                            impact=(
                                f"Retention status: {retention_result.status.value}.",
                                f"Reclaimed bytes: {reclaimed_bytes}.",
                            ),
                            fixes=("Review the retention result and its reported conflicts.",),
                            commands=(("Preview safely", f"lf retain {arguments.config}"),),
                            context={"config": str(arguments.config)},
                            operation="artifact retention",
                            details=retention_errors,
                        )
                    )
                print(
                    json.dumps(retention_result.to_dict(), indent=2)
                    if arguments.json
                    else retention_result.summary()
                )
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command in {"results", "plot", "artifact"}:
            return guarded(lambda: run_evidence_command(arguments))
        if arguments.command == "debug":
            try:
                debug_report = PreprocessingDebugService().debug(
                    arguments.config,
                    records=arguments.records,
                    intermediates=arguments.intermediates,
                )
                if not debug_report.ok:
                    failed_record = next(
                        record for record in debug_report.records if record.get("exception")
                    )
                    return report_diagnostic(
                        execution_failure_diagnostic(
                            kind="preprocessing debug",
                            name=str(failed_record.get("source_key", "sample")),
                            source=arguments.config,
                            error=failed_record.get("exception"),
                            run_dir=arguments.intermediates,
                        )
                    )
                print(json.dumps(debug_report.to_dict(), indent=2))
                return 0
            except Exception as error:
                return report_error(error)
        if arguments.command == "run" and (
            arguments.on is not None or arguments.profile is not None
        ):
            try:
                run_catalog = ClusterCatalog.load(arguments.clusters)
                selected_clusters = tuple(arguments.on or ())
                if len(selected_clusters) > 1:
                    group = MultiClusterSubmissionService(run_catalog).submit(
                        arguments.config,
                        selected_clusters,
                        resources=cls._resource_request(arguments),
                        dry_run=arguments.dry_run,
                        independent_hpo=arguments.independent_hpo,
                        wait_for_submit=arguments.wait_for_submit,
                        allow_duplicate=arguments.allow_duplicate,
                    )
                    print(json.dumps(group.to_dict(), indent=2))
                    return 0
                cluster = selected_clusters[0] if selected_clusters else None
                base_resources = None
                if arguments.profile is not None:
                    execution_profile = run_catalog.execution_profile(arguments.profile)
                    cluster = execution_profile.cluster
                    base_resources = execution_profile.resources
                request = cls._resource_request(arguments, base=base_resources)
                if not arguments.dry_run and not arguments.wait_for_submit:
                    handle = SubmissionService(run_catalog).enqueue(
                        arguments.config,
                        cluster=cluster,
                        resources=request,
                        run_arguments=cls._remote_run_arguments(arguments),
                        allow_duplicate=arguments.allow_duplicate,
                    )
                    bundle_payload = None
                else:
                    handle, bundle = ControlPlane(run_catalog).submit(
                        arguments.config,
                        cluster=cluster,
                        resources=request,
                        dry_run=arguments.dry_run,
                        run_arguments=cls._remote_run_arguments(arguments),
                        allow_duplicate=arguments.allow_duplicate,
                    )
                    bundle_payload = bundle.to_dict()
                submission_payload = {
                    "job": handle.to_dict(),
                    "bundle": bundle_payload,
                    "submission": {
                        "mode": (
                            "asynchronous"
                            if handle.state is JobState.PREPARING
                            else "scheduler-acknowledged"
                        ),
                        "phase": "queued-locally" if handle.state is JobState.PREPARING else None,
                    },
                    "target": {
                        "cluster": cluster,
                        "source": arguments.default_cluster_source or "explicit",
                    },
                    "next": {
                        "status": f"lf jobs show {handle.job_id}",
                        "logs": f"lf jobs logs {handle.job_id} --tail 300",
                    },
                }
                if handle.state is JobState.FAILED:
                    service = JobService(run_catalog)
                    record = service.get(handle.job_id, refresh=False)
                    return report_diagnostic(
                        job_failure_diagnostic(record, service.logs(handle.job_id, tail=300))
                    )
                print(json.dumps(submission_payload, indent=2))
                return 0
            except Exception as error:
                return report_error(error)
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
                return report_error(error)
        materialized_kind = AuthoringConfig.from_yaml(arguments.config).materialize().kind
        if materialized_kind is ConfigurationKind.DATASET:
            recipe = DatasetRecipe.from_yaml(arguments.config)
            if arguments.command == "inspect" or arguments.dry_run:
                print(
                    json.dumps(
                        recipe.inspect(
                            force=getattr(arguments, "force", False),
                            force_stages=getattr(arguments, "force_stage", ()),
                        ).to_dict(),
                        indent=2,
                    )
                )
                return 0
            handle = DatasetBuildService().submit(
                recipe.config,
                cluster="local",
                force=arguments.force,
                force_stages=arguments.force_stage,
            )
            print(json.dumps(handle.to_dict(), indent=2))
            return 0
        if cls._is_workflow(arguments.config):
            if arguments.command == "aggregate":
                raise ValueError("aggregate applies only to training experiment YAML.")
            workflow = Workflow.from_yaml(arguments.config)
            workflow_outcome = workflow.run(
                dry_run=arguments.command == "inspect" or arguments.dry_run
            )
            if isinstance(workflow_outcome, WorkflowPlan) or workflow_outcome.status == "ok":
                print(json.dumps(workflow_outcome.to_dict(), indent=2, default=str))
                return 0
            return report_diagnostic(
                execution_failure_diagnostic(
                    kind="workflow",
                    name=workflow_outcome.name,
                    source=arguments.config,
                    nodes=workflow_outcome.nodes,
                    run_dir=workflow_outcome.run_dir,
                )
            )
        if TaskConfig.is_task_file(arguments.config):
            if arguments.command == "aggregate":
                raise ValueError("aggregate applies only to training experiment YAML.")
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
                raise ValueError(
                    "Experiment GPU overrides do not apply to kind: task; use task resources."
                )
            task.config = task.config.with_execution_policy(
                force=arguments.force,
                restart=arguments.restart,
                no_resume=arguments.no_resume,
            )
            task_outcome = task.run(dry_run=arguments.dry_run)
            if isinstance(task_outcome, TaskExecutionPlan):
                print(json.dumps(task_outcome.to_dict(), indent=2))
                return 0
            if task_outcome.status is TaskStatus.OK:
                print(
                    f"LambdaForge task {task_outcome.name!r} finished successfully; "
                    f"artifacts={len(task_outcome.artifacts)}."
                )
                return 0
            return report_diagnostic(
                execution_failure_diagnostic(
                    kind="task",
                    name=task_outcome.name,
                    source=arguments.config,
                    error=task_outcome.error,
                    run_dir=task_outcome.run_dir,
                )
            )

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
        run_outcome: Any = experiment.run(
            dry_run=arguments.dry_run,
            execution_overrides=experiment_overrides,
            aggregate_plots=not arguments.no_plots,
        )
        experiment_name = str(
            ExperimentConfig.get_value(
                experiment.config.as_dict(), "experiment.name", arguments.config.stem
            )
        )
        if hasattr(run_outcome, "to_dict"):
            summary = getattr(run_outcome, "summary", {})
            if summary.get("status") == "failed":
                return report_diagnostic(
                    execution_failure_diagnostic(
                        kind="adaptive experiment",
                        name=experiment_name,
                        source=arguments.config,
                        error=str(summary.get("error") or "Adaptive optimization failed."),
                    )
                )
            print(json.dumps(run_outcome.to_dict(), indent=2, default=str))
            return 0
        failed_results = [
            result for result in run_outcome if result.get("status") == "failed"
        ]
        if failed_results:
            first = failed_results[0]
            return report_diagnostic(
                execution_failure_diagnostic(
                    kind="experiment",
                    name=str(first.get("name", experiment_name)),
                    source=arguments.config,
                    error=str(first.get("error") or "Training run failed."),
                    run_dir=first.get("run_dir"),
                )
            )
        print(f"LambdaForge finished {len(run_outcome)} run(s); failed=0.")
        return 0

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
        values: dict[str, object] = (
            base.to_dict()
            if base is not None
            else ConfigurationResourceResolver.resolve(arguments.config).to_dict()
        )
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
        for stage in arguments.force_stage:
            values.extend(("--force-stage", stage))
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
    def _print_overview(payload: dict[str, Any]) -> None:
        jobs = payload["jobs"]
        datasets = payload["datasets"]
        states = jobs.get("by_state", {})
        print(
            f"LambdaForge: running={states.get('running', 0)} "
            f"queued={states.get('queued', 0)} staging={states.get('staging', 0)} "
            f"preparing={states.get('preparing', 0)} "
            f"paused={states.get('paused', 0)} / {jobs['total']} jobs; "
            f"{datasets['versions']} dataset versions; "
            f"offline={len(payload['offline_clusters'])}"
        )
        for cluster in payload["clusters"]:
            observed = cluster.get("observed", {})
            print(
                f"{cluster['cluster']:<16} "
                f"{'online' if cluster['online'] else 'offline':<8} "
                f"cpu={observed.get('cpu_total', 'unknown')} "
                f"ram={observed.get('ram_available_bytes', 'unknown')} "
                f"gpus={len(observed.get('gpus', []))}"
            )

    @staticmethod
    def _print_experiment_list(records: list[dict[str, Any]]) -> None:
        """Render the project experiment catalog in research terms."""
        print(
            "EXPERIMENT                 REVISION      TARGETS              "
            "STATE       RUNS  ATTEMPTS"
        )
        for record in records:
            progress = record.get("progress", {})
            progress = progress if isinstance(progress, dict) else {}
            completed, total = progress.get("completed", 0), progress.get("total")
            runs = f"{completed}/{total if total is not None else '?'}"
            targets = ",".join(record.get("active_clusters", ())) or "-"
            print(
                f"{str(record.get('name', '-')):<26.26} "
                f"{str(record.get('scientific_revision') or '-'):<13.13} "
                f"{targets:<20.20} {str(record.get('state', '-')):<11.11} "
                f"{runs:<5.5} {record.get('attempt_count', 0)}"
            )

    @staticmethod
    def _print_experiment(payload: dict[str, Any]) -> None:
        """Render one experiment with revision, progress and execution history."""
        progress = payload.get("progress", {})
        progress = progress if isinstance(progress, dict) else {}
        total = progress.get("total")
        completed = progress.get("completed", 0)
        print(f"Experiment: {payload.get('name', '-')}")
        print(f"Revision: {payload.get('scientific_revision') or 'unknown'}")
        print(f"State: {payload.get('state', 'not_run')}")
        print(
            f"Progress: {completed}/{total if total is not None else '?'} "
            f"{progress.get('unit', 'runs')}"
        )
        print(f"Datasets: {', '.join(payload.get('datasets', ())) or 'none'}")
        print(f"Configuration: {payload.get('path', '-')}")
        print("Executions:")
        executions = payload.get("executions", ())
        if not executions:
            print("  none")
            return
        for execution in executions:
            print(
                f"  {execution.get('cluster', '-')}  {execution.get('state', '-')}  "
                f"revision={execution.get('scientific_revision') or 'legacy'}  "
                f"attempt={execution.get('attempt', 1)}  job={execution.get('job_id', '-')}"
            )

    @staticmethod
    def _print_configuration_explanation(payload: dict[str, Any]) -> None:
        """Render materialized scientific intent before operational identifiers."""
        print(f"{str(payload.get('kind', 'configuration')).title()}: {payload.get('name', '-')}")
        print(f"Revision: {payload.get('scientific_revision', 'unknown')}")
        datasets = payload.get("datasets", ())
        print(f"Datasets: {', '.join(datasets) if datasets else 'none'}")
        planned = payload.get("planned_work", {})
        if isinstance(planned, dict):
            print(
                f"Planned work: {planned.get('total', '?')} "
                f"{planned.get('unit', 'units')}"
            )
        if payload.get("model"):
            model = payload["model"]
            print(f"Model: {model.get('target')}({model.get('params', {})})")
        if payload.get("training"):
            training = payload["training"]
            print(f"Training: maximum {training.get('max_epochs', '?')} epochs")
        if payload.get("seeds"):
            print(f"Seeds: {', '.join(str(value) for value in payload['seeds'])}")
        if payload.get("objective"):
            print(f"Objective: {payload['objective']}")
        nodes = payload.get("stages", payload.get("nodes", ()))
        if nodes:
            print("Stages:" if "stages" in payload else "Nodes:")
            for node in nodes:
                dependencies = ", ".join(node.get("depends_on", ())) or "start"
                print(f"  {node.get('name')}  after: {dependencies}")
        print(f"Resources: {payload.get('resources', {})}")

    @staticmethod
    def _initialize(directory: Path, *, force: bool, template: str = "minimal") -> int:
        """Create a consumer project using the canonical scaffold implementation."""
        return initialize(directory, force=force, template=template)

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

    @classmethod
    def _normalize_arguments(cls, values: list[str]) -> list[str]:
        """Apply documented aliases and the root plan shortcut before one canonical parser."""
        if not values:
            return values
        resources = {"ds": "datasets", "exp": "experiments", "env": "environments"}
        values[0] = resources.get(values[0], values[0])
        if (
            len(values) > 1
            and values[1] == "ls"
            and values[0]
            in {
                "datasets",
                "experiments",
                "tasks",
                "configs",
                "clusters",
                "jobs",
                "environments",
            }
        ):
            values[1] = "list"
        if values[0] == "plan":
            values[0] = "run"
            if "--dry-run" not in values:
                values.append("--dry-run")
        default = cls._project_default_cluster()
        needs_default = values[0] == "run" or (
            len(values) > 1
            and values[0] == "datasets"
            and values[1] in {"plan", "build", "materialize"}
        )
        if default and needs_default and "--on" not in values and "--profile" not in values:
            values.extend(("--on", default))
        return values

    @classmethod
    def _uses_default_cluster(cls, values: Sequence[str]) -> bool:
        """Return whether this command accepts the project/user execution preference."""
        if not values:
            return False
        resource = {"ds": "datasets", "exp": "experiments", "env": "environments"}.get(
            values[0], values[0]
        )
        return resource in {"run", "plan"} or (
            len(values) > 1
            and resource == "datasets"
            and values[1] in {"plan", "build", "materialize"}
        )

    @classmethod
    def _default_cluster(cls) -> tuple[str | None, str | None]:
        """Resolve project before user preference and retain provenance for human/JSON output."""
        current = Path.cwd().resolve()
        for directory in (current, *current.parents):
            for name in ("lambdaforge.yaml", "lambdaforge.clusters.yaml"):
                path = directory / name
                if not path.is_file():
                    continue
                try:
                    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except (OSError, yaml.YAMLError):
                    continue
                if isinstance(value, dict) and value.get("default_cluster"):
                    return str(value["default_cluster"]), "project default"
            if (directory / "pyproject.toml").is_file():
                break
        user_path = ClusterCatalog.user_path()
        if user_path.is_file():
            try:
                value = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                value = {}
            if isinstance(value, dict) and value.get("default_cluster"):
                return str(value["default_cluster"]), "user default"
        return None, None

    @classmethod
    def _project_default_cluster(cls) -> str | None:
        """Backward-compatible value-only view used by project status and argument normalization."""
        return cls._default_cluster()[0]

    @staticmethod
    def _completion(shell: str) -> str:
        """Generate conservative dependency-free completion for stable CLI resources."""
        base = (
            "run plan status top doctor clusters datasets ds experiments exp tasks configs "
            "jobs results artifacts environments env resources storage project completion"
        )
        dynamic: set[str] = set()
        for provider in (
            lambda: ClusterCatalog.load().names(),
            lambda: tuple(record.key for record in DatasetRegistry().records()),
            lambda: tuple(record.name for record in ProjectConfigService().list() if record.valid),
            lambda: tuple(record.job_id for record in JobService().store.records()),
        ):
            try:
                dynamic.update(str(value) for value in provider())
            except Exception:
                continue
        commands = " ".join((base, *sorted(dynamic)))
        if shell == "bash":
            return (
                "_lambdaforge_complete() {\n"
                f'  COMPREPLY=( $(compgen -W "{commands}" -- "${{COMP_WORDS[COMP_CWORD]}}") )\n'
                "}\ncomplete -F _lambdaforge_complete lambdaforge lf\n"
            )
        if shell == "zsh":
            return f"#compdef lambdaforge lf\n_arguments '1:command:({commands})' '*::arg:->args'\n"
        return f"complete -c lambdaforge -c lf -f -a '{commands}'\n"

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        """Return the canonical parser used by both console entry points."""
        return build_parser()
