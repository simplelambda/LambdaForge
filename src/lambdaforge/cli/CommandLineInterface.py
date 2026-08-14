"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import importlib
import inspect as python_inspect
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.artifacts.ArtifactPluginRegistry import ArtifactPluginRegistry
from lambdaforge.artifacts.ArtifactService import ArtifactService
from lambdaforge.artifacts.NumpyArtifactValidator import NumpyArtifactValidator
from lambdaforge.artifacts.RemoteArtifactService import RemoteArtifactService
from lambdaforge.cli.DatasetCommands import DatasetCommands
from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.AuthoringSchemaCatalog import AuthoringSchemaCatalog
from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.configuration.ProjectConfigService import ProjectConfigService
from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterService import ClusterService
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobState import JobState
from lambdaforge.controlplane.MultiClusterSubmissionService import MultiClusterSubmissionService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.SshConnectionPolicy import SshConnectionPolicy
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataService import DataService
from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetRecipe import DatasetRecipe
from lambdaforge.data.DatasetRegistry import DatasetRegistry
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
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.preprocessing.PreprocessingDebugService import PreprocessingDebugService
from lambdaforge.registry.ExperimentRegistry import ExperimentRegistry
from lambdaforge.registry.LocalDashboard import LocalDashboard
from lambdaforge.reproducibility.IdentityExplainer import IdentityExplainer
from lambdaforge.results.RemoteResultService import RemoteResultService
from lambdaforge.results.ResultService import ResultService
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
from lambdaforge.tasks.TaskStatus import TaskStatus
from lambdaforge.tasks.TaskValidator import TaskValidator
from lambdaforge.visualization.VisualizationService import VisualizationService
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
        arguments.default_cluster_source = default_source if implicit_default else None
        if arguments.command in {"run", "validate", "inspect"} and not arguments.config.exists():
            try:
                arguments.config = ProjectConfigService().resolve(arguments.config)
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                doctor_report = Doctor(ClusterCatalog.load(arguments.clusters)).check(
                    arguments.on, config_path=arguments.config
                )
                print(
                    json.dumps(doctor_report.to_dict(), indent=2)
                    if arguments.json
                    else doctor_report.summary()
                )
                return 0 if doctor_report.ok else 1
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command in {"overview", "top"}:
            try:
                while True:
                    overview_payload = OverviewService(
                        ClusterCatalog.load(arguments.clusters)
                    ).snapshot()
                    if arguments.json:
                        print(json.dumps(overview_payload, indent=2))
                    else:
                        cls._print_overview(overview_payload)
                    if arguments.command != "top" or not arguments.follow:
                        return 0
                    time.sleep(arguments.interval)
            except KeyboardInterrupt:
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                    cls._print_resources(resource_payload)
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                    if arguments.json:
                        print(json.dumps(storage_payload, indent=2))
                    else:
                        cls._print_storage(storage_payload)
                    return 0 if all(value.online for value in reports) else 1
                plan = storage_service.gc(arguments.on, apply=arguments.apply)
                print(json.dumps(plan.to_dict(), indent=2))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "project":
            try:
                configs = ProjectConfigService(arguments.root)
                job_values = JobService().list(refresh=False)
                config_records = [record.to_dict() for record in configs.list()]
                active_jobs = [
                    value.to_dict() for value in job_values if not value.state.terminal
                ]
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
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "datasets":
            try:
                return DatasetCommands.run(arguments)
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command in {"configs", "experiments", "tasks"}:
            try:
                kind = (
                    None if arguments.command == "configs" else arguments.command.removesuffix("s")
                )
                config_service = ProjectConfigService(arguments.root)
                if arguments.entity_command == "list":
                    print(
                        json.dumps(
                            [value.to_dict() for value in config_service.list(kind=kind)], indent=2
                        )
                    )
                    return 0
                config_record = config_service.show(arguments.selector)
                if kind is not None and config_record.kind != kind:
                    raise ValueError(f"{config_record.name!r} is {config_record.kind}, not {kind}.")
                if arguments.entity_command == "show":
                    print(json.dumps(config_record.to_dict(), indent=2))
                    return 0
                if arguments.entity_command == "status":
                    entity_jobs = JobService().list(name=config_record.name)
                    print(json.dumps([value.to_dict() for value in entity_jobs], indent=2))
                    return 0
                if arguments.entity_command in {"history", "results"}:
                    print(json.dumps(ResultService().show(config_record.path), indent=2))
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
                return cls.main(command)
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "clusters":
            try:
                cluster_catalog = ClusterCatalog.load(arguments.catalog)
                if arguments.cluster_command == "add":
                    destination = arguments.catalog or (
                        ClusterCatalog.user_path()
                        if arguments.scope == "user"
                        else ClusterCatalog.project_path()
                    )
                    credential = arguments.credential
                    credentials = CredentialService()
                    if arguments.store_password:
                        if arguments.auth != "password":
                            raise ValueError("--store-password requires --auth password.")
                        credential = credential or cls._keyring_reference(
                            arguments.name, arguments.user, arguments.host
                        )
                        if not credential.startswith("keyring:"):
                            raise ValueError("--store-password requires a keyring: reference.")
                    cluster_profile = ClusterProfile(
                        arguments.name,
                        transport="ssh" if arguments.host else "local",
                        scheduler=arguments.scheduler,
                        host=arguments.host,
                        user=arguments.user,
                        port=arguments.port,
                        auth=ClusterAuthentication(arguments.auth, credential),
                        known_hosts=(str(arguments.known_hosts) if arguments.known_hosts else None),
                        ssh_timeout=arguments.ssh_timeout,
                        connection=SshConnectionPolicy(
                            connect_timeout=arguments.connect_timeout,
                            auth_timeout=arguments.auth_timeout,
                            banner_timeout=arguments.banner_timeout,
                            keepalive=arguments.keepalive_interval,
                            multiplex=not arguments.no_multiplex,
                            persist=arguments.control_persist,
                            command_timeout=arguments.command_timeout,
                        ),
                        workspace=arguments.workspace,
                        storage=ClusterStoragePolicy.from_mapping(
                            {
                                key: value
                                for key, value in {
                                    "state_root": arguments.state_root,
                                    "cache_root": arguments.cache_root,
                                    "run_root": arguments.run_root,
                                    "dataset_root": arguments.dataset_root,
                                    "cache_max_size": arguments.cache_max_size,
                                    "cache_max_age": arguments.cache_max_age,
                                }.items()
                                if value is not None
                            },
                            workspace=arguments.workspace,
                        ),
                        python=arguments.python,
                        environment=arguments.environment,
                        wheelhouse=(str(arguments.wheelhouse) if arguments.wheelhouse else None),
                        pytorch=TorchInstallationPolicy(
                            arguments.torch_channel,
                            (
                                True
                                if arguments.require_cuda
                                else False
                                if arguments.no_require_cuda
                                else None
                            ),
                        ),
                        project_module=arguments.project_module,
                        data_environment=arguments.data_environment,
                    )
                    if arguments.store_password:
                        secret = credentials.interactive.get(
                            "interactive",
                            prompt=f"Password for {arguments.user or ''}@{arguments.host}: ",
                        )
                        credentials.store(credential or "", secret)
                        del secret
                    print(ClusterCatalog.add(destination, cluster_profile))
                    return 0
                if arguments.cluster_command == "list":
                    cluster_payload = [
                        cluster_catalog.inspect(name) for name in cluster_catalog.names()
                    ]
                    if arguments.json:
                        print(json.dumps(cluster_payload, indent=2))
                    else:
                        for inspected in cluster_payload:
                            profile_payload = inspected["profile"]
                            print(
                                f"{profile_payload['name']:<16} "
                                f"{profile_payload['transport']:<6} "
                                f"{profile_payload['scheduler']:<6} "
                                f"{profile_payload['host'] or '-'} "
                                f"[{inspected['source']}]"
                            )
                    return 0
                if arguments.cluster_command == "resources":
                    cluster_resource = ResourceService(cluster_catalog).get(arguments.name)
                    value = cluster_resource.to_dict()
                    if arguments.json:
                        print(json.dumps(value, indent=2))
                    else:
                        cls._print_resources(value)
                    return 0 if cluster_resource.online else 1
                if arguments.cluster_command == "storage":
                    cluster_storage = StorageService(cluster_catalog).status(arguments.name)
                    value = cluster_storage.to_dict()
                    if arguments.json:
                        print(json.dumps(value, indent=2))
                    else:
                        cls._print_storage([value])
                    return 0 if cluster_storage.online else 1
                cluster_profile = cluster_catalog.get(arguments.name)
                if arguments.cluster_command in {"set", "unset", "remove"}:
                    source = cluster_catalog.source(cluster_profile.name)
                    if source is None:
                        raise ValueError("The built-in local profile cannot be modified.")
                    if arguments.cluster_command == "remove":
                        ClusterCatalog.remove(source, cluster_profile.name)
                        print(source)
                        return 0
                    descriptor = cluster_profile.to_dict(include_defaults=False)
                    descriptor.pop("name", None)
                    if arguments.cluster_command == "set":
                        cls._set_dotted(
                            descriptor,
                            arguments.key,
                            yaml.safe_load(arguments.value),
                        )
                    else:
                        cls._delete_dotted(descriptor, arguments.key)
                    ClusterCatalog.add(
                        source,
                        ClusterProfile.from_mapping(cluster_profile.name, descriptor),
                    )
                    print(source)
                    return 0
                if arguments.cluster_command == "show":
                    print(json.dumps(cluster_profile.to_dict(), indent=2))
                    return 0
                if arguments.cluster_command == "inspect":
                    print(json.dumps(cluster_catalog.inspect(arguments.name), indent=2))
                    return 0
                if arguments.cluster_command == "export":
                    cluster_export_payload = cluster_profile.to_dict()
                    cluster_export_payload.pop("name", None)
                    exported = {"clusters": {cluster_profile.name: cluster_export_payload}}
                    if arguments.output is None:
                        print(yaml.safe_dump(exported, sort_keys=False, allow_unicode=True), end="")
                    else:
                        ClusterCatalog.export(arguments.output, cluster_profile)
                        print(arguments.output.expanduser().resolve())
                    return 0
                if arguments.cluster_command == "credentials":
                    if cluster_profile.name == "local":
                        raise ValueError("The built-in local profile has no SSH credentials.")
                    source = cluster_catalog.source(cluster_profile.name)
                    if source is None:
                        raise ValueError("The selected profile has no writable catalog source.")
                    credentials = CredentialService()
                    reference = cluster_profile.auth.credential or cls._keyring_reference(
                        cluster_profile.name, cluster_profile.user, cluster_profile.host
                    )
                    if arguments.credential_command == "set":
                        if not reference.startswith("keyring:"):
                            reference = cls._keyring_reference(
                                cluster_profile.name, cluster_profile.user, cluster_profile.host
                            )
                        secret = credentials.interactive.get(
                            "interactive",
                            prompt=(
                                f"Password for {cluster_profile.user or ''}@"
                                f"{cluster_profile.host}: "
                            ),
                        )
                        credentials.store(reference, secret)
                        del secret
                        updated = replace(
                            cluster_profile,
                            auth=ClusterAuthentication("password", reference),
                        )
                        ClusterCatalog.add(source, updated)
                        print(
                            f"Stored credential reference {reference!r} for "
                            f"{cluster_profile.name!r}."
                        )
                        return 0
                    if not reference.startswith("keyring:"):
                        raise ValueError("Only keyring: credentials can be deleted from storage.")
                    credentials.delete(reference)
                    updated = replace(cluster_profile, auth=ClusterAuthentication("password", None))
                    ClusterCatalog.add(source, updated)
                    print(
                        f"Deleted credential for {cluster_profile.name!r}; password mode "
                        "is now interactive."
                    )
                    return 0
                if arguments.cluster_command == "test":
                    cluster_report = Doctor(cluster_catalog).check(cluster_profile.name)
                    print(
                        json.dumps(cluster_report.to_dict(), indent=2)
                        if arguments.json
                        else cluster_report.summary()
                    )
                    return 0 if cluster_report.ok else 1
                bootstrapped = ClusterService(cluster_catalog).bootstrap(
                    cluster_profile.name, wheelhouse=arguments.wheelhouse
                )
                print(
                    json.dumps(bootstrapped.to_dict(), indent=2)
                    if arguments.json
                    else (
                        f"Cluster {bootstrapped.cluster!r} is ready with "
                        f"{bootstrapped.environment_id} "
                        f"({'reused' if bootstrapped.reused else 'created'})."
                    )
                )
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "jobs":
            try:
                jobs = JobService(ClusterCatalog.load(arguments.clusters))
                if arguments.job_command == "list":
                    if arguments.all:
                        jobs.reconcile(all_clusters=True)
                    job_records = jobs.list(
                        cluster=arguments.cluster,
                        state=arguments.state,
                        name=arguments.name,
                    )
                    if arguments.json:
                        print(json.dumps([record.to_dict() for record in job_records], indent=2))
                    else:
                        print(
                            "JOB                             NAME              TYPE           "
                            "STATE      CLUSTER      AGE       RESOURCES"
                        )
                        for job_record in job_records:
                            name = str(job_record.metadata.get("name", "-"))
                            resources = cls._job_resources(job_record.resources)
                            print(
                                f"{job_record.job_id:<32} {name:<17.17} "
                                f"{job_record.job_type:<14.14} {job_record.state.value:<10} "
                                f"{job_record.cluster:<12.12} "
                                f"{cls._age(job_record.created_at_utc):<9} "
                                f"{resources}"
                            )
                    return 0
                selected_job_id: str = (
                    jobs.resolve_selector(arguments.job_id)
                    if hasattr(arguments, "job_id")
                    else ""
                )
                if arguments.job_command in {"status", "show"}:
                    job_record = jobs.get(selected_job_id)
                    print(
                        json.dumps(job_record.to_dict(), indent=2)
                        if arguments.json
                        else (
                            f"{job_record.job_id}: {job_record.state.value} on {job_record.cluster}"
                        )
                    )
                    return 0
                if arguments.job_command == "logs":
                    if arguments.follow:
                        return cls._follow_job_logs(jobs, selected_job_id, tail=arguments.tail)
                    print(jobs.logs(selected_job_id, tail=arguments.tail), end="")
                    return 0
                if arguments.job_command == "cancel":
                    cancelled_job = jobs.cancel(selected_job_id)
                    print(f"{cancelled_job.job_id}: {cancelled_job.state.value}")
                    return 0
                if arguments.job_command == "pause":
                    paused = jobs.pause(selected_job_id)
                    print(json.dumps(paused.to_dict(), indent=2))
                    return 0
                if arguments.job_command == "resume":
                    resumed = jobs.resume(selected_job_id)
                    print(json.dumps(resumed.to_dict(), indent=2))
                    return 0
                if arguments.job_command == "delete":
                    jobs.delete(selected_job_id)
                    print(json.dumps({"deleted": selected_job_id}, indent=2))
                    return 0
                if arguments.job_command == "reconcile":
                    reconciled = jobs.reconcile(
                        cluster=arguments.cluster, all_clusters=arguments.all
                    )
                    print(json.dumps([value.to_dict() for value in reconciled], indent=2))
                    return 0
                if arguments.job_command in {"group", "groups"}:
                    from lambdaforge.controlplane.JobGroupStore import JobGroupStore

                    groups = JobGroupStore()
                    group_id = getattr(arguments, "group_id", None)
                    group_payload: object = (
                        groups.get(group_id).to_dict()
                        if group_id
                        else [value.to_dict() for value in groups.list()]
                    )
                    print(json.dumps(group_payload, indent=2))
                    return 0
                handle = jobs.retry(selected_job_id, dry_run=arguments.dry_run)
                print(json.dumps(handle.to_dict(), indent=2))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                if arguments.results_command == "list":
                    records = ResultService(arguments.root or ("runs",)).records(
                        status=arguments.status
                    )
                    result_payload = [record.to_dict() for record in records]
                    if arguments.json:
                        print(json.dumps(result_payload, indent=2))
                    else:
                        for result_record in records:
                            print(
                                f"{result_record.attempt_id:<32} {result_record.status:<11} "
                                f"{result_record.result.name} seed={result_record.result.seed}"
                            )
                    return 0
                if arguments.results_command == "show":
                    shown = ResultService(arguments.root or ("runs",)).show(arguments.selector)
                    if arguments.json:
                        print(json.dumps(shown, indent=2))
                    else:
                        print(f"Selector: {shown['selector']}  ambiguous={shown['ambiguous']}")
                        for record in shown["records"]:
                            metrics = ", ".join(
                                f"{key}={value}" for key, value in record.get("metrics", {}).items()
                            )
                            print(
                                f"{record['attempt_id']:<32} {record['status']:<11} "
                                f"{record['name']} seed={record.get('seed')}  {metrics}"
                            )
                    return 0
                if arguments.results_command == "compare":
                    compared = ResultService(arguments.root or ("runs",)).compare(
                        arguments.selectors,
                        metrics=arguments.metric,
                        confidence_level=arguments.confidence,
                        direction=arguments.direction,
                    )
                    if arguments.json:
                        print(json.dumps(compared, indent=2))
                    else:
                        for comparison in compared["comparisons"]:
                            print(f"Metric: {comparison['metric']}")
                            print("GROUP  N  MEAN  STD  CI  DELTA VS BASELINE")
                            for label, summary in comparison["groups"].items():
                                interval = summary["confidence_interval"]
                                print(
                                    f"{label}  {summary['count']}  {summary['mean']:.6g}  "
                                    f"{summary['stdev']:.6g}  [{interval[0]:.6g}, "
                                    f"{interval[1]:.6g}]  {summary['delta_vs_baseline']:+.6g}"
                                )
                            if comparison.get("direction"):
                                print(
                                    f"Best: {comparison['best_group']}  "
                                    f"Worst: {comparison['worst_group']}"
                                )
                    return 0
                if arguments.results_command == "export":
                    suffix = arguments.format
                    destination = arguments.output or Path(
                        f"{str(arguments.selector).replace('/', '-')}-results.{suffix}"
                    )
                    print(
                        ResultService(arguments.root or ("runs",)).export(
                            arguments.selector,
                            destination,
                            metric_series=arguments.series,
                        )
                    )
                    return 0
                if arguments.results_command == "sync":
                    synced = RemoteResultService(root=arguments.destination).sync(arguments.job_id)
                    print(json.dumps(synced.to_dict(), indent=2))
                    return 0
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
        if arguments.command == "plot":
            try:
                visualization = VisualizationService()
                if arguments.plot_command == "learning":
                    selector: str | Path = arguments.selector
                    if str(selector).startswith("job-"):
                        selector = Path(RemoteResultService().sync(str(selector)).destination)
                    spec = visualization.learning(
                        selector,
                        metrics=arguments.metric,
                        aggregate=arguments.aggregate,
                        uncertainty=arguments.uncertainty,
                    )
                elif arguments.plot_command == "sweep":
                    spec = visualization.sweep(
                        arguments.config,
                        x=arguments.x,
                        y=arguments.y,
                        metrics=arguments.metric,
                        uncertainty=arguments.uncertainty,
                        interpolate=arguments.interpolate,
                        view=arguments.kind,
                        normalize=arguments.normalize,
                    )
                elif arguments.plot_command == "seeds":
                    spec = visualization.seed_distribution(
                        arguments.selector, metric=arguments.metric, kind=arguments.kind
                    )
                elif arguments.plot_command == "hpo":
                    spec = visualization.hpo(
                        arguments.study,
                        parameter=arguments.parameter,
                        direction=arguments.direction,
                    )
                else:
                    spec = visualization.resources(arguments.selector)
                if arguments.json:
                    print(json.dumps(spec.to_dict(), indent=2))
                    return 0
                output = arguments.output or Path(f"{spec.plot_type}.png")
                if getattr(arguments, "follow", False):
                    return cls._follow_plot(arguments, visualization, output)
                print(visualization.render(spec, output))
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "artifact":
            try:
                artifact_service = ArtifactService()
                if arguments.artifact_command == "inspect":
                    artifact_inspection = artifact_service.inspect(
                        arguments.path,
                        array=arguments.array,
                        rows=arguments.rows,
                        slice_expression=arguments.slice,
                        inspector=arguments.inspector,
                    )
                    if arguments.json:
                        print(json.dumps(artifact_inspection.to_dict(), indent=2))
                    else:
                        print(
                            f"Type: {artifact_inspection.artifact_type}\n"
                            f"Path: {artifact_inspection.path}\n"
                            f"Size: {artifact_inspection.size_bytes} bytes"
                        )
                        for item in artifact_inspection.items:
                            if "name" in item:
                                print(
                                    f"{item['name']:<24} shape={tuple(item['shape'])!s:<18} "
                                    f"dtype={item['dtype']:<12} elements={item['elements']}"
                                )
                            else:
                                print(json.dumps(dict(item), indent=2))
                elif arguments.artifact_command == "export":
                    destination = arguments.output or arguments.path.with_suffix(
                        f".{arguments.format}"
                    )
                    print(
                        artifact_service.export_array(
                            arguments.path, array=arguments.array, destination=destination
                        )
                    )
                elif arguments.artifact_command == "visualize":
                    roles = {
                        key: value
                        for key, value in {
                            "positions": arguments.positions,
                            "nodes": arguments.nodes,
                            "edges": arguments.edges,
                        }.items()
                        if value is not None
                    }
                    spec = artifact_service.visualization_spec(
                        arguments.path,
                        visualization_type=arguments.type,
                        roles=roles,
                        visualizer=arguments.visualizer,
                    )
                    if arguments.json:
                        print(json.dumps(spec.to_dict(), indent=2))
                    else:
                        print(VisualizationService().render(spec, arguments.output))
                elif arguments.artifact_command == "list":
                    artifact_payload = (
                        RemoteArtifactService().list(arguments.selector)
                        if str(arguments.selector).startswith("job-")
                        else artifact_service.list(arguments.selector)
                    )
                    if arguments.json:
                        print(json.dumps(artifact_payload, indent=2))
                    else:
                        for artifact in artifact_payload:
                            print(
                                f"{str(artifact.get('logical_name')):<32} "
                                f"{str(artifact.get('type', 'artifact')):<14} "
                                f"{str(artifact.get('size_bytes', '?')):<12} "
                                f"{artifact.get('location', artifact.get('path', ''))}"
                            )
                elif arguments.artifact_command == "fetch":
                    print(
                        RemoteArtifactService().fetch(
                            arguments.job_id, arguments.logical_name, arguments.output
                        )
                    )
                elif arguments.artifact_command == "plugins":
                    print(json.dumps(ArtifactPluginRegistry().names(arguments.kind), indent=2))
                else:
                    shapes = cls._artifact_shapes(arguments.shape)
                    artifact_validator = NumpyArtifactValidator(
                        required_arrays=arguments.require_array,
                        shapes=shapes,
                        finite=arguments.finite,
                    )
                    print(
                        json.dumps(
                            artifact_service.validate(
                                arguments.path, (artifact_validator,)
                            ).to_dict(),
                            indent=2,
                        )
                    )
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "debug":
            try:
                debug_report = PreprocessingDebugService().debug(
                    arguments.config,
                    records=arguments.records,
                    intermediates=arguments.intermediates,
                )
                print(json.dumps(debug_report.to_dict(), indent=2))
                return 0 if debug_report.ok else 1
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
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
                handle, bundle = ControlPlane(run_catalog).submit(
                    arguments.config,
                    cluster=cluster,
                    resources=request,
                    dry_run=arguments.dry_run,
                    run_arguments=cls._remote_run_arguments(arguments),
                )
                submission_payload = {
                    "job": handle.to_dict(),
                    "bundle": bundle.to_dict(),
                    "target": {
                        "cluster": cluster,
                        "source": arguments.default_cluster_source or "explicit",
                    },
                }
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
        materialized_kind = AuthoringConfig.from_yaml(arguments.config).materialize().kind
        if materialized_kind is ConfigurationKind.DATASET:
            recipe = DatasetRecipe.from_yaml(arguments.config)
            if arguments.command == "inspect" or arguments.dry_run:
                print(json.dumps(recipe.inspect().to_dict(), indent=2))
                return 0
            handle = DatasetBuildService().submit(recipe.config, cluster="local")
            print(json.dumps(handle.to_dict(), indent=2))
            return 0
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
    def _keyring_reference(name: str, user: str | None, host: str | None) -> str:
        """Create a stable non-secret keyring identifier for one endpoint."""
        endpoint = f"{user or 'default'}@{host or 'local'}"
        return f"keyring:cluster/{name}/{endpoint}"

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
        if base is None:
            kind = AuthoringConfig.from_yaml(arguments.config).materialize().kind
            if kind is ConfigurationKind.DATASET:
                values.update({})
            elif TaskConfig.is_task_file(arguments.config):
                values.update(TaskConfig.from_yaml(arguments.config).resources.to_dict())
            elif not CommandLineInterface._is_workflow(arguments.config):
                values.update(ExperimentConfig.from_yaml(arguments.config).resources.to_dict())
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
    def _print_overview(payload: dict[str, Any]) -> None:
        jobs = payload["jobs"]
        datasets = payload["datasets"]
        states = jobs.get("by_state", {})
        print(
            f"LambdaForge: running={states.get('running', 0)} "
            f"queued={states.get('queued', 0)} staging={states.get('staging', 0)} "
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
    def _age(created: str) -> str:
        try:
            seconds = max(
                0,
                int((datetime.now(timezone.utc) - datetime.fromisoformat(created)).total_seconds()),
            )
        except ValueError:
            return "unknown"
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        if seconds < 86400:
            return f"{seconds // 3600}h"
        return f"{seconds // 86400}d"

    @staticmethod
    def _job_resources(resources: dict[str, Any] | object) -> str:
        if not isinstance(resources, dict):
            return "unknown"
        selected = [
            f"{key}={resources[key]}"
            for key in ("cpus", "memory", "gpus", "gpu_memory", "time")
            if resources.get(key) not in (None, 0, "")
        ]
        return ",".join(selected) or "default"

    @staticmethod
    def _print_resources(payload: object) -> None:
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict):
                print(value)
                continue
            observed = value.get("observed", {})
            print(
                f"{value.get('cluster', '-')}: "
                f"{'online' if value.get('online') else 'offline'}; "
                f"CPU={observed.get('cpu_total', '?')} "
                f"(load={observed.get('cpu_load', '?')}); "
                f"RAM={observed.get('ram_available_bytes', '?')}/"
                f"{observed.get('ram_total_bytes', '?')}; "
                f"GPUs={observed.get('gpus', [])}"
            )

    @staticmethod
    def _print_storage(payload: list[dict[str, Any]]) -> None:
        for report in payload:
            print(f"{report['cluster']} LambdaForge storage")
            if not report["online"]:
                print(f"  offline: {report.get('error')}")
                continue
            for name, usage in report["categories"].items():
                print(f"  {name:<18} {usage['bytes']:>12} bytes {usage['files']:>8} files")

    @staticmethod
    def _set_dotted(value: dict[str, Any], path: str, replacement: Any) -> None:
        parts = path.split(".")
        if not all(parts):
            raise ValueError("Cluster setting path cannot be empty.")
        current = value
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cluster setting {part!r} is not a mapping.")
            current = child
        current[parts[-1]] = replacement

    @staticmethod
    def _delete_dotted(value: dict[str, Any], path: str) -> None:
        parts = path.split(".")
        current = value
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                raise KeyError(f"Unknown cluster setting {path!r}.")
            current = child
        if parts[-1] not in current:
            raise KeyError(f"Unknown cluster setting {path!r}.")
        del current[parts[-1]]

    @staticmethod
    def _follow_job_logs(jobs: JobService, job_id: str, *, tail: int | None) -> int:
        """Stream scheduler logs by reconnecting through the persistent job record."""
        previous = ""
        try:
            while True:
                current = jobs.logs(job_id, tail=tail)
                delta = current[len(previous) :] if current.startswith(previous) else current
                if delta:
                    print(delta, end="" if delta.endswith("\n") else "\n", flush=True)
                previous = current
                if jobs.get(job_id).state.terminal:
                    return 0
                time.sleep(2.0)
        except KeyboardInterrupt:
            return 130

    @staticmethod
    def _follow_plot(
        arguments: argparse.Namespace,
        visualization: VisualizationService,
        output: Path,
    ) -> int:
        """Synchronize small metric files and atomically refresh a job learning plot."""
        job_id = str(arguments.selector)
        if not job_id.startswith("job-"):
            raise ValueError("--follow requires a persistent JOB selector.")
        remote = RemoteResultService()
        try:
            while True:
                synced = remote.sync(job_id)
                spec = visualization.learning(
                    synced.destination,
                    metrics=arguments.metric,
                    aggregate=arguments.aggregate,
                    uncertainty=arguments.uncertainty,
                )
                visualization.render(spec, output)
                record = remote.jobs.get(job_id)
                if record.state.terminal:
                    print(output)
                    return 0
                time.sleep(arguments.interval)
        except KeyboardInterrupt:
            return 130

    @staticmethod
    def _artifact_shapes(values: Sequence[str]) -> dict[str, tuple[int | None, ...]]:
        """Parse repeated ARRAY=DIM,DIM validation constraints without expressions."""
        output: dict[str, tuple[int | None, ...]] = {}
        for value in values:
            if "=" not in value:
                raise ValueError("--shape must use ARRAY=DIM,DIM with * for any dimension.")
            name, raw = value.split("=", 1)
            output[name] = tuple(
                None if part.strip() == "*" else int(part) for part in raw.split(",")
            )
        return output

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
dependencies = ["lambdaforge>=0.7,<0.8"]

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
        parser = argparse.ArgumentParser(
            prog="lambdaforge",
            description="Configure and run reproducible ML experiments and generic tasks.",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        completion = subparsers.add_parser(
            "completion", help="Generate shell completion for lambdaforge and lf."
        )
        completion.add_argument("shell", choices=("bash", "zsh", "fish"))
        project = subparsers.add_parser("project", help="Inspect project-local LambdaForge state.")
        project_commands = project.add_subparsers(dest="project_command", required=True)
        project_status = project_commands.add_parser("status")
        project_status.add_argument("--root", type=Path)
        project_status.add_argument("--json", action="store_true")
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
        doctor.add_argument(
            "--clusters",
            "--clusters-file",
            dest="clusters",
            type=Path,
            help="Highest-precedence cluster catalogue YAML.",
        )
        doctor.add_argument("--config", type=Path, help="Also check required logical datasets.")
        doctor.add_argument("--json", action="store_true")
        overview = subparsers.add_parser("overview", help="Summarize clusters, jobs and datasets.")
        overview.add_argument("--clusters", type=Path)
        overview.add_argument("--json", action="store_true")
        top = subparsers.add_parser("top", help="Show a refreshable global control-plane view.")
        top.add_argument("--clusters", type=Path)
        top.add_argument("--follow", action="store_true")
        top.add_argument("--interval", type=float, default=5.0)
        top.add_argument("--json", action="store_true")
        resources = subparsers.add_parser(
            "resources", help="Observe cluster CPU, RAM, GPU and jobs."
        )
        resources.add_argument("--on")
        resources.add_argument("--all", action="store_true")
        resources.add_argument("--processes", action="store_true")
        resources.add_argument("--clusters", type=Path)
        resources.add_argument("--json", action="store_true")
        storage = subparsers.add_parser(
            "storage", help="Inspect and conservatively collect internal storage."
        )
        storage.add_argument("--clusters", type=Path)
        storage_commands = storage.add_subparsers(dest="storage_command", required=True)
        storage_status = storage_commands.add_parser("status")
        storage_status.add_argument("--on", default="local")
        storage_status.add_argument("--all", action="store_true")
        storage_status.add_argument("--json", action="store_true")
        storage_gc = storage_commands.add_parser("gc")
        storage_gc.add_argument("--on", default="local")
        storage_gc.add_argument("--apply", action="store_true")
        storage_gc.add_argument("--json", action="store_true")
        environments = subparsers.add_parser("environments", help="Inspect managed environments.")
        environments.add_argument("--clusters", type=Path)
        environment_commands = environments.add_subparsers(
            dest="environment_command", required=True
        )
        environment_list = environment_commands.add_parser("list")
        environment_list.add_argument("--on", default="local")
        environment_show = environment_commands.add_parser("show")
        environment_show.add_argument("environment_id")
        environment_show.add_argument("--on", default="local")
        environment_gc = environment_commands.add_parser("gc")
        environment_gc.add_argument("--on", default="local")
        environment_gc.add_argument("--apply", action="store_true")
        clusters = subparsers.add_parser("clusters", help="Inspect and test cluster profiles.")
        clusters.add_argument(
            "--catalog",
            "--clusters-file",
            dest="catalog",
            type=Path,
            help="Highest-precedence cluster catalogue YAML.",
        )
        cluster_commands = clusters.add_subparsers(dest="cluster_command", required=True)
        cluster_add = cluster_commands.add_parser("add")
        cluster_add.add_argument("name")
        cluster_add.add_argument("--host")
        cluster_add.add_argument("--user")
        cluster_add.add_argument("--port", type=int, default=22)
        cluster_add.add_argument("--known-hosts", type=Path)
        cluster_add.add_argument("--ssh-timeout", type=float, default=15.0)
        cluster_add.add_argument("--connect-timeout", type=float, default=15.0)
        cluster_add.add_argument("--auth-timeout", type=float, default=30.0)
        cluster_add.add_argument("--banner-timeout", type=float, default=30.0)
        cluster_add.add_argument("--keepalive-interval", type=float, default=30.0)
        cluster_add.add_argument("--control-persist", type=float, default=60.0)
        cluster_add.add_argument("--command-timeout", type=float)
        cluster_add.add_argument("--no-multiplex", action="store_true")
        cluster_add.add_argument("--auth", choices=("openssh", "password"), default="openssh")
        cluster_add.add_argument(
            "--credential", help="Credential reference (keyring:... or env:...); never the value."
        )
        cluster_add.add_argument(
            "--store-password", action="store_true", help="Prompt and store in the system keyring."
        )
        cluster_add.add_argument("--scope", choices=("user", "project"), default="user")
        cluster_add.add_argument("--scheduler", choices=("local", "slurm"), default="slurm")
        cluster_add.add_argument("--workspace", required=True)
        cluster_add.add_argument("--state-root")
        cluster_add.add_argument("--cache-root")
        cluster_add.add_argument("--run-root")
        cluster_add.add_argument("--dataset-root")
        cluster_add.add_argument("--cache-max-size")
        cluster_add.add_argument("--cache-max-age")
        cluster_add.add_argument("--python", default="python3")
        cluster_add.add_argument(
            "--environment", choices=("existing", "managed"), default="managed"
        )
        cluster_add.add_argument("--wheelhouse", type=Path)
        cluster_add.add_argument(
            "--torch-channel",
            choices=tuple(sorted(TorchInstallationPolicy.CHANNELS)),
            default="auto",
            help="Official PyTorch wheel channel; auto probes driver, GPU and remote Python.",
        )
        cuda_requirement = cluster_add.add_mutually_exclusive_group()
        cuda_requirement.add_argument("--require-cuda", action="store_true")
        cuda_requirement.add_argument("--no-require-cuda", action="store_true")
        cluster_add.add_argument("--project-module")
        cluster_add.add_argument("--data-environment")
        cluster_list = cluster_commands.add_parser("list")
        cluster_list.add_argument("--json", action="store_true")
        cluster_show = cluster_commands.add_parser("show")
        cluster_show.add_argument("name")
        cluster_show.add_argument("--json", action="store_true")
        cluster_inspect = cluster_commands.add_parser("inspect")
        cluster_inspect.add_argument("name")
        cluster_inspect.add_argument("--json", action="store_true")
        cluster_set = cluster_commands.add_parser("set")
        cluster_set.add_argument("name")
        cluster_set.add_argument("key")
        cluster_set.add_argument("value")
        cluster_unset = cluster_commands.add_parser("unset")
        cluster_unset.add_argument("name")
        cluster_unset.add_argument("key")
        cluster_remove = cluster_commands.add_parser("remove")
        cluster_remove.add_argument("name")
        cluster_export = cluster_commands.add_parser("export")
        cluster_export.add_argument("name")
        cluster_export.add_argument("--output", type=Path)
        cluster_credentials = cluster_commands.add_parser("credentials")
        credential_commands = cluster_credentials.add_subparsers(
            dest="credential_command", required=True
        )
        credential_set = credential_commands.add_parser("set")
        credential_set.add_argument("name")
        credential_delete = credential_commands.add_parser("delete")
        credential_delete.add_argument("name")
        cluster_test = cluster_commands.add_parser("test")
        cluster_test.add_argument("name")
        cluster_test.add_argument("--json", action="store_true")
        cluster_bootstrap = cluster_commands.add_parser("bootstrap")
        cluster_bootstrap.add_argument("name")
        cluster_bootstrap.add_argument("--wheelhouse", type=Path)
        cluster_bootstrap.add_argument("--json", action="store_true")
        cluster_resources = cluster_commands.add_parser("resources")
        cluster_resources.add_argument("name")
        cluster_resources.add_argument("--json", action="store_true")
        cluster_storage = cluster_commands.add_parser("storage")
        cluster_storage.add_argument("name")
        cluster_storage.add_argument("--json", action="store_true")
        jobs = subparsers.add_parser("jobs", help="List and control persistent jobs.")
        jobs.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        job_commands = jobs.add_subparsers(dest="job_command", required=True)
        jobs_list = job_commands.add_parser("list")
        jobs_list.add_argument("--cluster", "--on", dest="cluster")
        jobs_list.add_argument("--state", choices=tuple(state.value for state in JobState))
        jobs_list.add_argument("--name")
        jobs_list.add_argument("--all", action="store_true")
        jobs_list.add_argument("--json", action="store_true")
        jobs_status = job_commands.add_parser("status")
        jobs_status.add_argument("job_id")
        jobs_status.add_argument("--json", action="store_true")
        jobs_show = job_commands.add_parser("show")
        jobs_show.add_argument("job_id")
        jobs_show.add_argument("--json", action="store_true")
        jobs_logs = job_commands.add_parser("logs")
        jobs_logs.add_argument("job_id")
        jobs_logs.add_argument("--tail", type=int)
        jobs_logs.add_argument("--follow", action="store_true")
        jobs_cancel = job_commands.add_parser("cancel")
        jobs_cancel.add_argument("job_id")
        jobs_pause = job_commands.add_parser("pause")
        jobs_pause.add_argument("job_id")
        jobs_resume = job_commands.add_parser("resume")
        jobs_resume.add_argument("job_id")
        jobs_delete = job_commands.add_parser("delete")
        jobs_delete.add_argument("job_id")
        jobs_reconcile = job_commands.add_parser("reconcile")
        jobs_reconcile.add_argument("--cluster", "--on", dest="cluster")
        jobs_reconcile.add_argument("--all", action="store_true")
        jobs_groups = job_commands.add_parser("groups")
        jobs_groups.add_argument("group_id", nargs="?")
        jobs_group = job_commands.add_parser("group")
        jobs_group_commands = jobs_group.add_subparsers(dest="group_command", required=True)
        jobs_group_show = jobs_group_commands.add_parser("show")
        jobs_group_show.add_argument("group_id")
        jobs_group_commands.add_parser("list")
        jobs_retry = job_commands.add_parser("retry")
        jobs_retry.add_argument("job_id")
        jobs_retry.add_argument("--dry-run", action="store_true")
        status = subparsers.add_parser("status", help="List jobs or show one persistent job.")
        status.add_argument("job_id", nargs="?")
        status.add_argument("--on")
        status.add_argument("--state", choices=tuple(state.value for state in JobState))
        status.add_argument("--name")
        status.add_argument("--clusters", type=Path)
        status.add_argument("--json", action="store_true")
        logs = subparsers.add_parser("logs", help="Read or follow one persistent job log.")
        logs.add_argument("job_id")
        logs.add_argument("--tail", type=int)
        logs.add_argument("--follow", action="store_true")
        logs.add_argument("--clusters", type=Path)
        cancel = subparsers.add_parser("cancel", help="Cancel one persistent job.")
        cancel.add_argument("job_id")
        cancel.add_argument("--clusters", type=Path)
        retry = subparsers.add_parser("retry", help="Retry one terminal persistent job.")
        retry.add_argument("job_id")
        retry.add_argument("--dry-run", action="store_true")
        retry.add_argument("--clusters", type=Path)
        data = subparsers.add_parser("data", help="Inspect or explicitly replicate datasets.")
        data.add_argument("--catalog", type=Path, required=True, help="Data catalogue YAML.")
        data.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
        data_commands = data.add_subparsers(dest="data_command", required=True)
        data_commands.add_parser("list")
        data_locations = data_commands.add_parser("locations")
        data_locations.add_argument("dataset")
        data_inspect = data_commands.add_parser("inspect")
        data_inspect.add_argument("dataset")
        data_replicate = data_commands.add_parser("replicate")
        data_replicate.add_argument("dataset")
        data_replicate.add_argument("--from", dest="source", default="local")
        data_replicate.add_argument("--to", dest="destination", required=True)
        data_replicate.add_argument(
            "--apply", action="store_true", help="Transfer bytes; omission is preview-only."
        )
        datasets = subparsers.add_parser(
            "datasets", help="Discover and safely manage registered dataset versions."
        )
        datasets.add_argument("--clusters", type=Path)
        dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
        dataset_list = dataset_commands.add_parser("list")
        dataset_list.add_argument("--on")
        dataset_list.add_argument("--all", action="store_true")
        dataset_list.add_argument("--json", action="store_true")
        dataset_show = dataset_commands.add_parser("show")
        dataset_show.add_argument("dataset")
        dataset_show.add_argument("--json", action="store_true")
        for build_operation in ("plan", "build"):
            dataset_build = dataset_commands.add_parser(build_operation)
            dataset_build.add_argument("dataset")
            dataset_build.add_argument("--on", default="local")
            dataset_build.add_argument("--force", action="store_true")
            dataset_build.add_argument("--force-stage", action="append", default=[])
            dataset_build.add_argument("--verbose", action="store_true")
            dataset_build.add_argument("--json", action="store_true")
            if build_operation == "build":
                dataset_build.add_argument("--dry-run", action="store_true")
        dataset_add = dataset_commands.add_parser("add")
        dataset_add.add_argument("manifest", type=Path)
        dataset_add.add_argument("--on", default="local")
        dataset_add.add_argument("--root", type=Path)
        dataset_locations = dataset_commands.add_parser("locations")
        dataset_locations.add_argument("dataset")
        dataset_locations.add_argument("--json", action="store_true")
        dataset_stats = dataset_commands.add_parser("stats")
        dataset_stats.add_argument("dataset")
        dataset_stats.add_argument("--on")
        dataset_stats.add_argument("--schema", type=Path)
        dataset_stats.add_argument("--json", action="store_true")
        dataset_members = dataset_commands.add_parser("members")
        dataset_members.add_argument("dataset")
        dataset_members.add_argument("--on")
        dataset_members.add_argument("--partition", action="append", default=[])
        dataset_members.add_argument("--offset", type=int, default=0)
        dataset_members.add_argument("--limit", type=int, default=100)
        dataset_members.add_argument("--json", action="store_true")
        dataset_member = dataset_commands.add_parser("member")
        dataset_member.add_argument("dataset")
        dataset_member.add_argument("member_id")
        dataset_member.add_argument("--on")
        dataset_member.add_argument("--json", action="store_true")
        dataset_diff = dataset_commands.add_parser("diff")
        dataset_diff.add_argument("left")
        dataset_diff.add_argument("right")
        dataset_diff.add_argument("--on")
        dataset_diff.add_argument("--json", action="store_true")
        dataset_verify = dataset_commands.add_parser("verify")
        dataset_verify.add_argument("dataset")
        dataset_verify.add_argument("--on")
        dataset_verify.add_argument("--json", action="store_true")
        dataset_lineage = dataset_commands.add_parser("lineage")
        dataset_lineage.add_argument("dataset")
        dataset_lineage.add_argument("--json", action="store_true")
        dataset_remove = dataset_commands.add_parser("remove")
        dataset_remove.add_argument("dataset")
        dataset_remove.add_argument("--on")
        dataset_delete = dataset_commands.add_parser("delete")
        dataset_delete.add_argument("dataset")
        dataset_delete.add_argument("--on", required=True)
        dataset_delete.add_argument("--apply", action="store_true")
        dataset_materialize = dataset_commands.add_parser("materialize")
        dataset_materialize.add_argument("dataset")
        dataset_materialize.add_argument("--on", required=True)
        dataset_materialize.add_argument(
            "--strategy", choices=("auto", "replicate", "build"), default="auto"
        )
        dataset_materialize.add_argument("--apply", action="store_true")
        dataset_materialize.add_argument("--json", action="store_true")
        dataset_replicate = dataset_commands.add_parser("replicate")
        dataset_replicate.add_argument("dataset")
        dataset_replicate.add_argument("--from", dest="source", required=True)
        dataset_replicate.add_argument("--to", dest="destination", required=True)
        dataset_replicate.add_argument("--apply", action="store_true")
        dataset_replicate.add_argument("--json", action="store_true")
        for entity in ("configs", "experiments", "tasks"):
            entity_parser = subparsers.add_parser(
                entity, help=f"Discover and operate project {entity} by name."
            )
            entity_parser.add_argument("--root", type=Path)
            entity_commands = entity_parser.add_subparsers(dest="entity_command", required=True)
            entity_list = entity_commands.add_parser("list")
            entity_list.add_argument("--json", action="store_true")
            entity_show = entity_commands.add_parser("show")
            entity_show.add_argument("selector")
            entity_show.add_argument("--json", action="store_true")
            for operation in ("validate", "plan"):
                operation_parser = entity_commands.add_parser(operation)
                operation_parser.add_argument("selector")
            entity_run = entity_commands.add_parser("run")
            entity_run.add_argument("selector")
            entity_run.add_argument("--on", action="append")
            entity_run.add_argument("--dry-run", action="store_true")
            entity_run.add_argument("--independent-hpo", action="store_true")
            if entity == "experiments":
                for operation in ("status", "history", "results"):
                    entity_query = entity_commands.add_parser(operation)
                    entity_query.add_argument("selector")
                    entity_query.add_argument("--json", action="store_true")
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
        placement.add_argument(
            "--on",
            action="append",
            help="Submit through a cluster profile; repeat for independent replicas.",
        )
        placement.add_argument("--profile", help="Use a named cluster/resource preset.")
        run.add_argument(
            "--independent-hpo",
            action="store_true",
            help="Allow repeated --on to create separate, uncoordinated HPO studies.",
        )
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
        results = subparsers.add_parser("results", help="Query, compare, export or sync results.")
        result_commands = results.add_subparsers(dest="results_command", required=True)
        result_audit = result_commands.add_parser(
            "audit", help="Audit attempts by scientific configuration identity."
        )
        result_audit.add_argument("source", type=Path, help="Experiment YAML or result-tree root.")
        result_audit.add_argument("--status", choices=("ok", "failed", "interrupted", "dry_run"))
        result_audit.add_argument(
            "--no-archived",
            action="store_true",
            help="Only show the canonical result.json for each run directory.",
        )
        result_audit.add_argument(
            "--duplicates",
            action="store_true",
            help="Only show identities with more than one attempt.",
        )
        result_audit.add_argument(
            "--write-index",
            action="store_true",
            help="Atomically write .lambdaforge/result-index.json under the suite root.",
        )
        result_audit.add_argument(
            "--fail-on-ambiguous",
            action="store_true",
            help="Return exit code 2 when one identity has multiple successful attempts.",
        )
        result_audit.add_argument(
            "--json", action="store_true", help="Print machine-readable JSON."
        )
        result_list = result_commands.add_parser("list")
        result_list.add_argument("--root", action="append", type=Path)
        result_list.add_argument("--status", choices=("ok", "failed", "interrupted", "dry_run"))
        result_list.add_argument("--json", action="store_true")
        result_show = result_commands.add_parser("show")
        result_show.add_argument("selector")
        result_show.add_argument("--root", action="append", type=Path)
        result_show.add_argument("--json", action="store_true")
        result_compare = result_commands.add_parser("compare")
        result_compare.add_argument("selectors", nargs="+")
        result_compare.add_argument("--metric", action="append", default=[])
        result_compare.add_argument("--confidence", type=float, default=0.95)
        result_compare.add_argument("--direction", choices=("minimize", "maximize"))
        result_compare.add_argument("--root", action="append", type=Path)
        result_compare.add_argument("--json", action="store_true")
        result_export = result_commands.add_parser("export")
        result_export.add_argument("selector")
        result_export.add_argument("--format", choices=("csv", "json", "parquet"), default="csv")
        result_export.add_argument("--output", type=Path)
        result_export.add_argument("--series", action="store_true")
        result_export.add_argument("--root", action="append", type=Path)
        result_sync = result_commands.add_parser("sync")
        result_sync.add_argument("job_id")
        result_sync.add_argument("--destination", type=Path)
        result_sync.add_argument("--json", action="store_true")
        plot = subparsers.add_parser("plot", help="Build reproducible scientific plots.")
        plot_commands = plot.add_subparsers(dest="plot_command", required=True)
        plot_learning = plot_commands.add_parser("learning")
        plot_learning.add_argument("selector")
        plot_learning.add_argument("--metric", action="append", default=[])
        plot_learning.add_argument("--aggregate", choices=("individual", "mean"), default="mean")
        plot_learning.add_argument("--uncertainty", choices=("none", "std", "ci"), default="std")
        plot_learning.add_argument("--output", type=Path)
        plot_learning.add_argument("--follow", action="store_true")
        plot_learning.add_argument("--interval", type=float, default=5.0)
        plot_learning.add_argument("--json", action="store_true")
        plot_sweep = plot_commands.add_parser("sweep")
        plot_sweep.add_argument("config", type=Path)
        plot_sweep.add_argument("--x", required=True)
        plot_sweep.add_argument("--y")
        plot_sweep.add_argument("--metric", action="append", required=True)
        plot_sweep.add_argument("--uncertainty", choices=("none", "std", "ci"), default="std")
        plot_sweep.add_argument("--interpolate", action="store_true")
        plot_sweep.add_argument(
            "--normalize",
            action="store_true",
            help=(
                "Min-max normalize each metric over observed cells; raw values remain in PlotSpec."
            ),
        )
        plot_sweep.add_argument(
            "--kind",
            choices=("auto", "line", "scatter", "heatmap", "contour", "surface"),
            default="auto",
        )
        plot_sweep.add_argument("--output", type=Path)
        plot_sweep.add_argument("--json", action="store_true")
        plot_seeds = plot_commands.add_parser("seeds")
        plot_seeds.add_argument("selector")
        plot_seeds.add_argument("--metric", required=True)
        plot_seeds.add_argument("--kind", choices=("box", "violin", "strip"), default="box")
        plot_seeds.add_argument("--output", type=Path)
        plot_seeds.add_argument("--json", action="store_true")
        plot_hpo = plot_commands.add_parser("hpo")
        plot_hpo.add_argument("study", type=Path)
        plot_hpo.add_argument("--parameter")
        plot_hpo.add_argument("--direction", choices=("minimize", "maximize"), default="minimize")
        plot_hpo.add_argument("--output", type=Path)
        plot_hpo.add_argument("--json", action="store_true")
        plot_resources = plot_commands.add_parser("resources")
        plot_resources.add_argument("selector")
        plot_resources.add_argument("--output", type=Path)
        plot_resources.add_argument("--json", action="store_true")
        artifact = subparsers.add_parser("artifact", help="Inspect and retrieve artifacts safely.")
        artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
        artifact_inspect = artifact_commands.add_parser("inspect")
        artifact_inspect.add_argument("path", type=Path)
        artifact_inspect.add_argument("--array")
        artifact_inspect.add_argument("--rows", type=int, default=20)
        artifact_inspect.add_argument("--slice")
        artifact_inspect.add_argument("--inspector")
        artifact_inspect.add_argument("--json", action="store_true")
        artifact_export = artifact_commands.add_parser("export")
        artifact_export.add_argument("path", type=Path)
        artifact_export.add_argument("--array", required=True)
        artifact_export.add_argument("--format", choices=("csv", "json", "npy"), default="csv")
        artifact_export.add_argument("--output", type=Path)
        artifact_visualize = artifact_commands.add_parser("visualize")
        artifact_visualize.add_argument("path", type=Path)
        artifact_visualize.add_argument(
            "--type", choices=("graph", "point-cloud", "mesh"), required=True
        )
        artifact_visualize.add_argument("--positions")
        artifact_visualize.add_argument("--nodes")
        artifact_visualize.add_argument("--edges")
        artifact_visualize.add_argument("--visualizer")
        artifact_visualize.add_argument("--output", type=Path, default=Path("artifact.png"))
        artifact_visualize.add_argument("--json", action="store_true")
        artifact_list = artifact_commands.add_parser("list")
        artifact_list.add_argument("selector")
        artifact_list.add_argument("--json", action="store_true")
        artifact_fetch = artifact_commands.add_parser("fetch")
        artifact_fetch.add_argument("job_id")
        artifact_fetch.add_argument("logical_name")
        artifact_fetch.add_argument("--output", type=Path, default=Path.cwd())
        artifact_plugins = artifact_commands.add_parser("plugins")
        artifact_plugins.add_argument("--kind", choices=tuple(ArtifactPluginRegistry.GROUPS))
        artifact_plugins.add_argument("--json", action="store_true")
        artifact_validate = artifact_commands.add_parser("validate")
        artifact_validate.add_argument("path", type=Path)
        artifact_validate.add_argument("--require-array", action="append", default=[])
        artifact_validate.add_argument("--shape", action="append", default=[])
        artifact_validate.add_argument("--finite", action="store_true")
        artifact_validate.add_argument("--json", action="store_true")
        debug = subparsers.add_parser("debug", help="Sample a preprocessing pipeline safely.")
        debug.add_argument("config", type=Path)
        debug.add_argument("--records", type=int, default=1)
        debug.add_argument("--intermediates", type=Path)
        debug.add_argument("--json", action="store_true")
        return parser
