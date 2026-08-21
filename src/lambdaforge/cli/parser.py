"""Canonical argparse tree shared by the ``lf`` and ``lambdaforge`` entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from lambdaforge.artifacts.ArtifactPluginRegistry import ArtifactPluginRegistry
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.experiments.migrations.MigrationPreviewFormat import MigrationPreviewFormat
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.plugins.PluginKind import PluginKind


class _LambdaForgeArgumentParser(argparse.ArgumentParser):
    """Route usage failures through the normal diagnostic boundary."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(f"Invalid command line: {message}")


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI grammar without importing command services."""
    parser = _LambdaForgeArgumentParser(
        prog="lambdaforge",
        description="Configure and run reproducible ML experiments and generic tasks.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {LambdaForgeVersion.CURRENT}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include exception type and traceback in errors (accepted anywhere).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional operational detail when a command supports it.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Render errors as a stable JSON envelope (accepted anywhere).",
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
        "explain", help="Explain a configuration or one dotted Schema property."
    )
    explain.add_argument(
        "subject",
        help="CONFIG, changes, or a Schema kind: authoring, experiment, task, workflow.",
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
    top.add_argument(
        "--history",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Keep this many seconds of CPU/RAM/GPU history in the interactive view.",
    )
    top.add_argument(
        "--once", action="store_true", help="Print one snapshot even in an interactive terminal."
    )
    top.add_argument("--json", action="store_true")
    resources = subparsers.add_parser("resources", help="Observe cluster CPU, RAM, GPU and jobs.")
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
    environment_commands = environments.add_subparsers(dest="environment_command", required=True)
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
        "--python-strategy",
        choices=("auto", "existing", "managed"),
        help="Python runtime policy; managed clusters default to auto.",
    )
    cluster_add.add_argument("--python-version", help="Optional managed Python minor, e.g. 3.13.")
    cluster_add.add_argument(
        "--no-managed-python",
        action="store_true",
        help="Allow discovery but prohibit provisioning a user-space Python runtime.",
    )
    cluster_add.add_argument("--environment", choices=("existing", "managed"), default="managed")
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
    cluster_bootstrap.add_argument("--dry-run", action="store_true")
    cluster_bootstrap.add_argument("--json", action="store_true")
    cluster_resources = cluster_commands.add_parser("resources")
    cluster_resources.add_argument("name")
    cluster_resources.add_argument("--json", action="store_true")
    cluster_storage = cluster_commands.add_parser("storage")
    cluster_storage.add_argument("name")
    cluster_storage.add_argument("--json", action="store_true")
    jobs = subparsers.add_parser(
        "jobs",
        help="Inspect and control low-level execution attempts.",
        description="Advanced operational access to scheduler/process jobs and their logs.",
        epilog=(
            "Examples:\n"
            "  lf jobs list --all\n"
            "  lf jobs show latest --json\n"
            "  lf jobs logs latest --follow\n"
            "  lf jobs retry latest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "datasets",
        help="Discover and safely manage immutable dataset versions.",
        description=(
            "Lifecycle: list/show registered versions, inspect or verify content, then "
            "materialize, replicate or remove explicit placements. Run recipe YAML with lf run."
        ),
        epilog=(
            "Examples:\n"
            "  lf datasets list\n"
            "  lf datasets show wisdom-dna@1\n"
            "  lf run datasets/dna.yaml --on citius\n"
            "  lf datasets verify wisdom-dna@1 --on citius"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
            dataset_build.add_argument(
                "--wait-for-submit",
                action="store_true",
                help="Wait for remote staging and scheduler acknowledgement.",
            )
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
        entity_epilog = (
            "Examples:\n"
            "  lf experiments list\n"
            "  lf experiments show baseline\n"
            "  lf experiments runs baseline --json\n"
            "  lf experiments results baseline --json"
            if entity == "experiments"
            else None
        )
        entity_parser = subparsers.add_parser(
            entity,
            help=f"Discover and operate project {entity} by name.",
            epilog=entity_epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        entity_parser.add_argument("--root", type=Path)
        entity_commands = entity_parser.add_subparsers(dest="entity_command", required=True)
        entity_list = entity_commands.add_parser("list")
        entity_list.add_argument("--json", action="store_true")
        entity_show = entity_commands.add_parser("show")
        entity_show.add_argument("selector")
        entity_show.add_argument("--json", action="store_true")
        if entity == "experiments":
            entity_show.add_argument("--revision")
        for operation in ("validate", "plan"):
            operation_parser = entity_commands.add_parser(operation)
            operation_parser.add_argument("selector")
        entity_run = entity_commands.add_parser("run")
        entity_run.add_argument("selector")
        entity_run.add_argument("--on", action="append")
        entity_run.add_argument("--dry-run", action="store_true")
        entity_run.add_argument("--independent-hpo", action="store_true")
        entity_run.add_argument(
            "--allow-duplicate",
            action="store_true",
            help="Intentionally allow the same experiment revision on the same target.",
        )
        if entity == "experiments":
            for operation in ("status", "history", "runs", "results"):
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
        "run", help="Execute a dataset recipe, experiment, task or workflow YAML file."
    )
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--wait-for-submit",
        action="store_true",
        help="Wait for remote staging and scheduler acknowledgement.",
    )
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
    run.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Intentionally allow the same experiment revision on the same target.",
    )
    run.add_argument("--clusters", type=Path, help="Cluster catalogue YAML.")
    lifecycle = run.add_mutually_exclusive_group()
    lifecycle.add_argument(
        "--rerun",
        "--force",
        dest="force",
        action="store_true",
        help="Deliberately create a new execution after scientific success.",
    )
    lifecycle.add_argument(
        "--restart", action="store_true", help="Run from scratch without partial state."
    )
    run.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="Force one dataset stage and its downstream dependants; repeat as needed.",
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
    validate.add_argument("--json", action="store_true", help="Print a machine-readable report.")
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
        help="Query, compare, export or synchronize scientific evidence.",
        description="Inspect immutable run evidence without selecting a scientific winner.",
        epilog=(
            "Examples:\n"
            "  lf results list --json\n"
            "  lf results show baseline\n"
            "  lf results compare baseline candidate --metric val_loss --direction minimize"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    result_audit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
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
        help=("Min-max normalize each metric over observed cells; raw values remain in PlotSpec."),
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
