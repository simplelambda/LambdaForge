"""Small argparse grammar for the Work-centric LambdaForge command line."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(f"Invalid command line: {message}")


def _cluster_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the only supported 0.13 command grammar."""
    parser = _Parser(
        prog="lf",
        description="Run reproducible scientific Work locally or on managed clusters.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {LambdaForgeVersion.CURRENT}"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    commands = parser.add_subparsers(dest="command")
    project = commands.add_parser(
        "project", help="Show the current project identity and discovered root."
    )
    project.add_argument("--json", action="store_true")

    seeds = commands.add_parser("seeds", help="Inspect deterministic project seed streams.")
    seeds.add_argument("--count", type=int, default=10)
    seeds.add_argument("--role", choices=("replicate", "confirmation"), default="replicate")
    seeds.add_argument("--json", action="store_true")

    config = commands.add_parser("config", help="Resolve Work configuration without execution.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    resolve = config_commands.add_parser(
        "resolve", help="Show the complete versioned configuration used by runtime."
    )
    resolve.add_argument("config", type=Path)
    resolve.add_argument("--format", choices=("human", "json", "yaml"), default="human")
    resolve.add_argument("--json", action="store_true")

    fleets = commands.add_parser("fleets", help="Inspect operational cluster fleets.")
    fleets.add_argument("--catalog", type=Path)
    fleet_commands = fleets.add_subparsers(dest="fleet_command", required=True)
    fleet_list = fleet_commands.add_parser("list", help="List configured fleets, without probing.")
    fleet_list.add_argument("--json", action="store_true")
    for name in ("show", "offers"):
        fleet_item = fleet_commands.add_parser(
            name,
            help="Show fleet configuration."
            if name == "show"
            else "Observe member resources; not a GPU grant.",
        )
        fleet_item.add_argument("name")
        fleet_item.add_argument("--json", action="store_true")
    for name, state in (("drain", "draining"), ("disable", "disabled"), ("enable", "online")):
        fleet_item = fleet_commands.add_parser(
            name, help="Change new-work eligibility; never stop active Jobs."
        )
        fleet_item.add_argument("name")
        fleet_item.add_argument("cluster")
        fleet_item.add_argument("--apply", action="store_true")
        fleet_item.add_argument("--json", action="store_true")
        fleet_item.set_defaults(member_state=state)

    init = commands.add_parser("init", help="Create an installable Work project.")
    init.add_argument("directory", type=Path)
    init.add_argument("--force", action="store_true")
    init.add_argument(
        "--template", choices=("minimal", "preprocessing", "training", "full"), default="minimal"
    )

    for name in ("validate", "explain"):
        item = commands.add_parser(name, help=f"{name.title()} current Work YAML.")
        item.add_argument("config", type=Path)
        item.add_argument("--json", action="store_true")

    run = commands.add_parser("run", help="Execute one Work configuration.")
    run.add_argument("config", type=Path)
    targets = run.add_mutually_exclusive_group()
    targets.add_argument("--on", default="local")
    targets.add_argument(
        "--on-fleet",
        help="Execute one fresh adaptive/fixed/repeated Study on a prepared Fleet.",
    )
    run.add_argument("--clusters", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--wait-for-submit", action="store_true")
    run.add_argument("--rerun", action="store_true")
    run.add_argument("--restart", action="store_true")
    run.add_argument("--resume-execution", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--input-pins", type=Path, help=argparse.SUPPRESS)
    run.add_argument(
        "--input-ref",
        action="append",
        default=[],
        metavar="PARAMETER=JSON",
        help="Bind a historical result/product input without editing YAML.",
    )
    run.add_argument("--resume-study-path", type=Path, help=argparse.SUPPRESS)
    run.add_argument(
        "--accept-code-change",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run.add_argument("--allow-duplicate", action="store_true")
    run.add_argument("--json", action="store_true")

    export_study = commands.add_parser(
        "export", help="Export a portable evidence snapshot of any Work/Study state."
    )
    export_study.add_argument(
        "selector", help="Exact Work ID, unambiguous Work name, revision or underlying Job ID."
    )
    export_study.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local parent directory in which to create NAME--EXECUTION_ID.",
    )
    export_study.add_argument(
        "--profile",
        choices=("default", "full"),
        default="default",
        help=(
            "default keeps complete scientific/decision evidence with summarized telemetry; "
            "full also includes duplicate high-frequency controller streams"
        ),
    )
    _cluster_selector(export_study)
    export_study.add_argument("--json", action="store_true")

    import_study = commands.add_parser(
        "import", help="Verify/register a portable Study and its products; never execute it."
    )
    import_study.add_argument("source", type=Path)
    import_study.add_argument("--results-root", type=Path)
    import_study.add_argument("--products-root", type=Path)
    import_study.add_argument(
        "--apply", action="store_true", help="Register after exact verification."
    )
    import_study.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor", help="Diagnose a local or remote runtime.")
    doctor.add_argument("--on", default="local")
    doctor.add_argument("--config", type=Path)
    _cluster_selector(doctor)
    doctor.add_argument("--json", action="store_true")

    overview = commands.add_parser(
        "overview", help="Machine-readable current-project Work overview."
    )
    _cluster_selector(overview)
    overview.add_argument("--json", action="store_true")
    resources = commands.add_parser("resources", help="Inspect cluster resources.")
    resources.add_argument("--on")
    resources.add_argument("--all", action="store_true")
    resources.add_argument("--processes", action="store_true")
    _cluster_selector(resources)
    resources.add_argument("--json", action="store_true")

    clean = commands.add_parser("clean", help="Preview removal of reconstructible cache.")
    clean.add_argument("--on", default="local")
    clean.add_argument("--apply", action="store_true")
    _cluster_selector(clean)
    clean.add_argument("--json", action="store_true")

    storage = commands.add_parser("storage", help="Inspect and reconcile owned storage.")
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    for operation in ("status", "reconcile"):
        storage_operation = storage_commands.add_parser(operation)
        storage_operation.add_argument("--on", default="local")
        _cluster_selector(storage_operation)
        storage_operation.add_argument("--json", action="store_true")
        if operation == "reconcile":
            storage_operation.add_argument(
                "--apply",
                action="store_true",
                help="Update the diagnostic ledger; never delete data.",
            )

    clusters = commands.add_parser("clusters", help="Manage execution targets.")
    clusters.add_argument("--catalog", type=Path)
    cluster_commands = clusters.add_subparsers(dest="cluster_command", required=True)
    add = cluster_commands.add_parser("add", help="Create a cluster profile non-interactively.")
    add.add_argument("name")
    add.add_argument("--host")
    add.add_argument("--user")
    add.add_argument("--port", type=int, default=22)
    add.add_argument("--known-hosts", type=Path)
    add.add_argument("--ssh-timeout", type=float, default=15.0)
    add.add_argument("--connect-timeout", type=float, default=15.0)
    add.add_argument("--auth-timeout", type=float, default=30.0)
    add.add_argument("--banner-timeout", type=float, default=30.0)
    add.add_argument("--keepalive-interval", type=float, default=30.0)
    add.add_argument("--control-persist", type=float, default=60.0)
    add.add_argument("--command-timeout", type=float)
    add.add_argument("--no-multiplex", action="store_true")
    add.add_argument("--auth", choices=("openssh", "password"), default="openssh")
    add.add_argument("--credential")
    add.add_argument("--store-password", action="store_true")
    add.add_argument("--scope", choices=("user", "project"), default="user")
    add.add_argument("--scheduler", choices=("local", "slurm"), default="slurm")
    add.add_argument("--workspace", required=True)
    add.add_argument("--state-root")
    add.add_argument("--cache-root")
    add.add_argument("--run-root")
    add.add_argument("--dataset-root")
    add.add_argument("--cache-max-size")
    add.add_argument("--cache-max-age")
    add.add_argument("--python", default="python3")
    add.add_argument("--python-strategy", choices=("auto", "existing", "managed"))
    add.add_argument("--python-version")
    add.add_argument("--no-managed-python", action="store_true")
    add.add_argument("--environment", choices=("existing", "managed"), default="managed")
    add.add_argument("--wheelhouse", type=Path)
    add.add_argument(
        "--torch-channel", choices=tuple(sorted(TorchInstallationPolicy.CHANNELS)), default="auto"
    )
    cuda = add.add_mutually_exclusive_group()
    cuda.add_argument("--require-cuda", action="store_true")
    cuda.add_argument("--no-require-cuda", action="store_true")
    add.add_argument("--project-module")
    add.add_argument(
        "--project-root",
        help="Absolute remote mirror of the consumer project for shared inputs/outputs.",
    )
    add.add_argument("--data-environment")
    add.add_argument(
        "--gpu-access",
        choices=("auto", "exclusive", "shared", "command", "scheduler"),
        default="auto",
        help="GPU admission policy; SLURM defaults to scheduler and direct hosts to exclusive.",
    )
    add.add_argument(
        "--gpu-command-prefix",
        nargs="+",
        help="argv prepended to GPU work when --gpu-access command (for example: gpu exec).",
    )
    add.add_argument(
        "--gpu-claim-command",
        nargs="+",
        help=(
            "optional argv run before each GPU submission; {gpu_count} expands to the requested "
            "count (for example: gpu claim {gpu_count})"
        ),
    )
    add.add_argument(
        "--gpu-release-command",
        nargs="+",
        help="cleanup argv paired with --gpu-claim-command (for example: gpu release)",
    )
    add.add_argument(
        "--gpu-visibility-command",
        nargs="+",
        help=(
            "argv that prints the currently granted CUDA tokens; required for dynamic grant "
            "shrink/growth unless a '... exec' launcher has a sibling '... env' command"
        ),
    )
    cluster_commands.add_parser("list").add_argument("--json", action="store_true")
    for operation in ("show", "inspect", "test", "resources", "storage"):
        item = cluster_commands.add_parser(operation)
        item.add_argument("name")
        item.add_argument("--json", action="store_true")
    for operation in ("set", "unset"):
        item = cluster_commands.add_parser(operation)
        item.add_argument("name")
        item.add_argument("key")
        if operation == "set":
            item.add_argument("value")
    cluster_commands.add_parser("remove").add_argument("name")
    export = cluster_commands.add_parser("export")
    export.add_argument("name")
    export.add_argument("--output", type=Path)
    credentials = cluster_commands.add_parser("credentials").add_subparsers(
        dest="credential_command", required=True
    )
    credentials.add_parser("set").add_argument("name")
    credentials.add_parser("delete").add_argument("name")
    bootstrap = cluster_commands.add_parser("bootstrap")
    bootstrap.add_argument("name")
    bootstrap.add_argument("--wheelhouse", type=Path)
    bootstrap.add_argument(
        "--project",
        type=Path,
        help="Consumer project whose wheel and declarative native environment are prepared.",
    )
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.add_argument("--json", action="store_true")

    jobs = commands.add_parser("jobs", help="Advanced low-level Job operations.")
    jobs.add_argument("--clusters", type=Path)
    job_commands = jobs.add_subparsers(dest="job_command", required=True)
    listing = job_commands.add_parser("list")
    listing.add_argument("--cluster")
    listing.add_argument("--state")
    listing.add_argument("--name")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--json", action="store_true")
    for operation in ("status", "show", "cancel", "pause", "resume", "delete"):
        item = job_commands.add_parser(operation)
        item.add_argument("job_id")
        item.add_argument("--json", action="store_true")
    logs = job_commands.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--tail", type=int)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--json", action="store_true")
    retry = job_commands.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--dry-run", action="store_true")
    retry.add_argument("--accept-code-change", action="store_true")
    reconcile = job_commands.add_parser("reconcile")
    reconcile.add_argument("--cluster")
    reconcile.add_argument("--all", action="store_true")
    clear = job_commands.add_parser("clear")
    clear.add_argument("--apply", action="store_true")
    clear.add_argument("--json", action="store_true")

    for operation in ("show", "logs", "cancel", "retry", "delete"):
        item = commands.add_parser(operation, help=f"Work-level {operation} operation.")
        item.add_argument("selector")
        item.add_argument("--clusters", type=Path)
        item.add_argument("--apply", action="store_true")
        item.add_argument("--tail", type=int)
        item.add_argument("--follow", action="store_true")
        item.add_argument("--dry-run", action="store_true")
        item.add_argument("--json", action="store_true")
        if operation == "retry":
            item.add_argument(
                "--accept-code-change",
                action="store_true",
                help="Reuse earlier evidence after an explicitly compatible consumer-code fix.",
            )
        if operation in {"show", "logs"}:
            item.add_argument(
                "--run",
                dest="study_run",
                help=(
                    "Exact study Run key shown by the Research Console/overview "
                    "(candidate plus seed)."
                ),
            )
            item.add_argument("--curve-points", type=int, default=80)

    datasets = commands.add_parser("datasets", help="Inspect immutable dataset versions.")
    datasets.add_argument("--clusters", type=Path)
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    preflight = dataset_commands.add_parser(
        "preflight", help="Check before expensive reconstruction."
    )
    preflight.add_argument("dataset")
    preflight.add_argument("--on", default="local")
    preflight.add_argument("--intent", choices=("publish", "reuse", "rebuild"), default="publish")
    preflight.add_argument("--json", action="store_true")
    compare = dataset_commands.add_parser(
        "compare", help="Compare two sealed, co-located datasets."
    )
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--on", default="local")
    compare.add_argument("--verifier", help="Explicit trusted project module:function.")
    compare.add_argument("--policy", type=Path, help="Variable-specific verification policy JSON.")
    compare.add_argument(
        "--contracts",
        type=Path,
        help="Project contracts keyed by exact content ID for historical data.",
    )
    compare.add_argument("--output", type=Path, help="Save the comparison report locally.")
    compare.add_argument("--json", action="store_true")
    publish = dataset_commands.add_parser(
        "publish-candidate", help="Publish saved bytes, not computation."
    )
    publish.add_argument("root", type=Path)
    publish.add_argument("--version", help="Explicit new logical version for different content.")
    publish.add_argument("--on", default="local")
    publish.add_argument("--apply", action="store_true")
    publish.add_argument("--json", action="store_true")
    listing = dataset_commands.add_parser("list")
    listing.add_argument("--on")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--json", action="store_true")
    adopt = dataset_commands.add_parser(
        "adopt", help="Explicitly choose a verified content identity as the project reference."
    )
    adopt.add_argument("dataset")
    adopt.add_argument("--on", required=True)
    adopt.add_argument(
        "--content-id", required=True, help="Exact sha256:… ID from target inventory."
    )
    adopt.add_argument("--apply", action="store_true")
    adopt.add_argument("--json", action="store_true")
    for operation in (
        "show",
        "locations",
        "stats",
        "verify",
        "lineage",
        "remove",
        "reconcile",
        "delete",
        "materialize",
    ):
        item = dataset_commands.add_parser(operation)
        item.add_argument("dataset")
        item.add_argument("--on", default="local")
        item.add_argument("--apply", action="store_true")
        item.add_argument("--schema", type=Path)
        item.add_argument("--strategy", choices=("auto", "replicate"), default="auto")
        item.add_argument("--json", action="store_true")
        if operation in {"delete", "remove"}:
            item.add_argument(
                "--content-id",
                help="Select this exact target identity despite conflicts; preview unless --apply.",
            )
    members = dataset_commands.add_parser("members")
    members.add_argument("dataset")
    members.add_argument("--on", default="local")
    members.add_argument("--partition", action="append", default=[])
    members.add_argument("--offset", type=int, default=0)
    members.add_argument("--limit", type=int, default=100)
    members.add_argument("--json", action="store_true")
    member = dataset_commands.add_parser("member")
    member.add_argument("dataset")
    member.add_argument("member_id")
    member.add_argument("--on", default="local")
    member.add_argument("--json", action="store_true")
    diff = dataset_commands.add_parser("diff")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--on", default="local")
    diff.add_argument("--json", action="store_true")
    add_dataset = dataset_commands.add_parser("add")
    add_dataset.add_argument("manifest", type=Path)
    add_dataset.add_argument("--root", type=Path)
    add_dataset.add_argument("--on", default="local")
    add_dataset.add_argument("--json", action="store_true")
    replicate = dataset_commands.add_parser("replicate")
    replicate.add_argument("dataset")
    replicate.add_argument("--source", required=True)
    replicate.add_argument("--destination", required=True)
    replicate.add_argument("--apply", action="store_true")
    replicate.add_argument(
        "--route",
        choices=("auto", "direct", "relay"),
        default="auto",
        help="Prefer/require site SSH, or relay compressed bytes without local disk staging.",
    )
    replicate.add_argument("--json", action="store_true")

    products = commands.add_parser(
        "products", help="Inspect and publish durable scientific products."
    )
    product_commands = products.add_subparsers(dest="product_command", required=True)
    for operation in (
        "list",
        "show",
        "verify",
        "provenance",
        "publish",
        "export",
        "import",
        "select",
        "decide",
        "consumers",
        "finalize",
        "status",
        "materialize",
    ):
        product_operation = product_commands.add_parser(operation)
        product_operation.add_argument("--root", type=Path)
        product_operation.add_argument("--json", action="store_true")
        if operation in {
            "show",
            "verify",
            "provenance",
            "consumers",
            "export",
            "select",
            "decide",
            "finalize",
            "status",
            "materialize",
        }:
            product_operation.add_argument("selector")
        if operation in {"list", "provenance", "consumers"}:
            product_operation.add_argument("--offset", type=int, default=0)
            product_operation.add_argument("--limit", type=int, default=100)
        if operation == "publish":
            product_operation.add_argument("manifest", type=Path)
            product_operation.add_argument(
                "--file",
                action="append",
                default=[],
                metavar="NAME=PATH",
                help="Explicit source of one declared artifact; copied and verified on apply.",
            )
            product_operation.add_argument("--apply", action="store_true")
        if operation == "export":
            product_operation.add_argument("--output", type=Path, required=True)
            product_operation.add_argument("--apply", action="store_true")
        if operation == "import":
            product_operation.add_argument("source", type=Path)
            product_operation.add_argument("--apply", action="store_true")
        if operation == "materialize":
            product_operation.add_argument("--on", required=True)
            product_operation.add_argument("--clusters", type=Path)
            product_operation.add_argument("--apply", action="store_true")
        if operation in {"select", "decide"}:
            product_operation.add_argument("--name", required=True)
            product_operation.add_argument("--contract", required=True)
            if operation == "select":
                product_operation.add_argument("--policy", type=Path, required=True)
            product_operation.add_argument("--results-root", type=Path)
            product_operation.add_argument("--apply", action="store_true")
        if operation in {"finalize", "status"}:
            product_operation.add_argument("--results-root", type=Path)
        if operation == "finalize":
            product_operation.add_argument("--apply", action="store_true")

    results = commands.add_parser("results", help="Inspect persisted Work results.")
    result_commands = results.add_subparsers(dest="result_command", required=True)
    result_reference = result_commands.add_parser(
        "reference", help="Resolve and pin a historical input without reading artifact bytes."
    )
    result_reference.add_argument("selector")
    result_reference.add_argument("--product")
    result_reference.add_argument(
        "--artifact", help="Exact registered member of the selected product."
    )
    result_reference.add_argument("--run")
    result_reference.add_argument("--attempt", type=int)
    result_reference.add_argument("--root", type=Path)
    result_reference.add_argument("--json", action="store_true")
    result_materialize = result_commands.add_parser(
        "materialize", help="Place verified historical evidence on a cluster; preview first."
    )
    result_materialize.add_argument("selector")
    result_materialize.add_argument("--on", required=True)
    result_materialize.add_argument("--clusters", type=Path)
    result_materialize.add_argument("--root", type=Path)
    result_materialize.add_argument("--apply", action="store_true")
    result_materialize.add_argument("--json", action="store_true")
    result_list = result_commands.add_parser("list")
    result_preview = result_commands.add_parser(
        "preview-output",
        help="Verify and preview one retained small UTF-8 output (64 KiB maximum).",
    )
    result_preview.add_argument("selector")
    result_preview.add_argument("name")
    result_preview.add_argument("--run", dest="run_id")
    result_preview.add_argument("--attempt", type=int)
    result_preview.add_argument("--root", type=Path)
    result_preview.add_argument("--json", action="store_true")
    result_list.add_argument("--root", type=Path)
    result_list.add_argument("--json", action="store_true")
    result_show = result_commands.add_parser("show")
    result_show.add_argument("selector")
    result_show.add_argument("--root", type=Path)
    result_show.add_argument("--json", action="store_true")
    result_show.add_argument(
        "--view", choices=("overview", "attempt", "metrics", "logs", "outputs", "provenance")
    )
    result_show.add_argument("--run", dest="run_id")
    result_show.add_argument("--attempt", type=int)
    result_compare = result_commands.add_parser("compare")
    result_compare.add_argument("selectors", nargs="+")
    result_compare.add_argument("--metric")
    result_compare.add_argument("--mode", choices=("min", "max"), default="max")
    result_compare.add_argument("--root", type=Path)
    result_compare.add_argument("--json", action="store_true")
    result_analyze = result_commands.add_parser("analyze")
    result_analyze.add_argument("selector")
    result_analyze.add_argument("--root", type=Path)
    result_analyze.add_argument("--recompute", action="store_true")
    result_analyze.add_argument("--json", action="store_true")
    result_report = result_commands.add_parser("report")
    result_report.add_argument("selector")
    result_report.add_argument("--output", type=Path, required=True)
    result_report.add_argument("--root", type=Path)
    result_report.add_argument("--recompute", action="store_true")
    result_replay = result_commands.add_parser("replay")
    result_replay.add_argument("selector")
    result_replay.add_argument(
        "--policy",
        choices=("recorded", "ari-v2-compat", "ari-v3-compat", "ari-v3.1"),
        default="ari-v3.1",
    )
    result_replay.add_argument("--root", type=Path)
    result_replay.add_argument("--json", action="store_true")
    return parser
