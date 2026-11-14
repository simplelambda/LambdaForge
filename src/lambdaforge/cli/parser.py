"""Small argparse grammar for the Work-centric LambdaForge command line."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import NoReturn

from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        suggestion = ""
        invalid = re.search(r"invalid choice: '([^']+)' \(choose from (.+)\)", message)
        if invalid:
            choices = re.findall(r"'([^']+)'", invalid.group(2))
            matches = difflib.get_close_matches(invalid.group(1), choices, n=1, cutoff=0.6)
            if matches:
                suggestion = f" Did you mean {matches[0]!r}?"
        elif message.startswith("unrecognized arguments:"):
            word = message.split(":", 1)[1].strip().split()[0]

            def options(parser: argparse.ArgumentParser) -> set[str]:
                names = set(parser._option_string_actions)
                for action in parser._actions:
                    if isinstance(action, argparse._SubParsersAction):
                        for child in action.choices.values():
                            names.update(options(child))
                return names

            matches = difflib.get_close_matches(word, options(self), n=1, cutoff=0.6)
            if matches:
                suggestion = f" Did you mean {matches[0]!r}?"
        raise ValueError(
            f"Invalid command line: {message}{suggestion} Use lf --help for real commands."
        )


def _cluster_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the sole installed Work-centric command grammar."""
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
    catalog = commands.add_parser(
        "catalog", help="Metadata-only native scientific operation contracts."
    )
    catalog.add_argument("operation", nargs="?")
    catalog.add_argument("--kind")
    catalog.add_argument("--json", action="store_true")
    from lambdaforge.cli.ai import add_ai_parser

    add_ai_parser(commands)
    intent_help = commands.add_parser("help", help="Find real operations by task intent.")
    intent_help.add_argument("--search", default="")
    intent_help.add_argument("--limit", type=int, default=10)
    intent_help.add_argument("--json", action="store_true")
    completion = commands.add_parser(
        "completion", help="Print shell completion; never edit shell settings."
    )
    completion.add_argument("shell", choices=("bash", "zsh", "fish"))
    completion_query = commands.add_parser("_complete", help=argparse.SUPPRESS)
    completion_query.add_argument("words", nargs=argparse.REMAINDER)
    for inspection_verb in ("inspect", "diagnose", "watch"):
        inspection = commands.add_parser(
            inspection_verb, help="Read native Work/Study/Execution/Job evidence and observation."
        )
        inspection.add_argument("selector")
        inspection.add_argument(
            "--kind", choices=("auto", "work", "study", "execution", "job"), default="auto"
        )
        inspection.add_argument(
            "--refresh",
            action="store_true",
            help="Explicit provider observation; never read artifacts.",
        )
        inspection.add_argument("--clusters", type=Path)
        inspection.add_argument("--json", action="store_true")
        if inspection_verb == "watch":
            inspection.add_argument("--interval", type=float, default=2)
            inspection.add_argument(
                "--count",
                type=int,
                help="Stop after this many observations; JSON output is NDJSON.",
            )
    studies = commands.add_parser(
        "studies", help="Review and submit verified incremental fixed Study deltas."
    )
    study_commands = studies.add_subparsers(dest="study_command", required=True)
    extension = study_commands.add_parser(
        "extend", help="Preview a fixed delta; explicitly apply a reviewed supported plan."
    )
    extension.add_argument(
        "study", help="Exact Series, Job, Execution ID or unambiguous native name."
    )
    extension.add_argument("--config", type=Path, required=True)
    extension.add_argument(
        "--on", help="Defaults to the exact origin placement; no implicit migration."
    )
    extension_mode = extension.add_mutually_exclusive_group()
    extension_mode.add_argument("--dry-run", action="store_true")
    extension_mode.add_argument(
        "--apply", action="store_true", help="Submit only new cells of a reviewed supported plan."
    )
    extension.add_argument("--plan-hash", help="Exact reviewed preview hash required with --apply.")
    extension.add_argument(
        "--code-policy",
        choices=("revisioned", "strict"),
        default="revisioned",
        help="Revisioned permits current code with separate inference cohorts; "
        "strict requires equivalence.",
    )
    extension.add_argument("--json", action="store_true")
    _cluster_selector(extension)
    for series_verb in ("history", "show", "refresh", "analyze", "report", "export"):
        series_action = study_commands.add_parser(
            series_verb,
            help=(
                "Reconcile native delta outcomes."
                if series_verb == "refresh"
                else "Read owned series metadata only."
            ),
        )
        series_action.add_argument("series", help="Exact Study series ID.")
        series_action.add_argument("--json", action="store_true")
        if series_verb == "report":
            series_action.add_argument("--output", type=Path, required=True)
        if series_verb == "export":
            series_action.add_argument("--destination", type=Path, required=True)
        _cluster_selector(series_action)
    reports = commands.add_parser("reports", help="Open generated reports with scoped lazy assets.")
    report_actions = reports.add_subparsers(dest="report_command", required=True)
    serve_report = report_actions.add_parser(
        "serve", help="Read-only loopback viewer; no science executed."
    )
    serve_report.add_argument("path", type=Path)
    serve_report.add_argument("--max-asset-mib", type=int, default=256)
    serve_report.add_argument("--idle-seconds", type=float, default=1800)
    serve_report.add_argument("--open", action="store_true")
    _cluster_selector(serve_report)
    project = commands.add_parser(
        "project", help="Show the current project identity and discovered root."
    )
    project.add_argument("--json", action="store_true")

    health = commands.add_parser(
        "health", help="Inspect operational evidence, freshness and ownership."
    )
    health.add_argument("selector", nargs="?")
    health.add_argument("--refresh", action="store_true")
    health.add_argument("--clusters", type=Path)
    health.add_argument("--json", action="store_true")

    incidents = commands.add_parser("incidents", help="Read durable local operational incidents.")
    incidents.add_argument("--job")
    incidents.add_argument("--clusters", type=Path)
    incidents.add_argument("--json", action="store_true")

    monitor = commands.add_parser(
        "monitor", help="Manage the independent project operational observer."
    )
    monitor.add_argument("--clusters", type=Path)
    monitor_actions = monitor.add_subparsers(dest="monitor_command", required=True)
    for name in (
        "start",
        "stop",
        "status",
        "diagnose",
        "run",
        "once",
        "install",
        "config",
        "test-alert",
    ):
        action = monitor_actions.add_parser(name)
        action.add_argument("--json", action="store_true")
        if name == "install":
            action.add_argument("--apply", action="store_true")
        if name == "config":
            action.add_argument(
                "--file", type=Path, help="Validated JSON; only secret references are accepted."
            )

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
    schema = config_commands.add_parser(
        "schema", help="Print/export the installed Work JSON Schema."
    )
    schema.add_argument("--output", type=Path)
    schema.add_argument("--kind", choices=("work", "cluster"), default="work")
    schema.add_argument("--json", action="store_true")
    field_help = config_commands.add_parser(
        "explain", help="Static schema field help without Work imports."
    )
    field_help.add_argument("config", type=Path)
    field_help.add_argument("--field", required=True)
    field_help.add_argument("--json", action="store_true")
    comparison = config_commands.add_parser(
        "diff", help="Native semantic diff, not recovery permission."
    )
    comparison.add_argument("previous", type=Path)
    comparison.add_argument("proposed", type=Path)
    comparison.add_argument("--json", action="store_true")

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
    init.add_argument("--agents", choices=("none", "portable", "codex"), default="none")
    init.add_argument("--knowledge", choices=("none", "markdown", "obsidian"), default="none")
    init.add_argument(
        "--install-deps",
        action="store_true",
        help="Consent to declared optional agent Python dependencies, never external software.",
    )
    init.add_argument(
        "--interactive", action="store_true", help="Ask about agent and memory setup."
    )
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
    run.add_argument(
        "--queue",
        action="store_true",
        help="Prepare now; run sequentially after earlier Jobs on this direct cluster.",
    )
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

    queue = commands.add_parser("queue", help="Inspect persistent project/cluster execution FIFO.")
    queue.add_argument("--on", default="local")
    queue.add_argument("--clusters", type=Path)
    queue.add_argument("--json", action="store_true")
    queue.add_argument(
        "--all", action="store_true", help="Include released/cancelled ticket history."
    )
    queue.add_argument(
        "--remove-pending",
        metavar="JOB_ID",
        help="Remove only an unstarted admission ticket; never stop a running Job.",
    )
    queue.add_argument("--apply", action="store_true")

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
    resources.add_argument(
        "--report",
        action="store_true",
        help="Sanitized saved-evidence diagnostic; no cluster query.",
    )
    resources.add_argument("--since", help="Report interval start, ISO-8601 UTC.")
    resources.add_argument("--until", help="Report interval end, ISO-8601 UTC.")
    resources.add_argument(
        "--limit", type=int, default=500, help="Maximum saved report Jobs (1..1000)."
    )
    _cluster_selector(resources)
    resources.add_argument("--json", action="store_true")

    clean = commands.add_parser("clean", help="Preview removal of reconstructible cache.")
    clean.add_argument("--on", default="local")
    clean.add_argument("--apply", action="store_true")
    clean.add_argument(
        "--environments-only",
        action="store_true",
        help="Collect only obsolete environment caches; do not compact Jobs or checkpoints.",
    )
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
    repair = storage_commands.add_parser(
        "repair",
        help="Preview retirement of exact abandoned cache references, preserving scientific files.",
    )
    repair.add_argument("--on", default="local")
    _cluster_selector(repair)
    repair.add_argument("--retire-job", action="append", default=[])
    repair.add_argument("--release-build", action="append", default=[])
    repair.add_argument("--apply", action="store_true")
    repair.add_argument("--json", action="store_true")

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
            "legacy claim configuration; automatic use is refused without a scoped ownership "
            "receipt. Prefer a provider-owned wrapper or manually established reservation."
        ),
    )
    add.add_argument(
        "--gpu-release-command",
        nargs="+",
        help="legacy paired release configuration; unscoped external release is never executed",
    )
    add.add_argument(
        "--gpu-visibility-command",
        nargs="+",
        help=(
            "argv that prints the currently granted CUDA tokens; required for dynamic grant "
            "shrink/growth. Exact 'gpu exec' derives 'gpu env --shell'; other launchers need "
            "an explicit command. CSV or explicit CUDA_VISIBLE_DEVICES assignment is accepted."
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
    recovery = job_commands.add_parser(
        "recovery", help="Saved recovery links and paged audit, without provider access."
    )
    recovery.add_argument("job_id")
    recovery.add_argument("--offset", type=int, default=0)
    recovery.add_argument("--limit", type=int, default=100)
    recovery.add_argument("--json", action="store_true")
    dependencies = job_commands.add_parser(
        "recovery-dependencies",
        help="Inspect retained source bytes and preview exact artifact repair; never re-download.",
    )
    dependencies.add_argument("job_id")
    dependencies.add_argument("--apply", action="store_true")
    dependencies.add_argument("--limit", type=int, default=100)
    dependencies.add_argument("--json", action="store_true")
    retry = job_commands.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--dry-run", action="store_true")
    retry.add_argument("--accept-code-change", action="store_true")
    retry.add_argument("--config", type=Path, help="Verify and use a relocated recovery YAML.")
    reconcile = job_commands.add_parser("reconcile")
    reconcile.add_argument("--cluster")
    reconcile.add_argument("--all", action="store_true")
    clear = job_commands.add_parser("clear")
    clear.add_argument("--apply", action="store_true")
    clear.add_argument("--json", action="store_true")

    for operation in ("show", "logs", "cancel", "retry", "delete"):
        description = (
            "Recover a failed/interrupted Work or Study with native retry; preserve completed Runs."
            if operation == "retry"
            else f"Work-level {operation} operation."
        )
        item = commands.add_parser(operation, help=description)
        item.add_argument("selector")
        item.add_argument("--clusters", type=Path)
        item.add_argument("--apply", action="store_true")
        item.add_argument("--tail", type=int)
        item.add_argument("--follow", action="store_true")
        item.add_argument("--dry-run", action="store_true")
        item.add_argument("--json", action="store_true")
        if operation == "retry":
            item.add_argument(
                "--config", type=Path, help="Verify and use a relocated recovery YAML."
            )
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
        "--ask-password",
        action="store_true",
        help="Prompt privately for the destination password; send it to the source "
        "only in memory. No credential is saved.",
    )
    replicate.add_argument(
        "--route",
        choices=("auto", "direct", "relay"),
        default="auto",
        help="auto probes push then pull; direct forbids PC relay; "
        "relay explicitly accepts PC/network dependency.",
    )
    replicate.add_argument("--json", action="store_true")
    trust = dataset_commands.add_parser(
        "trust-ssh",
        help="Review/explicitly trust first-use SSH host keys on an initiating cluster.",
    )
    trust.add_argument(
        "--source", required=True, help="Cluster initiating SSH (trust is stored here)."
    )
    trust.add_argument(
        "--destination", required=True, help="Peer cluster whose host keys are reviewed."
    )
    trust.add_argument(
        "--fingerprint",
        action="append",
        default=[],
        help="Verified SHA256 fingerprint; repeat for every offered key.",
    )
    trust.add_argument(
        "--known-hosts-file",
        help="Exact initiating-host file reported by preview; required for apply.",
    )
    trust.add_argument("--apply", action="store_true")
    trust.add_argument("--json", action="store_true")
    transfers = dataset_commands.add_parser("transfers", help="List durable Dataset transfer Jobs.")
    transfers.add_argument("dataset", nargs="?")
    transfers.add_argument("--json", action="store_true")
    transfer_status = dataset_commands.add_parser(
        "transfer-status", help="Reconnect to a Dataset transfer Job."
    )
    transfer_status.add_argument("job_id")
    transfer_status.add_argument("--json", action="store_true")

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
    result_table = result_commands.add_parser(
        "table", help="Export a bounded page of scalar Run evidence, not artifact bytes."
    )
    result_table.add_argument("selector")
    result_table.add_argument(
        "--metric", action="append", default=[], help="Exact metric; repeatable."
    )
    result_table.add_argument("--run", dest="run_id", action="append", default=[])
    result_table.add_argument("--trial", type=int, action="append", default=[])
    result_table.add_argument(
        "--columns", help="Comma-separated optional columns; provenance is retained."
    )
    result_table.add_argument("--attempts", choices=("latest", "all"), default="latest")
    result_table.add_argument("--offset", type=int, default=0)
    result_table.add_argument("--limit", type=int, default=100)
    result_table.add_argument(
        "--fingerprint", help="Refuse if evidence changed since the previous page."
    )
    result_table.add_argument("--format", choices=("json", "csv"), default="json")
    result_table.add_argument("--output", type=Path, help="Create a new file outside Run evidence.")
    result_table.add_argument("--root", type=Path)
    result_table.add_argument("--json", action="store_true")
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
    commands.metavar = (
        "{" + ",".join(name for name in commands.choices if not name.startswith("_")) + "}"
    )
    commands._choices_actions = [
        item for item in commands._choices_actions if item.help != argparse.SUPPRESS
    ]
    return parser
