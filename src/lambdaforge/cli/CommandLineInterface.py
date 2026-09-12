"""Work-centric LambdaForge command-line boundary."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from lambdaforge.cli.clusters import run_cluster_command
from lambdaforge.cli.common import (
    current_diagnostic_context,
    diagnostic_context,
    print_resources,
    report_error,
)
from lambdaforge.cli.DatasetCommands import DatasetCommands
from lambdaforge.cli.jobs import follow_job_logs, run_job_command
from lambdaforge.cli.parser import build_parser
from lambdaforge.cli.scaffold import initialize
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.Doctor import Doctor
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.diagnostics import DiagnosticContext
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner


class CommandLineInterface:
    """Parse commands and keep Work as the sole scientific execution route."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        supplied = list(argv) if argv is not None else sys.argv[1:]
        supplied = cls._normalize_help(supplied)
        context = DiagnosticContext.from_argv(supplied)
        parent = current_diagnostic_context()
        if parent.arguments:
            context = replace(
                context,
                json_output=context.json_output or parent.json_output,
                debug=context.debug or parent.debug,
                verbose=context.verbose or parent.verbose,
            )
        parsed = [item for item in supplied if item not in {"--debug", "--verbose", "--json"}]
        with diagnostic_context(context):
            try:
                return cls._dispatch(parsed, json_output=context.json_output)
            except SystemExit as error:
                # argparse implements successful help/version rendering by raising
                # SystemExit.  The console-script wrapper handles that, but direct API
                # callers should receive the same ordinary integer result as every other
                # LambdaForge CLI operation.
                if error.code in {None, 0}:
                    return 0
                raise
            except KeyboardInterrupt:
                print(
                    "Operation cancelled; submitted jobs continue until explicitly cancelled.",
                    file=sys.stderr,
                )
                return 130
            except Exception as error:
                return report_error(error)

    @staticmethod
    def _normalize_help(argv: Sequence[str]) -> list[str]:
        """Accept ``help`` in the natural command positions supported by users."""
        values = list(argv)
        if "help" not in values:
            return values
        return [value for value in values if value != "help"] + ["--help"]

    @classmethod
    def _dispatch(cls, argv: Sequence[str], *, json_output: bool) -> int:
        parser = build_parser()
        arguments = parser.parse_args(argv)
        arguments.json = bool(getattr(arguments, "json", False) or json_output)
        arguments.verbose = bool(getattr(arguments, "verbose", False))
        if arguments.command is None:
            if not arguments.json and sys.stdin.isatty() and sys.stdout.isatty():
                from lambdaforge.tui.App import run_console

                return run_console()
            parser.print_help()
            return 0
        if arguments.command == "init":
            return initialize(
                arguments.directory, force=arguments.force, template=arguments.template
            )
        if arguments.command == "project":
            project = ProjectContext.discover()
            cls._render(
                project.to_dict(),
                f"Project: {project.project_id}\nRoot: {project.root}",
                arguments.json,
            )
            return 0
        if arguments.command == "validate":
            validation = WorkConfig.validate_file(arguments.config)
            cls._render(validation.to_dict(), validation.summary(), arguments.json)
            return 0 if validation.valid else 2
        if arguments.command == "explain":
            config = WorkConfig.from_yaml(arguments.config)
            payload = config.explanation()
            cls._render(payload, cls._explanation(payload), arguments.json)
            return 0
        if arguments.command == "run":
            return cls._run(arguments)
        if arguments.command == "clusters":
            return run_cluster_command(arguments)
        if arguments.command == "jobs":
            return run_job_command(arguments)
        if arguments.command == "datasets":
            arguments.default_cluster_source = None
            return DatasetCommands.run(arguments)
        if arguments.command == "doctor":
            doctor_report = Doctor(ClusterCatalog.load(arguments.clusters)).check(
                arguments.on, config_path=arguments.config
            )
            cls._render(doctor_report.to_dict(), doctor_report.summary(), arguments.json)
            return doctor_report.exit_code
        if arguments.command == "overview":
            return cls._overview(arguments)
        if arguments.command == "resources":
            service = ResourceService(ClusterCatalog.load(arguments.clusters))
            if arguments.processes:
                resource_payload: Any = service.processes(arguments.on or "local")
            elif arguments.all:
                resource_payload = [value.to_dict() for value in service.all()]
            else:
                resource_payload = service.get(arguments.on or "local").to_dict()
            if arguments.json:
                print(json.dumps(resource_payload, indent=2))
            else:
                print_resources(resource_payload)
            return 0
        if arguments.command == "clean":
            plan = StorageService(ClusterCatalog.load(arguments.clusters)).gc(
                arguments.on, apply=arguments.apply
            )
            cls._render(plan.to_dict(), cls._clean_summary(plan.to_dict()), arguments.json)
            return 0
        if arguments.command in {"show", "logs", "cancel", "retry", "delete"}:
            return cls._work_operation(arguments)
        if arguments.command == "results":
            return cls._results(arguments)
        raise ValueError(f"Unknown command: {arguments.command}")

    @staticmethod
    def _run(arguments: Any) -> int:
        # Scheduler children invoke the same public CLI.  They execute inline; an interactive
        # top-level invocation is handed to the durable submission worker on every target,
        # including local, so terminal latency does not depend on scientific execution.
        supervised = os.environ.get("LAMBDAFORGE_EXECUTION_MODE") == "worker" or (
            bool(os.environ.get("LAMBDAFORGE_JOB_ID"))
            and os.environ.get("LAMBDAFORGE_BUNDLE") == "1"
        )
        if supervised or (arguments.on == "local" and arguments.dry_run):
            config = WorkConfig.from_yaml(arguments.config)
            outcome = WorkRunner().run(
                config,
                dry_run=arguments.dry_run,
                rerun=arguments.rerun,
                restart=arguments.restart,
            )
            payload = outcome.to_dict()
            if arguments.json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"{payload['name']}: {payload.get('status', 'planned')} "
                    f"({payload['execution_id']})"
                )
            return 0 if payload.get("status", "succeeded") == "succeeded" else 4
        run_arguments = tuple(
            flag
            for enabled, flag in (
                (arguments.rerun, "--rerun"),
                (arguments.restart, "--restart"),
            )
            if enabled
        )
        if arguments.dry_run or arguments.wait_for_submit:
            handle, bundle = ControlPlane(ClusterCatalog.load(arguments.clusters)).submit(
                arguments.config,
                cluster=arguments.on,
                dry_run=arguments.dry_run,
                run_arguments=run_arguments,
                allow_duplicate=arguments.allow_duplicate,
            )
            payload = {"job": handle.to_dict(), "bundle": bundle.to_dict()}
        else:
            handle = SubmissionService(ClusterCatalog.load(arguments.clusters)).enqueue(
                arguments.config,
                cluster=arguments.on,
                run_arguments=run_arguments,
                allow_duplicate=arguments.allow_duplicate,
            )
            payload = handle.to_dict()
        print(
            json.dumps(payload, indent=2)
            if arguments.json
            else f"Submitted {payload.get('job_id', payload)}"
        )
        return 0

    @staticmethod
    def _overview(arguments: Any) -> int:
        catalog = ClusterCatalog.load(arguments.clusters)
        service = OverviewService(catalog)
        payload = service.snapshot()
        if arguments.json:
            print(json.dumps(payload, indent=2), flush=True)
        else:
            works = payload.get("work", {}).get("items", [])
            print("WORK  TARGET  STATE  PROGRESS")
            for work in works:
                print(
                    f"{work.get('name', '-'):<24} {work.get('cluster', '-'):<12} "
                    f"{work.get('state', '-'):<10} {work.get('progress', '-')}"
                )
        return 0

    @staticmethod
    def _work_operation(arguments: Any) -> int:
        catalog = ClusterCatalog.load(arguments.clusters)
        works = WorkService(catalog)
        local = ResultStore()
        if arguments.command in {"show", "delete"}:
            try:
                if arguments.command == "show" and arguments.study_run:
                    raise KeyError(arguments.selector)
                payload = (
                    local.select(arguments.selector)
                    if arguments.command == "show"
                    else local.delete(arguments.selector, apply=arguments.apply)
                )
                payload.pop("_manifest_path", None)
            except KeyError:
                if arguments.command == "show":
                    remote_work = works.show(arguments.selector)
                    jobs = JobService(catalog)
                    if arguments.study_run:
                        payload = jobs.study_run(
                            remote_work.primary_job_id,
                            arguments.study_run,
                            tail=arguments.tail,
                            curve_points=arguments.curve_points,
                        )
                        print(
                            json.dumps(payload, indent=2)
                            if arguments.json
                            else CommandLineInterface._render_study_run(payload)
                        )
                        return 0
                    scientific = jobs.scientific_result(remote_work.primary_job_id)
                    payload = {
                        **remote_work.to_dict(),
                        "study": jobs.study(remote_work.primary_job_id),
                        "scientific_result_path": (
                            scientific.get("path") if scientific is not None else None
                        ),
                        "failure": (scientific.get("failure") if scientific is not None else None),
                        "failures": (
                            scientific.get("failures", []) if scientific is not None else []
                        ),
                    }
                else:
                    payload = works.delete(arguments.selector, apply=arguments.apply)
        elif arguments.command in {"logs", "retry"}:
            try:
                local_record = local.select(arguments.selector)
            except KeyError:
                local_record = None
            if local_record is not None:
                if arguments.command == "logs":
                    if arguments.study_run:
                        if arguments.follow:
                            raise ValueError(
                                "Use the Research Console for live per-Run logs; machine clients "
                                "can poll "
                                "'lf show WORK --run KEY --json'."
                            )
                        selected = works.show(arguments.selector)
                        detail = JobService(catalog).study_run(
                            selected.primary_job_id,
                            arguments.study_run,
                            tail=arguments.tail,
                            curve_points=arguments.curve_points,
                        )
                        if arguments.json:
                            print(json.dumps(detail, indent=2))
                        else:
                            log = str(detail["log"])
                            print(log, end="" if log.endswith("\n") else "\n")
                        return 0
                    context = current_diagnostic_context()
                    report = local.log_report(
                        arguments.selector,
                        tail=arguments.tail,
                        include_traceback=context.debug or context.verbose,
                    )
                    if arguments.json:
                        print(json.dumps(report, indent=2))
                    else:
                        print(report["text"], end="")
                    return 0
                if local_record.get("status") == "succeeded":
                    raise ValueError(
                        "A successful local Work is not retryable; use 'lf run CONFIG --rerun' "
                        "for a deliberate new Execution."
                    )
                outcome = WorkRunner().run(
                    local.configuration(arguments.selector),
                    dry_run=arguments.dry_run,
                )
                payload = outcome.to_dict()
            else:
                selected = works.show(arguments.selector)
                job_id = selected.primary_job_id
                jobs = JobService(catalog)
                if arguments.command == "logs":
                    if arguments.study_run:
                        if arguments.follow:
                            raise ValueError(
                                "Use the Research Console for live per-Run logs; machine clients "
                                "can poll "
                                "'lf show WORK --run KEY --json'."
                            )
                        detail = jobs.study_run(
                            job_id,
                            arguments.study_run,
                            tail=arguments.tail,
                            curve_points=arguments.curve_points,
                        )
                        if arguments.json:
                            print(json.dumps(detail, indent=2))
                        else:
                            log = str(detail["log"])
                            print(log, end="" if log.endswith("\n") else "\n")
                        return 0
                    if arguments.follow:
                        return follow_job_logs(jobs, job_id, tail=arguments.tail)
                    context = current_diagnostic_context()
                    report = jobs.log_report(
                        job_id,
                        tail=arguments.tail,
                        include_traceback=context.debug or context.verbose,
                    )
                    if arguments.json:
                        print(json.dumps(report, indent=2))
                    else:
                        print(report["text"], end="")
                    return 0
                payload = jobs.retry(job_id, dry_run=arguments.dry_run).to_dict()
        else:
            payload = works.cancel(arguments.selector)
        print(json.dumps(payload, indent=2) if arguments.json else json.dumps(payload, indent=2))
        return 0

    @staticmethod
    def _render_study_run(detail: Mapping[str, Any]) -> str:
        """Render one Run summary with its durable artifact locations."""
        lines = [
            f"LambdaForge Run {detail.get('key', 'unknown')}",
            f"State: {detail.get('state', 'unknown')}",
            f"Trial: {detail.get('trial', '-')}  Seed: {detail.get('seed', 'none')}  "
            f"GPU: {detail.get('gpu_index', '-')}",
        ]
        parameters = detail.get("parameters")
        if isinstance(parameters, Mapping) and parameters:
            lines.extend(("", "PARAMETERS"))
            lines.extend(f"  {name}: {value}" for name, value in sorted(parameters.items()))
        lines.extend(("", "ARTIFACTS"))
        artifacts = detail.get("artifacts", ())
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, str | bytes):
            visible = [value for value in artifacts if isinstance(value, Mapping)]
        else:
            visible = []
        if not visible:
            lines.append("  No finalized managed artifacts are recorded for this Run.")
        for artifact in visible:
            size = artifact.get("size_bytes")
            size_text = f"{size} bytes" if isinstance(size, int) else "size unavailable"
            media_type = artifact.get("media_type") or "unspecified media type"
            lines.extend(
                (
                    f"  {artifact.get('name', 'artifact')} "
                    f"[{artifact.get('role', 'artifact')}; {media_type}; {size_text}]",
                    f"    Path: {artifact.get('path', 'unavailable')}",
                )
            )
            published = artifact.get("published_path")
            managed = artifact.get("managed_path")
            if published and managed and published != managed:
                lines.append(f"    Managed source: {managed}")
        paths = detail.get("paths")
        if isinstance(paths, Mapping):
            lines.extend(("", "EVIDENCE"))
            lines.append(f"  Result: {paths.get('result') or 'unavailable'}")
            lines.append(f"  Log: {paths.get('log') or 'unavailable'}")
        lines.extend(("", "Use --json for curves, metrics, checksums and complete metadata."))
        return "\n".join(lines)

    @staticmethod
    def _results(arguments: Any) -> int:
        store = ResultStore(arguments.root)
        if arguments.result_command == "analyze":
            payload = store.analysis(arguments.selector, recompute=arguments.recompute)
            if arguments.json:
                print(json.dumps(payload, indent=2))
            else:
                winner = payload.get("winner", {}).get("screening_winner")
                print(
                    f"{payload['source']['status'].title()} Study Analysis · "
                    f"{payload['summary']['complete_candidate_count']} complete candidates"
                )
                if winner:
                    print(
                        f"Screening winner: trial {winner['trial']} "
                        f"objective={winner['mean']:.6g} n={winner['n']}"
                    )
                print(f"Findings: {len(payload.get('findings', []))}")
            return 0
        if arguments.result_command == "report":
            path = store.report(
                arguments.selector,
                arguments.output,
                recompute=arguments.recompute,
            )
            print(path)
            return 0
        if arguments.result_command == "compare":
            print(
                json.dumps(
                    store.compare(
                        tuple(arguments.selectors),
                        metric=arguments.metric,
                        mode=arguments.mode,
                    ),
                    indent=2,
                )
            )
            return 0
        records = [
            {key: value for key, value in record.items() if key != "_manifest_path"}
            for record in store.list()
        ]
        if arguments.result_command == "list":
            result_payload: Any = records
        else:
            selected = [
                item
                for item in records
                if arguments.selector
                in {
                    item.get("name"),
                    item.get("execution_id"),
                    item.get("scientific_fingerprint"),
                }
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Result selector must identify exactly one execution; found {len(selected)}."
                )
            result_payload = selected[0]
            analysis = store.analysis_summary(arguments.selector)
            if analysis is not None:
                result_payload = {**result_payload, "analysis": analysis}
        print(json.dumps(result_payload, indent=2))
        return 0

    @staticmethod
    def _render(payload: Mapping[str, Any], human: str, as_json: bool) -> None:
        print(json.dumps(payload, indent=2) if as_json else human)

    @staticmethod
    def _explanation(payload: Mapping[str, Any]) -> str:
        lines = [f"Work: {payload['name']}", f"Planned Runs: {payload['planned_runs']}"]
        for level_number, level in enumerate(payload["levels"], 1):
            lines.append(f"Level {level_number}:")
            for run in level:
                lines.append(f"  {run['name']}: {run['class']} ({run['runs']} Run(s))")
                for parameter in run["parameters"]:
                    state = (
                        "required" if parameter["required"] else f"default={parameter['default']!r}"
                    )
                    lines.append(f"    {parameter['name']}: {parameter['type']} ({state})")
        return "\n".join(lines)

    @staticmethod
    def _clean_summary(payload: Mapping[str, Any]) -> str:
        verb = "Removed" if payload.get("applied") else "Would remove"
        count = len(payload.get("candidates", []))
        reclaimable = payload.get("reclaimable_bytes", 0)
        return f"{verb} {count} safe storage entries ({reclaimable} bytes)."
