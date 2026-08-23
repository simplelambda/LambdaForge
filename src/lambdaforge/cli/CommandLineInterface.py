"""Work-centric LambdaForge command-line boundary."""

from __future__ import annotations

import json
import os
import sys
import time
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
from lambdaforge.cli.LiveJobMonitor import LiveJobMonitor
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
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner


class CommandLineInterface:
    """Parse commands and keep Work as the sole scientific execution route."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
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
        parsed = [item for item in supplied if item not in {"--debug", "--verbose", "--json"}]
        with diagnostic_context(context):
            try:
                return cls._dispatch(parsed, json_output=context.json_output)
            except KeyboardInterrupt:
                print(
                    "Operation cancelled; submitted jobs continue until explicitly cancelled.",
                    file=sys.stderr,
                )
                return 130
            except Exception as error:
                return report_error(error)

    @classmethod
    def _dispatch(cls, argv: Sequence[str], *, json_output: bool) -> int:
        arguments = build_parser().parse_args(argv)
        arguments.json = bool(getattr(arguments, "json", False) or json_output)
        arguments.verbose = bool(getattr(arguments, "verbose", False))
        if arguments.command == "init":
            return initialize(
                arguments.directory, force=arguments.force, template=arguments.template
            )
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
        if arguments.command in {"overview", "top"}:
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
        config = WorkConfig.from_yaml(arguments.config)
        if arguments.on == "local":
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
        if (
            arguments.command == "top"
            and not arguments.json
            and not arguments.once
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.name == "posix"
        ):
            return LiveJobMonitor(
                service,
                JobService(catalog),
                interval=arguments.interval,
                history_seconds=arguments.history,
            ).run()
        while True:
            payload = service.snapshot()
            if arguments.json:
                print(json.dumps(payload, indent=None if arguments.follow else 2), flush=True)
            else:
                works = payload.get("work", {}).get("items", [])
                print("WORK  TARGET  STATE  PROGRESS")
                for work in works:
                    print(
                        f"{work.get('name', '-'):<24} {work.get('cluster', '-'):<12} "
                        f"{work.get('state', '-'):<10} {work.get('progress', '-')}"
                    )
            if arguments.command != "top" or arguments.once or not arguments.follow:
                return 0
            time.sleep(arguments.interval)

    @staticmethod
    def _work_operation(arguments: Any) -> int:
        catalog = ClusterCatalog.load(arguments.clusters)
        works = WorkService(catalog)
        local = ResultStore()
        if arguments.command in {"show", "delete"}:
            try:
                payload = (
                    local.select(arguments.selector)
                    if arguments.command == "show"
                    else local.delete(arguments.selector, apply=arguments.apply)
                )
                payload.pop("_manifest_path", None)
            except KeyError:
                payload = (
                    works.show(arguments.selector).to_dict()
                    if arguments.command == "show"
                    else works.delete(arguments.selector, apply=arguments.apply)
                )
        elif arguments.command in {"logs", "retry"}:
            try:
                local_record = local.select(arguments.selector)
            except KeyError:
                local_record = None
            if local_record is not None:
                if arguments.command == "logs":
                    print(local.logs(arguments.selector, tail=arguments.tail), end="")
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
                    if arguments.follow:
                        return follow_job_logs(jobs, job_id, tail=arguments.tail)
                    print(jobs.logs(job_id, tail=arguments.tail), end="")
                    return 0
                payload = jobs.retry(job_id, dry_run=arguments.dry_run).to_dict()
        else:
            selected = works.show(arguments.selector)
            payload = JobService(catalog).cancel(selected.primary_job_id).to_dict()
        print(json.dumps(payload, indent=2) if arguments.json else json.dumps(payload, indent=2))
        return 0

    @staticmethod
    def _results(arguments: Any) -> int:
        store = ResultStore(arguments.root)
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
            payload: Any = records
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
                    "Result selector must identify exactly one execution; "
                    f"found {len(selected)}."
                )
            payload = selected[0]
        print(json.dumps(payload, indent=2))
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
        return f"{verb} {count} cache entries ({reclaimable} bytes)."
