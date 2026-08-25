"""Durable job lifecycle and group CLI commands."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone

from lambdaforge.cli.common import age, current_diagnostic_context, job_resources
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobGroupStore import JobGroupStore
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.diagnostics import LambdaForgeError, job_failure_diagnostic
from lambdaforge.work.failure import render_scientific_failures


def run_job_command(arguments: argparse.Namespace) -> int:
    """Execute one parsed ``jobs`` action through ``JobService``."""
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
                resources = job_resources(job_record.resources)
                print(
                    f"{job_record.job_id:<32} {name:<17.17} "
                    f"{job_record.job_type:<14.14} {job_record.state.value:<10} "
                    f"{job_record.cluster:<12.12} "
                    f"{age(job_record.created_at_utc):<9} "
                    f"{resources}"
                )
        return 0
    selected_job_id: str = (
        jobs.resolve_selector(arguments.job_id) if hasattr(arguments, "job_id") else ""
    )
    if arguments.job_command in {"status", "show"}:
        job_record = jobs.get(selected_job_id)
        if arguments.job_command == "show":
            scientific = jobs.scientific_result(selected_job_id)
            payload = {
                **job_record.to_dict(),
                "scientific_result_path": scientific.get("path") if scientific else None,
                "failure": scientific.get("failure") if scientific else None,
                "failures": scientific.get("failures", []) if scientific else [],
            }
            if arguments.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"{job_record.job_id}: {job_record.state.value} on {job_record.cluster}")
                failure = payload["failure"]
                if isinstance(failure, Mapping):
                    print(
                        f"Scientific failure: {failure.get('type')}: {failure.get('message')}\n"
                        f"Phase: {failure.get('phase')}\n"
                        f"Location: {failure.get('location')}\n"
                        f"Persisted result: {failure.get('result_path')}"
                    )
                    context = current_diagnostic_context()
                    if (context.debug or context.verbose) and failure.get("traceback"):
                        print(f"\n{failure['traceback']}", end="")
            return 0
        if job_record.state in {JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED}:
            try:
                job_logs = jobs.logs(selected_job_id, tail=300)
            except Exception:
                job_logs = job_record.stdout + job_record.stderr
            raise LambdaForgeError(job_failure_diagnostic(job_record, job_logs))
        print(
            json.dumps(job_record.to_dict(), indent=2)
            if arguments.json
            else (f"{job_record.job_id}: {job_record.state.value} on {job_record.cluster}")
        )
        return 0
    if arguments.job_command == "logs":
        if arguments.follow:
            return follow_job_logs(jobs, selected_job_id, tail=arguments.tail)
        context = current_diagnostic_context()
        report = jobs.log_report(
            selected_job_id,
            tail=arguments.tail,
            include_traceback=context.debug or context.verbose,
        )
        print(json.dumps(report, indent=2) if arguments.json else report["text"], end="")
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
        result = WorkService(jobs.catalog, jobs=jobs).delete_job(selected_job_id, apply=True)
        print(json.dumps(result, indent=2))
        return 0
    if arguments.job_command == "clear":
        result = WorkService(jobs.catalog, jobs=jobs).clear_history(apply=arguments.apply)
        print(json.dumps(result, indent=2))
        return 0
    if arguments.job_command == "reconcile":
        reconciled = jobs.reconcile(cluster=arguments.cluster, all_clusters=arguments.all)
        print(json.dumps([value.to_dict() for value in reconciled], indent=2))
        return 0
    if arguments.job_command in {"group", "groups"}:
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


def follow_job_logs(jobs: JobService, job_id: str, *, tail: int | None) -> int:
    """Stream lifecycle and science independently, including quiet-job health observations."""
    previous_scientific = ""
    seen_events = 0
    last_activity = time.monotonic()
    last_status = last_activity
    last_heartbeat: object = None
    while True:
        record = jobs.get(job_id)
        events = jobs.events(job_id)
        for event in events[seen_events:]:
            print(
                f"[{event.get('timestamp_utc', '-')}] "
                f"[{event.get('phase') or event.get('source') or 'lifecycle'}] "
                f"{event.get('state', 'unknown')}: {event.get('message', '')}",
                flush=True,
            )
        if len(events) > seen_events:
            seen_events = len(events)
            last_activity = time.monotonic()
        current = jobs.scientific_logs(job_id, tail=tail)
        delta = (
            current[len(previous_scientific) :]
            if current.startswith(previous_scientific)
            else current
        )
        if delta:
            print(delta, end="" if delta.endswith("\n") else "\n", flush=True)
            last_activity = time.monotonic()
        previous_scientific = current
        remote = record.metadata.get("remote_state", {})
        remote = remote if isinstance(remote, Mapping) else {}
        heartbeat = remote.get("heartbeat_at_utc")
        now = time.monotonic()
        if heartbeat and heartbeat != last_heartbeat:
            print(
                f"[{heartbeat}] [runtime] {record.state.value}: supervisor heartbeat observed; "
                "scientific output may remain quiet.",
                flush=True,
            )
            last_heartbeat = heartbeat
            last_status = now
        elif now - last_status >= 30 and now - last_activity >= 30:
            silent = int(now - last_activity)
            observed = datetime.now(timezone.utc).isoformat()
            provider = (
                "provider status remains unavailable"
                if record.state is JobState.UNKNOWN
                else "provider remains reachable"
            )
            print(
                f"[{observed}] [scheduler] {record.state.value}: {provider}; "
                f"no new lifecycle/scientific output for {silent}s.",
                flush=True,
            )
            last_status = now
        if record.state.terminal:
            if record.state in {JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED}:
                scientific = jobs.scientific_result(job_id)
                failures = scientific.get("failures", ()) if scientific is not None else ()
                context = current_diagnostic_context()
                section = render_scientific_failures(
                    tuple(value for value in failures if isinstance(value, Mapping)),
                    existing_output=current,
                    include_traceback=context.debug or context.verbose,
                )
                if section:
                    print(section, flush=True)
                raise LambdaForgeError(
                    job_failure_diagnostic(
                        record,
                        f"{current}\n{section}" if section else current,
                    )
                )
            return 0
        time.sleep(2.0)
