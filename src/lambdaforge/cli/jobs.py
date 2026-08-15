"""Durable job lifecycle and group CLI commands."""

from __future__ import annotations

import argparse
import json
import time

from lambdaforge.cli.common import age, job_resources
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobGroupStore import JobGroupStore
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.diagnostics import LambdaForgeError, job_failure_diagnostic


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
    """Stream scheduler logs by reconnecting through the persistent job record."""
    previous = ""
    while True:
        current = jobs.logs(job_id, tail=tail)
        delta = current[len(previous) :] if current.startswith(previous) else current
        if delta:
            print(delta, end="" if delta.endswith("\n") else "\n", flush=True)
        previous = current
        record = jobs.get(job_id)
        if record.state.terminal:
            if record.state in {JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED}:
                raise LambdaForgeError(job_failure_diagnostic(record, current))
            return 0
        time.sleep(2.0)
