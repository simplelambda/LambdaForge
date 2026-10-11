"""Native CLI routes for operational observations, monitor and incidents."""

from __future__ import annotations

import argparse
import json
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.MonitorService import MonitorService
from lambdaforge.controlplane.OperationalHealth import OperationalHealth
from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents


def run_operational_command(arguments: argparse.Namespace) -> int:
    jobs = JobService(ClusterCatalog.load(arguments.clusters))
    monitor = MonitorService(jobs)
    payload: Any
    failed = False
    if arguments.command == "health":
        records = (
            (
                jobs.get(
                    jobs.resolve_selector(arguments.selector),
                    refresh=arguments.refresh,
                    include_study=False,
                ),
            )
            if arguments.selector
            else jobs.list(refresh=False)
        )
        if arguments.refresh and not arguments.selector:
            from lambdaforge.controlplane.OverviewService import OverviewService

            records = OverviewService._jobs(jobs)
        payload = [OperationalHealth.describe(record) for record in records]
        failed = any(
            item["health"] in {"degraded", "unreachable", "lost_confirmed", "stale", "unknown"}
            for item in payload
        )
        if not arguments.json:
            for item in payload:
                age = item["observation_age_seconds"]
                print(
                    f"{item['job_id']} · {item['execution_state']} · {item['health']} · "
                    f"{item['reason']} · last observation {age if age is not None else 'unknown'}s"
                )
    elif arguments.command == "incidents":
        payload = list(OperationalIncidents(jobs.store).list(job_id=arguments.job))
        if not arguments.json:
            for item in payload:
                print(
                    f"{item['incident_id']} · {item['severity']} · {item['status']} · "
                    f"{item['job_id']} · {item['code']}"
                )
    else:
        operation = arguments.monitor_command
        if operation == "start":
            payload = monitor.start(clusters=arguments.clusters)
        elif operation == "stop":
            payload = monitor.stop()
        elif operation in {"status", "diagnose"}:
            payload = monitor.status()
            failed = payload["health"] != "healthy"
            if operation == "diagnose":
                payload = {
                    **payload,
                    "configuration": monitor.config(),
                    "incidents": len(monitor.incidents.list()),
                    "log": str(monitor.root / "monitor.log"),
                    "science_mutated": False,
                }
        elif operation == "config":
            settings = (
                json.loads(arguments.file.read_text(encoding="utf-8")) if arguments.file else None
            )
            if settings is not None and not isinstance(settings, dict):
                raise ValueError("Monitor configuration must be a JSON object.")
            payload = monitor.config(settings)
        elif operation == "install":
            payload = monitor.install(clusters=arguments.clusters, apply=arguments.apply)
        elif operation == "once":
            payload = monitor.tick()
        elif operation == "test-alert":
            payload = monitor.test_alert()
        elif operation == "run":
            monitor.run()
            return 0
        else:
            raise ValueError(f"Unsupported monitor operation {operation!r}.")
        if not arguments.json:
            print(json.dumps(payload, indent=2))
    if arguments.json:
        print(json.dumps(payload, indent=2))
    return 2 if failed else 0
