"""Read-only task-oriented inspection/diagnosis/watch routes over native services."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.InspectionService import InspectionService


def render_inspection(value: dict[str, Any]) -> str:
    entity = value.get("entity", value)
    lines = [
        f"{entity.get('name') or entity.get('job_id') or entity.get('execution_id')} · "
        f"{entity.get('kind')} · {entity.get('state', entity.get('status', 'unknown'))}"
    ]
    for key in ("work_id", "execution_id", "job_id", "cluster", "progress", "states"):
        if key in entity:
            lines.append(f"{key}: {entity[key]}")
    operational = (value.get("job") or entity).get("operational", {})
    if operational:
        lines.append(
            f"Observation: {operational.get('health')} · {operational.get('freshness')} · "
            f"age {operational.get('observation_age_seconds')}s"
        )
    if value.get("diagnostic"):
        from lambdaforge.diagnostics import DiagnosticRenderer, ErrorDiagnostic

        lines.append(DiagnosticRenderer().human(ErrorDiagnostic.from_dict(value["diagnostic"])))
    if value.get("recovery"):
        lines.append(value["recovery"]["eligibility"])
        if value["recovery"]["preview_command"]:
            lines.append("Inspect recovery: " + value["recovery"]["preview_command"])
    return "\n".join(lines)


def run_inspection_command(arguments: argparse.Namespace) -> int:
    if arguments.command == "watch":
        if not 1 <= arguments.interval <= 60:
            raise ValueError("Watch interval must be 1..60 seconds.")
        if arguments.count is not None and arguments.count < 1:
            raise ValueError("Watch count must be positive.")
    service = InspectionService(ClusterCatalog.load(arguments.clusters))
    if arguments.command == "diagnose":
        value = service.diagnose(arguments.selector, kind=arguments.kind, refresh=arguments.refresh)
    else:
        value = service.inspect(arguments.selector, kind=arguments.kind, refresh=arguments.refresh)
    encoded = _bounded_json(value)
    if arguments.command != "watch":
        print(encoded if arguments.json else render_inspection(value))
        return 0
    delay = arguments.interval
    previous = None
    number = 0
    try:
        while True:
            encoded = _bounded_json(value)

            def stable(item: Any) -> Any:
                if isinstance(item, dict):
                    return {
                        key: stable(child)
                        for key, child in item.items()
                        if key not in {"observation_age_seconds", "confirmation_age_seconds"}
                    }
                if isinstance(item, list):
                    return [stable(child) for child in item]
                return item

            signature = json.dumps(stable(value), sort_keys=True, allow_nan=False)
            changed = signature != previous
            if changed:
                print(encoded if arguments.json else render_inspection(value), flush=True)
            number += 1
            if arguments.count is not None and number >= arguments.count:
                break
            delay = arguments.interval if changed else min(60, max(arguments.interval, delay * 1.5))
            previous = signature
            time.sleep(delay)
            value = service.inspect(
                arguments.selector, kind=arguments.kind, refresh=arguments.refresh
            )
    except KeyboardInterrupt:
        pass  # Leaving an observer never cancels science or stops a native monitor.
    return 0


def _bounded_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(encoded.encode()) > 512 * 1024:
        raise ValueError("Inspection response exceeds 512 KiB; use an exact smaller identity.")
    return encoded
