"""Shared CLI rendering, error and small value-manipulation helpers."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any


def guarded(action: Callable[[], int]) -> int:
    """Render expected command failures consistently without swallowing interrupts."""
    try:
        return action()
    except Exception as error:
        print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
        return 1


def keyring_reference(name: str, user: str | None, host: str | None) -> str:
    """Create a stable non-secret keyring identifier for one endpoint."""
    endpoint = f"{user or 'default'}@{host or 'local'}"
    return f"keyring:cluster/{name}/{endpoint}"


def age(created: str) -> str:
    """Render an ISO timestamp as a compact non-negative age."""
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


def job_resources(resources: dict[str, Any] | object) -> str:
    """Render the portable non-empty job resource fields."""
    if not isinstance(resources, dict):
        return "unknown"
    selected = [
        f"{key}={resources[key]}"
        for key in ("cpus", "memory", "gpus", "gpu_memory", "time")
        if resources.get(key) not in (None, 0, "")
    ]
    return ",".join(selected) or "default"


def print_resources(payload: object) -> None:
    """Render one or more resource snapshots for terminal users."""
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


def print_storage(payload: list[dict[str, Any]]) -> None:
    """Render categorized internal-storage usage reports."""
    for report in payload:
        print(f"{report['cluster']} LambdaForge storage")
        if not report["online"]:
            print(f"  offline: {report.get('error')}")
            continue
        for name, usage in report["categories"].items():
            print(f"  {name:<18} {usage['bytes']:>12} bytes {usage['files']:>8} files")


def set_dotted(value: dict[str, Any], path: str, replacement: Any) -> None:
    """Set one validated dotted cluster-profile field."""
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


def delete_dotted(value: dict[str, Any], path: str) -> None:
    """Delete one existing dotted cluster-profile field."""
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
