"""Shared CLI rendering, error and small value-manipulation helpers."""

from __future__ import annotations

import shlex
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.diagnostics import (
    DiagnosticClassifier,
    DiagnosticContext,
    DiagnosticRecorder,
    DiagnosticRenderer,
    ErrorCategory,
    ErrorDiagnostic,
    LambdaForgeError,
    diagnostic,
)

_CONTEXT: ContextVar[DiagnosticContext | None] = ContextVar(
    "lambdaforge_cli_diagnostic_context",
    default=None,
)


@contextmanager
def diagnostic_context(context: DiagnosticContext) -> Iterator[None]:
    """Scope recursive CLI dispatch to one invocation without global mutable state."""
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_diagnostic_context() -> DiagnosticContext:
    """Expose the immutable invocation facts to command adapters."""
    return _CONTEXT.get() or DiagnosticContext((), "LambdaForge command")


def guarded(action: Callable[[], int]) -> int:
    """Render expected command failures consistently without swallowing interrupts."""
    try:
        return action()
    except Exception as error:
        return report_error(error)


def report_error(error: BaseException) -> int:
    """Classify, persist and render one boundary failure with consistent exit semantics."""
    context = current_diagnostic_context()
    value = DiagnosticClassifier().classify(error, context)
    record = DiagnosticRecorder().record(value, error, context)
    if record is not None:
        value = value.with_diagnostic_path(str(record))
    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    renderer = DiagnosticRenderer()
    rendered = (
        renderer.json(
            value,
            debug=context.debug,
            exception_type=type(error).__name__,
            traceback_text=traceback_text,
        )
        if context.json_output
        else renderer.human(
            value,
            debug=context.debug,
            exception_type=type(error).__name__,
            traceback_text=traceback_text,
        )
    )
    print(rendered, end="", file=sys.stdout if context.json_output else sys.stderr)
    return value.exit_code


def report_diagnostic(value: ErrorDiagnostic) -> int:
    """Render a known non-exception outcome through the same persistent boundary."""
    return report_error(LambdaForgeError(value))


def validation_diagnostic(
    source: str | Path,
    errors: Sequence[str],
    *,
    kind: str,
) -> ErrorDiagnostic:
    """Turn an existing validation report into actionable human/JSON semantics."""
    context = current_diagnostic_context()
    path = str(Path(source).expanduser().resolve())
    return diagnostic(
        ErrorCategory.VALIDATION,
        f"Invalid {kind} configuration.",
        str(errors[0]) if errors else "The configuration is invalid.",
        reason="The file was parsed, but it does not satisfy the selected LambdaForge contract.",
        impact=("No job was submitted and no scientific computation started.",),
        fixes=("Correct the reported field/value and validate the file again.",),
        commands=(("Validate again", f"lf validate {shlex.quote(path)}"),),
        context={"file": path, "kind": kind, "errors": list(errors)},
        operation=context.operation,
    )


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
