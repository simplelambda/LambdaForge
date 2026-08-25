"""Extract and render bounded scientific failures from Work result envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def scientific_failures(
    result: Mapping[str, Any], *, result_path: str
) -> tuple[dict[str, Any], ...]:
    """Return normalized failed Run views without losing their structured diagnostic."""
    raw_runs = result.get("runs", ())
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes | bytearray):
        return ()
    failures: list[dict[str, Any]] = []
    for run in raw_runs:
        if not isinstance(run, Mapping) or not isinstance(run.get("failure"), Mapping):
            continue
        failure = run["failure"]
        diagnostic = failure.get("diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
        operation = str(diagnostic.get("operation") or "Work.run")
        attempt = run.get("attempt_number")
        location = str(run.get("name") or result.get("name") or "Work")
        if isinstance(attempt, int):
            location = f"{location}, Attempt {attempt}"
        failures.append(
            {
                "type": str(failure.get("type") or "Exception"),
                "message": str(failure.get("message") or "Scientific execution failed."),
                "phase": operation,
                "location": location,
                "traceback": str(failure.get("traceback") or ""),
                "diagnostic": dict(diagnostic),
                "result_path": result_path,
                "run_id": run.get("run_id"),
                "attempt_id": run.get("attempt_id"),
            }
        )
    return tuple(failures)


def render_scientific_failures(
    failures: Sequence[Mapping[str, Any]],
    *,
    existing_output: str,
    include_traceback: bool,
) -> str:
    """Render only failure evidence that is not already present in captured output."""
    sections: list[str] = []
    for failure in failures:
        failure_type = str(failure.get("type") or "Exception")
        message = str(failure.get("message") or "Scientific execution failed.")
        traceback = str(failure.get("traceback") or "").strip()
        summary_is_visible = failure_type in existing_output and message in existing_output
        traceback_is_visible = bool(traceback and traceback in existing_output)
        lines: list[str] = []
        if not summary_is_visible:
            lines.extend(
                (
                    f"{failure_type}: {message}",
                    f"Phase: {failure.get('phase') or 'Work.run'}",
                    f"Location: {failure.get('location') or 'Work'}",
                    f"Persisted result: {failure.get('result_path') or '-'}",
                )
            )
        if include_traceback and traceback and not traceback_is_visible:
            if not lines:
                lines.append(f"Persisted result: {failure.get('result_path') or '-'}")
            lines.extend(("", traceback))
        if lines:
            sections.append("\n".join(lines))
    if not sections:
        return ""
    return "== Scientific failure ==\n" + "\n\n".join(sections)


__all__ = ["render_scientific_failures", "scientific_failures"]
