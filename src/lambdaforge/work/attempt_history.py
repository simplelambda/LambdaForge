"""Bounded Attempt display history with cumulative, idempotent physical accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TERMINAL = {"succeeded", "failed", "pruned", "paused", "aborted"}


def attempt_number(value: Any) -> int:
    """Read an owned native Attempt ordinal; legacy non-ordinal IDs remain unknown."""
    try:
        return int(str(value).removeprefix("attempt-"))
    except ValueError:
        return 0


def retain_attempt(record: Mapping[str, Any]) -> dict[str, Any]:
    """Archive a terminal physical Attempt before replacing its logical Run state.

    This is a read-model fold, not budget accounting or authoritative recovery evidence. Full
    result.json files remain immutable. Old truncated histories are explicitly lower bounds.
    """
    result = dict(record)
    attempt = record.get("attempt_id")
    if not attempt or record.get("state") not in _TERMINAL:
        return result
    history = [
        dict(item) for item in record.get("attempt_history", ()) if isinstance(item, Mapping)
    ]
    statistics = record.get("attempt_statistics")
    if not isinstance(statistics, Mapping):
        first_attempt = attempt_number(attempt) == 1
        statistics = {
            "completed": len(history),
            "failed": sum(item.get("state") == "failed" for item in history),
            "pruned": sum(item.get("state") == "pruned" for item in history),
            "duration_seconds": sum(
                float(item.get("duration_seconds", 0) or 0) for item in history
            ),
            "history_complete": not bool(history) and first_attempt,
            "latest_attempt_number": max(
                (attempt_number(item.get("attempt_id")) for item in history), default=0
            ),
        }
    statistics = dict(statistics)
    result["attempt_statistics"] = statistics
    ordinal = attempt_number(attempt)
    if (
        record.get("archived_attempt_id") == attempt
        or (ordinal and ordinal <= int(statistics.get("latest_attempt_number", 0)))
        or any(item.get("attempt_id") == attempt for item in history)
    ):
        return result
    history.append(
        {
            key: record.get(key)
            for key in (
                "attempt_id",
                "attempt_number",
                "phase",
                "state",
                "termination_type",
                "termination",
                "failure",
                "failure_disposition",
                "duration_seconds",
                "finished_at_utc",
            )
        }
    )
    for counter, condition in (
        ("completed", True),
        ("failed", record.get("state") == "failed"),
        ("pruned", record.get("state") == "pruned"),
    ):
        statistics[counter] = int(statistics.get(counter, 0)) + int(condition)
    statistics["duration_seconds"] = float(statistics.get("duration_seconds", 0)) + float(
        record.get("duration_seconds", 0) or 0
    )
    if ordinal > int(statistics.get("latest_attempt_number", 0)) + 1:
        statistics["history_complete"] = False
    statistics["latest_attempt_number"] = ordinal
    result.update(
        attempt_history=history[-8:], attempt_statistics=statistics, archived_attempt_id=attempt
    )
    return result
