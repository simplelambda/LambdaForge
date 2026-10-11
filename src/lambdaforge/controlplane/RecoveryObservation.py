"""Recovery request presentation derived from the one native Job lifecycle."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.OperationalHealth import timestamp


class RecoveryObservation:
    """An audit link and honest phases, never a second recovery scheduler/state store."""

    @staticmethod
    def eligible(state: str, operational: Mapping[str, Any]) -> bool:
        """Cached preview eligibility; apply still revalidates provider ownership under lock."""
        return state in {"failed", "cancelled", "timeout"} or (
            state == "unknown"
            and operational.get("health") == "lost_confirmed"
            and operational.get("freshness") == "fresh"
        )

    @staticmethod
    def advanced(previous: Any, current: Mapping[str, Any]) -> bool:
        """A loaded positive count/epoch is a baseline, not new recovery computation.

        Only a strict increase on a finite scientific coordinate already observed in
        this Job is progression. Total/queue changes and a reset are not advancement.
        """
        if not isinstance(previous, Mapping):
            return False

        def valid(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (not isinstance(value, float) or math.isfinite(value))
                and value >= 0
            )

        return any(
            valid(previous.get(key)) and valid(current.get(key)) and current[key] > previous[key]
            for key in ("completed", "step", "epoch")
        )

    @staticmethod
    def describe(record: JobRecord) -> dict[str, Any] | None:
        request = record.metadata.get("recovery_request")
        if not isinstance(request, Mapping):
            return None
        observation = record.metadata.get("operational_observation", {})
        observation = observation if isinstance(observation, Mapping) else {}
        progressed_at = timestamp(observation.get("recovery_progress_at_utc"))
        requested_at = timestamp(request.get("requested_at_utc"))
        progressed = (
            progressed_at is not None
            and requested_at is not None
            and progressed_at >= requested_at
            and observation.get("recovery_progress_request_id") == request.get("request_id")
        )
        phase = "accepted"
        if record.state in {JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT}:
            phase = record.state.value
        elif record.state is JobState.SUCCEEDED:
            phase = "completed"
        elif progressed:
            phase = "scientific_progress_observed"
        elif timestamp(observation.get("heartbeat_at_utc")) is not None:
            phase = "runtime_observed"
        elif record.scheduler_id is not None:
            phase = "scheduler_acknowledged"
        elif record.metadata.get("submission_phase") not in {None, "queued-locally"}:
            phase = "preparing"
        return {
            "recovery_version": 1,
            **dict(request),
            "job_id": record.job_id,
            "scheduler_id": record.scheduler_id,
            "phase": phase,
            "preparation_phase": record.metadata.get("submission_phase"),
            "failure_phase": record.metadata.get("failure_phase"),
            "failure_category": record.metadata.get("failure_category"),
            "scientific_progress_observed": progressed,
            "progress_observed_at_utc": observation.get("recovery_progress_at_utc"),
            "continuation": "Application checkpoint support is required for epoch continuation.",
            "diagnostic_command": f"lf jobs show {record.job_id} --json",
        }
