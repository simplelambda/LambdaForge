"""Pure operational observation policy, independent of scientific lifecycle state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.diagnostics.failure import operational_diagnostic


def timestamp(value: object) -> datetime | None:
    """Read UTC clocks without accepting malformed observations as fresh."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class ObservationPolicy:
    """Operational deadlines; progress silence has no expectation by default."""

    freshness_seconds: float = 180.0
    heartbeat_seconds: float = 45.0
    grace_seconds: float = 120.0
    progress_seconds: float | None = None
    unreachable_failures: int = 2

    def __post_init__(self) -> None:
        for name in ("freshness_seconds", "heartbeat_seconds", "grace_seconds", "progress_seconds"):
            value = getattr(self, name)
            if value is None and name == "progress_seconds":
                continue
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Observation {name} must be finite and positive.")
        if (
            isinstance(self.unreachable_failures, bool)
            or not isinstance(self.unreachable_failures, int)
            or self.unreachable_failures < 1
        ):
            raise ValueError("Observation unreachable_failures must be positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ObservationPolicy:
        """Reject silently ignored policy controls."""
        data = dict(value or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown observation policy fields: {sorted(unknown)}")
        return cls(**data)


class OperationalHealth:
    """One shared source of operational freshness, never scientific success or loss."""

    @staticmethod
    def observation(
        record: JobRecord,
        *,
        started: str,
        finished: str,
        state: JobState | None,
        details: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        source: str,
    ) -> dict[str, Any]:
        """Build a versioned fact; a failed read cannot manufacture last success."""
        old = record.metadata.get("operational_observation", {})
        old = dict(old) if isinstance(old, Mapping) else {}
        value = {
            **old,
            "observation_version": 1,
            "last_attempt_started_at_utc": started,
            "last_attempt_at_utc": finished,
            "source": source,
            "last_error": (
                {"type": type(error).__name__, "message": SecretRedactor.redact(str(error))[:1500]}
                if error
                else None
            ),
            "consecutive_failures": int(old.get("consecutive_failures", 0)) + 1 if error else 0,
        }
        if error is None:
            value["last_success_at_utc"] = finished
            facts = dict(details or {})
            value["provider_evidence"] = {
                key: facts[key]
                for key in (
                    "source",
                    "raw_state",
                    "reason",
                    "exit_code",
                    "node",
                    "started_at_utc",
                    "finished_at_utc",
                    "provider_observation",
                    "observation_reason",
                )
                if key in facts
            }
            if (
                state is not None
                and state is not JobState.UNKNOWN
                and facts.get("provider_observation") not in {"unknown", "degraded"}
            ):
                value["last_confirmed_state"] = state.value
                value["last_confirmed_at_utc"] = finished
                value["confirmed_by"] = source
            # Remote clocks are not the successful observer clock. Keep them separate and
            # never allow a future/skewed heartbeat to certify indefinite liveness.
            value["heartbeat_at_utc"] = facts.get("heartbeat_at_utc")
            progress = facts.get("progress")
            if isinstance(progress, Mapping):
                signature = {
                    key: progress[key]
                    for key in ("completed", "total", "step", "epoch")
                    if key in progress
                }
                if signature:
                    request = record.metadata.get("recovery_request")
                    if (
                        isinstance(request, Mapping)
                        and state in {JobState.RUNNING, JobState.PAUSED, JobState.SUCCEEDED}
                        and facts.get("provider_observation")
                        not in {"unknown", "degraded", "lost_confirmed"}
                    ):
                        from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation

                        baseline = (
                            old.get("recovery_progress_signature")
                            if old.get("recovery_progress_baseline_request_id")
                            == request.get("request_id")
                            else None
                        )
                        plan = request.get("plan")
                        retained = (
                            plan.get("preserved_completed_runs", plan.get("reuse_runs"))
                            if isinstance(plan, Mapping)
                            else None
                        )
                        # Native Study progress counts latest succeeded logical Runs. Preview
                        # supplies that same authoritative baseline; restoring it is not work.
                        comparison = dict(signature)
                        completed = comparison.get("completed")
                        beyond_retained = False
                        if type(retained) is int and retained >= 0:
                            beyond_retained = type(completed) is int and completed > retained
                            if not beyond_retained:
                                comparison.pop("completed", None)
                        previous_completed = (
                            baseline.get("completed") if isinstance(baseline, Mapping) else None
                        )
                        newly_beyond_retained = (
                            type(completed) is int
                            and beyond_retained
                            and (
                                type(previous_completed) is not int
                                or completed > previous_completed
                            )
                        )
                        if newly_beyond_retained or RecoveryObservation.advanced(
                            baseline, comparison
                        ):
                            value["recovery_progress_at_utc"] = finished
                            value["recovery_progress_request_id"] = request.get("request_id")
                        value["recovery_progress_signature"] = signature
                        value["recovery_progress_baseline_request_id"] = request.get("request_id")
                    if signature != old.get("progress_signature"):
                        value["progress_signature"] = signature
                        value["last_progress_at_utc"] = finished
        return value

    @classmethod
    def describe(
        cls,
        record: JobRecord,
        *,
        now: datetime | None = None,
        policy: ObservationPolicy | None = None,
    ) -> dict[str, Any]:
        """Age stored facts honestly; reading a cached record is not a fresh probe."""
        now = now or datetime.now(timezone.utc)
        policy = policy or ObservationPolicy.from_mapping(record.metadata.get("observation_policy"))
        raw = record.metadata.get("operational_observation", {})
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        successful = timestamp(raw.get("last_success_at_utc"))
        confirmed = timestamp(raw.get("last_confirmed_at_utc"))
        heartbeat = timestamp(raw.get("heartbeat_at_utc"))
        progress = timestamp(raw.get("last_progress_at_utc"))
        created = timestamp(record.created_at_utc)
        age = (now - successful).total_seconds() if successful else None
        freshness = (
            "unknown"
            if age is None
            else "clock_skew"
            if age < -5
            else "stale"
            if age > policy.freshness_seconds
            else "fresh"
        )
        health, reason, connectivity = "unknown", "NO_PROVIDER_OBSERVATION", "unknown"
        error = raw.get("last_error")
        evidence = raw.get("provider_evidence", {})
        evidence = evidence if isinstance(evidence, Mapping) else {}
        if error:
            from lambdaforge.diagnostics.failure import provider_observation_category
            from lambdaforge.diagnostics.models import ErrorCategory

            count = int(raw.get("consecutive_failures", 0))
            inaccessible = isinstance(error, Mapping) and provider_observation_category(error) in {
                ErrorCategory.CONNECTION,
                ErrorCategory.AUTHENTICATION,
            }
            malformed = isinstance(error, Mapping) and error.get("type") in {
                "JSONDecodeError",
                "ValueError",
                "TypeError",
            }
            health = (
                "unreachable"
                if inaccessible and count >= policy.unreachable_failures
                else "degraded"
            )
            reason = "INVALID_PROVIDER_OBSERVATION" if malformed else "PROVIDER_OBSERVATION_FAILED"
            connectivity = "unreachable" if inaccessible else "unknown"
        elif successful:
            connectivity = "reachable" if freshness == "fresh" else "unknown"
            if freshness != "fresh":
                health, reason = "stale", "PROVIDER_OBSERVATION_STALE"
            elif evidence.get("provider_observation") == "lost_confirmed":
                health, reason = "lost_confirmed", "PROCESS_LOST_CONFIRMED"
            elif evidence.get("provider_observation") == "degraded":
                health, reason = "degraded", str(evidence.get("observation_reason"))
            elif raw.get("last_confirmed_state") is None or (
                record.state is JobState.UNKNOWN
                or evidence.get("provider_observation") == "unknown"
            ):
                health, reason = (
                    "unknown",
                    str(evidence.get("observation_reason") or "PROVIDER_STATE_UNKNOWN"),
                )
            elif record.state.terminal:
                health, reason = "healthy", "SCHEDULER_TERMINAL"
            else:
                health, reason = "healthy", "PROVIDER_CONFIRMED"
                direct = record.scheduler != "slurm"
                grace = (
                    created is not None and (now - created).total_seconds() < policy.grace_seconds
                )
                if (
                    direct
                    and not grace
                    and (
                        heartbeat is None
                        or (now - heartbeat).total_seconds() > policy.heartbeat_seconds
                        or (heartbeat - now).total_seconds() > 5
                    )
                ):
                    health, reason = "degraded", "HEARTBEAT_STALE"
                if (
                    policy.progress_seconds is not None
                    and not grace
                    and record.state is JobState.RUNNING
                    and (
                        progress is None
                        or (now - progress).total_seconds() > policy.progress_seconds
                    )
                ):
                    health, reason = "degraded", "SCIENTIFIC_PROGRESS_STALLED"
        # Historical/old records remain unknown, even if locally rewritten just now.
        execution_state = (
            raw.get("last_confirmed_state")
            or record.metadata.get("last_known_state")
            or record.state.value
        )
        result = {
            **raw,
            "health": health,
            "reason": reason,
            "connectivity": connectivity,
            "freshness": freshness,
            "observation_age_seconds": max(0.0, age) if age is not None else None,
            "confirmation_age_seconds": max(0.0, (now - confirmed).total_seconds())
            if confirmed
            else None,
            "execution_state": execution_state,
            "job_id": record.job_id,
            "cluster": record.cluster,
            "scheduler": record.scheduler,
            "scheduler_id": record.scheduler_id,
            "execution_id": record.metadata.get("execution_id"),
            "policy": asdict(policy),
        }
        failure = record.metadata.get("failure")
        result["diagnostic"] = operational_diagnostic(
            result,
            state=record.state.value,
            failure=failure if isinstance(failure, Mapping) else None,
            failure_phase=record.metadata.get("failure_phase"),
            failure_category=record.metadata.get("failure_category"),
        )
        return result
