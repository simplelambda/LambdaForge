"""Durable operational incidents; no scientific execution or result authority."""

from __future__ import annotations

import hashlib
import json
import os
import smtplib
import ssl
from collections.abc import Mapping
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalHealth import OperationalHealth, timestamp
from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


def read_document(path: Path) -> dict[str, Any]:
    """Read only a regular owned operational document; corruption is not an empty catalog."""
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"Unsafe operational document: {path}")
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Corrupt operational document: {path}")
    return value


def write_document(path: Path, value: Mapping[str, Any]) -> None:
    """Atomic durable operational facts, never a replacement JobStore."""
    if path.resolve() != path.absolute() or path.is_symlink():
        raise ValueError(f"Unsafe operational document: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(value), handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OperationalIncidents:
    """Deduplicated project incidents with durable notification delivery state."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.root = store.root / ".operational" / "incidents"

    def observe(self, record: JobRecord, *, now: datetime | None = None) -> dict[str, Any] | None:
        """Record a transition from verified facts; silence never fabricates a failure."""
        now = now or datetime.now(timezone.utc)
        health = OperationalHealth.describe(record, now=now)
        raw = (
            str(health.get("provider_evidence", {}).get("raw_state", "")).split(" ")[0].rstrip("+")
        )
        code: str | None = None
        severity = "warning"
        if health["health"] in {"degraded", "unreachable"}:
            code = (
                "CONNECTIVITY_DEGRADED"
                if health["connectivity"] == "unreachable"
                else health["reason"]
            )
            severity = "error" if health["health"] == "unreachable" else "warning"
        elif not record.state.terminal and health["health"] in {"stale", "lost_confirmed"}:
            code = health["reason"]
            severity = "critical" if health["health"] == "lost_confirmed" else "warning"
        elif record.state in {JobState.FAILED, JobState.TIMEOUT} or (
            record.state is JobState.CANCELLED and raw == "PREEMPTED"
        ):
            code = {
                "NODE_FAIL": "INFRASTRUCTURE_NODE_FAILURE",
                "OUT_OF_MEMORY": "RESOURCE_OOM",
                "PREEMPTED": "SCHEDULER_PREEMPTION",
            }.get(
                raw, "EXECUTION_FAILED" if record.state is JobState.FAILED else "EXECUTION_TIMEOUT"
            )
            severity = "error"
        result = (
            self.record(
                record,
                code,
                severity=severity,
                summary=str(health["reason"]),
                now=now,
                diagnosis=health.get("diagnostic"),
            )
            if code
            else None
        )
        # Resolve only from a positive fresh observation, not disappearance or uncertainty.
        if health["health"] == "healthy":
            for incident in self.list(job_id=record.job_id):
                if incident["status"] == "open" and incident["code"] not in {
                    "EXECUTION_FAILED",
                    "EXECUTION_TIMEOUT",
                    "INFRASTRUCTURE_NODE_FAILURE",
                    "RESOURCE_OOM",
                    "SCHEDULER_PREEMPTION",
                }:
                    self.resolve(str(incident["incident_id"]), now=now)
            self._resolve_recovered_origin(record, now=now)
        return result

    def _resolve_recovered_origin(self, record: JobRecord, *, now: datetime) -> None:
        """Resolve an audited origin only after fresh progress in its exact recovery Job.

        Enqueue, a scheduler acknowledgement and a runtime heartbeat are not scientific
        recovery. Historical Job/Attempt outcomes are never rewritten by incident resolution.
        """
        from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation

        recovery = RecoveryObservation.describe(record)
        if (
            not recovery
            or not recovery["scientific_progress_observed"]
            or record.state not in {JobState.RUNNING, JobState.PAUSED, JobState.SUCCEEDED}
        ):
            return
        requested = timestamp(recovery.get("requested_at_utc"))
        progressed = timestamp(recovery.get("progress_observed_at_utc"))
        if requested is None or progressed is None or progressed < requested:
            return
        origin_id = recovery.get("origin_job_id")
        request_id = recovery.get("request_id")
        if (
            not isinstance(origin_id, str)
            or not isinstance(request_id, str)
            or record.retry_of != origin_id
            or origin_id not in record.metadata.get("recovery_dependencies", ())
        ):
            return
        open_incidents = [
            incident for incident in self.list(job_id=origin_id) if incident["status"] == "open"
        ]
        if not open_incidents:
            return  # Resolved origins must not rescan the append-only audit on every poll.
        try:
            origin = self.store.get(origin_id)
        except (FileNotFoundError, ValueError):
            return
        expected = origin.metadata.get("recovery_work_identity") or origin.metadata.get(
            "scientific_identity"
        )
        if (
            record.cluster != origin.cluster
            or not expected
            or record.metadata.get("recovery_work_identity") != expected
            or not any(
                event.get("phase") == "RECOVERY_ACCEPTED"
                and event.get("recovery_request_id") == request_id
                for event in self.store.events(origin_id)
            )
        ):
            return
        for incident in open_incidents:
            self.resolve(
                str(incident["incident_id"]),
                now=now,
                recovered_by={
                    "job_id": record.job_id,
                    "request_id": request_id,
                    "progress_observed_at_utc": recovery["progress_observed_at_utc"],
                },
            )

    def record(
        self,
        record: JobRecord,
        code: str,
        *,
        severity: str,
        summary: str,
        now: datetime | None = None,
        diagnosis: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upsert one exact Job/code; polling changes neither identity nor first occurrence."""
        now = now or datetime.now(timezone.utc)
        identifier = (
            "incident-" + hashlib.sha256(f"{record.job_id}\0{code}".encode()).hexdigest()[:24]
        )
        path = self._path(identifier)
        with self._lock():
            old = read_document(path)
            changed = not old or old.get("status") != "open" or old.get("severity") != severity
            value = {
                **old,
                "incident_version": 1,
                "incident_id": identifier,
                "job_id": record.job_id,
                "cluster": record.cluster,
                "scheduler_id": record.scheduler_id,
                "execution_id": record.metadata.get("execution_id"),
                "code": code,
                "severity": severity,
                "status": "open",
                "summary": SecretRedactor.redact(summary)[:1500],
                "first_seen_at_utc": old.get("first_seen_at_utc") or now.isoformat(),
                "last_seen_at_utc": now.isoformat(),
                "occurrences": int(old.get("occurrences", 0)) + (1 if changed else 0),
                "diagnostic": dict(diagnosis) if diagnosis else None,
                "recommendation": " ".join(str(fix) for fix in diagnosis.get("fixes", ()))
                if diagnosis
                else (
                    "Diagnose/reconcile before deliberate recovery; do not assume lost ownership."
                ),
                "diagnostic_command": f"lf health {record.job_id} --refresh --json",
            }
            if changed:
                value["notification"] = {
                    **old.get("notification", {}),
                    "status": "pending",
                    "attempts": 0,
                    "attempt_id": None,
                }
            write_document(path, value)
            if changed:
                self.store.append_event(
                    record.job_id,
                    state=record.state.value,
                    source="operational-incident",
                    phase=code,
                    message=f"{identifier}: {severity} · {value['summary']}",
                )
            return value

    def list(self, *, job_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Read local compact facts, without contacting clusters or opening science bytes."""
        values = [read_document(path) for path in sorted(self.root.glob("incident-*.json"))]
        return tuple(
            value for value in values if value and (job_id is None or value.get("job_id") == job_id)
        )

    def resolve(
        self,
        identifier: str,
        *,
        now: datetime | None = None,
        recovered_by: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            path = self._path(identifier)
            value = read_document(path)
            if not value:
                raise FileNotFoundError(identifier)
            if value["status"] == "resolved":
                return value
            value.update(
                status="resolved", resolved_at_utc=(now or datetime.now(timezone.utc)).isoformat()
            )
            if recovered_by is not None:
                value["recovered_by"] = dict(recovered_by)
            value["notification"] = {
                **value.get("notification", {}),
                "status": "pending",
                "attempts": 0,
                "attempt_id": None,
            }
            write_document(path, value)
            self.store.append_event(
                str(value["job_id"]),
                state=self.store.get(str(value["job_id"])).state.value,
                source="operational-incident",
                phase="INCIDENT_RESOLVED",
                recovery_request_id=(
                    str(recovered_by["request_id"]) if recovered_by is not None else None
                ),
                message=(
                    f"{identifier}: scientific progress verified in recovery Job "
                    f"{recovered_by['job_id']}; historical outcome preserved."
                    if recovered_by is not None
                    else f"{identifier}: resolved from a positive provider observation."
                ),
            )
            return value

    def deliver(self, settings: Mapping[str, Any], *, now: datetime | None = None) -> None:
        """Optional SMTP with bounded retry/cooldown; failure never modifies scientific Jobs.

        SMTP cannot provide exactly-once delivery across an interrupted send. The stable
        Message-ID permits receiver deduplication; acceptance, not intent, marks delivery.
        """
        if not settings.get("enabled"):
            return
        now = now or datetime.now(timezone.utc)
        for incident in self.list():
            path = self._path(str(incident["incident_id"]))
            with self._lock():
                value = read_document(path)
                delivery = dict(value.get("notification", {}))
                attempts = int(delivery.get("attempts", 0))
                last = timestamp(delivery.get("last_attempt_at_utc"))
                cooldown = float(settings.get("cooldown_seconds", 900))
                if (
                    delivery.get("status") == "sent"
                    or attempts >= 3
                    or (last is not None and (now - last).total_seconds() < cooldown)
                ):
                    continue
                attempt_id = uuid4().hex
                delivery.update(
                    attempts=attempts + 1,
                    last_attempt_at_utc=now.isoformat(),
                    status="in_flight",
                    attempt_id=attempt_id,
                )
                value["notification"] = delivery
                write_document(path, value)
            # Never hold the incident lock through SMTP. A slow relay must not delay
            # provider fact commits, incident escalation or last-good Console reads.
            error_type: str | None = None
            try:
                self.send(settings, value)
            except Exception as error:
                error_type = type(error).__name__  # Server text may echo authentication secrets.
            with self._lock():
                current = read_document(path)
                delivery = dict(current.get("notification", {}))
                if delivery.get("attempt_id") != attempt_id:
                    continue  # A resolution/reopening/escalation superseded this delivery.
                if error_type:
                    delivery.update(status="failed", error_type=error_type)
                else:
                    delivery.pop("error_type", None)
                    delivery.update(status="sent", accepted_at_utc=now.isoformat())
                current["notification"] = delivery
                write_document(path, current)

    @staticmethod
    def send(settings: Mapping[str, Any], incident: Mapping[str, Any]) -> None:
        """TLS-only SMTP; a credential reference is resolved in memory through native providers."""
        message = EmailMessage()
        message["From"] = str(settings["sender"])
        message["To"] = ", ".join(settings["recipients"])
        message["Subject"] = (
            f"LambdaForge {incident['status']} · {incident['code']} · {incident['job_id']}"
        )
        message["Message-ID"] = (
            f"<{incident['incident_id']}-{incident.get('occurrences', 1)}-"
            f"{incident['status']}@lambdaforge>"
        )
        message.set_content(
            f"{incident['summary']}\n{incident['diagnostic_command']}\n{incident['recommendation']}"
        )
        context = ssl.create_default_context()
        if settings.get("tls", "starttls") not in {"ssl", "starttls"}:
            raise ValueError("SMTP requires ssl or starttls; plaintext credentials are refused.")
        constructor = smtplib.SMTP_SSL if settings.get("tls", "starttls") == "ssl" else smtplib.SMTP
        options: dict[str, Any] = {"timeout": 15}
        if constructor is smtplib.SMTP_SSL:
            options["context"] = context
        with constructor(str(settings["host"]), int(settings.get("port", 587)), **options) as smtp:
            if settings.get("tls", "starttls") == "starttls":
                smtp.starttls(context=context)
            if settings.get("username"):
                secret = CredentialService().resolve_reference(str(settings["credential"]))
                smtp.login(str(settings["username"]), secret)
            refused = smtp.send_message(message)
            if refused:
                raise RuntimeError("SMTP rejected at least one configured recipient.")

    def _path(self, identifier: str) -> Path:
        if (
            not identifier.startswith("incident-")
            or len(identifier) != 33
            or any(char not in "0123456789abcdef" for char in identifier[9:])
        ):
            raise ValueError("Invalid operational incident identifier.")
        return self.root / f"{identifier}.json"

    def _lock(self) -> CrossProcessFileLock:
        return CrossProcessFileLock(
            self.root / ".lock", shared=False, timeout_seconds=20, poll_interval_seconds=0.05
        )
