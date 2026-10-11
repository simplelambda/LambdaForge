"""Read native recovery Job links and append-only events, never a second catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalHealth import OperationalHealth
from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation
from lambdaforge.diagnostics.service import DiagnosticRecorder


def recovery_history(
    store: JobStore, job_id: str, *, offset: int = 0, limit: int = 100
) -> dict[str, Any]:
    """Inspect exact linked Jobs and a page of the selected Job's saved audit.

    Available without the execution host, application imports, artifacts or checkpoints.
    Links require the native retry/dependency relation, same cluster and exact scientific
    identity; shared display names never join histories. Missing ancestors stay explicit.
    Each Job's event stream remains append-ordered and independently pageable.
    """
    selected = store.get(job_id)
    records = {record.job_id: record for record in store.records()}
    family = {job_id}

    def linked(child: JobRecord, parent: JobRecord) -> bool:
        request = child.metadata.get("recovery_request")
        dependencies = child.metadata.get("recovery_dependencies", ())
        identity = parent.metadata.get("recovery_work_identity") or parent.metadata.get(
            "scientific_identity"
        )
        child_identity = child.metadata.get("recovery_work_identity") or child.metadata.get(
            "scientific_identity"
        )
        return (
            bool(identity)
            and identity == child_identity
            and child.cluster == parent.cluster
            and child.retry_of == parent.job_id
            and isinstance(dependencies, Sequence)
            and not isinstance(dependencies, str)
            and parent.job_id in dependencies
            and isinstance(request, Mapping)
            and request.get("origin_job_id") == parent.job_id
        )

    while True:
        expanded = set(family)
        for child in records.values():
            parent = records.get(child.retry_of or "")
            if parent is not None and linked(child, parent):
                if child.job_id in family or parent.job_id in family:
                    expanded.update((child.job_id, parent.job_id))
        if expanded == family:
            break
        family = expanded
    jobs = sorted((records[key] for key in family), key=lambda item: item.created_at_utc)
    page = store.event_page(job_id, offset=offset, limit=limit)
    result = {
        "recovery_audit_version": 1,
        "selected_job_id": job_id,
        "cluster": selected.cluster,
        "source": "saved_native_job_records_and_events",
        "provider_refreshed": False,
        "science_mutated": False,
        "jobs": [
            {
                "job_id": record.job_id,
                "retry_of": record.retry_of,
                "state": record.state.value,
                "scheduler_id": record.scheduler_id,
                "operational": OperationalHealth.describe(record),
                "recovery": RecoveryObservation.describe(record),
                "configuration": record.metadata.get("recovery_configuration"),
            }
            for record in jobs
        ],
        "missing_dependencies": sorted(
            {
                key
                for record in jobs
                for key in record.metadata.get("recovery_dependencies", ())
                if isinstance(key, str)
                if key not in records
            }
        ),
        "events": page,
        "continuation": (
            "Checkpoint availability is not proof of application restoration or epoch continuation."
        ),
    }
    return DiagnosticRecorder._redact(result)
