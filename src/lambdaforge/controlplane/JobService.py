"""Application service for persistent job operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobHandle import JobHandle
from lambdaforge.controlplane.JobRecord import JobRecord
from lambdaforge.controlplane.JobState import JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.execution.ResourceRequest import ResourceRequest


class JobService:
    """Submit and reconnect to jobs without keeping a resident daemon."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        store: JobStore | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.store = store or JobStore()
        self.factory = factory or ControlPlaneFactory()

    def submit(
        self,
        command: Sequence[str],
        *,
        cluster: str,
        resources: ResourceRequest,
        work_dir: str | Path,
        dry_run: bool = False,
        bundle_id: str | None = None,
        config_path: str | None = None,
        retry_of: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> JobHandle:
        """Submit one schedulable unit and persist its complete reconnection data."""
        profile = self.catalog.get(cluster)
        transport = self.factory.transport(profile)
        scheduler = self.factory.scheduler(profile, transport)
        now = datetime.now(timezone.utc).isoformat()
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        submission = scheduler.submit(command, resources, work_dir=work_dir, dry_run=dry_run)
        record_metadata = dict(metadata or {})
        if dry_run:
            record_metadata["scheduler_preview"] = submission.to_dict()
        record = JobRecord(
            job_id=job_id,
            cluster=cluster,
            scheduler=profile.scheduler,
            scheduler_id=submission.scheduler_id,
            state=submission.state,
            command=tuple(command),
            work_dir=str(work_dir),
            resources=resources.to_dict(),
            created_at_utc=now,
            updated_at_utc=now,
            bundle_id=bundle_id,
            config_path=config_path,
            retry_of=retry_of,
            stdout=submission.stdout,
            stderr=submission.stderr,
            metadata=record_metadata,
        )
        self.store.write(record)
        return JobHandle(
            record.job_id,
            cluster,
            record.state,
            record.scheduler_id,
            submission.to_dict() if dry_run else None,
        )

    def get(self, job_id: str, *, refresh: bool = True) -> JobRecord:
        """Load and optionally refresh a non-terminal scheduler state."""
        record = self.store.get(job_id)
        if not refresh or record.scheduler_id is None or record.state.terminal:
            return record
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        state = scheduler.state(record.scheduler_id)
        if state is not record.state:
            record = record.with_updates(
                state=state, updated_at_utc=datetime.now(timezone.utc).isoformat()
            )
            self.store.write(record)
        return record

    def list(
        self,
        *,
        cluster: str | None = None,
        state: JobState | str | None = None,
        name: str | None = None,
        refresh: bool = True,
    ) -> tuple[JobRecord, ...]:
        """Return persistent jobs, reconnecting non-terminal scheduler records by default."""
        selected_state = JobState(state) if state is not None else None
        records = tuple(
            self.get(record.job_id) if refresh and not record.state.terminal else record
            for record in self.store.records()
        )
        return tuple(
            record
            for record in records
            if (cluster is None or record.cluster == cluster)
            and (selected_state is None or record.state is selected_state)
            and (
                name is None
                or name.lower()
                in str(record.metadata.get("name", record.config_path or "")).lower()
            )
        )

    def logs(self, job_id: str, *, tail: int | None = None) -> str:
        """Return captured local output or reconnect to scheduler logs."""
        record = self.get(job_id, refresh=False)
        if record.scheduler == "local" or record.scheduler_id is None:
            value = record.stdout + record.stderr
            return "\n".join(value.splitlines()[-tail:]) if tail else value
        profile = self.catalog.get(record.cluster)
        transport = self.factory.transport(profile)
        filename = f"lambdaforge-{record.scheduler_id}.out"
        command = ("tail", "-n", str(tail), filename) if tail else ("cat", filename)
        result = transport.run(command, cwd=record.work_dir)
        return result.stdout if result.returncode == 0 else result.stderr

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel through the provider and persist the transition."""
        record = self.get(job_id)
        if record.state.terminal or record.scheduler_id is None:
            return record
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        scheduler.cancel(record.scheduler_id)
        record = record.with_updates(
            state=JobState.CANCELLED,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(record)
        return record

    def retry(self, job_id: str, *, dry_run: bool = False) -> JobHandle:
        """Create a new auditable job from one terminal job's exact request."""
        previous = self.get(job_id)
        if not previous.state.terminal:
            raise ValueError("Only terminal jobs can be retried.")
        return self.submit(
            previous.command,
            cluster=previous.cluster,
            resources=ResourceRequest.from_mapping(previous.resources),
            work_dir=previous.work_dir,
            dry_run=dry_run,
            bundle_id=previous.bundle_id,
            config_path=previous.config_path,
            retry_of=previous.job_id,
            metadata=previous.metadata,
        )
