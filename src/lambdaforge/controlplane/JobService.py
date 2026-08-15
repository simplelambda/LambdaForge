"""Application service for persistent job operations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.jobs import JobHandle, JobRecord, JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.diagnostics import (
    DiagnosticClassifier,
    DiagnosticContext,
    LambdaForgeError,
)
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
        job_type: str = "command",
        group_id: str | None = None,
        job_id: str | None = None,
    ) -> JobHandle:
        """Submit one schedulable unit and persist its complete reconnection data."""
        profile = self.catalog.get(cluster)
        transport = self.factory.transport(profile)
        scheduler = self.factory.scheduler(profile, transport)
        now = datetime.now(timezone.utc).isoformat()
        job_id = job_id or self.new_id()
        if not job_id.startswith("job-"):
            raise ValueError("LambdaForge job ids must start with 'job-'.")
        record_metadata = dict(metadata or {})
        record = JobRecord(
            job_id=job_id,
            cluster=cluster,
            scheduler=profile.scheduler,
            scheduler_id=None,
            state=JobState.CREATED,
            command=tuple(command),
            work_dir=str(work_dir),
            resources=resources.to_dict(),
            created_at_utc=now,
            updated_at_utc=now,
            bundle_id=bundle_id,
            config_path=config_path,
            retry_of=retry_of,
            metadata=record_metadata,
            job_type=job_type,
            group_id=group_id,
        )
        self.store.write(record)
        try:
            parameters = inspect.signature(scheduler.submit).parameters
            kwargs: dict[str, Any] = {
                "work_dir": work_dir,
                "dry_run": dry_run,
            }
            if "job_id" in parameters:
                kwargs["job_id"] = job_id
            submission = scheduler.submit(command, resources, **kwargs)
        except Exception as error:
            context = DiagnosticContext(
                (),
                "job submission",
                cluster,
            )
            base = DiagnosticClassifier().classify(error, context)
            failed = record.with_updates(
                state=JobState.FAILED,
                stderr=f"{error.__class__.__name__}: {error}",
                metadata={
                    **record_metadata,
                    "failure_phase": "submission",
                    "failure_category": base.category.value,
                },
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            self.store.write(failed)
            commands = (
                ("Job details", f"lf jobs show {job_id} --json"),
                ("Submission record", f"lf jobs logs {job_id} --tail 300"),
                ("Diagnose cluster", f"lf doctor --on {cluster}"),
                ("Retry after fixing", f"lf jobs retry {job_id}"),
                ("Retry with internals", f"lf jobs retry {job_id} --debug"),
            )
            raise LambdaForgeError(
                replace(
                    base,
                    title=f"Job {job_id} could not be submitted: {base.title}",
                    impact=(
                        "A durable failed submission record was created for inspection.",
                        "The scheduler did not acknowledge a scientific job.",
                    ),
                    commands=commands,
                    context={**dict(base.context), "cluster": cluster, "job": job_id},
                    operation="job submission",
                    job_id=job_id,
                    details=(*base.details, "Failure occurred before scheduler acknowledgement."),
                )
            ) from error
        if dry_run:
            record_metadata["scheduler_preview"] = submission.to_dict()
        record = record.with_updates(
            scheduler_id=submission.scheduler_id,
            state=JobState.PLANNED if dry_run else submission.state,
            work_dir=submission.work_dir or str(work_dir),
            stdout=submission.stdout,
            stderr=submission.stderr,
            metadata=record_metadata,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(record)
        return JobHandle(
            record.job_id,
            cluster,
            record.state,
            record.scheduler_id,
            submission.to_dict() if dry_run else None,
        )

    def resolve_selector(self, selector: str) -> str:
        """Resolve an exact ID, ``latest`` or one unambiguous job name."""
        try:
            return self.store.get(selector).job_id
        except (KeyError, FileNotFoundError):
            pass
        records = self.store.records()
        if selector == "latest":
            if not records:
                raise KeyError("No persistent jobs exist.")
            return max(records, key=lambda value: value.created_at_utc).job_id
        matches = tuple(
            value
            for value in records
            if value.metadata.get("name") == selector or value.job_id.startswith(selector)
        )
        if not matches:
            raise KeyError(f"Unknown job selector {selector!r}.")
        if len(matches) > 1:
            raise ValueError(
                f"Job selector {selector!r} is ambiguous: "
                f"{tuple(value.job_id for value in matches)}."
            )
        return matches[0].job_id

    @staticmethod
    def new_id() -> str:
        """Create a collision-resistant human-sortable job identifier."""
        return f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"

    def get(self, job_id: str, *, refresh: bool = True) -> JobRecord:
        """Load and optionally refresh a non-terminal scheduler state."""
        record = self.store.get(job_id)
        if not refresh or record.scheduler_id is None or record.state.terminal:
            return record
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        try:
            state = scheduler.state(record.scheduler_id)
        except Exception as error:
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "last_known_state": record.state.value,
                    "unreachable_error": f"{error.__class__.__name__}: {error}",
                    "last_refresh_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            record = record.with_updates(
                state=JobState.UNKNOWN,
                metadata=metadata,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            self.store.write(record)
            return record
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
        stored = self.store.records()
        if refresh:
            refreshable = tuple(record for record in stored if not record.state.terminal)
            refreshed: dict[str, JobRecord] = {}
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(refreshable)))) as executor:
                futures = {
                    executor.submit(self.get, record.job_id): record for record in refreshable
                }
                for future in as_completed(futures):
                    original = futures[future]
                    try:
                        refreshed[original.job_id] = future.result()
                    except Exception:
                        refreshed[original.job_id] = original
            records = tuple(refreshed.get(record.job_id, record) for record in stored)
        else:
            records = stored
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
        if record.scheduler_id is None:
            value = record.stdout + record.stderr
            return "\n".join(value.splitlines()[-tail:]) if tail else value
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        return scheduler.logs(record.scheduler_id, tail=tail)

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

    def pause(self, job_id: str) -> JobRecord:
        """Pause only when the authoritative scheduler advertises support."""
        return self._lifecycle(job_id, "pause", JobState.PAUSED)

    def resume(self, job_id: str) -> JobRecord:
        """Resume only when the authoritative scheduler advertises support."""
        return self._lifecycle(job_id, "resume", JobState.RUNNING)

    def delete(self, job_id: str) -> None:
        """Delete local metadata only; results and remote job bytes remain untouched."""
        record = self.get(job_id, refresh=False)
        if not record.state.terminal:
            raise ValueError("Only terminal job metadata can be deleted.")
        self.store.delete(job_id)

    def reconcile(
        self, *, cluster: str | None = None, all_clusters: bool = False
    ) -> tuple[JobRecord, ...]:
        """Discover durable provider jobs and rebuild/update the local index."""
        names = self.catalog.names() if all_clusters else (cluster or "local",)
        discovered: list[JobRecord] = []
        inventories: dict[str, tuple[Mapping[str, object], ...]] = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(names)))) as executor:
            futures = {executor.submit(self._inventory, name): name for name in names}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    inventories[name] = future.result()
                except Exception:
                    # One unavailable cluster must not prevent reconciliation elsewhere.
                    inventories[name] = ()
        for name in names:
            profile = self.catalog.get(name)
            for item in inventories.get(name, ()):
                request = item.get("request")
                state_value = item.get("state")
                if not isinstance(request, Mapping) or not isinstance(state_value, Mapping):
                    continue
                job_id = str(request.get("job_id", ""))
                if not job_id.startswith("job-"):
                    continue
                try:
                    previous = self.store.get(job_id)
                except FileNotFoundError:
                    previous = None
                now = datetime.now(timezone.utc).isoformat()
                state = JobState(str(state_value.get("state", JobState.UNKNOWN.value)))
                record = (
                    previous.with_updates(state=state, updated_at_utc=now)
                    if previous is not None
                    else JobRecord(
                        job_id=job_id,
                        cluster=name,
                        scheduler=profile.scheduler,
                        scheduler_id=job_id,
                        state=state,
                        command=tuple(str(value) for value in request.get("command", ())),
                        work_dir=str(request.get("work_dir", "")),
                        resources=request.get("resources", {}),
                        created_at_utc=str(request.get("created_at_utc", now)),
                        updated_at_utc=now,
                        metadata={"reconciled": True, "remote_state": state_value},
                        job_type="command",
                    )
                )
                self.store.write(record)
                discovered.append(record)
        return tuple(discovered)

    def _inventory(self, cluster: str) -> tuple[Mapping[str, object], ...]:
        profile = self.catalog.get(cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        return tuple(scheduler.inventory())

    def _lifecycle(self, job_id: str, operation: str, target: JobState) -> JobRecord:
        record = self.get(job_id)
        if record.scheduler_id is None:
            raise ValueError("The job has no scheduler identity.")
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        capability = getattr(scheduler.capabilities, f"supports_{operation}")
        if not capability:
            raise LambdaForgeError(
                DiagnosticClassifier().classify(
                    RuntimeError(f"{operation.title()} is not supported by this scheduler."),
                    DiagnosticContext(
                        ("jobs", operation, job_id),
                        f"job {operation}",
                        record.cluster,
                    ),
                )
            )
        getattr(scheduler, operation)(record.scheduler_id)
        metadata = dict(record.metadata)
        if target is JobState.PAUSED:
            metadata["pause_warning"] = "RAM/VRAM and GPU leases may remain allocated."
        record = record.with_updates(
            state=target,
            metadata=metadata,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(record)
        return record

    def retry(self, job_id: str, *, dry_run: bool = False) -> JobHandle:
        """Create a new auditable job from one terminal job's exact request."""
        previous = self.get(job_id)
        if not previous.state.terminal:
            raise ValueError("Only terminal jobs can be retried.")
        retry_metadata = dict(previous.metadata)
        retry_metadata.pop("failure_phase", None)
        retry_metadata.pop("failure_category", None)
        return self.submit(
            previous.command,
            cluster=previous.cluster,
            resources=ResourceRequest.from_mapping(previous.resources),
            work_dir=previous.work_dir,
            dry_run=dry_run,
            bundle_id=previous.bundle_id,
            config_path=previous.config_path,
            retry_of=previous.job_id,
            metadata=retry_metadata,
            job_type=previous.job_type,
            group_id=previous.group_id,
        )
