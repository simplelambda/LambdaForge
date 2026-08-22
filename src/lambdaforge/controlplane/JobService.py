"""Application service for persistent job operations."""

from __future__ import annotations

import inspect
import json
import shlex
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
    ErrorCategory,
    LambdaForgeError,
    RetryDisposition,
    diagnostic,
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
        try:
            reserved = self.store.get(job_id)
        except FileNotFoundError:
            reserved = None
        if reserved is not None and reserved.state is JobState.CANCELLED:
            raise RuntimeError(
                f"Submission {job_id} was cancelled before scheduler acknowledgement."
            )
        record_metadata = {
            **(dict(reserved.metadata) if reserved is not None else {}),
            **dict(metadata or {}),
        }
        record_metadata.setdefault("attempt", self._attempt_number(retry_of))
        record = JobRecord(
            job_id=job_id,
            cluster=cluster,
            scheduler=profile.scheduler,
            scheduler_id=None,
            state=JobState.CREATED,
            command=tuple(command),
            work_dir=str(work_dir),
            resources=resources.to_dict(),
            created_at_utc=reserved.created_at_utc if reserved is not None else now,
            updated_at_utc=now,
            bundle_id=bundle_id,
            config_path=config_path,
            retry_of=retry_of,
            metadata=record_metadata,
            job_type=job_type,
            group_id=group_id,
        )
        self.store.write(record)
        if reserved is None:
            self.store.append_event(
                job_id,
                state=record.state.value,
                phase="scheduler",
                message=f"Submission record created; contacting {profile.scheduler} scheduler.",
            )
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
            self.store.append_event(
                job_id,
                state=failed.state.value,
                phase="scheduler",
                message="Scheduler submission failed; inspect the diagnostic and controller log.",
            )
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
        self.store.append_event(
            job_id,
            state=record.state.value,
            phase="scheduler",
            message=(
                "Read-only scheduler plan completed."
                if dry_run
                else f"Scheduler acknowledged the job as {record.state.value}."
            ),
        )
        return JobHandle(
            record.job_id,
            cluster,
            record.state,
            record.scheduler_id,
            submission.to_dict() if dry_run else None,
        )

    def reserve(
        self,
        *,
        cluster: str,
        resources: ResourceRequest,
        config_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        job_type: str = "command",
        group_id: str | None = None,
        retry_of: str | None = None,
    ) -> JobHandle:
        """Persist a controller-side submission before slow preparation begins."""
        profile = self.catalog.get(cluster)
        now = datetime.now(timezone.utc).isoformat()
        job_id = self.new_id()
        record_metadata = dict(metadata or {})
        record_metadata.setdefault("attempt", self._attempt_number(retry_of))
        record = JobRecord(
            job_id=job_id,
            cluster=cluster,
            scheduler=profile.scheduler,
            scheduler_id=None,
            state=JobState.PREPARING,
            command=(),
            work_dir=str(Path(config_path).resolve().parent),
            resources=resources.to_dict(),
            created_at_utc=now,
            updated_at_utc=now,
            config_path=str(Path(config_path).resolve()),
            retry_of=retry_of,
            metadata=record_metadata,
            job_type=job_type,
            group_id=group_id,
        )
        self.store.write(record)
        self.store.append_event(
            job_id,
            state=record.state.value,
            phase="queued-locally",
            message="Submission accepted locally; background preparation started.",
        )
        return JobHandle(job_id, cluster, JobState.PREPARING)

    def active_execution(
        self,
        scientific_identity: str,
        cluster: str,
        *,
        exclude_job_id: str | None = None,
    ) -> tuple[JobRecord, ...]:
        """Return locally known active attempts for one scientific revision and target."""
        return tuple(
            record
            for record in self.store.records()
            if record.job_id != exclude_job_id
            and record.cluster == cluster
            and not record.state.terminal
            and record.metadata.get("scientific_identity") == scientific_identity
        )

    def refuse_active_execution(
        self,
        scientific_identity: str,
        cluster: str,
        *,
        name: str,
        source: str | Path | None = None,
        exclude_job_id: str | None = None,
    ) -> None:
        """Reject an accidental duplicate while retaining an explicit escape hatch."""
        active = self.active_execution(scientific_identity, cluster, exclude_job_id=exclude_job_id)
        if not active:
            return
        selected = active[0]
        revision = scientific_identity.removeprefix("sha256:")[:12]
        rerun = (
            shlex.join(("lf", "run", str(source), "--on", cluster, "--allow-duplicate"))
            if source is not None
            else "lf run CONFIG --on " + shlex.quote(cluster) + " --allow-duplicate"
        )
        raise LambdaForgeError(
            diagnostic(
                ErrorCategory.OPERATION_REFUSED,
                f"Work {name!r} is already active on {cluster!r}.",
                "No duplicate execution was submitted.",
                reason=(
                    f"Scientific revision {revision} already has active job {selected.job_id} "
                    f"in state {selected.state.value}."
                ),
                impact=("Existing scientific work continues unchanged.",),
                fixes=(
                    "Watch or inspect the existing execution.",
                    "Use --allow-duplicate only when concurrent duplicate work is intentional.",
                ),
                commands=(
                    ("Work status", f"lf show {shlex.quote(name)}"),
                    ("Existing job", f"lf jobs show {selected.job_id}"),
                    ("Intentional duplicate", rerun),
                ),
                context={
                    "work": name,
                    "scientific_identity": scientific_identity,
                    "scientific_revision": revision,
                    "cluster": cluster,
                    "active_job_id": selected.job_id,
                },
                retryable=RetryDisposition.NO,
                operation="work submission",
                job_id=selected.job_id,
            )
        )

    def _attempt_number(self, retry_of: str | None) -> int:
        if retry_of is None:
            return 1
        try:
            previous = self.store.get(retry_of)
        except (FileNotFoundError, KeyError, ValueError):
            return 2
        try:
            return int(previous.metadata.get("attempt", 1)) + 1
        except (TypeError, ValueError):
            return 2

    def update_preparation(self, job_id: str, phase: str) -> JobRecord:
        """Advance one reserved submission without inventing scheduler acknowledgement."""
        record = self.store.get(job_id)
        if record.state is JobState.CANCELLED:
            raise RuntimeError(f"Submission {job_id} was cancelled.")
        metadata = {**dict(record.metadata), "submission_phase": phase}
        updated = record.with_updates(
            state=JobState.STAGING if phase == "staging" else JobState.PREPARING,
            metadata=metadata,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(updated)
        if record.metadata.get("submission_phase") != phase:
            self.store.append_event(
                job_id,
                state=updated.state.value,
                phase=phase,
                message=self._phase_message(phase),
            )
        return updated

    def fail_preparation(self, job_id: str, error: BaseException) -> JobRecord:
        """Persist an asynchronous pre-scheduler failure for ordinary job diagnostics."""
        record = self.store.get(job_id)
        if record.state is JobState.CANCELLED:
            return record
        context = DiagnosticContext((), "asynchronous job preparation", record.cluster)
        classified = DiagnosticClassifier().classify(error, context)
        updated = record.with_updates(
            state=JobState.FAILED,
            stderr=f"{error.__class__.__name__}: {error}\n",
            metadata={
                **dict(record.metadata),
                "failure_phase": "preparation",
                "failure_category": classified.category.value,
            },
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(updated)
        self.store.append_event(
            job_id,
            state=updated.state.value,
            phase=str(updated.metadata.get("failure_phase", "preparation")),
            message=("Background preparation failed; inspect the diagnostic and controller log."),
        )
        return updated

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
            previous_state = record.state
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "last_known_state": (
                        metadata.get("last_known_state", JobState.UNKNOWN.value)
                        if record.state is JobState.UNKNOWN
                        else record.state.value
                    ),
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
            if previous_state is not JobState.UNKNOWN:
                self.store.append_event(
                    job_id,
                    state=record.state.value,
                    message=(
                        "The scheduler could not be reached; the last known state was preserved."
                    ),
                    source="scheduler",
                )
            return record
        if state is not record.state:
            previous_state = record.state
            record = record.with_updates(
                state=state, updated_at_utc=datetime.now(timezone.utc).isoformat()
            )
            self.store.write(record)
            self.store.append_event(
                job_id,
                state=state.value,
                message=f"Scheduler state changed from {previous_state.value} to {state.value}.",
                source="scheduler",
            )
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

    def events(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Return the append-only framework/scheduler lifecycle facts for one job."""
        return self.store.events(job_id)

    def scientific_logs(self, job_id: str, *, tail: int | None = None) -> str:
        """Return only consumer/scientific output, without lifecycle annotations."""
        record = self.get(job_id, refresh=False)
        if record.scheduler_id is None:
            value = record.stdout + record.stderr
            return "\n".join(value.splitlines()[-tail:]) if tail else value
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        return scheduler.logs(record.scheduler_id, tail=tail)

    def logs(self, job_id: str, *, tail: int | None = None) -> str:
        """Return lifecycle, preparation and scientific logs as clearly labelled streams."""
        record = self.get(job_id)
        lifecycle = [self._format_event(value) for value in self.events(job_id)]
        remote = record.metadata.get("remote_state", {})
        remote = remote if isinstance(remote, Mapping) else {}
        heartbeat = remote.get("heartbeat_at_utc")
        if heartbeat:
            lifecycle.append(
                f"[{heartbeat}] [runtime] {record.state.value}: supervisor heartbeat observed."
            )
        observed = datetime.now(timezone.utc).isoformat()
        lifecycle.append(
            f"[{observed}] [observation] {record.state.value}: "
            + (
                "current provider observation completed; scientific progress requires "
                "application output or artifacts."
                if record.scheduler_id is not None
                else "local controller record is available; the scheduler has not acknowledged "
                "this job yet."
            )
        )
        parts = ["== LambdaForge lifecycle ==", *(lifecycle or ["No lifecycle events recorded."])]
        controller_log = self.store.root / "submissions" / job_id / "controller.log"
        controller = ""
        if controller_log.is_file() and not controller_log.is_symlink():
            try:
                controller = controller_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                controller = ""
        if controller.strip():
            controller_lines = controller.splitlines()
            parts.extend(
                (
                    "",
                    "== LambdaForge submission worker ==",
                    *(controller_lines[-tail:] if tail else controller_lines),
                )
            )
        scientific = self.scientific_logs(job_id, tail=tail)
        parts.extend(
            (
                "",
                "== Scientific output (consumer code) ==",
                scientific.rstrip() or "No scientific output has been emitted yet.",
            )
        )
        return "\n".join(parts).rstrip() + "\n"

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel through the provider and persist the transition."""
        record = self.get(job_id)
        if record.state.terminal:
            return record
        if record.scheduler_id is None:
            record = record.with_updates(
                state=JobState.CANCELLED,
                metadata={**dict(record.metadata), "cancelled_before_scheduler": True},
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            self.store.write(record)
            self.store.append_event(
                job_id,
                state=record.state.value,
                phase=str(record.metadata.get("submission_phase", "preparation")),
                message="Job cancelled before scheduler acknowledgement.",
            )
            return record
        profile = self.catalog.get(record.cluster)
        scheduler = self.factory.scheduler(profile, self.factory.transport(profile))
        scheduler.cancel(record.scheduler_id)
        record = record.with_updates(
            state=JobState.CANCELLED,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.store.write(record)
        self.store.append_event(
            job_id,
            state=record.state.value,
            message="Scheduler cancellation requested.",
            source="scheduler",
        )
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
                    previous.with_updates(
                        state=state,
                        metadata={
                            **dict(previous.metadata),
                            "remote_state": dict(state_value),
                            "last_refresh_at_utc": now,
                        },
                        updated_at_utc=now,
                    )
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
                if previous is None or previous.state is not state:
                    self.store.append_event(
                        job_id,
                        state=state.value,
                        message=(
                            "Durable supervisor state discovered during reconciliation."
                            if previous is None
                            else (
                                "Provider state changed from "
                                f"{previous.state.value} to {state.value}."
                            )
                        ),
                        source="provider",
                    )
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
        self.store.append_event(
            job_id,
            state=record.state.value,
            message=f"Job {operation} requested through the scheduler.",
            source="scheduler",
        )
        return record

    @staticmethod
    def _phase_message(phase: str) -> str:
        return {
            "validation": "Configuration validated; resolving the execution context.",
            "runtime": "Resolving a compatible remote Python and PyTorch/CUDA plan.",
            "bundle": "Building or reusing the content-addressed execution bundle.",
            "staging": "Staging the execution bundle and declared bounded inputs.",
            "environment": "Preparing or verifying the immutable remote environment.",
            "scheduler": "Environment ready; handing the job to the scheduler.",
        }.get(phase, f"Preparation entered phase {phase}.")

    @staticmethod
    def _format_event(value: Mapping[str, Any]) -> str:
        timestamp = str(value.get("timestamp_utc", "-"))
        phase = str(value.get("phase") or value.get("source") or "lifecycle")
        state = str(value.get("state", "unknown"))
        return f"[{timestamp}] [{phase}] {state}: {value.get('message', '')}"

    def retry(self, job_id: str, *, dry_run: bool = False) -> JobHandle:
        """Create a new auditable job from one terminal job's exact request."""
        previous = self.get(job_id)
        if previous.state not in {JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT}:
            raise ValueError(
                "Retry applies only to failed, cancelled or timed-out attempts. "
                "Use 'lf run CONFIG --rerun' for deliberate repetition after success."
            )
        retry_metadata = dict(previous.metadata)
        retry_metadata.pop("failure_phase", None)
        retry_metadata.pop("failure_category", None)
        retry_metadata["attempt"] = self._attempt_number(previous.job_id)
        if previous.metadata.get("submission_mode") == "asynchronous":
            source_config = self._retry_source_config(previous)
            if source_config is None:
                raise ValueError("The asynchronous submission has no source configuration path.")
            arguments = previous.metadata.get("run_arguments", ())
            if not isinstance(arguments, Sequence) or isinstance(
                arguments, (str, bytes, bytearray)
            ):
                raise TypeError("Persisted asynchronous run_arguments are invalid.")
            request = ResourceRequest.from_mapping(previous.resources)
            run_arguments = tuple(str(item) for item in arguments)
            if dry_run:
                from lambdaforge.controlplane.ControlPlane import ControlPlane

                handle, _ = ControlPlane(self.catalog, jobs=self).submit(
                    str(source_config),
                    cluster=previous.cluster,
                    resources=request,
                    dry_run=True,
                    run_arguments=run_arguments,
                    group_id=previous.group_id,
                )
                return handle
            from lambdaforge.controlplane.SubmissionService import SubmissionService

            return SubmissionService(self.catalog, self).enqueue(
                str(source_config),
                cluster=previous.cluster,
                resources=request,
                run_arguments=run_arguments,
                group_id=previous.group_id,
                retry_of=previous.job_id,
            )
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

    def _retry_source_config(self, previous: JobRecord) -> str | None:
        """Recover the controller-side config from current or 0.9.x async records."""
        configured = previous.metadata.get("source_config_path")
        if configured:
            return str(configured)
        request = self.store.root / "submissions" / previous.job_id / "request.json"
        if request.is_file() and not request.is_symlink():
            try:
                payload = json.loads(request.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, Mapping) and payload.get("config"):
                return str(payload["config"])
        return previous.config_path
