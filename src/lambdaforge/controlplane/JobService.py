"""Application service for persistent job operations."""

from __future__ import annotations

import inspect
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
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
from lambdaforge.work.failure import render_scientific_failures, scientific_failures


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
        profile = self._submission_profile(
            self.catalog.get(cluster),
            config_path=config_path,
            work_dir=work_dir,
        )
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
        if profile.transport == "local":
            assert profile.storage is not None
            record_metadata["local_workspace"] = profile.workspace
            record_metadata["local_storage"] = profile.storage.to_dict()
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
        profile, transport, scheduler = self._provider(record)
        try:
            state = scheduler.state(record.scheduler_id)
            provider_details = scheduler.details(record.scheduler_id)
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
        metadata = dict(record.metadata)
        metadata.pop("unreachable_error", None)
        metadata.pop("last_known_state", None)
        metadata["last_refresh_at_utc"] = datetime.now(timezone.utc).isoformat()
        if provider_details:
            metadata["remote_state"] = dict(provider_details)
        if profile.scheduler == "slurm" and state in {JobState.RUNNING, JobState.PAUSED}:
            progress_path = str(PurePosixPath(record.work_dir).parent / "progress.json")
            progress_result = transport.run(("cat", progress_path), timeout=5.0)
            if progress_result.returncode == 0:
                try:
                    progress = json.loads(progress_result.stdout)
                except (TypeError, ValueError):
                    progress = None
                if isinstance(progress, Mapping):
                    remote_state = metadata.get("remote_state", {})
                    remote_state = dict(remote_state) if isinstance(remote_state, Mapping) else {}
                    remote_state["progress"] = dict(progress)
                    metadata["remote_state"] = remote_state
        study = self._load_study_summary(record, transport)
        if study is not None:
            remote_state = metadata.get("remote_state", {})
            remote_state = dict(remote_state) if isinstance(remote_state, Mapping) else {}
            remote_state["study"] = study
            metadata["remote_state"] = remote_state
        if state is not record.state or metadata != record.metadata:
            previous_state = record.state
            record = record.with_updates(
                state=state,
                metadata=metadata,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            self.store.write(record)
            if state is not previous_state:
                self.store.append_event(
                    job_id,
                    state=state.value,
                    message=(
                        f"Scheduler state changed from {previous_state.value} to {state.value}."
                    ),
                    source="scheduler",
                )
                if state.terminal and state is not JobState.PLANNED:
                    try:
                        from lambdaforge.controlplane.StorageService import StorageService

                        compacted = StorageService(
                            self.catalog, self.factory, jobs=self
                        ).compact_job(
                            record.cluster, record.job_id, apply=True
                        )
                        metadata = {
                            **dict(record.metadata),
                            "workspace_compaction": {
                                "reclaimed_bytes": int(compacted["reclaimed_bytes"]),
                                "preserved": list(compacted["preserved"]),
                            },
                        }
                    except Exception as error:
                        metadata = {
                            **dict(record.metadata),
                            "workspace_compaction_warning": f"{type(error).__name__}: {error}",
                        }
                    record = record.with_updates(metadata=metadata)
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

    def events(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Return the append-only framework/scheduler lifecycle facts for one job."""
        return self.store.events(job_id)

    def scientific_logs(self, job_id: str, *, tail: int | None = None) -> str:
        """Return only consumer/scientific output, without lifecycle annotations."""
        record = self.get(job_id, refresh=False)
        if record.scheduler_id is None:
            value = record.stdout + record.stderr
            return "\n".join(value.splitlines()[-tail:]) if tail else value
        _profile, _transport, scheduler = self._provider(record)
        return scheduler.logs(record.scheduler_id, tail=tail)

    def scientific_result(self, job_id: str) -> dict[str, Any] | None:
        """Read the bounded structured Work result owned by one provider Job, if present."""
        return self._scientific_result(self.get(job_id, refresh=False))

    def study(self, job_id: str) -> dict[str, Any] | None:
        """Return one bounded live/terminal study snapshot for machine clients and the TUI."""
        record = self.get(job_id, refresh=False)
        _profile, transport, _scheduler = self._provider(record)
        return self._load_study_summary(record, transport)

    def study_run(
        self,
        job_id: str,
        run_key: str,
        *,
        tail: int | None = None,
        curve_points: int = 80,
    ) -> dict[str, Any]:
        """Read one internal Run's isolated log and bounded scalar learning curves."""
        if re.fullmatch(r"trial-\d{5}-seed-(?:none|n?\d+)", run_key) is None:
            raise ValueError("Invalid study Run key.")
        if tail is not None and tail < 1:
            raise ValueError("Run log tail must be positive or null.")
        if not 10 <= curve_points <= 500:
            raise ValueError("curve_points must be between 10 and 500.")
        record = self.get(job_id, refresh=False)
        _profile, transport, _scheduler = self._provider(record)
        summary = self._load_study_summary(record, transport)
        if summary is None:
            raise KeyError(f"Job {job_id} has no study telemetry.")
        selected: dict[str, Any] | None = None
        candidate_parameters: dict[str, Any] = {}
        for candidate in summary.get("candidates", ()):
            if not isinstance(candidate, Mapping):
                continue
            for run in candidate.get("runs", ()):
                if isinstance(run, Mapping) and run.get("key") == run_key:
                    selected = dict(run)
                    candidate_parameters = dict(candidate.get("parameters", {}))
                    break
            if selected is not None:
                break
        if selected is None:
            raise KeyError(f"Unknown study Run {run_key!r} in Job {job_id}.")
        log_path = self._owned_study_path(record, selected.get("log_path"))
        run_dir = self._owned_study_path(record, selected.get("run_dir"))
        result_path = run_dir / "result.json" if run_dir is not None else None
        metric_paths = tuple(
            path
            for path in (
                self._owned_study_path(record, selected.get("metrics_path")),
                self._owned_study_path(record, selected.get("training_metrics_path")),
            )
            if path is not None
        )
        log_text, log_truncated = self._read_study_log(transport, log_path, tail=tail)
        failure = selected.get("failure")
        failure = dict(failure) if isinstance(failure, Mapping) else None
        if result_path is not None and failure is not None:
            result_text, _result_truncated = self._read_bounded_file(
                transport, result_path, limit=8 * 1024 * 1024
            )
            if result_text:
                try:
                    persisted = json.loads(result_text)
                except (json.JSONDecodeError, TypeError):
                    persisted = None
                if isinstance(persisted, Mapping) and isinstance(
                    persisted.get("failure"), Mapping
                ):
                    failure = dict(persisted["failure"])
        if failure is not None:
            diagnostic = failure.get("diagnostic")
            diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
            failure.setdefault("phase", diagnostic.get("operation") or "Work.run")
            failure["result_path"] = str(result_path) if result_path is not None else None
        observations: list[dict[str, Any]] = []
        chart_filter: dict[str, list[str]] = {}
        metrics_truncated = False
        for path in metric_paths:
            text, truncated = self._read_bounded_file(transport, path, limit=16 * 1024 * 1024)
            metrics_truncated = metrics_truncated or truncated
            for line in text.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, Mapping):
                    continue
                if value.get("kind") == "chart-filter":
                    chart_filter = {
                        field: [str(item) for item in value.get(field, ())]
                        for field in ("include", "exclude")
                        if isinstance(value.get(field), list)
                    }
                    continue
                metric = value.get("value")
                step = value.get("step")
                if (
                    isinstance(metric, int | float)
                    and not isinstance(metric, bool)
                    and isinstance(step, int)
                ):
                    name = str(value.get("name", ""))
                    key = f"{value['split']}_{name}" if value.get("split") else name
                    observations.append({"name": key, "step": step, "value": float(metric)})
        series: dict[str, list[dict[str, float | int]]] = {}
        for observation in observations:
            series.setdefault(str(observation["name"]), []).append(
                {"step": int(observation["step"]), "value": float(observation["value"])}
            )
        objective = summary.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        objective_metric = str(objective.get("metric", ""))
        objective_mode = str(objective.get("mode", "max"))
        objective_values = series.get(objective_metric, ())
        best_observation = (
            (min if objective_mode == "min" else max)(
                objective_values, key=lambda value: float(value["value"])
            )
            if objective_values
            else None
        )
        best_step = (
            int(best_observation["step"])
            if isinstance(best_observation, Mapping)
            else selected.get("best_step")
        )
        best_objective = (
            float(best_observation["value"])
            if isinstance(best_observation, Mapping)
            else selected.get("best_objective")
        )
        normalized_series: dict[str, Sequence[dict[str, float | int]]] = {
            name: self._downsample_curve(values, curve_points, preserve_step=best_step)
            for name, values in sorted(series.items())
        }
        latest = {
            name: values[-1]["value"] for name, values in normalized_series.items() if values
        }
        return {
            "study_run_version": 1,
            "job_id": job_id,
            "cluster": record.cluster,
            "key": run_key,
            "trial": selected.get("trial"),
            "seed": selected.get("seed"),
            "state": selected.get("state"),
            "gpu_index": selected.get("gpu_index"),
            "gpu_token": selected.get("gpu_token"),
            "parameters": candidate_parameters or dict(selected.get("parameters", {})),
            "objective": dict(objective),
            "best_step": best_step,
            "best_objective": best_objective,
            "chart_filter": chart_filter,
            "started_at_utc": selected.get("started_at_utc"),
            "finished_at_utc": selected.get("finished_at_utc"),
            "duration_seconds": selected.get("duration_seconds"),
            "failure": failure,
            "prune_reason": selected.get("prune_reason"),
            "latest_metrics": {**dict(selected.get("latest_metrics", {})), **latest},
            "curves": normalized_series,
            "log": log_text,
            "log_truncated": log_truncated,
            "metrics_truncated": metrics_truncated,
            "paths": {
                "run_dir": selected.get("run_dir"),
                "log": str(log_path) if log_path is not None else None,
                "metrics": [str(value) for value in metric_paths],
                "result": str(result_path) if result_path is not None else None,
            },
        }

    @staticmethod
    def _load_study_summary(record: JobRecord, transport: Any) -> dict[str, Any] | None:
        path = str(PurePosixPath(record.work_dir).parent / "study" / "summary.json")
        linked = transport.run(("test", "-L", path), timeout=10.0)
        if linked.returncode == 0:
            return None
        loaded = transport.run(("cat", path), timeout=15.0)
        if loaded.returncode != 0 or not loaded.stdout.strip():
            return None
        if len(loaded.stdout.encode("utf-8")) > 8 * 1024 * 1024:
            raise RuntimeError(f"Study telemetry exceeds the 8 MiB summary limit: {path}")
        try:
            value = json.loads(loaded.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(f"Corrupt study telemetry for {record.job_id}: {path}") from error
        if not isinstance(value, dict) or value.get("study_telemetry_version") != 1:
            raise RuntimeError(f"Unsupported study telemetry for {record.job_id}: {path}")
        return value

    @staticmethod
    def _owned_study_path(record: JobRecord, value: Any) -> PurePosixPath | None:
        if not isinstance(value, str) or not value:
            return None
        root = PurePosixPath(record.work_dir)
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe study evidence path for {record.job_id}.")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Study evidence escaped the Job work root for {record.job_id}."
            ) from error
        return path

    @staticmethod
    def _read_study_log(
        transport: Any, path: PurePosixPath | None, *, tail: int | None
    ) -> tuple[str, bool]:
        if path is None:
            return "", False
        if tail is not None:
            result = transport.run(("tail", "-n", str(tail), str(path)), timeout=30.0)
            return (result.stdout if result.returncode == 0 else ""), False
        return JobService._read_bounded_file(transport, path, limit=8 * 1024 * 1024)

    @staticmethod
    def _read_bounded_file(
        transport: Any, path: PurePosixPath, *, limit: int
    ) -> tuple[str, bool]:
        linked = transport.run(("test", "-L", str(path)), timeout=10.0)
        if linked.returncode == 0:
            return "", False
        size_result = transport.run(("wc", "-c", str(path)), timeout=10.0)
        if size_result.returncode != 0:
            return "", False
        try:
            size = int(size_result.stdout.strip().split()[0])
        except (ValueError, IndexError):
            return "", False
        command = (
            ("cat", str(path))
            if size <= limit
            else ("tail", "-c", str(limit), str(path))
        )
        result = transport.run(command, timeout=30.0)
        return (result.stdout if result.returncode == 0 else ""), size > limit

    @staticmethod
    def _downsample_curve(
        values: Sequence[Mapping[str, Any]],
        maximum: int,
        *,
        preserve_step: int | None = None,
    ) -> Sequence[dict[str, float | int]]:
        ordered = sorted(
            ({"step": int(value["step"]), "value": float(value["value"])} for value in values),
            key=lambda value: value["step"],
        )
        # Last writer wins for duplicate metric/step observations (for example final adoption).
        unique = {int(value["step"]): value for value in ordered}
        compact = [unique[key] for key in sorted(unique)]
        if len(compact) <= maximum:
            return compact
        indices = {
            round(index * (len(compact) - 1) / (maximum - 1)) for index in range(maximum)
        }
        preserved = next(
            (
                index
                for index, value in enumerate(compact)
                if preserve_step is not None and int(value["step"]) == preserve_step
            ),
            None,
        )
        if preserved is not None and preserved not in indices:
            indices.add(preserved)
            removable = sorted(
                indices - {0, len(compact) - 1, preserved},
                key=lambda index: (abs(index - preserved), index),
            )
            if removable:
                indices.remove(removable[0])
        return [compact[index] for index in sorted(indices)]

    def _scientific_result(self, record: JobRecord) -> dict[str, Any] | None:
        _profile, transport, _scheduler = self._provider(record)
        exact = str(PurePosixPath(record.work_dir).parent / "result.json")

        def load(path: str, *, require_job_match: bool) -> dict[str, Any] | None:
            linked = transport.run(("test", "-L", path), timeout=15.0)
            if linked.returncode == 0:
                return None
            loaded = transport.run(("cat", path), timeout=30.0)
            if loaded.returncode != 0 or not loaded.stdout.strip():
                return None
            if len(loaded.stdout.encode("utf-8")) > 32 * 1024 * 1024:
                raise RuntimeError(f"Scientific result exceeds the 32 MiB diagnostic limit: {path}")
            try:
                value = json.loads(loaded.stdout)
            except (ValueError, TypeError) as error:
                raise RuntimeError(
                    f"Corrupt scientific result for {record.job_id}: {path}"
                ) from error
            if not isinstance(value, Mapping) or value.get("execution_result_version") != 1:
                raise RuntimeError(f"Unsupported scientific result for {record.job_id}: {path}")
            if require_job_match:
                runs = value.get("runs", ())
                if not isinstance(runs, Sequence) or not any(
                    isinstance(run, Mapping) and run.get("job_id") == record.job_id
                    for run in runs
                ):
                    return None
            failures = scientific_failures(value, result_path=path)
            return {
                "path": path,
                "status": value.get("status"),
                "failure": failures[0] if failures else None,
                "failures": list(failures),
                "result": dict(value),
            }

        current = load(exact, require_job_match=False)
        if current is not None:
            return current
        if not record.state.terminal:
            # Legacy discovery is only needed once an old Work has finished. Avoid a remote
            # directory scan on every live-log refresh while no aggregate result can exist yet.
            return None

        legacy_root = str(PurePosixPath(record.work_dir) / ".lambdaforge" / "runs")
        discovered = transport.run(
            (
                "find",
                legacy_root,
                "-mindepth",
                "3",
                "-maxdepth",
                "3",
                "-type",
                "f",
                "-name",
                "result.json",
                "-print",
            ),
            timeout=15.0,
        )
        if discovered.returncode != 0:
            return None
        for path in discovered.stdout.splitlines()[:64]:
            selected = path.strip()
            if selected and selected != exact:
                legacy = load(selected, require_job_match=True)
                if legacy is not None:
                    return legacy
        return None

    def log_report(
        self,
        job_id: str,
        *,
        tail: int | None = None,
        include_traceback: bool = False,
    ) -> dict[str, Any]:
        """Return human log text plus its structured terminal scientific failure."""
        record = self.get(job_id)
        lifecycle = [self._format_event(value) for value in self.events(job_id)]
        remote = record.metadata.get("remote_state", {})
        remote = remote if isinstance(remote, Mapping) else {}
        heartbeat = remote.get("heartbeat_at_utc")
        if heartbeat:
            lifecycle.append(
                f"[{heartbeat}] [runtime] {record.state.value}: supervisor heartbeat observed."
            )
        provider_message = remote.get("message")
        if provider_message:
            lifecycle.append(
                f"[{remote.get('updated_at_utc', record.updated_at_utc)}] [runtime] "
                f"{remote.get('state', record.state.value)}: {provider_message}"
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
        result = (
            self._scientific_result(record)
            if record.scheduler_id is not None
            and (record.state.terminal or record.state is JobState.UNKNOWN)
            else None
        )
        failures = result.get("failures", ()) if result is not None else ()
        if isinstance(failures, Sequence):
            failure_section = render_scientific_failures(
                tuple(value for value in failures if isinstance(value, Mapping)),
                existing_output=scientific,
                include_traceback=include_traceback,
            )
            if failure_section:
                parts.extend(("", failure_section))
        text = "\n".join(parts).rstrip() + "\n"
        return {
            "job_id": record.job_id,
            "cluster": record.cluster,
            "state": record.state.value,
            "text": text,
            "failure": result.get("failure") if result is not None else None,
            "failures": list(failures) if isinstance(failures, Sequence) else [],
            "result_path": result.get("path") if result is not None else None,
        }

    def logs(
        self,
        job_id: str,
        *,
        tail: int | None = None,
        include_traceback: bool = False,
    ) -> str:
        """Return labelled lifecycle/science streams and any persisted terminal failure."""
        return str(
            self.log_report(
                job_id,
                tail=tail,
                include_traceback=include_traceback,
            )["text"]
        )

    def cancel(self, job_id: str) -> JobRecord:
        """Cancel through the provider and persist the transition."""
        record = self.get(job_id)
        reconciling_cancelled = (
            record.state is JobState.CANCELLED
            and record.scheduler == "local"
            and record.scheduler_id is not None
        )
        if record.state.terminal and not reconciling_cancelled:
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
        _profile, _transport, scheduler = self._provider(record)
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
        profile, _transport, scheduler = self._provider(record)
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

    def job_root(self, record: JobRecord) -> str:
        """Return the exact provider Job root recorded for one durable local Job.

        Current records contain an absolute local storage descriptor.  The source-relative
        fallback intentionally supports Jobs submitted before that descriptor existed, when the
        detached submission worker resolved the built-in local workspace from the YAML directory.
        """
        profile = self.catalog.get(record.cluster)
        if profile.transport != "local":
            assert profile.storage is not None
            return profile.storage.job_root
        storage = record.metadata.get("local_storage")
        if isinstance(storage, Mapping) and storage.get("run_root"):
            root = Path(str(storage["run_root"])).expanduser()
            if root.is_absolute():
                return str(root.resolve())
        work = Path(record.work_dir).expanduser()
        candidates: list[Path] = []
        if work.is_absolute():
            candidates.append(work)
        else:
            source = record.metadata.get("source_config_path") or record.config_path
            if source:
                candidates.append(Path(str(source)).expanduser().resolve().parent / work)
            candidates.append(Path.cwd().resolve() / work)
        valid: list[Path] = []
        for candidate in candidates:
            selected = candidate.resolve(strict=False)
            if selected.name == "work" and selected.parent.name == record.job_id:
                valid.append(selected.parent.parent)
        if not valid:
            raise RuntimeError(
                f"Persisted local work directory is not owned by {record.job_id}: "
                f"{record.work_dir}"
            )
        for root in valid:
            if (root / record.job_id / "state.json").is_file():
                return str(root)
        return str(valid[0])

    def _provider(self, record: JobRecord) -> tuple[Any, Any, Any]:
        """Resolve a provider using per-Job local roots and current remote credentials."""
        profile = self.catalog.get(record.cluster)
        if profile.transport == "local":
            storage = record.metadata.get("local_storage")
            workspace = str(record.metadata.get("local_workspace") or profile.workspace)
            if isinstance(storage, Mapping):
                local_storage = ClusterStoragePolicy.from_mapping(storage, workspace=workspace)
            else:
                assert profile.storage is not None
                local_storage = replace(profile.storage, run_root=self.job_root(record))
            profile = replace(profile, workspace=workspace, storage=local_storage)
        transport = self.factory.transport(profile)
        return profile, transport, self.factory.scheduler(profile, transport)

    @classmethod
    def _submission_profile(
        cls,
        profile: Any,
        *,
        config_path: str | None,
        work_dir: str | Path,
    ) -> Any:
        """Anchor built-in local storage to the consumer project before detaching."""
        if profile.transport != "local":
            return profile
        start = (
            Path(config_path).expanduser().resolve().parent
            if config_path is not None
            else Path(work_dir).expanduser().resolve()
        )
        base = next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            start,
        )

        def anchored(value: str | None) -> str | None:
            if value is None:
                return None
            selected = Path(value).expanduser()
            resolved = selected.resolve() if selected.is_absolute() else (base / selected).resolve()
            return str(resolved)

        assert profile.storage is not None
        storage = replace(
            profile.storage,
            state_root=str(anchored(profile.storage.state_root)),
            cache_root=str(anchored(profile.storage.cache_root)),
            run_root=str(anchored(profile.storage.run_root)),
            dataset_root=anchored(profile.storage.dataset_root),
        )
        return replace(profile, workspace=str(anchored(profile.workspace)), storage=storage)

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
        """Recover the controller-side Work config recorded for asynchronous retry."""
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
