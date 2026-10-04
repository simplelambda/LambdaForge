"""Prepared local CPU shards through the existing durable ProcessScheduler.

This is not remote preparation and does not authorize GPU work. No environment is installed,
no input is replicated and no scientific planner is created by this adapter. It provides an
actual provider integration for an already verified local execution stratum.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.Fleet import ClusterHealth, FleetMember
from lambdaforge.controlplane.FleetPlacement import ClusterOffer, ExecutionEquivalence
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.StudyCoordinator import (
    RemoteObservation,
    ShardRejectedError,
    StudyShard,
)
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json


class PreparedCpuShardExecutor:
    """Durable direct CPU provider; ambiguous acceptance never triggers another submit.

    ``invocation`` supplies exact already-prepared native specifications. Equivalence belongs
    to the caller's verified preparation, not to an observation of physical hardware. Pending
    Jobs consume this executor's cap until terminal evidence is actually read.
    """

    def __init__(
        self,
        profile: ClusterProfile,
        member: FleetMember,
        *,
        root: Path,
        resources: ResourceRequest,
        equivalence: ExecutionEquivalence,
        invocation: Callable[[str], Mapping[str, Any]],
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        if profile.transport != "local" or profile.scheduler != "local":
            raise ValueError("Prepared CPU executor supports local direct profiles only.")
        if (
            profile.environment != "existing"
            or Path(profile.python).absolute() != Path(sys.executable).absolute()
        ):
            raise ValueError(
                "Prepared local CPU execution requires the verified current interpreter."
            )
        if resources.gpu_count or profile.name != member.cluster:
            raise ValueError("Prepared CPU executor cannot claim GPU or another member's work.")
        self.root = root.absolute()
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError("Prepared shard storage cannot be symlinked.")
        self.profile = profile
        self.member = member
        self.resources = resources
        self.equivalence = equivalence
        self.invocation = invocation
        factory = factory or ControlPlaneFactory()
        self.scheduler = factory.scheduler(profile, factory.transport(profile))

    def _directory(self, shard: StudyShard) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", shard.shard_id) or shard.cluster != self.profile.name:
            raise ValueError("Shard must have an exact owned digest and executor target.")
        directory = self.root / shard.shard_id
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("Owned shard directory is symlinked.")
        return directory

    def submit(self, shard: StudyShard) -> str:
        """Persist intent before using ProcessScheduler; never silently replay an intent."""
        directory = self._directory(shard)
        parallelism = len(shard.leases)
        if parallelism > self.resources.cpu_cores or (
            self.member.max_runs is not None and parallelism > self.member.max_runs
        ):
            raise ShardRejectedError("Prepared shard exceeds its local CPU/Run cap.")
        if any(
            lease.run.requires_gpu
            or lease.run.equivalence != self.equivalence
            or lease.attempt != 1
            for lease in shard.leases
        ):
            raise ShardRejectedError("Only fresh equivalent CPU Runs have prepared ownership.")
        invocations = {
            lease.run.key: dict(self.invocation(lease.run.key)) for lease in shard.leases
        }
        from lambdaforge.work.shard import prepare_concrete_shard

        try:
            prepare_concrete_shard(
                shard,
                invocations,
                root=directory / "worker",
                resources=self.resources,
                verified_equivalence=self.equivalence,
                parallelism=parallelism,
            )
        except (ValueError, TypeError, OSError) as error:
            raise ShardRejectedError(f"Prepared invocation preflight failed: {error}") from error
        manifest = {
            "manifest_version": 1,
            "shard": shard.to_dict(),
            "invocations": invocations,
            "root": str(directory / "worker"),
            "resources": self.resources.to_dict(),
            "equivalence": self.equivalence.to_dict(),
            "parallelism": parallelism,
        }
        # Validate a strict portable manifest before touching any scheduler.
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        if len(json.dumps(manifest).encode("utf-8")) > 8 * 1024**2:
            raise ShardRejectedError("Prepared shard manifest exceeds the control-envelope limit.")
        job_id = "job-fleet-" + shard.shard_id
        with CrossProcessFileLock(
            directory / ".submit.lock",
            shared=False,
            timeout_seconds=5,
            poll_interval_seconds=0.05,
        ):
            path = directory / "manifest.json"
            intent = directory / "submission.json"
            if path.is_symlink() or intent.is_symlink():
                raise ValueError("Owned shard submission metadata is symlinked.")
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("Prepared shard manifest is immutable.")
            if intent.exists():
                previous = json.loads(intent.read_text())
                if previous.get("job_id") != job_id:
                    raise ValueError("Shard submission belongs to another Job.")
                if previous.get("acknowledged"):
                    return job_id
                raise RuntimeError("Ambiguous shard acceptance must be reconciled, not submitted.")
            atomic_write_json(path, manifest)
            atomic_write_json(intent, {"job_id": job_id, "acknowledged": False})
            submission = self.scheduler.submit(
                (self.profile.python, "-m", "lambdaforge.work.shard", str(path)),
                self.resources,
                work_dir=directory,
                job_id=job_id,
            )
            if submission.scheduler_id != job_id:
                raise RuntimeError("Process provider did not acknowledge the exact shard Job.")
            atomic_write_json(intent, {"job_id": job_id, "acknowledged": True})
        return job_id

    def observe(self, shard: StudyShard, job_id: str | None) -> Sequence[RemoteObservation]:
        """Read one outcome batch; missing evidence or terminal Job alone does not prove loss."""
        directory = self._directory(shard)
        expected_job = "job-fleet-" + shard.shard_id
        if job_id is not None and job_id != expected_job:
            raise ValueError("Provider Job does not belong to this shard.")
        path = directory / "worker" / "worker.json"
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Owned worker evidence is symlinked.")
        state = json.loads(path.read_text()) if path.exists() else {}
        if state and state.get("manifest", {}).get("shard") != shard.to_dict():
            raise ValueError("Worker evidence does not belong to its exact leased shard.")
        try:
            provider_state = self.scheduler.state(expected_job)
        except (RuntimeError, OSError):
            provider_state = JobState.UNKNOWN
        observed = (
            "running"
            if provider_state is JobState.RUNNING
            else "queued"
            if provider_state in {JobState.CREATED, JobState.STAGING, JobState.QUEUED}
            else "unknown_remote"
        )
        results = state.get("results", {})
        for lease in shard.leases:
            envelope = results.get(lease.run.key)
            if envelope is None:
                continue
            native = envelope.get("result", {})
            invocation = self.invocation(lease.run.key)
            result_path = Path(str(native.get("run_dir", "")))
            expected_root = directory / "worker" / "execution"
            if (
                native.get("execution_id") != invocation["execution_id"]
                or native.get("seed") != lease.run.seed
                or native.get("attempt_number") != lease.attempt
                or (native.get("trial") or {}).get("index") != invocation["trial_index"]
                or not result_path.is_absolute()
                or not result_path.is_relative_to(expected_root)
                or any(path.is_symlink() for path in (result_path, *result_path.parents))
            ):
                raise ValueError("Native worker result differs from its exact owned invocation.")
        return tuple(
            RemoteObservation(
                lease.run.key,
                lease.attempt,
                lease.lease_id,
                results[lease.run.key]["state"] if lease.run.key in results else observed,
                result=results.get(lease.run.key),
            )
            for lease in shard.leases
        )

    def offer(self, records: Sequence[Mapping[str, Any]]) -> ClusterOffer:
        """Apply local hard caps before attesting additional CPU dispatch capacity."""
        active = [
            record
            for record in records
            if record.get("state") in {"leased", "queued", "running", "unknown_remote"}
            and record["attempts"][-1]["cluster"] == self.member.cluster
        ]
        cap = min(self.resources.cpu_cores, self.member.max_runs or self.resources.cpu_cores)
        slots = max(0, cap - len(active))
        jobs = {record["attempts"][-1]["shard_id"] for record in active}
        if self.member.max_inflight_jobs is not None and len(jobs) >= self.member.max_inflight_jobs:
            slots = 0
        now = time.time()
        acknowledged = []
        for record in active:
            attempt = record["attempts"][-1]
            receipt = self.root / attempt["shard_id"] / "submission.json"
            if receipt.is_symlink() or receipt.parent.is_symlink():
                raise ValueError("Owned executor receipt is symlinked.")
            if receipt.is_file():
                value = json.loads(receipt.read_text())
                if value == {
                    "job_id": "job-fleet-" + attempt["shard_id"],
                    "acknowledged": True,
                }:
                    acknowledged.append(attempt["lease_id"])
        return ClusterOffer(
            self.member.cluster,
            ClusterHealth.ONLINE,
            "local",
            "auto",
            now,
            now + 10,
            slots=slots,
            equivalence=self.equivalence,
            environment_ready=True,
            inputs_ready=True,
            local_admission_verified=True,
            acknowledged_leases=tuple(acknowledged),
            diagnostics={"authority": "prepared-local-cpu-process-provider", "gpu_support": False},
        )
