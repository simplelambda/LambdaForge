"""Prepared concrete shards through the existing provider and preparation boundaries.

Local existing CPU invocations keep their verified direct-provider path. Passing ControlPlane
uses ordinary immutable bundle/environment/input preparation and JobService for remote providers.
This adapter never creates a planner or infers GPU offers from unallocated physical devices.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.Fleet import ClusterHealth, FleetMember
from lambdaforge.controlplane.FleetPlacement import ClusterOffer, ExecutionEquivalence
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.PreparedWork import PreparedWork
from lambdaforge.controlplane.ShardPreparation import prepared_input_bindings
from lambdaforge.controlplane.StudyCoordinator import (
    RemoteObservation,
    ShardRejectedError,
    StudyShard,
)
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json


class PreparedShardExecutor:
    """One provider adapter; ambiguous acceptance never triggers another submit.

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
        control_plane: ControlPlane | None = None,
        allocation_id: str | None = None,
        discover_equivalence: bool = False,
    ) -> None:
        if control_plane is None and (profile.transport != "local" or profile.scheduler != "local"):
            raise ValueError("Prepared CPU executor supports local direct profiles only.")
        if control_plane is None and (
            profile.environment != "existing"
            or Path(profile.python).absolute() != Path(sys.executable).absolute()
        ):
            raise ValueError(
                "Prepared local CPU execution requires the verified current interpreter."
            )
        if (resources.gpu_count and control_plane is None) or profile.name != member.cluster:
            raise ValueError("Prepared CPU executor cannot claim GPU or another member's work.")
        if control_plane is not None and control_plane.catalog.get(member.cluster) != profile:
            raise ValueError("Prepared executor must use its exact ControlPlane profile.")
        self.root = root.absolute()
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError("Prepared shard storage cannot be symlinked.")
        self.profile = profile
        self.member = member
        self.resources = resources
        self.equivalence = equivalence
        self.invocation = invocation
        self.control_plane = control_plane
        if allocation_id is not None and (
            control_plane is None or re.fullmatch(r"[a-f0-9]{64}", allocation_id) is None
        ):
            raise ValueError("A persistent allocation requires ControlPlane and an exact identity.")
        if member.max_gpus is not None and resources.gpu_count > member.max_gpus:
            raise ValueError("Prepared allocation exceeds the member GPU cap.")
        self.allocation_id = allocation_id
        if discover_equivalence and allocation_id is None:
            raise ValueError("Equivalence discovery requires a persistent prepared allocation.")
        self.discover_equivalence = discover_equivalence
        self._run_observations: dict[str, dict[str, Any]] = {}
        factory = (
            control_plane.factory if control_plane is not None else factory or ControlPlaneFactory()
        )
        self.scheduler = (
            None
            if control_plane is not None
            else factory.scheduler(profile, factory.transport(profile))
        )

    def _directory(self, shard: StudyShard) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", shard.shard_id) or shard.cluster != self.profile.name:
            raise ValueError("Shard must have an exact owned digest and executor target.")
        directory = self.root / shard.shard_id
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("Owned shard directory is symlinked.")
        return directory

    def submit(self, shard: StudyShard) -> str:
        """Persist intent before using ProcessScheduler; never silently replay an intent."""
        if self.allocation_id is not None:
            return self._submit_allocated(shard)
        directory = self._directory(shard)
        parallelism = len(shard.leases)
        if parallelism > self.resources.cpu_cores or (
            self.member.max_runs is not None and parallelism > self.member.max_runs
        ):
            raise ShardRejectedError("Prepared shard exceeds its local CPU/Run cap.")
        if any(
            lease.run.requires_gpu != bool(self.resources.gpu_count)
            or lease.run.equivalence != self.equivalence
            or lease.attempt != 1
            for lease in shard.leases
        ):
            raise ShardRejectedError("Only fresh equivalent Runs have prepared ownership.")
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
        manifest: dict[str, Any] = {
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
            if self.control_plane is not None:
                receipt: dict[str, Any] = {"job_id": job_id, "acknowledged": False}

                def entrypoint(prepared: PreparedWork) -> Sequence[str]:
                    if prepared.dry_run or prepared.job_id != job_id:
                        raise ValueError("Shard preparation needs its exact non-preview Job.")
                    sources = {value["source"] for value in invocations.values()}
                    if len(sources) != 1:
                        raise ShardRejectedError("One shard must share a prepared source bundle.")
                    try:
                        bindings = prepared_input_bindings(
                            Path(next(iter(sources))),
                            prepared,
                            self.equivalence,
                            invocations=invocations,
                        )
                    except (ValueError, TypeError, OSError) as error:
                        raise ShardRejectedError(f"Bundle equivalence failed: {error}") from error
                    # Operational paths change only at this already-prepared boundary. Scientific
                    # parameters stay authored; the worker verifies content before binding paths.
                    remote_path = str(PurePosixPath(prepared.work_dir) / ".fleet-shard.json")
                    worker_root = str(PurePosixPath(prepared.work_dir) / "worker")
                    values = {
                        key: {**value, "source": prepared.config}
                        for key, value in invocations.items()
                    }
                    remote_manifest = {
                        **manifest,
                        "root": worker_root,
                        "invocations": values,
                        "input_bindings": bindings,
                    }
                    encoded = json.dumps(remote_manifest, allow_nan=False)
                    if len(encoded.encode("utf-8")) > 8 * 1024**2:
                        raise ShardRejectedError("Prepared manifest exceeds the envelope limit.")
                    upload = directory / "prepared-manifest.json"
                    atomic_write_json(upload, remote_manifest)
                    prepared.transport.put(upload, remote_path)
                    receipt.update({"worker_root": worker_root, "python": prepared.python})
                    atomic_write_json(intent, receipt)
                    return ("lambdaforge.work.shard", remote_path)

                handle, _bundle = self.control_plane.submit(
                    next(iter(invocations.values()))["source"],
                    cluster=self.profile.name,
                    resources=self.resources,
                    reserved_job_id=job_id,
                    allow_duplicate=True,
                    entrypoint_builder=entrypoint,
                )
                if handle.job_id != job_id or not handle.scheduler_id:
                    raise RuntimeError("Provider did not acknowledge the exact shard Job.")
                receipt.update({"acknowledged": True, "scheduler_id": handle.scheduler_id})
                atomic_write_json(intent, receipt)
                return job_id
            assert self.scheduler is not None
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
        if self.allocation_id is not None:
            return self._observe_allocated(shard, job_id)
        directory = self._directory(shard)
        expected_job = "job-fleet-" + shard.shard_id
        if job_id is not None and job_id != expected_job:
            raise ValueError("Provider Job does not belong to this shard.")
        worker_root: Path | PurePosixPath = directory / "worker"
        if self.control_plane is not None:
            receipt_path = directory / "submission.json"
            if receipt_path.is_symlink():
                raise ValueError("Owned executor receipt is symlinked.")
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            state: dict[str, Any] = {"results": {}}
            if receipt.get("worker_root") and receipt.get("python"):
                worker_root = PurePosixPath(receipt["worker_root"])
                transport = self.control_plane.factory.transport(self.profile)
                for offset in range(0, len(shard.leases), 64):
                    batch = shard.leases[offset : offset + 64]
                    response = transport.run(
                        (
                            *self.profile.command_prefix,
                            receipt["python"],
                            "-m",
                            "lambdaforge.work.shard",
                            "--observe",
                            str(worker_root),
                            "--keys",
                            *(lease.run.key for lease in batch),
                        ),
                        timeout=30,
                    )
                    if response.returncode:
                        # Transport failure is absence of evidence, never an owned Run loss.
                        continue
                    if len(response.stdout.encode("utf-8")) > 8 * 1024**2:
                        raise ValueError("Provider observation exceeds the control-envelope limit.")
                    payload = json.loads(response.stdout)
                    if payload["shard_id"] not in {None, shard.shard_id}:
                        raise ValueError("Worker observation belongs to another shard.")
                    if payload["results"] and payload["shard_id"] != shard.shard_id:
                        raise ValueError("Published results need an exact observed shard identity.")
                    if set(payload["results"]) - {lease.run.key for lease in batch}:
                        raise ValueError("Observation contains another batch's results.")
                    state["results"].update(payload["results"])
        else:
            path = directory / "worker" / "worker.json"
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("Owned worker evidence is symlinked.")
            state = json.loads(path.read_text()) if path.exists() else {}
            if state and state.get("manifest", {}).get("shard") != shard.to_dict():
                raise ValueError("Worker evidence does not belong to its exact leased shard.")
        try:
            if self.control_plane is not None:
                provider_state = self.control_plane.jobs.get(
                    expected_job, include_study=False
                ).state
            else:
                assert self.scheduler is not None
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
            result_path = (
                PurePosixPath(str(native.get("run_dir", "")))
                if self.control_plane is not None
                else Path(str(native.get("run_dir", "")))
            )
            expected_root = worker_root / "execution"
            if (
                native.get("execution_id") != invocation["execution_id"]
                or native.get("seed") != lease.run.seed
                or native.get("attempt_number") != lease.attempt
                or (native.get("trial") or {}).get("index") != invocation["trial_index"]
                or not result_path.is_absolute()
                or not result_path.is_relative_to(expected_root)
                or (
                    self.control_plane is None
                    and any(Path(path).is_symlink() for path in (result_path, *result_path.parents))
                )
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
        if self.allocation_id is not None:
            return self._allocation_offer()
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
                if (
                    value.get("job_id") == "job-fleet-" + attempt["shard_id"]
                    and value.get("acknowledged") is True
                ):
                    acknowledged.append(attempt["lease_id"])
        return ClusterOffer(
            self.member.cluster,
            ClusterHealth.ONLINE,
            self.profile.scheduler,
            self.profile.gpu_access.effective_mode(self.profile.scheduler),
            now,
            now + 10,
            # GPU preparation/launch is not a live allocation. Until the persistent member
            # allocation authority is integrated, never turn physical inventory into an offer.
            slots=0 if self.resources.gpu_count else slots,
            equivalence=self.equivalence,
            environment_ready=True,
            inputs_ready=True,
            local_admission_verified=True,
            acknowledged_leases=tuple(acknowledged),
            diagnostics={
                "authority": "prepared-shard-provider",
                "gpu_support": False,
                "gpu_pending": "owned-allocation-offer" if self.resources.gpu_count else None,
            },
        )

    @property
    def allocation_job_id(self) -> str:
        if self.allocation_id is None:
            raise ValueError("Executor has no persistent allocation identity.")
        return "job-fleet-allocation-" + self.allocation_id

    def start_allocation(
        self,
        source: str | Path,
        study_identity: str,
        *,
        parent_job_id: str | None = None,
    ) -> str:
        """Prepare ONE durable owner through normal direct/command/SLURM Job submission.

        This ACK is not readiness: offers stay zero until a fresh owner-side attestation exists.
        Persisted ambiguous intent may only reconnect to its exact Job, never start another one.
        """
        assert self.control_plane is not None
        job_id = self.allocation_job_id
        receipt_path = self.root / "allocation-receipt.json"
        with CrossProcessFileLock(
            self.root / ".allocation.lock",
            shared=False,
            timeout_seconds=5,
            poll_interval_seconds=0.05,
        ):
            if receipt_path.is_symlink():
                raise ValueError("Owned allocation receipt is symlinked.")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("job_id"), receipt.get("study_identity")) != (
                    job_id,
                    study_identity,
                ):
                    raise ValueError("Allocation receipt belongs to another Study/Job.")
                if (
                    self.discover_equivalence
                    and receipt.get("equivalence")
                    and self.equivalence.hardware == "unattested"
                ):
                    self.equivalence = ExecutionEquivalence(**receipt["equivalence"])
                # Even without ACK this is a reconnect, not a new submission.
                return job_id
            receipt = {"job_id": job_id, "study_identity": study_identity, "acknowledged": False}
            atomic_write_json(receipt_path, receipt)
            if parent_job_id is not None:
                self.control_plane.jobs.reserve(
                    cluster=self.member.cluster,
                    resources=self.resources,
                    config_path=source,
                    job_id=job_id,
                    metadata={
                        "fleet_role": "member",
                        "fleet_parent_job": parent_job_id,
                        "source_config_path": str(source),
                    },
                )

            def entrypoint(prepared: PreparedWork) -> Sequence[str]:
                if prepared.dry_run or prepared.job_id != job_id:
                    raise ValueError("Member preparation requires its exact owned Job.")
                if self.discover_equivalence:
                    from lambdaforge.controlplane.ShardPreparation import prepared_equivalence

                    self.equivalence = prepared_equivalence(Path(source), prepared)
                bindings = prepared_input_bindings(Path(source), prepared, self.equivalence)
                owner_root = str(PurePosixPath(prepared.work_dir) / "member")
                manifest = {
                    "allocation_version": 1,
                    "job_id": job_id,
                    "study_identity": study_identity,
                    "member": self.member.to_dict(),
                    "root": owner_root,
                    "source": prepared.config,
                    "resources": self.resources.to_dict(),
                    "equivalence": self.equivalence.to_dict(),
                    "input_bindings": bindings,
                    "discover_hardware": self.discover_equivalence,
                }
                upload = self.root / "allocation-manifest.json"
                atomic_write_json(upload, manifest)
                created = prepared.transport.run(("mkdir", "-p", owner_root))
                if created.returncode:
                    raise RuntimeError("Could not create the owned member allocation root.")
                prepared.transport.put(upload, str(PurePosixPath(owner_root) / "allocation.json"))
                receipt.update(
                    root=owner_root,
                    python=prepared.python,
                    equivalence=self.equivalence.to_dict(),
                )
                atomic_write_json(receipt_path, receipt)
                return ("lambdaforge.work.member", owner_root)

            handle, _bundle = self.control_plane.submit(
                source,
                cluster=self.profile.name,
                resources=self.resources,
                reserved_job_id=job_id,
                allow_duplicate=True,
                entrypoint_builder=entrypoint,
            )
            if handle.job_id != job_id or not handle.scheduler_id:
                raise RuntimeError("Provider did not acknowledge its exact allocation Job.")
            receipt.update(acknowledged=True, scheduler_id=handle.scheduler_id)
            atomic_write_json(receipt_path, receipt)
        return job_id

    def _member_receipt(self) -> dict[str, Any]:
        from lambdaforge.work.member import read_envelope

        receipt = read_envelope(self.root / "allocation-receipt.json")
        if receipt.get("job_id") != self.allocation_job_id:
            raise ValueError("Member receipt does not belong to its exact executor.")
        if not receipt.get("root") or not receipt.get("python"):
            raise RuntimeError("Allocation preparation is not yet observable; do not resubmit.")
        return receipt

    def _member_call(self, *arguments: str) -> dict[str, Any]:
        assert self.control_plane is not None
        receipt = self._member_receipt()
        response = self.control_plane.factory.transport(self.profile).run(
            (
                *self.profile.command_prefix,
                receipt["python"],
                "-m",
                "lambdaforge.work.member",
                receipt["root"],
                *arguments,
            ),
            timeout=30,
        )
        if response.returncode == 255 and self.profile.transport == "ssh":
            # OpenSSH's transport failure is unknown ownership, not failed science.
            raise ConnectionError(f"Owned member transport unavailable: {response.stderr[:1000]}")
        if response.returncode:
            raise RuntimeError(f"Owned member operation failed: {response.stderr[:1000]}")
        if len(response.stdout.encode("utf-8")) > 8 * 1024**2:
            raise ValueError("Member observation exceeds its bounded envelope limit.")
        payload = json.loads(response.stdout or "{}")
        if payload and payload.get("job_id") != self.allocation_job_id:
            raise ValueError("Member operation returned another provider owner.")
        return dict(payload)

    def _submit_allocated(self, shard: StudyShard) -> str:
        directory = self._directory(shard)
        invocations = {
            lease.run.key: dict(self.invocation(lease.run.key)) for lease in shard.leases
        }
        if any(
            lease.attempt != 1
            or lease.run.equivalence != self.equivalence
            or lease.run.requires_gpu != bool(self.resources.gpu_count)
            for lease in shard.leases
        ):
            raise ShardRejectedError("Allocation accepts only exact equivalent fresh leases.")
        cap = min(self.resources.cpu_cores, self.member.max_runs or self.resources.cpu_cores, 64)
        if len(shard.leases) > cap:
            raise ShardRejectedError("Member wave exceeds its bounded Run cap.")
        manifest = {
            "allocation_version": 1,
            "job_id": self.allocation_job_id,
            "shard": shard.to_dict(),
            "invocations": invocations,
            "equivalence": self.equivalence.to_dict(),
        }
        path = directory / "allocation-wave.json"
        with CrossProcessFileLock(
            directory / ".submit.lock",
            shared=False,
            timeout_seconds=5,
            poll_interval_seconds=0.05,
        ):
            if path.exists():
                from lambdaforge.work.member import read_envelope

                if read_envelope(path) != manifest:
                    raise ValueError("An accepted wave cannot change its immutable invocation.")
            else:
                atomic_write_json(path, manifest)
            receipt = self._member_receipt()
            remote_path = str(PurePosixPath(receipt["root"]) / (shard.shard_id + ".upload.json"))
            assert self.control_plane is not None
            self.control_plane.factory.transport(self.profile).put(path, remote_path)
            response = self._member_call("--enqueue", remote_path)
            if response.get("rejected"):
                raise ShardRejectedError(str(response["rejected"]))
            if response.get("job_id") != self.allocation_job_id:
                raise RuntimeError("Member did not acknowledge its exact wave owner.")
        return self.allocation_job_id

    def _allocation_offer(self) -> ClusterOffer:
        now = time.time()
        health, reason = ClusterHealth.ONLINE, "allocation-not-ready"
        value: dict[str, Any] = {}
        assert self.control_plane is not None
        try:
            state = self.control_plane.jobs.get(self.allocation_job_id, include_study=False).state
            if state is JobState.RUNNING:
                value = self._member_call("--observe").get("offer") or {}
            elif state is JobState.UNKNOWN:
                health, reason = ClusterHealth.UNREACHABLE, "provider-unreachable"
            elif state.terminal:
                health, reason = ClusterHealth.DEGRADED, "allocation-terminal"
        except (OSError, RuntimeError, TimeoutError, KeyError):
            health, reason = ClusterHealth.UNREACHABLE, "member-unreachable"
        # The owner can publish during transport; compare freshness at receipt, not before
        # the request, otherwise a healthy newly written heartbeat looks future-dated.
        now = time.time()
        if self.discover_equivalence and value.get("equivalence"):
            actual = ExecutionEquivalence(**value["equivalence"])
            expected = ExecutionEquivalence(**self._member_receipt()["equivalence"])
            if actual.hardware == "unattested" or any(
                getattr(actual, name) != getattr(expected, name)
                for name in ("code", "environment", "inputs", "numerics")
            ):
                raise ValueError("Live member attestation differs from prepared equivalence.")
            self.equivalence = actual
        valid = (
            value.get("offer_version") == 1
            and value.get("job_id") == self.allocation_job_id
            and value.get("equivalence") == self.equivalence.to_dict()
            and value.get("observed_at", float("inf")) <= now
            and value.get("valid_until", 0) > now
        )
        cap = min(self.resources.cpu_cores, self.member.max_runs or self.resources.cpu_cores, 64)
        return ClusterOffer(
            self.member.cluster,
            health,
            self.profile.scheduler,
            self.profile.gpu_access.effective_mode(self.profile.scheduler),
            now,
            min(now + 15, value["valid_until"]) if valid else now + 1,
            slots=min(cap, value["slots"]) if valid else 0,
            admissible_gpus=value["admissible_gpus"] if valid else 0,
            free_memory_bytes=value["free_memory_bytes"] if valid else None,
            equivalence=self.equivalence,
            environment_ready=bool(valid),
            inputs_ready=bool(valid),
            local_admission_verified=bool(valid),
            acknowledged_leases=tuple(value.get("acknowledged_leases", ())) if valid else (),
            diagnostics={
                "authority": "owned-provider-allocation",
                "job_id": self.allocation_job_id,
                "reason": value.get("reason", reason) if valid else reason,
            },
        )

    def _observe_allocated(
        self,
        shard: StudyShard,
        job_id: str | None,
    ) -> Sequence[RemoteObservation]:
        self._directory(shard)
        if job_id is not None and job_id != self.allocation_job_id:
            raise ValueError("Wave belongs to another allocation Job.")
        payload = self._member_call("--observe", "--shard", shard.shard_id)
        observation = payload.get("observation", {})
        results = observation.get("results", {})
        runs = observation.get("runs", {})
        assert self.control_plane is not None
        owner = self.control_plane.jobs.get(self.allocation_job_id, include_study=False)
        if owner.state.terminal and any(lease.run.key not in results for lease in shard.leases):
            raise RuntimeError(
                f"Member allocation {self.allocation_job_id} ended as {owner.state.value} "
                "without all leased results. Exact leases/evidence are retained; "
                "no unproven replacement Attempt was submitted."
            )
        if (
            observation.get("shard_id") not in {None, shard.shard_id}
            or (results and observation.get("shard_id") != shard.shard_id)
            or set(results) - {lease.run.key for lease in shard.leases}
            or not isinstance(runs, Mapping)
            or set(runs) - {lease.run.key for lease in shard.leases}
        ):
            raise ValueError("Member observation differs from its exact leased wave.")
        expected_root = (
            PurePosixPath(self._member_receipt()["root"])
            / "waves"
            / shard.shard_id
            / "worker"
            / "execution"
        )
        for lease in shard.leases:
            invocation = self.invocation(lease.run.key)
            live = runs.get(lease.run.key)
            if live is not None:
                if (
                    not isinstance(live, Mapping)
                    or live.get("trial") != invocation["trial_index"]
                    or live.get("seed") != lease.run.seed
                    or live.get("phase") != lease.run.phase
                    or live.get("attempt_id") != f"attempt-{lease.attempt:04d}"
                    or live.get("state")
                    not in {"running", "succeeded", "failed", "pruned", "aborted"}
                ):
                    raise ValueError("Live Run metadata differs from its exact invocation.")
                for field in ("run_dir", "log_path", "metrics_path", "training_metrics_path"):
                    path = PurePosixPath(str(live.get(field, "")))
                    if (
                        not path.is_absolute()
                        or ".." in path.parts
                        or not path.is_relative_to(expected_root)
                    ):
                        raise ValueError("Live Run metadata contains a foreign owner path.")
                self._run_observations[lease.run.key] = dict(live)
            envelope = results.get(lease.run.key)
            if envelope is None:
                continue
            native = envelope["result"]
            path = PurePosixPath(native.get("run_dir", ""))
            if (
                native.get("execution_id") != invocation["execution_id"]
                or native.get("seed") != lease.run.seed
                or native.get("attempt_number") != lease.attempt
                or (native.get("trial") or {}).get("index") != invocation["trial_index"]
                or not path.is_absolute()
                or ".." in path.parts
                or not path.is_relative_to(expected_root)
            ):
                raise ValueError("Native allocation result differs from its exact invocation.")
        return tuple(
            RemoteObservation(
                lease.run.key,
                lease.attempt,
                lease.lease_id,
                results[lease.run.key]["state"]
                if lease.run.key in results
                else "running"
                if lease.run.key in runs and owner.state is JobState.RUNNING
                else "queued"
                if payload.get("accepted")
                else "unknown_remote",
                result=results.get(lease.run.key),
            )
            for lease in shard.leases
        )

    def run_observation(self, run_key: str) -> Mapping[str, Any] | None:
        """Last verified bounded native Run view; never an extra provider request/result store."""
        value = self._run_observations.get(run_key)
        return dict(value) if value is not None else None

    def pull_scalars(self, shard_id: str, receivers: Mapping[str, Any]) -> dict[str, bool]:
        """Incrementally ingest verified native metrics; retries reuse durable byte cursors."""
        from lambdaforge.work.fleet_stream import ScalarStream

        if self.allocation_id is None or not receivers or len(receivers) > 64:
            raise ValueError("Scalar reads require a bounded exact allocation wave.")
        if any(not isinstance(value, ScalarStream) for value in receivers.values()):
            raise TypeError("Scalar reads require owned stream receivers.")
        response = self._member_call(
            "--shard",
            shard_id,
            "--stream",
            json.dumps({key: value.cursors() for key, value in receivers.items()}, allow_nan=False),
        )
        streams = response.get("streams")
        if not isinstance(streams, Mapping) or set(streams) != set(receivers):
            raise ValueError("Scalar response omitted or invented a leased Run.")
        return {key: receivers[key].ingest(value) for key, value in streams.items()}

    def request_pruning(self, shard_id: str, command: Mapping[str, Any]) -> str:
        """Forward the native planner's fenced stop request, not another pruning decision."""
        response = self._member_call(
            "--shard",
            shard_id,
            "--prune",
            json.dumps(dict(command), allow_nan=False),
        )
        status = response.get("status")
        if status not in {"requested", "already-terminal", "not-running"}:
            raise ValueError("Member omitted the exact pruning request acknowledgement.")
        return str(status)

    def drain_allocation(self) -> None:
        """Stop acceptance and release the allocation only after its accepted wave completes."""
        self._member_call("--shutdown")
