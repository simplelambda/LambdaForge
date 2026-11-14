"""Durable local hand-off for remote submissions with expensive preparation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.jobs import JobHandle
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver
from lambdaforge.execution.ResourceRequest import ResourceRequest


class SubmissionService:
    """Queue slow bundle/runtime/transfer work in one auditable detached controller process."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        jobs: JobService | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.jobs = jobs or JobService(self.catalog)

    def enqueue(
        self,
        config: str | Path,
        *,
        cluster: str,
        resources: ResourceRequest | None = None,
        run_arguments: Sequence[str] = (),
        group_id: str | None = None,
        retry_of: str | None = None,
        allow_duplicate: bool = False,
        metadata: Mapping[str, Any] | None = None,
        fleet: str | None = None,
        input_bindings: Mapping[str, Any] | None = None,
        job_id: str | None = None,
        config_sha256: str | None = None,
        execution_queue: bool = False,
    ) -> JobHandle:
        """Persist and launch preparation without blocking on the selected execution target."""
        source = Path(config).expanduser().resolve()
        recovery_config = (metadata or {}).get("recovery_configuration")
        pin = config_sha256 or (
            recovery_config.get("sha256") if isinstance(recovery_config, Mapping) else None
        )
        if execution_queue and pin is None:
            pin = hashlib.sha256(source.read_bytes()).hexdigest()
        if pin is not None and hashlib.sha256(source.read_bytes()).hexdigest() != pin:
            raise ValueError("Recovery YAML changed after preview; no Job was enqueued.")
        if self.catalog.project is not None and not self.catalog.project.owns_source(str(source)):
            raise ValueError(
                "The YAML belongs to another project. Run lf from that project's directory."
            )
        if not source.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {source}")
        profile = self.catalog.get(cluster)
        if execution_queue and (fleet is not None or profile.scheduler != "local"):
            raise ValueError(
                "Sequential queues support direct process clusters only, not Fleet/SLURM."
            )
        if input_bindings:
            from lambdaforge.work.dependency_bindings import pin_bindings

            input_bindings = pin_bindings(input_bindings, source)
            if fleet is not None:
                raise ValueError("Declare Fleet dependency inputs in the authored Study YAML.")
        fleet_preview = None
        selected_fleet = None
        if fleet is not None:
            from lambdaforge.controlplane.FleetStudyService import fleet_preflight
            from lambdaforge.work.config import WorkConfig

            selected_fleet = self.catalog.fleet(fleet)
            if cluster != selected_fleet.coordinator:
                raise ValueError("Fleet submission must use its configured coordinator.")
            if run_arguments:
                raise ValueError("Fleet restart/rerun options are not yet supported.")
            fleet_preview = fleet_preflight(
                WorkConfig.from_yaml(source), self.catalog, selected_fleet
            )
        if (
            profile.transport == "ssh"
            and profile.auth.mode == "password"
            and profile.auth.credential is None
        ):
            raise ValueError(
                "Detached submission cannot prompt for a password. Store a keyring credential "
                "reference or use OpenSSH authentication first."
            )
        # Parse and identify first so authoring errors and duplicate scientific work remain
        # immediate; expensive remote preparation still happens in the detached controller.
        descriptor = (
            ConfigurationDescriptor.from_path(source, input_bindings=input_bindings)
            if input_bindings
            else ConfigurationDescriptor.from_path(source)
        )
        if not allow_duplicate:
            self.jobs.refuse_active_execution(
                descriptor.scientific_identity,
                "fleet:" + fleet if fleet is not None else cluster,
                name=descriptor.name,
                source=source,
            )
        request_resources = (
            ResourceRequest(cpu_cores=1)
            if selected_fleet is not None
            else resources or ConfigurationResourceResolver.resolve(source)
        )
        handle = self.jobs.reserve(
            cluster=cluster,
            resources=request_resources,
            config_path=source,
            retry_of=retry_of,
            metadata={
                **descriptor.metadata(),
                "submission_phase": "queued-locally",
                "submission_mode": "asynchronous",
                "run_arguments": list(run_arguments),
                **({"execution_queue_requested": True} if execution_queue else {}),
                **({"input_bindings": dict(input_bindings)} if input_bindings else {}),
                **dict(metadata or {}),
                **(
                    {"pinned_scientific_identity": descriptor.scientific_identity}
                    if config_sha256 is not None or execution_queue
                    else {}
                ),
                **(
                    {"fleet": fleet, "execution_target": "fleet:" + fleet}
                    if fleet is not None
                    else {}
                ),
            },
            job_type=descriptor.job_type,
            group_id=group_id,
            job_id=job_id,
        )
        if execution_queue:
            from lambdaforge.controlplane.ExecutionQueueService import ExecutionQueueService

            try:
                current = self.jobs.store.get(handle.job_id)
                blockers = [
                    record.job_id
                    for record in self.jobs.store.records()
                    if record.job_id != handle.job_id
                    and record.cluster == cluster
                    and not record.state.terminal
                    and not record.metadata.get("execution_queue_requested")
                    and record.created_at_utc <= current.created_at_utc
                ]
                admission = ExecutionQueueService(
                    self.catalog, self.jobs.factory, source_dir=source.parent
                ).register(cluster, handle.job_id, blockers)
                self.jobs.store.write(
                    current.with_updates(
                        metadata={**current.metadata, "execution_queue": admission}
                    )
                )
                # A predecessor can fail before its provider request exists while the
                # queue registration RPC is in flight. Close that preparation race.
                for previous_id in blockers:
                    previous = self.jobs.store.get(previous_id)
                    if previous.state.terminal and previous.scheduler_id is None:
                        ExecutionQueueService(
                            self.catalog, self.jobs.factory, source_dir=source.parent
                        ).abort(cluster, previous_id)
            except Exception as error:
                self.jobs.fail_preparation(handle.job_id, error)
                raise
        directory = self.jobs.store.root / "submissions" / handle.job_id
        profile_descriptor = profile.to_dict()
        profile_descriptor.pop("name", None)
        payload: dict[str, Any] = {
            "submission_request_version": 1,
            "job_id": handle.job_id,
            "cluster": cluster,
            "profile": profile_descriptor,
            "config": str(source),
            **({"config_sha256": pin} if pin is not None else {}),
            "resources": request_resources.to_dict(),
            "run_arguments": list(run_arguments),
            **({"input_bindings": dict(input_bindings)} if input_bindings else {}),
            "group_id": group_id,
            "allow_duplicate": allow_duplicate,
            "job_store": str(self.jobs.store.root),
            "project": self.jobs.store.project.to_dict() if self.jobs.store.project else None,
        }
        if selected_fleet is not None:
            payload.update(
                {
                    "fleet": selected_fleet.to_dict(),
                    "fleet_preflight": fleet_preview,
                    "member_profiles": {
                        name: self.catalog.get(name).to_dict()
                        for name in {
                            cluster,
                            *(member.cluster for member in selected_fleet.members),
                        }
                    },
                }
            )
        try:
            self._launch_preparation(directory, source.parent, payload)
        except Exception as error:
            self.jobs.fail_preparation(handle.job_id, error)
            raise
        return handle

    @staticmethod
    def _launch_preparation(directory: Path, cwd: Path, payload: Mapping[str, Any]) -> None:
        """Persist the controller request before detaching; callers own failure reconciliation."""
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(directory, 0o700)
        request_path = directory / "request.json"
        temporary = request_path.with_name(f".{request_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, request_path)
        finally:
            temporary.unlink(missing_ok=True)
        log_path = directory / "controller.log"
        log_path.touch(mode=0o600, exist_ok=False)
        os.chmod(log_path, 0o600)
        with log_path.open("ab", buffering=0) as log:
            subprocess.Popen(
                (
                    sys.executable,
                    "-m",
                    "lambdaforge.controlplane.SubmissionWorker",
                    str(request_path),
                ),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
