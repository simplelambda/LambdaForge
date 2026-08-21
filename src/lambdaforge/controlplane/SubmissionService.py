"""Durable local hand-off for remote submissions with expensive preparation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
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
    ) -> JobHandle:
        """Persist and launch a remote submission request without waiting for SSH preparation."""
        source = Path(config).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {source}")
        if cluster == "local":
            raise ValueError("Local execution does not use asynchronous remote preparation.")
        profile = self.catalog.get(cluster)
        if profile.auth.mode == "password" and profile.auth.credential is None:
            raise ValueError(
                "Detached submission cannot prompt for a password. Store a keyring credential "
                "reference or use OpenSSH authentication first."
            )
        # Parse first so authoring errors remain immediate and no doomed request is queued.
        # Tasks do not need active-experiment deduplication. Their complete input/code
        # fingerprint is attached by the worker, keeping the foreground hand-off cheap.
        descriptor = ConfigurationDescriptor.from_path(source, resolve_task_code_identity=False)
        if descriptor.job_type in {"experiment", "hpo"} and not allow_duplicate:
            self.jobs.refuse_active_execution(
                descriptor.scientific_identity,
                cluster,
                name=descriptor.name,
                source=source,
            )
        request_resources = resources or ConfigurationResourceResolver.resolve(source)
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
            },
            job_type=descriptor.job_type,
            group_id=group_id,
        )
        directory = self.jobs.store.root / "submissions" / handle.job_id
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(directory, 0o700)
        request_path = directory / "request.json"
        profile_descriptor = profile.to_dict()
        profile_descriptor.pop("name", None)
        payload: dict[str, Any] = {
            "submission_request_version": 1,
            "job_id": handle.job_id,
            "cluster": cluster,
            "profile": profile_descriptor,
            "config": str(source),
            "resources": request_resources.to_dict(),
            "run_arguments": list(run_arguments),
            "group_id": group_id,
            "allow_duplicate": allow_duplicate,
            "job_store": str(self.jobs.store.root),
        }
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
        try:
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
                    cwd=source.parent,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as error:
            self.jobs.fail_preparation(handle.job_id, error)
            raise
        return handle
