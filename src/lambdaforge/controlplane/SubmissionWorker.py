"""Internal entry point that completes one durable remote submission request."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.execution.ResourceRequest import ResourceRequest


def serve(request_path: str | Path) -> int:
    """Complete one request and persist every outcome under its preallocated job ID."""
    path = Path(request_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("submission_request_version") != 1:
        raise ValueError(f"Invalid LambdaForge submission request: {path}")
    job_id = str(value["job_id"])
    cluster = str(value["cluster"])
    store = JobStore(str(value["job_store"]))
    jobs = JobService(ClusterCatalog({cluster: ClusterProfile(cluster)}), store=store)
    try:
        raw_profile = value.get("profile")
        if not isinstance(raw_profile, Mapping):
            raise TypeError("Submission request profile must be a mapping.")
        profile = ClusterProfile.from_mapping(cluster, raw_profile)
        catalog = ClusterCatalog({cluster: profile})
        jobs = JobService(catalog, store=store)
        jobs.update_preparation(job_id, "validation")
        arguments = value.get("run_arguments", ())
        if not isinstance(arguments, Sequence) or isinstance(
            arguments, (str, bytes, bytearray)
        ):
            raise TypeError("Submission run_arguments must be a sequence.")

        def progress(phase: str) -> None:
            jobs.update_preparation(job_id, phase)

        ControlPlane(catalog, jobs=jobs).submit(
            str(value["config"]),
            cluster=cluster,
            resources=ResourceRequest.from_mapping(value.get("resources", {})),
            run_arguments=tuple(str(item) for item in arguments),
            group_id=str(value["group_id"]) if value.get("group_id") else None,
            reserved_job_id=job_id,
            progress=progress,
        )
        return 0
    except Exception as error:
        current = jobs.get(job_id, refresh=False)
        if current.state not in {JobState.FAILED, JobState.CANCELLED}:
            jobs.fail_preparation(job_id, error)
        traceback.print_exc()
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the private worker CLI."""
    values = tuple(argv if argv is not None else sys.argv[1:])
    if len(values) != 1:
        raise ValueError("SubmissionWorker requires exactly one request path.")
    return serve(values[0])


if __name__ == "__main__":  # pragma: no cover - exercised through process integration tests
    raise SystemExit(main())
