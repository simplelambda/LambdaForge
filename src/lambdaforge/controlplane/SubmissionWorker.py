"""Internal entry point that completes one durable remote submission request."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.PreparedWork import PreparedWork
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.ProjectContext import ProjectContext


def serve(request_path: str | Path) -> int:
    """Complete one request and persist every outcome under its preallocated job ID."""
    path = Path(request_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("submission_request_version") != 1:
        raise ValueError(f"Invalid LambdaForge submission request: {path}")
    job_id = str(value["job_id"])
    cluster = str(value["cluster"])
    raw_project = value.get("project")
    project = (
        ProjectContext(Path(str(raw_project["root"])), str(raw_project["project_id"]))
        if isinstance(raw_project, Mapping)
        else None
    )
    store = JobStore(str(value["job_store"]), project=project)
    jobs = JobService(ClusterCatalog({cluster: ClusterProfile(cluster)}), store=store)
    try:
        raw_profile = value.get("profile")
        if not isinstance(raw_profile, Mapping):
            raise TypeError("Submission request profile must be a mapping.")
        profile = ClusterProfile.from_mapping(cluster, raw_profile)
        catalog = ClusterCatalog({cluster: profile})
        jobs = JobService(catalog, store=store)
        jobs.update_preparation(job_id, "validation")
        pin = value.get("config_sha256")
        if (
            pin is not None
            and hashlib.sha256(Path(str(value["config"])).read_bytes()).hexdigest() != pin
        ):
            raise ValueError(
                "Recovery YAML changed after submission; re-preview before recovering."
            )
        arguments = value.get("run_arguments", ())
        if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes, bytearray)):
            raise TypeError("Submission run_arguments must be a sequence.")

        def progress(phase: str) -> None:
            jobs.update_preparation(job_id, phase)

        builder: Callable[[PreparedWork], Sequence[str]] | None = None
        if value.get("fleet") is not None:

            def builder(prepared: PreparedWork) -> Sequence[str]:
                return ("lambdaforge.controlplane.FleetStudyService", str(path))

        extension = jobs.get(job_id, refresh=False).metadata.get("study_extension")
        if isinstance(extension, Mapping) and extension.get("code_review_pin"):
            from lambdaforge.work.extension_compatibility import code_review_pin

            if project is None or code_review_pin(project) != extension["code_review_pin"]:
                raise ValueError(
                    "Extension source changed after review; no native dispatch occurred."
                )
        if isinstance(extension, Mapping) and extension.get("retained_runtime"):
            from lambdaforge.controlplane.StudyExtensionDispatch import (
                prepared_entrypoint,
                retained_profile,
            )

            if value.get("fleet") is not None:
                raise ValueError("Fleet extension dispatch is not supported.")
            proof = extension["retained_runtime"]
            origin = jobs.get(str(extension["origin_job_id"]), refresh=False)
            if (
                origin.cluster != cluster
                or origin.metadata.get("environment_id") != proof["environment_id"]
            ):
                raise ValueError("Retained runtime differs from its native origin Job.")
            profile = retained_profile(profile, proof)
            catalog = ClusterCatalog({cluster: profile}, project=project)
            jobs = JobService(catalog, store=store)

            def builder(prepared: PreparedWork) -> Sequence[str]:
                return prepared_entrypoint(prepared, proof, tuple(str(item) for item in arguments))

        if (
            isinstance(extension, Mapping) and extension.get("delta_contract")
            and cluster != "local"
        ):
            from lambdaforge.controlplane.StudyExtensionDispatch import (
                current_entrypoint,
                prepare_native_delta,
            )

            def builder(prepared: PreparedWork) -> Sequence[str]:
                # Recheck after ordinary managed preparation, before provider submission.
                if extension.get("code_review_pin"):
                    from lambdaforge.work.extension_compatibility import code_review_pin

                    if project is None or code_review_pin(project) != extension["code_review_pin"]:
                        raise ValueError("Extension source changed during native preparation.")
                attestation = prepare_native_delta(prepared, extension["delta_contract"])
                current = jobs.get(job_id, refresh=False)
                if current.state is JobState.CANCELLED:
                    raise ValueError("Extension submission was cancelled during preparation.")
                if current.metadata.get("recovery_execution_dir"):
                    # Native recovery restores the ORIGINAL namespace/Execution; its new
                    # staged configuration must not replace that execution's identity.
                    previous = current.metadata.get("prepared_extension_attestation")
                    if not isinstance(previous, Mapping):
                        raise ValueError("Missing original extension preparation attestation.")
                    attestation["scientific_identity"] = str(previous["scientific_identity"])
                jobs.store.write(current.with_updates(metadata={
                    **current.metadata, "prepared_extension_attestation": attestation,
                }))
                forwarded = tuple(str(item) for item in arguments)
                if extension.get("retained_runtime"):
                    return prepared_entrypoint(
                        prepared, extension["retained_runtime"], forwarded,
                        attestation=attestation,
                    )
                return current_entrypoint(prepared, attestation, forwarded)

        native_arguments = (
            ()
            if isinstance(extension, Mapping) and (
                extension.get("retained_runtime")
                or extension.get("delta_contract") and cluster != "local"
            )
            else arguments
        )

        ControlPlane(catalog, jobs=jobs).submit(
            str(value["config"]),
            cluster=cluster,
            resources=ResourceRequest.from_mapping(value.get("resources", {})),
            run_arguments=tuple(str(item) for item in native_arguments),
            group_id=str(value["group_id"]) if value.get("group_id") else None,
            reserved_job_id=job_id,
            allow_duplicate=bool(value.get("allow_duplicate", False)),
            progress=progress,
            entrypoint_builder=builder,
            input_bindings=value.get("input_bindings"),
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
