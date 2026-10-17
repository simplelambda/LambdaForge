"""Native verified fixed Study revisions, submission and cumulative evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lambdaforge.analysis.Evidence import normalize_evidence
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.config import WorkConfig, import_work_class
from lambdaforge.work.extension_compatibility import (
    code_review_pin,
    review_extension_code,
    verify_code,
    verify_local_runtime,
)
from lambdaforge.work.extension_design import (
    candidate_rows,
    compare_fixed_designs,
    exact,
    fixed_definition,
    plain,
)
from lambdaforge.work.recovery import (
    fixed_inventory,
    read_owned_json,
    restore_seed_context,
    validate_execution,
    validate_recovery_configuration,
)
from lambdaforge.work.ResultStore import ResultStore
from lambdaforge.work.runner import _identity_values, _resolve_inputs, _validate_fixed_recovery
from lambdaforge.work.StudySeriesStore import StudySeriesStore

if TYPE_CHECKING:
    from lambdaforge.controlplane.JobService import JobService


class StudyExtensionService:
    """Plan exact fixed-sweep deltas through native configuration/evidence contracts.

    Planning imports the selected Work for its native signature; callers must authorize
    validation. It writes nothing, downloads no artifact and launches no computation.
    Apply delegates to SubmissionService, retaining proven environments or deploying an
    explicitly reviewed new code/runtime cohort through the ordinary managed pipeline.
    """

    def __init__(
        self,
        *,
        project: ProjectContext | None = None,
        results: ResultStore | None = None,
        catalog: ClusterCatalog | None = None,
        jobs: JobService | None = None,
    ) -> None:
        self.project = project or ProjectContext.discover()
        self.results = results or ResultStore(self.project.root / ".lambdaforge" / "runs")
        self.catalog = catalog or ClusterCatalog.load(project=self.project)
        if jobs is None:
            from lambdaforge.controlplane.JobService import JobService

            jobs = JobService(self.catalog)
        self.jobs = jobs

    def apply(
        self,
        study: str,
        config: str | Path,
        *,
        plan_hash: str,
        target: str | None = None,
        code_policy: str = "revisioned",
    ) -> dict[str, Any]:
        """Revalidate, reserve and enqueue only the delta through native SubmissionService.

        Explicit reviewed-plan approval is mandatory. A durable preallocated Job ID closes
        duplicate handoffs: uncertain preparation is reported, never blindly replayed.
        """
        from lambdaforge.controlplane.SubmissionService import SubmissionService

        target = self._target(study, target)
        if not plan_hash.startswith("sha256:"):
            raise ValueError("Review an extension preview and supply its exact plan hash.")
        lock = (
            self.project.root
            / ".lambdaforge"
            / "study-series"
            / ".locks"
            / ("apply-" + hashlib.sha256(plan_hash.encode()).hexdigest() + ".lock")
        )
        if lock.resolve() != lock:
            raise ValueError("Unsafe extension apply storage.")
        with CrossProcessFileLock(
            lock, shared=False, timeout_seconds=2, poll_interval_seconds=0.02
        ):
            store = StudySeriesStore(self.project)
            # Returning an already owned request never reimports the Work or starts another Job.
            for directory in sorted(store.root.iterdir()):
                if directory.name == ".locks":
                    continue
                value = store.read(directory.name)
                for operation in value["operations"]:
                    if operation["plan_hash"] != plan_hash:
                        continue
                    if operation.get("source_yaml") not in (
                        None,
                        str(Path(config).expanduser().absolute()),
                    ):
                        raise ValueError("Reviewed plan belongs to a different YAML.")
                    if operation.get("job_id"):
                        try:
                            job = self.jobs.get(operation["job_id"], refresh=False)
                        except (KeyError, FileNotFoundError):
                            return self._resume_handoff(value, operation, target=target)
                        ownership = job.metadata.get("study_extension", {})
                        if (
                            ownership.get("series_id") != value["study_series_id"]
                            or ownership.get("plan_hash") != plan_hash
                            or job.cluster != target
                        ):
                            raise ValueError(
                                "Native extension Job ownership differs from its journal."
                            )
                        return {
                            **operation,
                            "state": job.state.value if job.state.terminal else operation["state"],
                            "study_series_id": value["study_series_id"],
                            "job": job.to_dict(),
                            "created": False,
                        }
            plan = self.preview(study, config, target=target, code_policy=code_policy)
            if plan["plan_hash"] != plan_hash:
                raise ValueError("Extension plan changed; review a fresh preview before applying.")
            if not plan["apply_supported"]:
                raise ValueError(
                    "This placement's extension ownership/retention is not integrated; "
                    "no Job was submitted."
                )
            if plan["comparison"]["counts"]["new"] == 0:
                return {
                    "state": "no-op",
                    "created": False,
                    "study_series_id": plan["study_series_id"],
                }
            request = "extension-" + uuid4().hex
            if plan.get("expected_revision", 0):
                # Operational reconciliation is an apply action, never a preview mutation.
                # The revision/design itself does not change when native completion is sealed.
                self.refresh(plan["study_series_id"])
            reserved = store.reserve(
                plan,
                request_id=request,
                expected_revision=plan.get("expected_revision", 0),
                verify_remote=(lambda: self._revalidate_remote(plan))
                if plan.get("placement") == "remote"
                else None,
            )
            request = reserved["request_id"]
            selected = Path(plan["proposed_yaml"])
            snapshot = selected.parent / (
                ".lf-extension-" + request.removeprefix("extension-") + ".yaml"
            )
            import yaml

            raw = yaml.safe_dump(
                plan["comparison"]["delta_configuration"], sort_keys=False
            ).encode()
            snapshot_hash = hashlib.sha256(raw).hexdigest()
            job_id = self.jobs.new_id()
            store.transition(
                plan["study_series_id"],
                request,
                state="dispatching",
                facts={
                    "job_id": job_id,
                    "snapshot": str(snapshot),
                    "snapshot_sha256": snapshot_hash,
                    "source_yaml": str(selected),
                },
            )
            if snapshot.resolve() != snapshot:
                raise ValueError("Unsafe extension snapshot path.")
            descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_reviewed_code(plan)
            handle = SubmissionService(self.catalog, self.jobs).enqueue(
                snapshot,
                cluster=target,
                job_id=job_id,
                config_sha256=snapshot_hash,
                metadata={
                    "study_extension": {
                        "series_id": plan["study_series_id"],
                        "request_id": request,
                        "origin_execution_id": plan["origin_execution_id"],
                        "origin_job_id": plan.get("origin_job_id"),
                        "proposed_yaml": str(selected),
                        "plan_hash": plan_hash,
                        "code_review_pin": plan["code_compatibility"].get("review_pin"),
                        "retained_runtime": plan["runtime_compatibility"].get("retained_runtime"),
                        "delta_contract": plan.get("delta_contract"),
                    },
                },
            )
            operation = store.transition(
                plan["study_series_id"],
                request,
                state="submitted",
                facts={"job_id": handle.job_id},
            )
            return {
                **operation,
                "study_series_id": plan["study_series_id"],
                "job": handle.to_dict(),
                "created": True,
            }

    def _resume_handoff(
        self,
        journal: Mapping[str, Any],
        operation: Mapping[str, Any],
        *,
        target: str,
    ) -> dict[str, Any]:
        """Repair only a proven pre-Job crash, keeping its preallocated native Job ID.

        A missing Job alongside any request/provider evidence is uncertain ownership, not
        permission to replay. No already launched preparation process is relaunched here.
        """
        from lambdaforge.controlplane.SubmissionService import SubmissionService

        revision = journal["revisions"][operation["revision"]]
        plan = revision.get("reviewed_plan")
        if not isinstance(plan, Mapping) or operation["state"] != "dispatching":
            raise ValueError(
                "Missing native handoff authority; preserve the journal for inspection."
            )
        if target != plan["target"]:
            raise ValueError("Interrupted extension belongs to another target.")
        request = self.jobs.store.root / "submissions" / str(operation["job_id"])
        if request.exists() or request.is_symlink():
            raise ValueError(
                "Missing native Job has submission evidence; refusing uncertain replay."
            )
        profile = self.catalog.get(target)
        if (
            ScientificIdentity.from_payload(profile.to_dict()).digest
            != plan["target_profile_fingerprint"]
        ):
            raise ValueError("Target changed after review; interrupted handoff cannot be replayed.")
        assert profile.storage is not None
        from pathlib import PurePosixPath

        provider_profile = self.jobs._submission_profile(
            profile,
            config_path=str(plan["proposed_yaml"]),
            work_dir=str(Path(plan["proposed_yaml"]).parent),
        )
        transport = self.jobs.factory.transport(profile)
        absent = transport.run(
            (
                "test",
                "-e",
                str(PurePosixPath(provider_profile.storage.run_root) / str(operation["job_id"])),
            )
        )
        if absent.returncode != 1:
            raise ValueError(
                "Provider Job state exists or is uncertain; refusing duplicate dispatch."
            )
        selected = Path(plan["proposed_yaml"])
        if (
            selected.resolve() != selected
            or not self.project.owns_source(str(selected))
            or hashlib.sha256(selected.read_bytes()).hexdigest() != plan["proposed_yaml_sha256"]
        ):
            raise ValueError("Reviewed YAML changed; interrupted handoff cannot be resumed.")
        import yaml

        raw = yaml.safe_dump(plan["comparison"]["delta_configuration"], sort_keys=False).encode()
        snapshot = Path(operation["snapshot"])
        if (
            snapshot.resolve() != snapshot
            or snapshot.parent != selected.parent
            or hashlib.sha256(raw).hexdigest() != operation["snapshot_sha256"]
        ):
            raise ValueError("Unsafe/corrupt interrupted configuration snapshot.")
        if snapshot.exists():
            if snapshot.read_bytes() != raw:
                raise ValueError("Interrupted snapshot was altered.")
        else:
            fd = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        self._check_reviewed_code(plan)
        definition = fixed_definition(WorkConfig.from_yaml(selected))
        if exact(_identity_values(definition.parameters, selected.parent)) != exact(
            plan["fixed_input_identities"]
        ):
            raise ValueError("Reviewed scientific inputs changed; no delta was submitted.")
        remote_refs = {}
        if plan.get("placement") == "remote":
            for ref in plan["reused_evidence"]:
                job_id = str(ref["job_id"])
                if job_id not in remote_refs:
                    snap = self.jobs.extension_evidence(job_id)
                    remote_refs[job_id] = {
                        row["result_path"]: row["result_sha256"] for row in snap["evidence"]
                    }
                if remote_refs[job_id].get(ref["result_path"]) != ref["result_sha256"]:
                    raise ValueError("Reviewed remote evidence changed before interrupted handoff.")
        else:
            origin = Path(plan["origin_execution_dir"])
            validate_execution(origin)
            if (
                hashlib.sha256((origin / "execution.json").read_bytes()).hexdigest()
                != plan["origin_manifest_sha256"]
            ):
                raise ValueError("Original Execution changed before interrupted handoff.")
            for ref in plan["reused_evidence"]:
                path = Path(ref["result_path"])
                if (
                    path.resolve() != path
                    or not path.is_relative_to(self.results.root)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != ref["result_sha256"]
                ):
                    raise ValueError(
                        "Reviewed historical Attempt changed before interrupted handoff."
                    )
        handle = SubmissionService(self.catalog, self.jobs).enqueue(
            snapshot,
            cluster=target,
            job_id=operation["job_id"],
            config_sha256=operation["snapshot_sha256"],
            metadata={
                "study_extension": {
                    "series_id": plan["study_series_id"],
                    "request_id": operation["request_id"],
                    "origin_execution_id": plan["origin_execution_id"],
                    "origin_job_id": plan.get("origin_job_id"),
                    "proposed_yaml": str(selected),
                    "plan_hash": plan["plan_hash"],
                    "code_review_pin": plan["code_compatibility"].get("review_pin"),
                    "retained_runtime": plan["runtime_compatibility"].get("retained_runtime"),
                    "delta_contract": plan.get("delta_contract"),
                }
            },
        )
        accepted = StudySeriesStore(self.project).transition(
            plan["study_series_id"],
            operation["request_id"],
            state="submitted",
            facts={"job_id": handle.job_id},
        )
        return {
            **accepted,
            "study_series_id": plan["study_series_id"],
            "job": handle.to_dict(),
            "created": True,
            "reconciled_pre_job_handoff": True,
        }

    def show(self, series: str) -> dict[str, Any]:
        """Read an exact owned journal; reservations do not imply successful submission."""
        return StudySeriesStore(self.project).read(series)

    def history(self, series: str) -> dict[str, Any]:
        """Return append-only operation facts without opening Runs or artifact bytes."""
        value = self.show(series)
        return {
            "study_series_id": value["study_series_id"],
            "project_id": value["project_id"],
            "history": value["history"],
        }

    @staticmethod
    def _execution_pin(metadata: Mapping[str, Any]) -> str | None:
        """Prefer the actual host's native identity, not a controller path/seed namespace.

        Preparation attests the exact reviewed semantic delta before recording this pin.
        It never replaces the original controller submission identity or origin records.
        """
        if "prepared_extension_attestation" in metadata:
            attestation = metadata["prepared_extension_attestation"]
            if not isinstance(attestation, Mapping) or not str(
                attestation.get("scientific_identity", "")
            ).startswith("sha256:"):
                raise ValueError("Invalid extension preparation attestation.")
            return str(attestation["scientific_identity"])
        value = metadata.get("pinned_scientific_identity")
        return str(value) if value is not None else None

    def refresh(self, series: str) -> dict[str, Any]:
        """Explicitly reconcile submitted operations with their native retained outcomes."""
        store = StudySeriesStore(self.project)
        value = store.read(series)
        for operation in value["operations"]:
            if operation["state"] not in {"dispatching", "submitted", "failed"}:
                continue
            separate = value["revisions"][operation["revision"]].get(
                "code_compatibility", {}
            ).get("kind") == "revisioned-current"
            job, recovery_jobs = self._operation_job(operation)
            job = self.jobs.get(job.job_id, refresh=True, include_study=False)
            binding = job.metadata.get("study_extension", {})
            if (
                binding.get("series_id") != series
                or binding.get("request_id") != operation["request_id"]
            ):
                raise ValueError("Extension Job is not owned by its exact series request.")
            if recovery_jobs != operation.get("recovery_jobs", []):
                operation = store.transition(
                    series,
                    operation["request_id"],
                    state="submitted",
                    facts={"recovery_jobs": recovery_jobs},
                )
            if not job.state.terminal:
                continue
            if job.state.value != "succeeded":
                store.transition(series, operation["request_id"], state="failed", facts={})
                continue
            if job.cluster != "local":
                snapshot = self.jobs.extension_evidence(job.job_id)
                runtime = self._remote_runtime(job, snapshot)
                from lambdaforge.work.extension_design import candidate_id, cell_id

                expected = {
                    cell["cell_id"]
                    for cell in value["revisions"][operation["revision"]]["new_cells"]
                }
                actual = {
                    cell_id(candidate_id(row["trial"]["parameters"]), row["seed"])
                    for row in snapshot["evidence"]
                }
                if (
                    actual != expected
                    or (not separate and runtime["fingerprint"]
                        != value["revisions"][0]["runtime_fingerprint"])
                    or snapshot["manifest"]["scientific_fingerprint"]
                    != self._execution_pin(job.metadata)
                ):
                    raise ValueError(
                        "Completed remote delta differs from its reviewed cells/runtime/identity."
                    )
                if operation["state"] == "dispatching":
                    store.transition(series, operation["request_id"], state="submitted", facts={})
                store.transition(
                    series,
                    operation["request_id"],
                    state="succeeded",
                    facts={
                        "execution_id": snapshot["manifest"]["execution_id"],
                        "evidence": self._cell_references(snapshot["evidence"]),
                        "spent_runs": snapshot["spent_runs"],
                        "elapsed_seconds": snapshot["elapsed_seconds"],
                        "manifest_sha256": snapshot["manifest_sha256"],
                    },
                )
                continue
            detail = self.jobs.work_result_view(job.job_id, view="overview")
            execution_id = str(detail["execution_id"])
            native = self.results.select(execution_id)
            if native.get("scientific_fingerprint") != job.metadata.get(
                "pinned_scientific_identity"
            ):
                raise ValueError(
                    "Completed delta scientific identity differs from its pinned submission; "
                    "refusing to accumulate incompatible evidence."
                )
            directory = self.results.execution_directory(execution_id)
            manifest = validate_execution(directory)
            if manifest.get("job_id") != job.metadata.get("recovery_owner_job", job.job_id):
                raise ValueError("Extension Execution does not belong to its submitted Job.")
            frozen = restore_seed_context(
                WorkConfig.from_mapping(
                    self._origin_configuration(directory),
                    source=Path(str(manifest["source"])),
                ),
                manifest,
            )
            validate_recovery_configuration(frozen, manifest)
            _validate_fixed_recovery(
                fixed_definition(frozen), Path(str(manifest["source"])), directory, manifest
            )
            inventory = fixed_inventory(directory, manifest=manifest)
            runtime = verify_local_runtime(list(inventory["latest"].values()), directory)
            if not separate and runtime["fingerprint"] != value["revisions"][0].get(
                "runtime_fingerprint"
            ):
                raise ValueError(
                    "Delta runtime differs from its attested origin; cannot accumulate it."
                )
            from lambdaforge.work.extension_design import candidate_id, cell_id

            expected = {
                cell["cell_id"] for cell in value["revisions"][operation["revision"]]["new_cells"]
            }
            cells = {
                cell_id(candidate_id((run.get("trial") or {}).get("parameters", {})), run["seed"])
                for run in inventory["latest"].values()
                if run.get("status") == "succeeded" and not run.get("pruned")
            }
            if (
                native.get("status") != "succeeded"
                or cells != expected
                or len(inventory["latest"]) != len(expected)
            ):
                raise ValueError(
                    "Extension completion does not cover exactly its required new cells."
                )
            if operation["state"] == "dispatching":
                store.transition(
                    series, operation["request_id"], state="submitted", facts={"job_id": job.job_id}
                )
            store.transition(
                series,
                operation["request_id"],
                state="succeeded",
                facts={
                    "execution_id": execution_id,
                    "evidence": self._cell_references(
                        self._local_snapshot(directory, manifest)["evidence"]
                    ),
                    "spent_runs": inventory["spent_runs"],
                    "elapsed_seconds": inventory["elapsed_seconds"],
                },
            )
        return store.read(series)

    @staticmethod
    def _local_snapshot(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
        from lambdaforge.work.extension_evidence import inspect_extension_execution

        return inspect_extension_execution(str(directory), manifest.get("job_id"))

    def overview(self, series: str, *, include_live: bool = False) -> dict[str, Any]:
        """Cumulative metadata projection; physical provenance is never renumbered on disk."""
        from collections import Counter

        value = self.show(series)
        origin = value["revisions"][0]
        required = value["revisions"][-1]["required_cells"]
        refs = [
            {**row, "origin_revision": 0, "job_id": origin.get("job_id")}
            for row in origin["evidence"]
        ]
        operations = []
        spent = value["revisions"][1]["budget"]["historical_attempts"]
        elapsed = value["revisions"][1]["budget"]["historical_elapsed_seconds"]
        for operation in value["operations"]:
            job, lineage = self._operation_job(operation) if operation.get("job_id") else (None, [])
            operations.append(
                {
                    **operation,
                    "native_state": job.state.value if job else "planned",
                    "latest_job_id": job.job_id if job else None,
                    "recovery_jobs": lineage,
                }
            )
            refs.extend(
                {
                    **row,
                    "origin_revision": operation["revision"],
                    "job_id": job.job_id if job else operation.get("job_id"),
                }
                for row in operation.get("evidence", [])
            )
            spent += operation.get("spent_runs", 0)
            elapsed += operation.get("elapsed_seconds", 0)
        by_cell = {row["cell_id"]: row for row in refs}
        if len(by_cell) != len(refs):
            raise ValueError("Duplicate completed evidence across series revisions.")
        candidates: dict[str, dict[str, Any]] = {}
        runs = []
        pending_by_cell = {
            cell["cell_id"]: operation
            for operation in operations
            for cell in value["revisions"][operation["revision"]]["new_cells"]
        }
        for cell in sorted(required, key=lambda row: (row["proposed_trial"], row["seed"])):
            reference = by_cell.get(cell["cell_id"])
            candidate = candidates.setdefault(
                cell["candidate_id"],
                {
                    "trial": cell["proposed_trial"],
                    "candidate_id": cell["candidate_id"],
                    "parameters": cell["parameters"],
                    "runs": [],
                },
            )
            pending = pending_by_cell.get(cell["cell_id"], {})
            pending_revision = pending.get("revision")
            projected = (
                {**reference}
                if reference
                else {
                    **cell,
                    "status": "incomplete"
                    if pending.get("native_state") in {"failed", "cancelled", "timeout"}
                    else "pending",
                    "job_state": pending.get("native_state"),
                    "origin_revision": pending_revision,
                    "job_id": pending.get("latest_job_id"),
                    "metrics": {},
                }
            )
            projected["presentation_trial"] = cell["proposed_trial"]
            projected["code_cohort"] = f"revision-{projected.get('origin_revision', 0)}"
            runs.append(projected)
            candidate["runs"].append(projected)
        configuration = value["revisions"][-1]["configuration"]
        normalized, _ = normalize_evidence(
            {"candidates": list(candidates.values())}, configuration["objective"]
        )
        mixed_code = any(
            revision.get("code_compatibility", {}).get("kind") == "revisioned-current"
            for revision in value["revisions"]
        )
        for candidate, native in zip(candidates.values(), normalized, strict=True):
            observed = [row.get("final_objective") for row in native["runs"]]
            comparable = bool(observed) and all(
                isinstance(number, int | float) and not isinstance(number, bool)
                for number in observed
            )
            complete_candidate = comparable
            if mixed_code and len({row["code_cohort"] for row in candidate["runs"]}) > 1:
                comparable = False
                candidate["comparison_status"] = "separate-code-cohorts"
            candidate["selection_objective"] = sum(observed) / len(observed) if comparable else None
            candidate["state"] = "succeeded" if complete_candidate else "incomplete"
        states = Counter(row["status"] for row in runs)
        complete = len(refs) == len(required)
        output = {
            "study_series_view_version": 1,
            "study_series_id": series,
            "name": configuration.get("name") or "Study series",
            "configuration": configuration,
            "objective": configuration["objective"],
            "state": "succeeded"
            if complete
            else (operations[-1]["native_state"] if operations else "planned"),
            "revisions": value["revisions"],
            "operations": operations,
            "counts": {
                "required": len(required),
                "completed": len(refs),
                "pending": len(required) - len(refs),
                "candidates": len(candidates),
            },
            "run_states": dict(states),
            "runs": runs,
            "candidates": list(candidates.values()),
            "mixed_code": mixed_code,
            "physical_cost": {"spent_attempts": spent, "elapsed_seconds": elapsed},
            "interpretation": "exploratory post-hoc extension; not a confirmatory redesign",
            "evidence_fingerprint": ScientificIdentity.from_payload(
                {
                    "series": series,
                    "revision": value["current_revision"],
                    "evidence": refs,
                    "required_cells": required,
                }
            ).digest,
        }
        return self._live_overview(output) if include_live else output

    def _live_overview(self, output: dict[str, Any]) -> dict[str, Any]:
        """Overlay native live metadata for display only, not accumulated scientific evidence."""
        from collections import Counter

        from lambdaforge.work.extension_design import candidate_id, cell_id

        output["live_observation_errors"] = []
        presented = {row["cell_id"]: row for row in output["runs"]}
        for operation in output["operations"]:
            if operation.get("evidence") or not operation.get("latest_job_id"):
                continue
            required = {
                cell["cell_id"] for cell in output["revisions"][operation["revision"]]["new_cells"]
            }
            try:
                detail = self.jobs.work_result_view(operation["latest_job_id"], view="overview")
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                output["live_observation_errors"].append(str(error)[:500])
                continue
            latest: dict[str, Any] = {}
            for attempt in detail.get("attempts", []):
                parameters = (attempt.get("trial") or {}).get("parameters")
                if not isinstance(parameters, Mapping) or type(attempt.get("seed")) is not int:
                    continue
                key = cell_id(candidate_id(parameters), attempt["seed"])
                if key not in required:
                    raise ValueError("Live delta Attempt is outside its exact series revision.")
                if int(attempt.get("attempt_number") or 0) >= int(
                    latest.get(key, {}).get("attempt_number") or 0
                ):
                    latest[key] = attempt
            for key, attempt in latest.items():
                presented[key] = {
                    **presented[key],
                    **{
                        field: attempt.get(field)
                        for field in (
                            "run_id",
                            "attempt_id",
                            "status",
                            "objective_observation",
                        )
                    },
                    "metrics": attempt.get("metrics") or {},
                    "live_unsealed": True,
                    "execution_id": detail["execution_id"],
                }
        output["runs"] = list(presented.values())
        output["run_states"] = dict(Counter(row["status"] for row in output["runs"]))
        return output

    def analysis(self, series: str) -> dict[str, Any]:
        """Use the single native analysis engine on a derived, provenance-preserving view."""
        view = self.overview(series)
        if view.get("mixed_code"):
            cohorts = {
                f"revision-{row['number']}": self._analysis_view(series, view, row["number"])
                for row in view["revisions"]
            }
            latest = f"revision-{view['revisions'][-1]['number']}"
            return {**cohorts[latest], "code_cohort_analyses": cohorts, "pooled_inference": False}
        return self._analysis_view(series, view)

    def _analysis_view(
        self, series: str, view: Mapping[str, Any], cohort: int | None = None
    ) -> dict[str, Any]:
        """Analyse one compatible evidence group, never infer across code revisions."""
        from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

        view = dict(view)
        configuration = view["configuration"]
        if cohort is not None:
            revision = view["revisions"][cohort]
            configuration = revision[
                "configuration" if cohort == 0 else "delta_configuration"
            ]
            own_runs = [row for row in view["runs"] if row.get("origin_revision") == cohort]
            view["runs"] = own_runs
            view["candidates"] = [
                {**candidate, "runs": [row for row in candidate["runs"]
                                       if row.get("origin_revision") == cohort]}
                for candidate in view["candidates"]
                if any(row.get("origin_revision") == cohort for row in candidate["runs"])
            ]
            view["state"] = (
                "succeeded" if own_runs and all(row["status"] == "succeeded" for row in own_runs)
                else "incomplete"
            )
            view["counts"] = {"pending": sum(row["status"] != "succeeded" for row in own_runs)}
            view["interpretation"] = (
                f"Code/runtime cohort revision-{cohort} only; cross-revision pooling disabled."
            )
            view.pop("physical_cost", None)
        source = {**view, "objective": view["objective"], "status": view["state"]}
        # Candidate indices are presentation coordinates only. The Run's native trial,
        # Execution, Run and Attempt identities remain verbatim in each candidate record.
        config = WorkConfig.from_mapping(
            configuration, source=Path(self.show(series)["revisions"][-1]["proposed_yaml"])
        )
        definition = fixed_definition(config)
        assert definition.study_design is not None
        if cohort is not None:
            indices = {row["candidate_id"]: row["original_trial"]
                       for row in candidate_rows(definition)}
            for candidate in source["candidates"]:
                candidate["trial"] = indices[candidate["candidate_id"]]
        source["design"] = plain(definition.study_design.to_dict())
        semantics = definition.analysis_semantics
        if cohort == 0:
            semantics = view["revisions"][0].get("analysis_semantics", semantics)
        elif cohort is not None:
            semantics = view["revisions"][cohort].get("analysis_semantics", semantics)
        source["analysis_semantics"] = plain(semantics)
        analysis = StudyAnalysis.compute(
            source,
            objective=view["objective"],
            authored_space=plain(definition.study_design.space),
            status="final" if not view["counts"]["pending"] else "provisional",
        )
        analysis["study_series"] = {
            "study_series_id": series,
            "evidence_fingerprint": view["evidence_fingerprint"],
            "interpretation": view["interpretation"],
            "revision_count": len(view["revisions"]),
            "mixed_code": view.get("mixed_code", False),
            "code_cohort": f"revision-{cohort}" if cohort is not None else "compatible-series",
            "pooled_inference": cohort is None,
            "provenance": [
                {
                    key: row.get(key)
                    for key in (
                        "cell_id",
                        "origin_revision",
                        "execution_id",
                        "run_id",
                        "attempt_id",
                        "job_id",
                        "original_trial",
                        "presentation_trial",
                    )
                }
                for row in view["runs"]
            ],
        }
        analysis["source"].update(study_series_id=series, study_series_name=view["name"])
        analysis["inference_scope"] = view["interpretation"]
        if isinstance(analysis.get("sweep_analysis"), dict):
            analysis["sweep_analysis"]["post_hoc_exploratory"] = True
        return analysis

    def export(self, series: str, destination: str | Path) -> dict[str, Any]:
        """Export the family as a read-only collection of exact native Execution packages.

        An individual native package is still an individual Execution, not a counterfeit
        executable series. Partial export is preserved with an explicit failure receipt.
        """
        from lambdaforge.analysis.Report import write_html
        from lambdaforge.controlplane.StudyExportService import StudyExportService
        from lambdaforge.work.models import atomic_json

        journal_snapshot = self.refresh(series)
        value = self.overview(series)
        parent = Path(destination).expanduser().absolute()
        if parent.resolve() != parent:
            raise ValueError("Series export destination cannot contain symbolic paths.")
        parent.mkdir(parents=True, exist_ok=True)
        stage = parent / (series + "--" + uuid4().hex[:12] + ".partial")
        stage.mkdir(mode=0o700)
        try:
            collection = []
            origin = value["revisions"][0]
            physical = [
                {
                    "revision": 0,
                    "execution_id": origin["execution_id"],
                    "latest_job_id": origin.get("job_id"),
                },
                *value["operations"],
            ]
            native = StudyExportService(self.catalog, jobs=self.jobs, factory=self.jobs.factory)
            for row in physical:
                target = stage / ("revision-" + str(row["revision"]))
                target.mkdir()
                if row.get("latest_job_id"):
                    receipt = native.export(str(row["latest_job_id"]), target)
                elif row.get("execution_id"):
                    receipt = self.results.export(str(row["execution_id"]), target)
                else:
                    receipt = {"state": "not-submitted", "no_execution_package": True}

                def portable(item: Any) -> Any:
                    if isinstance(item, Mapping):
                        return {key: portable(val) for key, val in item.items()}
                    if isinstance(item, (list, tuple)):
                        return [portable(val) for val in item]
                    if isinstance(item, str) and item.startswith(str(stage) + os.sep):
                        return str(Path(item).relative_to(stage))
                    return item

                collection.append({"revision": row["revision"], "package": portable(receipt)})
            journal = self.show(series)
            if journal["checksum"] != journal_snapshot["checksum"]:
                raise ValueError(
                    "Series changed during export; retry a consistent collection snapshot."
                )
            # Read-only provenance only: never treat archived local paths as owned state.
            atomic_json(stage / "series.json", journal)
            analysis = self.analysis(series)
            atomic_json(stage / "analysis.json", analysis)
            write_html(analysis, stage / "analysis.html")
            if (
                self.show(series)["checksum"] != journal_snapshot["checksum"]
                or analysis["study_series"]["evidence_fingerprint"] != value["evidence_fingerprint"]
            ):
                raise ValueError(
                    "Series evidence changed during report export; retry the snapshot."
                )
            atomic_json(
                stage / "collection.json",
                {
                    "study_series_export_version": 1,
                    "read_only": True,
                    "study_series_id": series,
                    "evidence_fingerprint": value["evidence_fingerprint"],
                    "state": value["state"],
                    "executions": collection,
                    "import_policy": "Native Execution packages may be imported individually "
                    "as read-only evidence. "
                    "Executable series adoption/migration is not supported.",
                },
            )
            final = stage.with_suffix("")
            stage.rename(final)
            return {
                "study_series_id": series,
                "path": str(final),
                "read_only": True,
                "execution_packages": len(collection),
            }
        except Exception as error:
            atomic_json(
                stage / "export-failure.json", {"status": "incomplete", "error": str(error)}
            )
            raise RuntimeError(
                f"Series export incomplete; retained partial package at {stage}: {error}"
            ) from error

    def _operation_job(self, operation: Mapping[str, Any]) -> tuple[Any, list[str]]:
        """Follow exact native retry lineage; never choose a new Job by name or timestamp."""
        job = self.jobs.get(str(operation["job_id"]), refresh=False)
        cluster = job.cluster
        binding = job.metadata.get("study_extension", {})
        records = self.jobs.list(refresh=False)
        recovered = []
        seen = {job.job_id}
        while True:
            children = [record for record in records if record.retry_of == job.job_id]
            if not children:
                break
            if len(children) != 1 or children[0].job_id in seen:
                raise ValueError("Ambiguous/cyclic native recovery lineage; reconcile exact Jobs.")
            job = children[0]
            if job.cluster != cluster or job.metadata.get("study_extension", {}) != binding:
                raise ValueError("Native retry lineage changed extension ownership or placement.")
            recovered.append(job.job_id)
            seen.add(job.job_id)
        return job, recovered

    def _target(self, study: str, target: str | None) -> str:
        if target is not None:
            return target
        if study.startswith("series-"):
            return str(self.show(study)["revisions"][0].get("target", "local"))
        if study.startswith("job-"):
            return self.jobs.get(study, refresh=False).cluster
        from lambdaforge.controlplane.WorkService import WorkService

        try:
            return WorkService(self.catalog, jobs=self.jobs).show(study).cluster
        except KeyError:
            return "local"

    def preview(
        self,
        study: str,
        config: str | Path,
        *,
        target: str | None = None,
        code_policy: str = "revisioned",
    ) -> dict[str, Any]:
        """Read-only verified plan. Exact IDs work; existing native selectors reject ambiguity."""
        selected = Path(config).expanduser().absolute()
        if code_policy not in {"strict", "revisioned"}:
            raise ValueError("Extension code policy must be strict or revisioned.")
        if (
            selected.resolve() != selected
            or not selected.is_file()
            or selected.stat().st_size > 1024**2
        ):
            raise ValueError("Extension YAML requires a bounded regular non-symbolic owned path.")
        if not self.project.owns_source(str(selected)):
            raise ValueError("Extension YAML belongs to another project.")
        target = self._target(study, target)
        profile = self.catalog.get(target)
        store = StudySeriesStore(self.project)
        if study.startswith("series-"):
            origin = store.read(study)["revisions"][0]
            if origin.get("target", "local") != target:
                raise ValueError("Extension must use the original supported placement.")
            study = str(origin.get("job_id") or origin["execution_id"])
        elif store.root.exists():
            for directory in sorted(store.root.iterdir()):
                if directory.name == ".locks":
                    continue
                value = store.read(directory.name)
                if any(
                    study in {row.get("job_id"), row.get("execution_id")}
                    for row in value["operations"]
                ):
                    origin = value["revisions"][0]
                    study = str(origin.get("job_id") or origin["execution_id"])
                    break
        if target != "local":
            return self._preview_remote(study, selected, profile, code_policy=code_policy)
        if profile.transport != "local" or profile.scheduler != "local":
            raise ValueError("Local extension requires the built-in local process placement.")
        if study.startswith("job-"):
            study = str(self.jobs.work_result_view(study, view="overview")["execution_id"])
        executable = shutil.which(profile.python)
        if (
            profile.environment != "existing"
            or profile.command_prefix
            or executable is None
            or Path(executable).resolve() != Path(sys.executable).resolve()
        ):
            raise ValueError(
                "Selected execution environment cannot be attested as this CPU interpreter."
            )
        source_bytes = selected.read_bytes()
        proposed = WorkConfig.from_yaml(selected)
        new_definition = fixed_definition(proposed)
        if new_definition.resources.gpu_count:
            raise ValueError("GPU extension runtime attestation is not integrated yet.")
        original = self.results.select(study)
        if original.get("imported") or original.get("already_deleted"):
            raise ValueError("Imported/deleted evidence is not executable Study ownership.")
        if original.get("status") != "succeeded":
            raise ValueError(
                "Complete required evidence with native retry before extending the Study."
            )
        execution = self.results.execution_directory(study)
        with CrossProcessFileLock(
            execution / ".controller.lock",
            shared=True,
            create=False,
            timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        ):
            return self._preview_owned(
                execution, original, selected, proposed, source_bytes, target, profile,
                code_policy=code_policy,
            )

    def _remote_runtime(
        self, job: Any, snapshot: Mapping[str, Any], *, require_sparse: bool = False
    ) -> dict[str, Any]:
        """Reuse an exact owned managed prefix, not a newly solved dependency set."""
        from pathlib import PurePosixPath

        from lambdaforge.work.extension_runtime import inspect_retained_runtime, runtime_contract

        profile, transport, _ = self.jobs._provider(job)
        assert profile.storage is not None
        environment_id = job.metadata.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id.startswith("env-"):
            raise ValueError(
                "Remote extension requires retained native managed environment provenance."
            )
        prefix = str(PurePosixPath(profile.storage.environment_root) / environment_id)
        contracts = [runtime_contract(env) for env in snapshot["environments"]]
        from lambdaforge._version import VERSION

        if not contracts or contracts[0]["framework"] != VERSION:
            raise ValueError(
                "Origin framework/runtime is not compatible with this extension controller."
            )
        if any(
            env.get("python", {}).get("executable") != prefix + "/bin/python"
            for env in snapshot["environments"]
        ):
            raise ValueError("Historical Python is outside its exact managed environment.")
        script = (
            "from __future__ import annotations\nfrom typing import Any\nimport json,sys\n"
            + inspect.getsource(inspect_retained_runtime)
            + "\nprint(json.dumps(inspect_retained_runtime(*json.loads(sys.argv[1]))))\n"
        )
        # No GPU import/probe/claim here: hardware is checked only inside the ordinary launcher.
        response = transport.run(
            (
                prefix + "/bin/python", "-c", script,
                json.dumps([prefix, environment_id, contracts, require_sparse]),
            ),
            timeout=30,
        )
        if response.returncode:
            raise ValueError("Cannot attest retained extension runtime: " + response.stderr[-1500:])
        retained = json.loads(response.stdout)
        return {
            "scope": "same-placement retained managed environment; GPU checked in launcher",
            "fingerprint": ScientificIdentity.from_payload(contracts[0]).digest,
            "retained_runtime": retained,
        }

    def _preview_remote(
        self, study: str, selected: Path, profile: ClusterProfile, *, code_policy: str
    ) -> dict[str, Any]:
        from lambdaforge.controlplane.WorkService import WorkService

        if not study.startswith("job-"):
            work = WorkService(jobs=self.jobs, catalog=self.catalog).show(study)
            study = str(work.primary_job_id)
        job = self.jobs.get(study, refresh=False)
        if job.cluster != profile.name:
            raise ValueError("Extension must stay on its verified original placement.")
        if job.metadata.get("fleet") or job.metadata.get("fleet_role"):
            raise ValueError("Fleet extension requires native coordinator integration.")
        snapshot = self.jobs.extension_evidence(job.job_id)
        manifest = snapshot["manifest"]
        source_bytes = selected.read_bytes()
        namespace = manifest["study_designs"][0]["seed_source"]["namespace"]
        ordered = json.loads(snapshot["configuration_ordered_json"])
        if exact(ordered) != exact(snapshot["configuration"]):
            raise ValueError("Remote ordered configuration differs from its frozen evidence.")
        previous = WorkConfig.from_mapping(
            ordered, source=selected, seed_namespace=namespace
        )
        proposed = WorkConfig.from_mapping(
            WorkConfig.from_yaml(selected).raw, source=selected, seed_namespace=namespace
        )
        fixed_definition(previous)
        definition = fixed_definition(proposed)
        source = inspect.getsourcefile(import_work_class(definition.work_class))
        if source is None or not Path(source).resolve().is_relative_to(self.project.root):
            raise ValueError("Executable Work source is outside the attested consumer project.")
        code = review_extension_code(manifest["code_identity"], self.project, selected, code_policy)
        revisioned = code["kind"] == "revisioned-current"
        if revisioned:
            if profile.environment != "managed":
                raise ValueError(
                    "Changed-code remote extension requires native managed deployment."
                )
            code, runtime = self._new_runtime_cohort(snapshot, code, code.get("reason", ""))
        else:
            try:
                runtime = self._remote_runtime(job, snapshot)
            except ValueError as error:
                if code_policy != "revisioned" or profile.environment != "managed":
                    raise
                code, runtime = self._new_runtime_cohort(snapshot, code, str(error))
                revisioned = True
        comparison, refs, spent, elapsed, revision = self._accumulated_plan(
            previous,
            proposed,
            selected,
            manifest,
            [
                {**row, "origin_revision": 0, "job_id": job.job_id}
                for row in self._cell_references(snapshot["evidence"])
            ],
            runtime,
            snapshot["spent_runs"],
            snapshot["elapsed_seconds"],
            profile.name,
            snapshot["fixed_identities"],
        )
        budget = definition.execution_policy
        remaining = budget.max_runs - spent if budget.max_runs is not None else None
        seconds = budget.max_time_seconds - elapsed if budget.max_time_seconds is not None else None
        new = comparison["counts"]["new"]
        if new and (
            remaining is not None and remaining < new or seconds is not None and seconds <= 0
        ):
            raise ValueError(
                "Extension budget is insufficient after preserving historical Attempt cost."
            )
        delta = comparison["delta_configuration"]
        if delta is not None:
            if not revisioned and ("candidates" in delta["sweep"] or "include" in delta["sweep"]):
                try:
                    runtime = self._remote_runtime(job, snapshot, require_sparse=True)
                except ValueError as error:
                    if code_policy != "revisioned" or profile.environment != "managed":
                        raise
                    code, runtime = self._new_runtime_cohort(snapshot, code, str(error))
            delta.setdefault("name", proposed.name)
            operational = delta.setdefault("execution", {})
            if remaining is not None:
                operational["max_runs"] = remaining
            if seconds is not None:
                operational["max_time"] = seconds
            WorkConfig.from_mapping(delta, source=selected)
        if selected.read_bytes() != source_bytes:
            raise ValueError("Proposed YAML changed during preview; inspect a new plan.")
        from lambdaforge.controlplane.StudyExtensionDispatch import native_delta_contract

        payload = {
            "study_extension_plan_version": 1,
            "code_policy": code_policy,
            "study_series_id": self._series_id(self.project.project_id, manifest),
            "project": self.project.to_dict(),
            "target": profile.name,
            "placement": "remote",
            "delta_contract": native_delta_contract(WorkConfig.from_mapping(delta, source=selected))
            if delta is not None else None,
            "expected_revision": revision,
            "origin_execution_id": manifest["execution_id"],
            "origin_execution_dir": snapshot["execution_dir"],
            "origin_job_id": job.job_id,
            "origin_manifest": manifest,
            "origin_configuration": snapshot["configuration"],
            "origin_analysis_semantics": snapshot.get("analysis_semantics", {}),
            "delta_analysis_semantics": plain(definition.analysis_semantics),
            "origin_manifest_sha256": snapshot["manifest_sha256"],
            "proposed_yaml": str(selected),
            "proposed_yaml_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "target_profile_fingerprint": ScientificIdentity.from_payload(profile.to_dict()).digest,
            "comparison": comparison,
            "reused_evidence": refs,
            "code_compatibility": code,
            "fixed_input_identities": _identity_values(definition.parameters, selected.parent),
            "runtime_compatibility": runtime,
            "budget": {
                "historical_attempts": spent,
                "historical_elapsed_seconds": elapsed,
                "new_required_runs": new,
                "remaining_runs": remaining,
                "remaining_seconds": seconds,
            },
            "apply_supported": True,
            "series_persisted": False,
            "boundary": (
                "native same-placement managed deployment; retain only with compatibility proof; "
                "no migration or Fleet"
            ),
        }
        payload["plan_hash"] = ScientificIdentity.from_payload(payload).digest
        return json.loads(json.dumps(payload, allow_nan=False))

    def _new_runtime_cohort(
        self, snapshot: Mapping[str, Any], proof: Mapping[str, Any], reason: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from lambdaforge.work.extension_runtime import runtime_contract

        return (
            {
                **proof, "kind": "revisioned-current", "equivalent": False,
                "review_pin": code_review_pin(self.project), "reason": reason,
                "warning": "New current-project code/runtime cohort. Prior evidence is retained; "
                           "cross-revision statistical pooling is disabled.",
            },
            {
                "scope": "new native managed code/runtime cohort; no old-environment reuse",
                "fingerprint": ScientificIdentity.from_payload(
                    runtime_contract(snapshot["environments"][0])
                ).digest,
            },
        )

    def _revalidate_remote(self, plan: Mapping[str, Any]) -> None:
        fresh = self.preview(
            str(plan["origin_job_id"]), str(plan["proposed_yaml"]), target=str(plan["target"]),
            code_policy=str(plan.get("code_policy", "strict")),
        )
        if fresh["plan_hash"] != plan["plan_hash"]:
            raise ValueError("Remote scientific evidence changed since review; inspect a new plan.")

    def _preview_owned(
        self,
        execution: Path,
        original: Mapping[str, Any],
        selected: Path,
        proposed: WorkConfig,
        source_bytes: bytes,
        target: str,
        profile: ClusterProfile,
        *,
        code_policy: str,
    ) -> dict[str, Any]:
        new_definition = fixed_definition(proposed)
        manifest = validate_execution(execution)
        job_id = manifest.get("job_id")
        if job_id:
            record = self.jobs.get(str(job_id), refresh=False)
            if record.metadata.get("study_extension"):
                raise ValueError(
                    "This Execution is already a series delta. Further expansion requires "
                    "the accumulated series planner; do not replay it as a new origin."
                )
            if record.metadata.get("fleet") or record.metadata.get("fleet_role"):
                raise ValueError("Fleet extension requires native coordinator integration.")
            if record.cluster != target or record.state.value != "succeeded":
                raise ValueError(
                    "Original Job ownership/state is not a confirmed successful local target; "
                    "UNKNOWN is not permission to extend."
                )
        original_source = Path(str(manifest.get("source", "")))
        if original_source.resolve() != original_source or not self.project.owns_source(
            str(original_source)
        ):
            raise ValueError("Original Execution belongs to another/unverifiable project.")
        previous = restore_seed_context(
            WorkConfig.from_mapping(
                self._origin_configuration(execution), source=original_source
            ),
            manifest,
        )
        old_definition = fixed_definition(previous)
        source = inspect.getsourcefile(import_work_class(old_definition.work_class))
        if (
            source is None
            or Path(source).absolute() != Path(source).resolve()
            or not Path(source).resolve().is_relative_to(self.project.root)
        ):
            raise ValueError("Executable Work source is outside the attested consumer project.")
        origin_code = manifest.get("code_identity")
        if not isinstance(origin_code, Mapping):
            raise ValueError("Origin code compatibility evidence is missing or corrupt.")
        code = review_extension_code(origin_code, self.project, selected, code_policy)
        # These are the existing fail-closed contracts, not an extension exception to recovery.
        validate_recovery_configuration(previous, manifest)
        _validate_fixed_recovery(old_definition, original_source, execution, manifest)
        comparison = compare_fixed_designs(
            previous,
            previous,
            fixed_identities=(
                _identity_values(old_definition.parameters, original_source.parent),
                _identity_values(old_definition.parameters, original_source.parent),
            ),
        ).to_dict()
        inventory = fixed_inventory(execution, manifest=manifest)
        rows = {row["original_trial"]: row for row in candidate_rows(old_definition)}
        identities = {}
        for trial, variant in enumerate(old_definition.variants, 1):
            _parameters, inputs, normalized = _resolve_inputs(
                {**old_definition.parameters, **variant}, original_source.parent
            )
            for seed in old_definition.seeds:
                identities[trial, seed] = ScientificIdentity.from_payload(
                    {
                        "identity_version": 1,
                        "work_class": old_definition.work_class,
                        "code": manifest["code_identity"],
                        "parameters": normalized,
                        "inputs": [value.identity_dict() for value in inputs.values()],
                        "seed": seed,
                        "trial_parameters": plain(variant),
                    }
                ).digest
        attested = {
            (value["execution_id"], value["run_id"], value["attempt_id"]): value
            for value in original.get("runs", ())
            if isinstance(value, Mapping)
        }
        assert old_definition.objective is not None
        evidence = []
        for cell in comparison["reused_cells"]:
            run = inventory["latest"].get((cell["original_trial"], cell["seed"]))
            if run is None or run.get("status") != "succeeded" or run.get("pruned"):
                raise ValueError(
                    "Required historical cell is incomplete; repair it with native retry."
                )
            if rows[cell["original_trial"]]["candidate_id"] != cell["candidate_id"]:
                raise ValueError("Historical candidate identity does not match its cell.")
            if run["scientific_fingerprint"] != identities[cell["original_trial"], cell["seed"]]:
                raise ValueError(
                    "Historical Run scientific fingerprint/code differs from its origin contract."
                )
            saved = attested.get((run["execution_id"], run["run_id"], run["attempt_id"]))
            if saved is None or any(
                exact(saved.get(field)) != exact(run.get(field))
                for field in (
                    "status",
                    "scientific_fingerprint",
                    "metrics",
                    "parameters",
                    "inputs",
                    "seed_metadata",
                    "objective_observation",
                )
            ):
                raise ValueError(
                    "Aggregate and authoritative Attempt evidence disagree or are incomplete."
                )
            _candidates, observations = normalize_evidence(
                {"runs": [run]}, old_definition.objective
            )
            if not observations or observations[0].get("final_objective") is None:
                raise ValueError("Historical completed Run lacks the declared objective evidence.")
            evidence.append(
                {
                    **cell,
                    "execution_id": run["execution_id"],
                    "run_id": run["run_id"],
                    "attempt_id": run["attempt_id"],
                    "scientific_fingerprint": run["scientific_fingerprint"],
                    "result_path": str(Path(run["run_dir"]) / "result.json"),
                    "result_sha256": hashlib.sha256(
                        (Path(run["run_dir"]) / "result.json").read_bytes()
                    ).hexdigest(),
                    "status": "succeeded",
                    "metrics": plain(run.get("metrics", {})),
                    "objective_observation": plain(run.get("objective_observation")),
                    "inputs": plain(run.get("inputs", [])),
                    "seed_metadata": plain(run.get("seed_metadata", {})),
                }
            )
        runtime = verify_local_runtime(list(inventory["latest"].values()), execution)
        comparison, evidence, spent, elapsed, revision = self._accumulated_plan(
            previous,
            proposed,
            selected,
            manifest,
            evidence,
            runtime,
            inventory["spent_runs"],
            inventory["elapsed_seconds"],
            target,
            _identity_values(old_definition.parameters, original_source.parent),
        )
        new_count = comparison["counts"]["new"]
        budget = new_definition.execution_policy
        remaining = budget.max_runs - spent if budget.max_runs is not None else None
        if remaining is not None and remaining < new_count:
            raise ValueError(
                "Extension Run budget is insufficient after preserving historical Attempt cost."
            )
        remaining_time = (
            budget.max_time_seconds - elapsed if budget.max_time_seconds is not None else None
        )
        if new_count and remaining_time is not None and remaining_time <= 0:
            raise ValueError(
                "Extension time budget was already spent; no historical cost is refunded."
            )
        delta = comparison["delta_configuration"]
        if delta is not None:
            delta.setdefault("name", proposed.name)
            delta_execution = delta.setdefault("execution", {})
            if remaining is not None:
                delta_execution["max_runs"] = remaining
            if remaining_time is not None:
                delta_execution["max_time"] = remaining_time
            # Sparse and rectangular deltas share the native configuration contract.
            WorkConfig.from_mapping(delta, source=selected)
        if selected.read_bytes() != source_bytes:
            raise ValueError("Proposed YAML changed during preview; inspect a new plan.")
        series = (
            "series-"
            + ScientificIdentity.from_payload(
                {
                    "series_identity_version": 1,
                    "project_id": self.project.project_id,
                    "origin_execution": manifest["execution_id"],
                    "origin_scientific_identity": manifest["scientific_fingerprint"],
                }
            ).digest.removeprefix("sha256:")[:24]
        )
        payload = {
            "study_extension_plan_version": 1,
            "code_policy": code_policy,
            "study_series_id": series,
            "project": self.project.to_dict(),
            "origin_execution_id": manifest["execution_id"],
            "origin_job_id": job_id,
            "origin_execution_dir": str(execution),
            "origin_manifest": plain(manifest),
            "origin_configuration": plain(previous.raw),
            "origin_analysis_semantics": read_owned_json(
                execution / "analysis-semantics.json"
            ).get(old_definition.name, {}),
            "delta_analysis_semantics": plain(new_definition.analysis_semantics),
            "placement": "local",
            "expected_revision": revision,
            "origin_manifest_sha256": hashlib.sha256(
                (execution / "execution.json").read_bytes()
            ).hexdigest(),
            "proposed_yaml": str(selected),
            "proposed_yaml_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "target": target,
            "target_profile_fingerprint": ScientificIdentity.from_payload(profile.to_dict()).digest,
            "comparison": comparison,
            "reused_evidence": evidence,
            "code_compatibility": code,
            "fixed_input_identities": _identity_values(new_definition.parameters, selected.parent),
            "runtime_compatibility": runtime,
            "budget": {
                "historical_attempts": spent,
                "historical_elapsed_seconds": elapsed,
                "new_required_runs": new_count,
                "remaining_runs": remaining,
                "remaining_seconds": remaining_time,
            },
            "apply_supported": True,
            "series_persisted": False,
            "boundary": (
                "verified same-interpreter local CPU delta through native asynchronous submission"
            ),
        }
        payload["plan_hash"] = ScientificIdentity.from_payload(payload).digest
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    @staticmethod
    def _check_reviewed_code(plan: Mapping[str, Any]) -> None:
        project = ProjectContext(Path(plan["project"]["root"]), plan["project"]["project_id"])
        proof = plan["code_compatibility"]
        if proof.get("review_pin"):
            if code_review_pin(project) != proof["review_pin"]:
                raise ValueError("Current code changed after extension review; review a new plan.")
        else:
            verify_code(proof["origin"], project, Path(plan["proposed_yaml"]))

    @staticmethod
    def _origin_configuration(directory: Path) -> dict[str, Any]:
        from lambdaforge.work.extension_evidence import read_extension_configuration

        return read_extension_configuration(directory)

    @staticmethod
    def _series_id(project_id: str, manifest: Mapping[str, Any]) -> str:
        return (
            "series-"
            + ScientificIdentity.from_payload(
                {
                    "series_identity_version": 1,
                    "project_id": project_id,
                    "origin_execution": manifest["execution_id"],
                    "origin_scientific_identity": manifest["scientific_fingerprint"],
                }
            ).digest.removeprefix("sha256:")[:24]
        )

    def _accumulated_plan(
        self,
        previous: WorkConfig,
        proposed: WorkConfig,
        selected: Path,
        manifest: Mapping[str, Any],
        evidence: list[dict[str, Any]],
        runtime: Mapping[str, Any],
        spent: int,
        elapsed: float,
        target: str,
        fixed_identities: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int, float, int]:
        """Subtract the entire last design, not just the first Execution's cells."""
        from lambdaforge.work.extension_design import candidate_id, cell_id
        from lambdaforge.work.extension_evidence import inspect_extension_execution

        series = self._series_id(self.project.project_id, manifest)
        store = StudySeriesStore(self.project)
        revision = 0
        namespace = manifest["study_designs"][0]["seed_source"]["namespace"]
        if (store.root / series).exists():
            prior = store.read(series)
            committed = [operation for operation in prior["operations"] if operation.get("job_id")]
            if committed:
                for operation in committed:
                    job, _ = self._operation_job(operation)
                    if job.cluster != target or job.state.value != "succeeded":
                        raise ValueError(
                            "A pending/failed Study extension owns this series; finish it with "
                            "native retry before adding another revision."
                        )
                    if target != "local":
                        snapshot = self.jobs.extension_evidence(job.job_id)
                    else:
                        execution_id = (
                            operation.get("execution_id")
                            or self.jobs.work_result_view(job.job_id, view="overview")[
                                "execution_id"
                            ]
                        )
                        snapshot = inspect_extension_execution(
                            str(self.results.execution_directory(str(execution_id))),
                            job.metadata.get("recovery_owner_job", job.job_id),
                        )
                    if snapshot["manifest"]["scientific_fingerprint"] != self._execution_pin(
                        job.metadata
                    ):
                        raise ValueError("Delta identity differs from its pinned submission.")
                    required = prior["revisions"][operation["revision"]]["new_cells"]
                    actual = self._cell_references(snapshot["evidence"])
                    if {row["cell_id"] for row in actual} != {row["cell_id"] for row in required}:
                        raise ValueError(
                            "Completed delta does not satisfy its exact revision cells."
                        )
                    checked_runtime = (
                        self._remote_runtime(job, snapshot)
                        if target != "local"
                        else verify_local_runtime(
                            list(
                                fixed_inventory(Path(snapshot["execution_dir"]))["latest"].values()
                            ),
                            Path(snapshot["execution_dir"]),
                        )
                    )
                    separate = prior["revisions"][operation["revision"]].get(
                        "code_compatibility", {}
                    ).get("kind") == "revisioned-current"
                    if not separate and checked_runtime["fingerprint"] != runtime["fingerprint"]:
                        raise ValueError("Delta runtime differs from its attested origin.")
                    evidence.extend(
                        {**row, "origin_revision": operation["revision"], "job_id": job.job_id}
                        for row in actual
                    )
                    spent += snapshot["spent_runs"]
                    elapsed += snapshot["elapsed_seconds"]
                revision = prior["current_revision"]
                previous = WorkConfig.from_mapping(
                    prior["revisions"][-1]["configuration"],
                    source=selected,
                    seed_namespace=namespace,
                )
        proposed = WorkConfig.from_mapping(proposed.raw, source=selected, seed_namespace=namespace)
        comparison = compare_fixed_designs(
            previous,
            proposed,
            fixed_identities=(
                fixed_identities,
                _identity_values(fixed_definition(proposed).parameters, selected.parent),
            ),
        ).to_dict()
        by_cell = {}
        for row in evidence:
            key = row.get("cell_id") or cell_id(
                candidate_id(row["trial"]["parameters"]), row["seed"]
            )
            if key in by_cell:
                raise ValueError("Duplicate scientific cell across Study revisions.")
            by_cell[key] = row
        if set(by_cell) != {cell["cell_id"] for cell in comparison["reused_cells"]}:
            raise ValueError("Accumulated evidence does not exactly cover the previous design.")
        refs = [{**cell, **by_cell[cell["cell_id"]]} for cell in comparison["reused_cells"]]
        return comparison, refs, spent, elapsed, revision

    @staticmethod
    def _cell_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from lambdaforge.work.extension_design import candidate_id, cell_id

        return [
            {
                **row,
                "candidate_id": candidate_id(row["trial"]["parameters"]),
                "cell_id": cell_id(candidate_id(row["trial"]["parameters"]), row["seed"]),
                "parameters": row["trial"]["parameters"],
                "original_trial": row["trial"]["index"],
            }
            for row in rows
        ]
