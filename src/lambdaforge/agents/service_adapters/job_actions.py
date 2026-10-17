"""Exact project-owned Job actions; native providers retain process/ownership authority."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from lambdaforge.agents.errors import AgentApprovalError, AgentOperationError

if TYPE_CHECKING:
    from lambdaforge.agents.scope import ProjectScope
    from lambdaforge.controlplane.jobs import JobRecord
    from lambdaforge.controlplane.JobService import JobService


class JobActions:
    """Metadata previews and cancellation only; never launch, recover or delete science."""

    def __init__(self, scope: ProjectScope, jobs: JobService | None = None) -> None:
        self.scope = scope
        if jobs is None:
            from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
            from lambdaforge.controlplane.JobService import JobService

            jobs = JobService(ClusterCatalog.load(project=scope.project))
        self.jobs = jobs
        if jobs.store.project != scope.project or jobs.catalog.project != scope.project:
            raise PermissionError("Job service must be pinned to the approved project.")

    def _record(self, job_id: str, *, refresh: bool = False) -> JobRecord:
        self.scope.assert_current()
        if not isinstance(job_id, str) or not job_id.startswith("job-"):
            raise ValueError("Cancellation requires an exact Job ID, not a Work name.")
        try:
            record = self.jobs.get(job_id, refresh=refresh, include_study=False)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise AgentOperationError(
                "EVIDENCE_CORRUPT",
                "Job metadata is corrupt; inspect it through native diagnostics.",
            ) from error
        if (
            record.job_id != job_id
            or record.metadata.get("project_id") != self.scope.project.project_id
        ):
            raise PermissionError("Cancellation requires authoritative project-owned Job metadata.")
        if (
            record.metadata.get("fleet")
            or record.metadata.get("fleet_role")
            or record.metadata.get("fleet_parent_job")
        ):
            raise AgentOperationError(
                "UNSUPPORTED", "Agent cancellation of coordinated Fleet Jobs is not supported."
            )
        return record

    def preview(self, job_id: str) -> dict[str, Any]:
        """No provider call, artifact read or write. Volatile observation is not plan identity."""
        record = self._record(job_id)
        self._eligible(record)
        return self._binding(record)

    @staticmethod
    def _eligible(record: JobRecord) -> None:
        from lambdaforge.controlplane.jobs import JobState

        if record.state is JobState.UNKNOWN:
            raise AgentOperationError(
                "PROVIDER_UNCERTAIN",
                "Job ownership/state is unknown; refresh native health before cancellation.",
            )
        if record.state.terminal:
            raise AgentOperationError("OPERATION_CONFLICT", "Job is already terminal.")

    def _binding(self, record: JobRecord) -> dict[str, Any]:
        # Never copy raw command arguments (which can contain secrets) to approval records.
        profile = self.jobs.catalog.get(record.cluster)
        policy_hash = hashlib.sha256(
            json.dumps(profile.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        storage = {
            name: deepcopy(record.metadata.get(name))
            for name in (
                "provider_storage",
                "local_storage",
                "local_workspace",
                "source_config_path",
            )
        }
        return {
            "job_id": record.job_id,
            "project_id": record.metadata["project_id"],
            "cluster": record.cluster,
            "scheduler": record.scheduler,
            "scheduler_id": record.scheduler_id,
            "created_at_utc": record.created_at_utc,
            "work_dir": record.work_dir,
            "bundle_id": record.bundle_id,
            "scientific_identity": record.metadata.get("scientific_identity"),
            "command_hash": hashlib.sha256(json.dumps(list(record.command)).encode()).hexdigest(),
            "storage_hash": hashlib.sha256(
                json.dumps(storage, sort_keys=True).encode()
            ).hexdigest(),
            "provider": {
                "transport": profile.transport,
                "host": profile.host,
                "user": profile.user,
                "port": profile.port,
                "policy_hash": policy_hash,
            },
            "scope": "exact Job only; not every Job of a semantic Work",
        }

    def cancel(self, job_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Reobserve and validate the exact target before native process-tree cancellation."""
        record = self._record(job_id, refresh=True)
        self._eligible(record)
        if self._binding(record) != evidence:
            raise AgentApprovalError("PLAN_STALE", "Job ownership changed; request a new plan.")
        result = self.jobs.cancel(job_id)
        return {
            "job_id": result.job_id,
            "cluster": result.cluster,
            "state": result.state.value,
            "outcome": "cancellation_requested"
            if result.state.value == "cancelled"
            else "already_terminal",
            "scientific_evidence_deleted": False,
        }
