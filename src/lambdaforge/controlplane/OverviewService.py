"""Concise global control-plane overview."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobObservation import JobObservation
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.data.DatasetService import DatasetService


class OverviewService:
    """Compose existing services; do not create another source of truth."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()

    def snapshot(self) -> dict[str, Any]:
        jobs = JobService(self.catalog, factory=self.factory)
        resources = ResourceService(self.catalog, self.factory)
        datasets = DatasetService(clusters=self.catalog, factory=self.factory)
        with ThreadPoolExecutor(max_workers=3) as executor:
            job_future = executor.submit(self._jobs, jobs)
            resource_future = executor.submit(resources.all)
            # The overview needs registry counts, not remote manifests or member
            # inventories. Dataset details have their own explicit screen.
            dataset_future = executor.submit(datasets.list)
            job_values = job_future.result()
            resource_values = resource_future.result()
            dataset_values = dataset_future.result()
        return {
            "snapshot_version": 1,
            "project": self.catalog.project.to_dict() if self.catalog.project else None,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "clusters": [value.to_dict() for value in resource_values],
            "jobs": {
                "active": sum(
                    value.state.value in {"preparing", "staging", "queued", "running", "paused"}
                    for value in job_values
                ),
                "total": len(job_values),
                "by_state": {
                    state: sum(value.state.value == state for value in job_values)
                    for state in sorted({value.state.value for value in job_values})
                },
                "items": [self._job_overview(value) for value in job_values],
            },
            "work": {
                "items": [
                    value.to_dict(study_detail="overview")
                    for value in aggregate_research_work(job_values)
                ],
            },
            "datasets": {
                "versions": len(dataset_values),
                "placements": sum(len(value.placements) for value in dataset_values),
            },
            "offline_clusters": [value.cluster for value in resource_values if not value.online],
        }

    @staticmethod
    def _jobs(service: JobService) -> tuple[Any, ...]:
        """Use provider inventory once; refresh only schedulers without inventory."""
        service.reconcile(all_clusters=True)
        records = service.list(refresh=False)
        refreshable = tuple(
            value for value in records if value.scheduler == "slurm" and not value.state.terminal
        )
        if not refreshable:
            return records
        refreshed: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(refreshable))) as executor:
            futures = {
                executor.submit(service.get, value.job_id, include_study=False): value
                for value in refreshable
            }
            for future, original in futures.items():
                try:
                    refreshed[original.job_id] = future.result()
                except Exception:
                    refreshed[original.job_id] = original
        return tuple(refreshed.get(value.job_id, value) for value in records)

    def research_snapshot(self) -> dict[str, Any]:
        """Return the lightweight Work/Study collection without unrelated probes."""
        jobs = JobService(self.catalog, factory=self.factory)
        job_values = self._jobs(jobs)
        return {
            "snapshot_version": 1,
            "project": self.catalog.project.to_dict() if self.catalog.project else None,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "jobs": {
                "active": sum(not value.state.terminal for value in job_values),
                "total": len(job_values),
                "by_state": {
                    state: sum(value.state.value == state for value in job_values)
                    for state in sorted({value.state.value for value in job_values})
                },
            },
            "work": {
                "items": [
                    value.to_dict(study_detail="overview")
                    for value in aggregate_research_work(job_values)
                ]
            },
        }

    @staticmethod
    def _job_overview(value: Any) -> dict[str, Any]:
        """Project durable Job records without logs, commands or Study telemetry."""
        metadata = value.metadata
        return {
            "job_record_version": 2,
            "detail_level": "overview",
            "job_id": value.job_id,
            "cluster": value.cluster,
            "scheduler": value.scheduler,
            "scheduler_id": value.scheduler_id,
            "state": value.state.value,
            "resources": dict(value.resources),
            "created_at_utc": value.created_at_utc,
            "updated_at_utc": value.updated_at_utc,
            "retry_of": value.retry_of,
            "job_type": value.job_type,
            "group_id": value.group_id,
            "metadata": {
                key: metadata[key]
                for key in (
                    "name",
                    "scientific_identity",
                    "scientific_revision",
                    "study_expected",
                    "submission_phase",
                )
                if key in metadata
            },
            **JobObservation.describe(value),
        }
