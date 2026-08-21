"""Concise global control-plane overview."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobObservation import JobObservation
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.data.DatasetService import DatasetService


class OverviewService:
    """Compose existing services; do not create another source of truth."""

    def __init__(self, catalog: ClusterCatalog | None = None) -> None:
        self.catalog = catalog or ClusterCatalog.load()

    def snapshot(self) -> dict[str, Any]:
        jobs = JobService(self.catalog)
        resources = ResourceService(self.catalog)
        datasets = DatasetService(clusters=self.catalog)
        with ThreadPoolExecutor(max_workers=3) as executor:
            job_future = executor.submit(self._jobs, jobs)
            resource_future = executor.submit(resources.all)
            dataset_future = executor.submit(datasets.list, all_clusters=True)
            job_values = job_future.result()
            resource_values = resource_future.result()
            dataset_values = dataset_future.result()
        return {
            "snapshot_version": 1,
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
                "items": [
                    {**value.to_dict(), **JobObservation.describe(value)} for value in job_values
                ],
            },
            "work": {
                "items": [value.to_dict() for value in aggregate_research_work(job_values)],
            },
            "datasets": {
                "versions": len(dataset_values),
                "placements": sum(len(value.placements) for value in dataset_values),
            },
            "offline_clusters": [value.cluster for value in resource_values if not value.online],
        }

    @staticmethod
    def _jobs(service: JobService) -> tuple[Any, ...]:
        """Reconcile every reachable provider before composing the global view."""
        service.reconcile(all_clusters=True)
        return service.list(refresh=False)
