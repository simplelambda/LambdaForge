"""Domain-service facade shared by Research Console screens, never CLI subprocesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ResourceService import ResourceService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.work.ResultStore import ResultStore


class ConsoleServices:
    """Construct reusable domain services once for one console session."""

    def __init__(self, catalog: ClusterCatalog | None = None) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.overview = OverviewService(self.catalog)
        self.jobs = JobService(self.catalog)
        self.works = WorkService(self.catalog)
        self.resources = ResourceService(self.catalog)
        self.datasets = DatasetService(clusters=self.catalog)
        self.results = ResultStore()

    def overview_snapshot(self) -> dict[str, Any]:
        return self.overview.snapshot()

    def cluster_rows(self) -> list[dict[str, Any]]:
        return [self.catalog.inspect(name) for name in self.catalog.names()]

    def dataset_rows(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.datasets.list()]

    def result_rows(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in record.items() if key != "_manifest_path"}
            for record in self.results.list()
        ]

    def analyze(self, selector: str, *, recompute: bool = False) -> dict[str, Any]:
        return self.results.analysis(selector, recompute=recompute)

    def report(self, selector: str, output: Path) -> Path:
        return self.results.report(selector, output)


__all__ = ["ConsoleServices"]
