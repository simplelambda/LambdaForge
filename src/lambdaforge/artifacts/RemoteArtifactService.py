"""Explicit logical artifact retrieval from persistent remote jobs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.results.RemoteResultService import RemoteResultService


class RemoteArtifactService:
    """Resolve an artifact through synchronized result metadata, then fetch only that path."""

    def __init__(
        self,
        jobs: JobService | None = None,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        results: RemoteResultService | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.results = results or RemoteResultService(self.jobs, self.catalog, self.factory)

    def list(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """List logical artifacts from synchronized terminal envelopes."""
        synced = self.results.sync(job_id)
        root = Path(synced.destination)
        output: list[dict[str, Any]] = []
        for result_path in root.rglob("result.json"):
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                continue
            for logical, field in (
                ("best-checkpoint", "best_model_path"),
                ("last-checkpoint", "last_model_path"),
            ):
                if value.get(field):
                    output.append({"logical_name": logical, "path": value[field], "remote": True})
            artifacts = value.get("artifacts", ())
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, dict) and artifact.get("path"):
                        metadata = artifact.get("metadata", {})
                        output.append(
                            {
                                "logical_name": metadata.get("logical_name")
                                if isinstance(metadata, dict) and metadata.get("logical_name")
                                else Path(str(artifact["path"])).name,
                                "path": artifact["path"],
                                "size_bytes": artifact.get("size_bytes"),
                                "type": artifact.get("kind"),
                                "remote": True,
                            }
                        )
        return tuple(output)

    def fetch(self, job_id: str, logical_name: str, destination: str | Path) -> Path:
        """Fetch one unambiguous logical artifact and no other run bytes."""
        matches = [value for value in self.list(job_id) if value["logical_name"] == logical_name]
        if len(matches) != 1:
            available = tuple(sorted(str(value["logical_name"]) for value in self.list(job_id)))
            raise LookupError(
                f"Expected one artifact {logical_name!r}, found {len(matches)}. "
                f"Available: {available}."
            )
        record = self.jobs.get(job_id)
        raw = PurePosixPath(str(matches[0]["path"]))
        work_dir = PurePosixPath(record.work_dir)
        remote = raw if raw.is_absolute() else work_dir / raw
        if not str(remote).startswith(f"{str(work_dir).rstrip('/')}/"):
            raise ValueError("Remote artifact is outside the recorded job work directory.")
        output = Path(destination).resolve()
        if output.is_dir():
            output = output / raw.name
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = self.catalog.get(record.cluster)
        self.factory.transport(profile).get(str(remote), output)
        return output
