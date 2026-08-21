"""Global bounded-concurrency resource observation service."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.DirectClusterResourceProbe import DirectClusterResourceProbe
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.ResourceSnapshot import ResourceSnapshot
from lambdaforge.controlplane.SlurmClusterResourceProbe import SlurmClusterResourceProbe
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class ResourceService:
    """Query local/multiple clusters while preserving offline last-known facts."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        *,
        cache_path: str | Path | None = None,
        max_parallel: int = 4,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.probe = DirectClusterResourceProbe()
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        self.cache_path = Path(cache_path or state_home / "lambdaforge" / "resources.json")
        self.max_parallel = max(1, int(max_parallel))

    def get(self, cluster: str = "local") -> ResourceSnapshot:
        """Return one live observation or an explicit offline last-known snapshot."""
        profile = self.catalog.get(cluster)
        try:
            probe = SlurmClusterResourceProbe() if profile.scheduler == "slurm" else self.probe
            snapshot = probe.probe(profile, self.factory.transport(profile))
            records = tuple(
                record
                for record in JobService(self.catalog, factory=self.factory).list(
                    cluster=cluster, refresh=False
                )
                if not record.state.terminal
            )
            requested = tuple(
                {
                    "job_id": record.job_id,
                    "state": record.state.value,
                    **record.resources,
                }
                for record in records
            )
            personal = self._personal(records, profile.scheduler)
            snapshot = ResourceSnapshot(
                cluster=snapshot.cluster,
                online=snapshot.online,
                scheduler=snapshot.scheduler,
                observed=snapshot.observed,
                available=snapshot.available,
                scheduler_view=snapshot.scheduler_view,
                requested=requested,
                personal=personal,
                observed_at_utc=snapshot.observed_at_utc,
            )
            self._cache(snapshot)
            return snapshot
        except Exception as error:
            previous = self._cached(cluster)
            return ResourceSnapshot(
                cluster=cluster,
                online=False,
                scheduler=profile.scheduler,
                observed=previous.get("observed", {}) if previous else {},
                available=previous.get("available", {}) if previous else {},
                scheduler_view=previous.get("scheduler_view", {}) if previous else {},
                requested=tuple(previous.get("requested", ())) if previous else (),
                personal=previous.get("personal", {}) if previous else {},
                observed_at_utc=(
                    str(previous.get("observed_at_utc", datetime.now(timezone.utc).isoformat()))
                    if previous
                    else datetime.now(timezone.utc).isoformat()
                ),
                error=f"{error.__class__.__name__}: {error}",
            )

    @staticmethod
    def _personal(records: tuple[Any, ...], scheduler: str) -> dict[str, Any]:
        requested = {
            key: sum(int(record.resources.get(key, 0) or 0) for record in records)
            for key in ("cpu_cores", "ram_bytes", "gpu_count", "gpu_memory_bytes")
        }
        observations = []
        for record in records:
            remote = record.metadata.get("remote_state", {})
            remote = remote if isinstance(remote, Mapping) else {}
            usage = remote.get("observed_usage", {})
            if isinstance(usage, Mapping) and usage:
                observations.append(usage)
        return {
            "scope": "lambdaforge-jobs",
            "scheduler": scheduler,
            "active_jobs": len(records),
            "requested": requested,
            "observed": {
                "cpu_percent": sum(float(item.get("cpu_percent", 0) or 0) for item in observations),
                "rss_bytes": sum(int(item.get("rss_bytes", 0) or 0) for item in observations),
                "gpu_memory_bytes": sum(
                    int(item.get("gpu_memory_bytes", 0) or 0) for item in observations
                ),
                "job_count": len(observations),
            },
            "note": (
                "Observed values include current LambdaForge direct-process jobs only."
                if scheduler != "slurm"
                else "SLURM allocation requests are known; actual usage is provider-dependent."
            ),
        }

    def all(self) -> tuple[ResourceSnapshot, ...]:
        """Query configured clusters concurrently with a bounded worker count."""
        names = self.catalog.names()
        values: dict[str, ResourceSnapshot] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(names) or 1)) as executor:
            futures = {executor.submit(self.get, name): name for name in names}
            for future in as_completed(futures):
                values[futures[future]] = future.result()
        return tuple(values[name] for name in sorted(values))

    def processes(self, cluster: str) -> tuple[dict[str, Any], ...]:
        """Observe GPU processes without granting control over external PIDs."""
        profile = self.catalog.get(cluster)
        result = self.factory.transport(profile).run(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ),
            timeout=20.0,
        )
        if result.returncode:
            return ()
        values = []
        for line in result.stdout.splitlines():
            row = [item.strip() for item in line.split(",")]
            if len(row) == 4 and row[1].isdigit():
                values.append(
                    {
                        "gpu_uuid": row[0],
                        "pid": int(row[1]),
                        "name": row[2],
                        "memory_bytes": int(float(row[3]) * 1024**2),
                        "controllable": False,
                    }
                )
        return tuple(values)

    def _cache(self, snapshot: ResourceSnapshot) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            self.cache_path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            value = self._cache_value()
            value[snapshot.cluster] = snapshot.to_dict()
            temporary = self.cache_path.with_name(
                f".{self.cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            try:
                temporary.write_text(
                    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(temporary, self.cache_path)
            finally:
                temporary.unlink(missing_ok=True)

    def _cached(self, cluster: str) -> dict[str, Any] | None:
        value = self._cache_value().get(cluster)
        return value if isinstance(value, dict) else None

    def _cache_value(self) -> dict[str, Any]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
