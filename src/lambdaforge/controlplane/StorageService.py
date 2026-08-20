"""Application service for cluster storage status and conservative GC."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.StorageGcPlan import StorageGcPlan
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.controlplane.StorageReport import StorageReport


class StorageService:
    """Keep internal cache lifecycle separate from scientific data and results."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        *,
        max_parallel: int = 4,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = JobService(self.catalog, factory=self.factory)
        self.max_parallel = max(1, int(max_parallel))

    def status(self, cluster: str = "local") -> StorageReport:
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        try:
            categories = self._invoke(cluster, "status", profile.storage.to_dict())
            return StorageReport(cluster, True, categories)
        except Exception as error:
            return StorageReport(cluster, False, {}, f"{error.__class__.__name__}: {error}")

    def all(self) -> tuple[StorageReport, ...]:
        values: dict[str, StorageReport] = {}
        names = self.catalog.names()
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(names) or 1)) as executor:
            futures = {executor.submit(self.status, name): name for name in names}
            for future in as_completed(futures):
                values[futures[future]] = future.result()
        return tuple(values[name] for name in sorted(values))

    def gc(self, cluster: str = "local", *, apply: bool = False) -> StorageGcPlan:
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        active = tuple(
            record
            for record in self.jobs.list(cluster=cluster, refresh=False)
            if not record.state.terminal
        )
        references = {
            "bundles": [record.bundle_id for record in active if record.bundle_id],
            "environments": [
                str(record.metadata["environment_id"])
                for record in active
                if record.metadata.get("environment_id") not in {None, "existing"}
            ],
            "runtimes": [
                str(record.metadata["python_runtime_id"])
                for record in active
                if record.metadata.get("python_runtime_id")
            ],
            "stage_cache": ["*"]
            if any(record.job_type == "dataset-build" for record in active)
            else [],
        }
        payload = self._invoke(
            cluster, "gc", profile.storage.to_dict(), references=references, apply=apply
        )
        candidates = tuple(item for item in payload["candidates"] if isinstance(item, dict))
        return StorageGcPlan(
            cluster,
            candidates,
            int(payload["reclaimable_bytes"]),
            apply,
            str(payload["blocked_reason"]) if payload.get("blocked_reason") else None,
        )

    def environments(self, cluster: str = "local") -> tuple[dict[str, Any], ...]:
        report = self.status(cluster)
        if not report.online:
            return ()
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        root = profile.storage.environment_root
        code = (
            "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "print(json.dumps([{'environment_id':x.name,'complete':"
            "(x/'.lambdaforge-environment.json').is_file()} for x in sorted(p.glob('*')) "
            "if x.is_dir()]))"
        )
        transport = self.factory.transport(profile)
        result = transport.run((self._python(profile, transport), "-c", code, root), timeout=30.0)
        return tuple(json.loads(result.stdout or "[]")) if result.returncode == 0 else ()

    def prune_environments(
        self,
        cluster: str,
        *,
        keep: tuple[str, ...],
        apply: bool = False,
    ) -> dict[str, Any]:
        """Prune superseded environment caches while retaining every live job reference."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        active = tuple(
            record
            for record in self.jobs.list(cluster=cluster, refresh=False)
            if not record.state.terminal
        )
        protected = {
            *keep,
            *(
                str(record.metadata["environment_id"])
                for record in active
                if record.metadata.get("environment_id") not in {None, "existing"}
            ),
        }
        return self._invoke(
            cluster,
            "prune-environments",
            profile.storage.to_dict(),
            references={"environments": sorted(protected)},
            apply=apply,
        )

    def _invoke(
        self,
        cluster: str,
        operation: str,
        descriptor: dict[str, Any],
        *,
        references: dict[str, list[str]] | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        if cluster == "local":
            if operation == "status":
                return StorageOperations.status(descriptor)
            if operation == "prune-environments":
                return StorageOperations.prune_environments(
                    descriptor,
                    (references or {}).get("environments", ()),
                    apply=apply,
                )
            return StorageOperations.gc(descriptor, references or {}, apply=apply)
        profile = self.catalog.get(cluster)
        transport = self.factory.transport(profile)
        arguments = [
            self._python(profile, transport),
            "-m",
            "lambdaforge.controlplane.StorageOperations",
            operation,
            json.dumps(descriptor, separators=(",", ":")),
        ]
        if operation in {"gc", "prune-environments"}:
            arguments.extend(
                (json.dumps(references or {}, separators=(",", ":")), str(apply).lower())
            )
        result = transport.run(tuple(arguments), timeout=60.0)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise TypeError("Storage operation returned a non-object response.")
        return payload

    @staticmethod
    def _python(profile: Any, transport: Any) -> str:
        if profile.environment != "managed":
            return str(profile.python)
        assert profile.storage is not None
        pointer = PurePosixPath(profile.storage.state_root) / "active-environment"
        result = transport.run(("cat", str(pointer)), timeout=10.0)
        if result.returncode or not result.stdout.strip():
            legacy = PurePosixPath(profile.workspace) / ".lambdaforge" / "active-environment"
            result = transport.run(("cat", str(legacy)), timeout=10.0)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError(
                f"No managed environment is active on {profile.name}; run clusters bootstrap."
            )
        return result.stdout.strip()
