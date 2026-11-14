"""Application service for cluster storage status and conservative GC."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
        jobs: JobService | None = None,
        max_parallel: int = 4,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.max_parallel = max(1, int(max_parallel))

    def status(self, cluster: str = "local", *, metadata_only: bool = False) -> StorageReport:
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        try:
            categories = self._invoke(
                cluster, "status", {**profile.storage.to_dict(), "metadata_only": metadata_only}
            )
            filesystems = categories.pop("filesystems", {})
            return StorageReport(cluster, True, categories, filesystems=filesystems)
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

    def reconcile(self, cluster: str = "local", *, apply: bool = False) -> dict[str, Any]:
        """Measure owned roots explicitly; optionally update a diagnostic ledger only."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        return {
            "cluster": cluster,
            **self._invoke(cluster, "reconcile", profile.storage.to_dict(), apply=apply),
        }

    def gc(
        self,
        cluster: str = "local",
        *,
        apply: bool = False,
        environments_only: bool = False,
        reference_preview: Mapping[str, Any] | None = None,
    ) -> StorageGcPlan:
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        records = self.jobs.list(cluster=cluster, refresh=False)
        active = tuple(
            record
            for record in records
            if not record.state.terminal or record.state.value in {"failed", "cancelled", "timeout"}
        )
        retained = {
            job_id
            for record in records
            for job_id in record.metadata.get("recovery_dependencies", ())
        }
        if self.jobs.store.project is not None:
            from lambdaforge.work.StudySeriesStore import StudySeriesStore

            retained.update(StudySeriesStore(self.jobs.store.project).job_references())
        dependency_records = tuple(record for record in records if record.job_id in retained)
        references = {
            "bundles": [
                record.bundle_id for record in (*active, *dependency_records) if record.bundle_id
            ],
            "environments": [
                str(record.metadata["environment_id"])
                for record in (*active, *dependency_records)
                if record.metadata.get("environment_id") not in {None, "existing"}
            ],
            "runtimes": [
                str(record.metadata["python_runtime_id"])
                for record in (*active, *dependency_records)
                if record.metadata.get("python_runtime_id")
            ],
            "stage_cache": [],
            "protected_jobs": sorted(retained),
            "terminal_jobs": [
                record.job_id
                for record in records
                if record.state.terminal
                and record.state.value != "planned"
                and record.job_id not in retained
            ],
        }
        for record in (*active, *dependency_records):
            for binding in record.metadata.get("tool_runtimes", ()):
                references["environments"].append(str(binding["environment_id"]))
                references["bundles"].append(str(binding["entrypoint_bundle_id"]))
        payload = self._invoke(
            cluster,
            "gc",
            {
                **profile.storage.to_dict(),
                "environments_only": environments_only,
                "reference_repair_preview": reference_preview,
            },
            references=references,
            apply=apply,
        )
        candidates = tuple(item for item in payload["candidates"] if isinstance(item, dict))
        return StorageGcPlan(
            cluster,
            candidates,
            int(payload["reclaimable_bytes"]),
            apply,
            str(payload["blocked_reason"]) if payload.get("blocked_reason") else None,
            categories=payload.get("categories", {}),
            protected_items=tuple(payload.get("protected_items", ())),
            reclaimed_bytes=int(payload.get("reclaimed_bytes", 0)),
            quota_unresolved_bytes=int(payload.get("quota_unresolved_bytes", 0)),
            filesystems_before=payload.get("filesystems_before", {}),
            filesystems_after=payload.get("filesystems_after", {}),
            reference_issues=tuple(payload.get("reference_issues", ())),
            environment_count=int(payload.get("environment_count", 0)),
            environment_usage=payload.get("environment_usage", {}),
            reclaimable_allocated_bytes=int(payload.get("reclaimable_allocated_bytes", 0)),
            outcome=str(payload.get("outcome", "preview")),
            hypothetical_after_repairs=bool(payload.get("hypothetical_after_repairs", False)),
            observed_free_bytes_delta=payload.get("observed_free_bytes_delta", {}),
        )

    def repair(
        self,
        cluster: str = "local",
        *,
        retire_jobs: Sequence[str] = (),
        release_builds: Sequence[str] = (),
        apply: bool = False,
    ) -> dict[str, Any]:
        """Retire explicitly selected operational references after native ownership checks."""
        from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease

        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        descriptor = {**profile.storage.to_dict(), "reconciliation_scheduler": profile.scheduler}
        report = self._invoke(cluster, "reference-report", descriptor)
        by_name = {item["name"]: item for item in report["builds"]}
        builds = []
        for name in release_builds:
            item = by_name.get(name)
            if item is None:
                raise ValueError(f"Build marker not found: {name}")
            owner = item.get("owner")
            # A PID is checked ONLY on its controller host. Remote installer children
            # are checked independently by the build-host operation under the GC lock.
            if not isinstance(owner, dict) or not (
                item["orphan_on_build_host"] or CacheBuildLease._controller_dead(owner)
            ):
                raise ValueError(
                    f"Cannot prove controller death for {name}; "
                    f"owner host: {(owner or {}).get('host', 'unreadable')}. "
                    "No lock was removed."
                )
            builds.append(
                {
                    "name": name,
                    "owner": owner,
                    "controller_proof": {
                        "status": "dead-or-reused",
                        "host": owner.get("host"),
                        "pid": owner.get("pid"),
                    },
                }
            )
        # Local catalog absence is not proof of remote death; the host operation
        # independently checks exact Job/process/scheduler evidence for every selection.
        result = {
            "cluster": cluster,
            **self._invoke(
                cluster,
                "repair",
                descriptor,
                references={"retire_jobs": list(retire_jobs), "release_builds": builds},
                apply=apply,
            ),
        }
        if not apply and (result["retired_jobs"] or result["released_builds"]):
            result["cleanup_preview"] = self.gc(
                cluster, environments_only=True, reference_preview=result
            ).to_dict()
        return result

    def references(self, cluster: str = "local") -> dict[str, Any]:
        """Inspect current bounded ownership issues without a recursive disk measurement."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        return self._invoke(cluster, "reference-report", profile.storage.to_dict())

    def environments(self, cluster: str = "local") -> tuple[dict[str, Any], ...]:
        report = self.status(cluster, metadata_only=True)
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
        records = self.jobs.list(cluster=cluster, refresh=False)
        retained = {
            job_id
            for record in records
            for job_id in record.metadata.get("recovery_dependencies", ())
        }
        active = tuple(
            record for record in records if not record.state.terminal or record.job_id in retained
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

    def delete_job(
        self,
        cluster: str,
        job_id: str,
        *,
        apply: bool = False,
        local_run_root: str | None = None,
        deleting_jobs: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Preview or delete one exact job workspace without touching shared state."""
        if self.jobs.series_dependents(job_id):
            raise ValueError("Workspace is retained by a Study series.")
        if set(self.jobs.recovery_dependents(job_id)) - set(deleting_jobs):
            raise ValueError(
                "Workspace is retained by a recovered Study; delete its latest history first."
            )
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        descriptor = profile.storage.to_dict()
        if profile.transport != "local" and self.catalog.project is not None:
            # Resolve through the scoped store first; historic Jobs retain their original root.
            record = self.jobs.get(job_id, refresh=False)
            if record.cluster != cluster:
                raise ValueError("Job belongs to a different cluster.")
            descriptor["run_root"] = self.jobs.job_root(record)
        if local_run_root is not None:
            if profile.transport != "local":
                raise ValueError("A per-Job local run root cannot override remote storage.")
            descriptor["run_root"] = local_run_root
        return self._invoke(
            cluster,
            "delete-job",
            descriptor,
            references={"job_id": job_id},
            apply=apply,
        )

    def compact_job(self, cluster: str, job_id: str, *, apply: bool = False) -> dict[str, Any]:
        """Compact bulk Attempt data for one terminal Job and preserve its evidence."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        return self._invoke(
            cluster,
            "compact-job",
            profile.storage.to_dict(),
            references={"job_id": job_id},
            apply=apply,
        )

    def _invoke(
        self,
        cluster: str,
        operation: str,
        descriptor: dict[str, Any],
        *,
        references: dict[str, Any] | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        if cluster == "local":
            if operation == "reference-report":
                return StorageOperations.reference_report(descriptor)
            if operation == "repair":
                return StorageOperations.repair_references(
                    descriptor, references or {}, apply=apply
                )
            if operation == "status":
                return StorageOperations.status(descriptor)
            if operation == "reconcile":
                return StorageOperations.reconcile(descriptor, apply=apply)
            if operation == "prune-environments":
                return StorageOperations.prune_environments(
                    descriptor,
                    (references or {}).get("environments", ()),
                    apply=apply,
                )
            if operation == "delete-job":
                return StorageOperations.delete_job(
                    descriptor, str((references or {})["job_id"]), apply=apply
                )
            if operation == "compact-job":
                return StorageOperations.compact_job(
                    descriptor, str((references or {})["job_id"]), apply=apply
                )
            return StorageOperations.gc(descriptor, references or {}, apply=apply)
        profile = self.catalog.get(cluster)
        transport = self.factory.transport(profile)
        import inspect
        import sys

        # Execute this controller's current bounded operational collector, rather
        # than an arbitrary old immutable scientific environment's cleanup code.
        source = inspect.getsource(sys.modules[StorageOperations.__module__])
        from lambdaforge.work.recovery_dependencies import (
            cache_recovery_references,
            recovery_namespaces,
        )

        # Old immutable hosts lack this helper. Ship the same pure metadata implementation
        # beside the existing collector, never install or patch the scientific runtime.
        helper = (
            inspect.getsource(recovery_namespaces)
            + "\n"
            + inspect.getsource(cache_recovery_references)
        )
        source = source.replace(
            "        from lambdaforge.work.recovery_dependencies import cache_recovery_references",
            "",
        )
        source = source.replace("class StorageOperations:", helper + "\n\nclass StorageOperations:")
        arguments = [
            self._python(profile, transport),
            "-c",
            source.replace(
                "raise SystemExit(StorageOperations.main())",
                "sys.argv.pop(1)\n    raise SystemExit(StorageOperations.main())",
            ),
            "lambdaforge.controlplane.StorageOperations",
            operation,
            json.dumps(descriptor, separators=(",", ":")),
        ]
        if operation in {
            "gc",
            "prune-environments",
            "delete-job",
            "compact-job",
            "reconcile",
            "repair",
            "reference-report",
        }:
            arguments.extend(
                (json.dumps(references or {}, separators=(",", ":")), str(apply).lower())
            )
        result = transport.run(tuple(arguments), timeout=600.0)
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
