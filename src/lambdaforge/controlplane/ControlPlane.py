"""Local coordinator for materialization, staging and job submission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.JobHandle import JobHandle
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.execution.ResourceRequest import ResourceRequest


class ControlPlane:
    """Coordinate remote execution while reusing the ordinary LambdaForge CLI remotely."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        jobs: JobService | None = None,
        bundles: ExecutionBundleBuilder | None = None,
        factory: ControlPlaneFactory | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.bundles = bundles or ExecutionBundleBuilder()

    def submit(
        self,
        config_path: str | Path,
        *,
        cluster: str,
        resources: ResourceRequest | None = None,
        dry_run: bool = False,
        run_arguments: Sequence[str] = (),
    ) -> tuple[JobHandle, ExecutionBundle]:
        """Build/cache a bundle, stage it and submit the normal remote run command."""
        profile = self.catalog.get(cluster)
        bundle = self.bundles.build(config_path, profile)
        request = resources or ResourceRequest()
        work_dir: str | Path
        if cluster == "local":
            work_dir = Path(config_path).resolve().parent
            command = (
                *profile.command_prefix,
                profile.python,
                "-m",
                "lambdaforge",
                "run",
                str(Path(config_path).resolve()),
                *tuple(run_arguments),
            )
            config = str(Path(config_path).resolve())
        else:
            remote_dir = str(PurePosixPath(profile.workspace) / "bundles" / bundle.bundle_id)
            transport = self.factory.transport(profile)
            if not dry_run:
                created = transport.run(("mkdir", "-p", str(PurePosixPath(remote_dir).parent)))
                if created.returncode:
                    raise RuntimeError(f"Could not create remote bundle cache: {created.stderr}")
                cached = transport.run(
                    ("test", "-f", str(PurePosixPath(remote_dir) / "manifest.json"))
                )
                if cached.returncode != 0:
                    transport.put(bundle.directory, remote_dir)
            work_dir = remote_dir
            config = str(PurePosixPath(remote_dir) / "config.yaml")
            command = (
                *profile.command_prefix,
                profile.python,
                "-m",
                "lambdaforge",
                "run",
                config,
                *tuple(run_arguments),
            )
        handle = self.jobs.submit(
            command,
            cluster=cluster,
            resources=request,
            work_dir=work_dir,
            dry_run=dry_run,
            bundle_id=bundle.bundle_id,
            config_path=config,
            metadata={"bundle_size_bytes": bundle.size_bytes},
        )
        return handle, bundle
