"""Local coordinator for materialization, staging and job submission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
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
            state_root = PurePosixPath(profile.workspace) / ".lambdaforge"
            remote_dir = str(state_root / "bundles" / bundle.bundle_id)
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
            if dry_run:
                remote_python = (
                    str(state_root / "environments" / str(bundle.environment_id) / "bin" / "python")
                    if profile.environment == "managed"
                    else profile.python
                )
            else:
                prepared = self.factory.environment_provider(profile).prepare(
                    profile,
                    transport,
                    bundle,
                    remote_bundle_dir=remote_dir,
                )
                remote_python = prepared.python
            work_dir = remote_dir
            config = str(PurePosixPath(remote_dir) / "config.yaml")
            command = (
                *profile.command_prefix,
                remote_python,
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
            metadata={
                "bundle_size_bytes": bundle.size_bytes,
                "environment_id": bundle.environment_id or "existing",
                "name": self._configuration_name(config_path),
                "scientific_identity": self._scientific_identity(config_path),
                "execution_identity": f"{cluster}:{bundle.bundle_id}",
                "remote_config_path": config,
            },
        )
        return handle, bundle

    @staticmethod
    def _scientific_identity(config_path: str | Path) -> str:
        from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
        from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
        from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
        from lambdaforge.tasks.TaskConfig import TaskConfig

        materialized = AuthoringConfig.from_yaml(config_path).materialize()
        if materialized.kind is ConfigurationKind.TASK:
            return TaskConfig(materialized.values, source=config_path).fingerprint
        if materialized.kind is ConfigurationKind.EXPERIMENT:
            config = ExperimentConfig(materialized.values, source=config_path)
            return RunFingerprint.digest(
                {"runs": [RunFingerprint.payload(run) for run in config.expand()]}
            )
        return RunFingerprint.digest(materialized.to_dict())

    @staticmethod
    def _configuration_name(config_path: str | Path) -> str:
        materialized = AuthoringConfig.from_yaml(config_path).materialize()
        values = materialized.values
        experiment = values.get("experiment", {})
        if isinstance(experiment, dict) and experiment.get("name"):
            return str(experiment["name"])
        return str(values.get("name", Path(config_path).stem))
