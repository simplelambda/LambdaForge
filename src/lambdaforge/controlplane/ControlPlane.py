"""Local coordinator for materialization, staging and job submission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.CudaCompatibilityResolver import CudaCompatibilityResolver
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.jobs import JobHandle
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
        cuda_resolver: CudaCompatibilityResolver | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.bundles = bundles or ExecutionBundleBuilder()
        self.cuda_resolver = cuda_resolver or CudaCompatibilityResolver()

    def submit(
        self,
        config_path: str | Path,
        *,
        cluster: str,
        resources: ResourceRequest | None = None,
        dry_run: bool = False,
        run_arguments: Sequence[str] = (),
        group_id: str | None = None,
    ) -> tuple[JobHandle, ExecutionBundle]:
        """Build/cache a bundle, stage it and submit the normal remote run command."""
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        storage = profile.storage
        transport = self.factory.transport(profile) if cluster != "local" else None
        torch_plan = (
            self.cuda_resolver.resolve(profile, transport)
            if transport is not None and profile.environment == "managed"
            else None
        )
        bundle = self.bundles.build(
            config_path,
            profile,
            dependency_policy=(
                {"pytorch": torch_plan.to_dict()} if torch_plan is not None else None
            ),
        )
        request = resources or ResourceRequest()
        materialized_kind = AuthoringConfig.from_yaml(config_path).materialize().kind
        reserved_job_id: str | None = None
        work_dir: str | Path
        if cluster == "local":
            work_dir = Path(config_path).resolve().parent
            command = self._command(
                profile.command_prefix,
                profile.python,
                str(Path(config_path).resolve()),
                materialized_kind.value,
                run_arguments,
            )
            config = str(Path(config_path).resolve())
        else:
            storage = profile.storage
            assert storage is not None
            remote_dir = str(PurePosixPath(storage.bundle_root) / bundle.bundle_id)
            assert transport is not None
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
                    str(
                        PurePosixPath(storage.environment_root)
                        / str(bundle.environment_id)
                        / "bin"
                        / "python"
                    )
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
            if profile.scheduler == "slurm" and not dry_run:
                reserved_job_id = JobService.new_id()
                work_dir = str(PurePosixPath(storage.job_root) / reserved_job_id / "work")
                staged = transport.run(("mkdir", "-p", str(work_dir)))
                if staged.returncode:
                    raise RuntimeError(f"Could not create SLURM job workspace: {staged.stderr}")
                copied = transport.run(("cp", "-a", f"{remote_dir}/.", str(work_dir)))
                if copied.returncode:
                    raise RuntimeError(f"Could not stage SLURM job workspace: {copied.stderr}")
                config = str(PurePosixPath(str(work_dir)) / "config.yaml")
            environment_prefix: tuple[str, ...] = ()
            if materialized_kind.value == "dataset":
                if storage.dataset_root is None:
                    raise RuntimeError(
                        f"Cluster {cluster!r} must configure storage.dataset_root for builds."
                    )
                environment_prefix = (
                    "env",
                    "LAMBDAFORGE_DATASET_REGISTRY="
                    f"{PurePosixPath(storage.state_root) / 'datasets.json'}",
                    f"LAMBDAFORGE_DATASET_ROOT={storage.dataset_root}",
                    "LAMBDAFORGE_DATASET_BUILD_ROOT="
                    f"{PurePosixPath(storage.run_root) / 'dataset-builds'}",
                    "LAMBDAFORGE_STAGE_CACHE_ROOT="
                    f"{PurePosixPath(storage.cache_root) / 'dataset-stages'}",
                    f"LAMBDAFORGE_CLUSTER={cluster}",
                )
            command = self._command(
                (*profile.command_prefix, *environment_prefix),
                remote_python,
                config,
                materialized_kind.value,
                run_arguments,
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
                "pytorch": torch_plan.to_dict() if torch_plan is not None else None,
                "datasets": list(self._configuration_datasets(config_path)),
            },
            job_id=reserved_job_id,
            group_id=group_id,
            job_type=self._configuration_type(config_path),
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

    @staticmethod
    def _configuration_datasets(config_path: str | Path) -> tuple[str, ...]:
        from lambdaforge.configuration.ProjectConfigService import ProjectConfigService

        materialized = AuthoringConfig.from_yaml(config_path).materialize()
        return ProjectConfigService.datasets(materialized.to_dict())

    @staticmethod
    def _configuration_type(config_path: str | Path) -> str:
        materialized = AuthoringConfig.from_yaml(config_path).materialize()
        hpo = materialized.values.get("hpo", {})
        if isinstance(hpo, dict) and hpo.get("enabled"):
            return "hpo"
        if materialized.kind.value == "dataset":
            return "dataset-build"
        if materialized.kind.value == "task" and "preprocess" in materialized.values:
            return "preprocessing"
        return materialized.kind.value

    @staticmethod
    def _command(
        prefix: Sequence[str],
        python: str,
        config: str,
        kind: str,
        arguments: Sequence[str],
    ) -> tuple[str, ...]:
        if kind == "dataset":
            return (
                *prefix,
                python,
                "-m",
                "lambdaforge.data.DatasetBuildWorker",
                config,
                *tuple(arguments),
            )
        return (
            *prefix,
            python,
            "-m",
            "lambdaforge",
            "run",
            config,
            *tuple(arguments),
        )
