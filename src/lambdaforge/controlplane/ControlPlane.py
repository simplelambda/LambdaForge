"""Local coordinator for materialization, staging and job submission."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.CudaCompatibilityResolver import (
    CudaCompatibilityResolver,
    NoCompatibleTorchWheelError,
)
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.jobs import JobHandle
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.python_runtime import (
    NoCompatiblePythonRuntimeError,
    PythonRuntime,
    PythonRuntimePolicy,
    PythonRuntimeRequirements,
)
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.TlsTrust import TlsTrust
from lambdaforge.diagnostics import ErrorCategory, LambdaForgeError, diagnostic
from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver
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
        runtime_resolver: PythonRuntimeResolver | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.bundles = bundles or ExecutionBundleBuilder()
        self.cuda_resolver = cuda_resolver or CudaCompatibilityResolver()
        self.runtime_resolver = runtime_resolver or PythonRuntimeResolver()

    def submit(
        self,
        config_path: str | Path,
        *,
        cluster: str,
        resources: ResourceRequest | None = None,
        dry_run: bool = False,
        run_arguments: Sequence[str] = (),
        group_id: str | None = None,
        reserved_job_id: str | None = None,
        allow_duplicate: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[JobHandle, ExecutionBundle]:
        """Build/cache a bundle, stage it and submit the normal remote run command."""
        notify = progress or (lambda _phase: None)
        notify("validation")
        profile = self.catalog.get(cluster)
        assert profile.storage is not None
        storage = profile.storage
        descriptor = ConfigurationDescriptor.from_path(config_path)
        materialized_kind = descriptor.kind
        request = resources or ConfigurationResourceResolver.resolve(config_path)
        if descriptor.job_type in {"experiment", "hpo"} and not dry_run and not allow_duplicate:
            self.jobs.refuse_active_execution(
                descriptor.scientific_identity,
                cluster,
                name=descriptor.name,
                source=descriptor.source,
                exclude_job_id=reserved_job_id,
            )
        if (
            cluster != "local"
            and materialized_kind.value == "dataset"
            and storage.dataset_root is None
        ):
            dataset = descriptor.name
            raise LambdaForgeError(
                diagnostic(
                    ErrorCategory.CONFIGURATION,
                    f"Cannot build dataset {dataset!r} on {cluster!r}.",
                    "No permanent dataset storage location is configured.",
                    reason=(
                        "Dataset builds publish immutable DatasetVersions. LambdaForge needs a "
                        "persistent target directory instead of placing scientific data in cache."
                    ),
                    impact=(
                        "No job was submitted and no remote computation started.",
                        "No dataset, environment or bundle was created by this submission attempt.",
                    ),
                    fixes=(
                        "Configure storage.dataset_root as a persistent, writable cluster path.",
                    ),
                    commands=(
                        (
                            "Configure dataset storage",
                            "lf clusters set "
                            f"{shlex.quote(cluster)} storage.dataset_root "
                            "/persistent/path/to/datasets",
                        ),
                        (
                            "Retry after configuring",
                            shlex.join(
                                (
                                    "lf",
                                    "run",
                                    str(Path(config_path)),
                                    "--on",
                                    cluster,
                                )
                            ),
                        ),
                    ),
                    context={
                        "cluster": cluster,
                        "dataset": dataset,
                        "storage.dataset_root": "not configured",
                        "config": str(Path(config_path).resolve()),
                    },
                    operation="dataset build preflight",
                )
            )
        transport = self.factory.transport(profile) if cluster != "local" else None
        runtime: PythonRuntime | None = None
        effective_profile = profile
        torch_plan = None
        if transport is not None and profile.environment == "managed":
            notify("runtime")
            project = self._project_root(Path(config_path).resolve().parent)
            requirement = PythonRuntimeRequirements.project(project)
            rejected: list[str] = []
            while True:
                try:
                    runtime = self.runtime_resolver.resolve(
                        profile,
                        transport,
                        requirements=((requirement,) if requirement else ()),
                        excluded_runtime_ids=rejected,
                        dry_run=dry_run,
                    )
                except NoCompatiblePythonRuntimeError as error:
                    if rejected:
                        raise NoCompatibleTorchWheelError(
                            "No Python runtime satisfies the combined LambdaForge, consumer "
                            "project and official PyTorch wheel constraints. Candidate runtimes "
                            f"rejected by PyTorch: {tuple(rejected)}."
                        ) from error
                    raise
                if not runtime.ready:
                    raise RuntimeError(
                        "This read-only run plan requires a managed Python runtime that is not "
                        f"provisioned yet ({runtime.version}). Inspect with 'lf clusters bootstrap "
                        f"{cluster} --dry-run', then bootstrap the cluster before planning work."
                    )
                try:
                    torch_plan = self.cuda_resolver.resolve(
                        profile, transport, python_executable=runtime.executable
                    )
                    break
                except NoCompatibleTorchWheelError:
                    rejected.append(runtime.runtime_id)
                    if profile.runtime_policy.strategy == "existing":
                        raise
            effective_profile = replace(
                profile,
                python=runtime.executable,
                python_runtime=PythonRuntimePolicy("existing", runtime.executable),
            )
        notify("bundle")
        bundle = self.bundles.build(
            config_path,
            effective_profile,
            dependency_policy=(
                {
                    "python_runtime": runtime.to_dict(),
                    "pytorch": torch_plan.to_dict(),
                }
                if torch_plan is not None and runtime is not None
                else None
            ),
        )
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
            notify("staging")
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
                notify("environment")
                prepared = self.factory.environment_provider(effective_profile).prepare(
                    effective_profile,
                    transport,
                    bundle,
                    remote_bundle_dir=remote_dir,
                )
                remote_python = prepared.python
                if runtime is not None:
                    self.runtime_resolver.activate(profile, transport, runtime)
            work_dir = remote_dir
            config = str(PurePosixPath(remote_dir) / "config.yaml")
            if profile.scheduler == "slurm" and not dry_run:
                reserved_job_id = reserved_job_id or JobService.new_id()
                work_dir = str(PurePosixPath(storage.job_root) / reserved_job_id / "work")
                staged = transport.run(("mkdir", "-p", str(work_dir)))
                if staged.returncode:
                    raise RuntimeError(f"Could not create SLURM job workspace: {staged.stderr}")
                # Copy-on-write is safe for mutable job workspaces and avoids physically
                # copying an unchanged cached bundle on filesystems that support reflinks.
                # Portable clusters fall back to the established recursive copy.
                copied = transport.run(
                    ("cp", "-a", "--reflink=auto", f"{remote_dir}/.", str(work_dir))
                )
                if copied.returncode:
                    copied = transport.run(("cp", "-a", f"{remote_dir}/.", str(work_dir)))
                if copied.returncode:
                    raise RuntimeError(f"Could not stage SLURM job workspace: {copied.stderr}")
                config = str(PurePosixPath(str(work_dir)) / "config.yaml")
            environment_assignments: list[str] = []
            trust = runtime.tls_trust if runtime is not None else None
            if isinstance(trust, TlsTrust):
                environment_assignments.extend(trust.assignments())
            if materialized_kind.value == "dataset":
                assert storage.dataset_root is not None
                environment_assignments.extend(
                    (
                        "LAMBDAFORGE_DATASET_REGISTRY="
                        f"{PurePosixPath(storage.state_root) / 'datasets.json'}",
                        f"LAMBDAFORGE_DATASET_ROOT={storage.dataset_root}",
                        "LAMBDAFORGE_DATASET_BUILD_ROOT="
                        f"{PurePosixPath(storage.run_root) / 'dataset-builds'}",
                        "LAMBDAFORGE_STAGE_CACHE_ROOT="
                        f"{PurePosixPath(storage.cache_root) / 'dataset-stages'}",
                        f"LAMBDAFORGE_CLUSTER={cluster}",
                    )
                )
            environment_prefix = (
                ("env", *environment_assignments) if environment_assignments else ()
            )
            command = self._command(
                (*profile.command_prefix, *environment_prefix),
                remote_python,
                config,
                materialized_kind.value,
                run_arguments,
            )
        notify("scheduler")
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
                **descriptor.metadata(),
                "execution_identity": f"{cluster}:{bundle.bundle_id}",
                "remote_config_path": config,
                "pytorch": torch_plan.to_dict() if torch_plan is not None else None,
                "python_runtime_id": runtime.runtime_id if runtime is not None else None,
            },
            job_id=reserved_job_id,
            group_id=group_id,
            job_type=self._configuration_type(config_path),
        )
        return handle, bundle

    @staticmethod
    def _project_root(start: Path) -> Path | None:
        return next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            None,
        )

    @staticmethod
    def _scientific_identity(config_path: str | Path) -> str:
        return ConfigurationDescriptor.from_path(config_path).scientific_identity

    @staticmethod
    def _configuration_name(config_path: str | Path) -> str:
        return ConfigurationDescriptor.from_path(config_path).name

    @staticmethod
    def _configuration_datasets(config_path: str | Path) -> tuple[str, ...]:
        return ConfigurationDescriptor.from_path(config_path).datasets

    @staticmethod
    def _configuration_type(config_path: str | Path) -> str:
        return ConfigurationDescriptor.from_path(config_path).job_type

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
