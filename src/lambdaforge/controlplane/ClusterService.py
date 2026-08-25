"""Application service for cluster workspace and environment preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.ClusterBootstrapResult import ClusterBootstrapResult
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.CudaCompatibilityResolver import (
    CudaCompatibilityResolver,
    NoCompatibleTorchWheelError,
)
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.MicromambaArtifactStore import MicromambaArtifactStore
from lambdaforge.controlplane.NativeEnvironment import (
    NativeEnvironmentPlan,
    NativeEnvironmentPlanner,
    NativeEnvironmentSpecification,
)
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder
from lambdaforge.controlplane.python_runtime import (
    NoCompatiblePythonRuntimeError,
    PythonRuntime,
    PythonRuntimePolicy,
    PythonRuntimeRequirements,
)
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class ClusterService:
    """Prepare a cluster entirely in user space through configured providers."""

    def __init__(
        self,
        catalog: ClusterCatalog | None = None,
        factory: ControlPlaneFactory | None = None,
        cache_root: str | Path = ".lambdaforge/control",
        cuda_resolver: CudaCompatibilityResolver | None = None,
        wheel_builder: ProjectWheelBuilder | None = None,
        runtime_resolver: PythonRuntimeResolver | None = None,
        native_planner: NativeEnvironmentPlanner | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.cache_root = Path(cache_root).resolve()
        self.cuda_resolver = cuda_resolver or CudaCompatibilityResolver()
        self.wheel_builder = wheel_builder or ProjectWheelBuilder(self.cache_root / "wheels")
        self.runtime_resolver = runtime_resolver or PythonRuntimeResolver(
            MicromambaArtifactStore(self.cache_root / "runtime-installers")
        )
        self.native_planner = native_planner or NativeEnvironmentPlanner(
            self.cache_root / "native-environments",
            self.runtime_resolver,
        )

    def bootstrap(
        self,
        cluster: str,
        *,
        wheelhouse: str | Path | None = None,
        project: str | Path | None = None,
        dry_run: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> ClusterBootstrapResult:
        """Create workspace and idempotently verify/install the configured environment."""
        report = progress or (lambda _message: None)
        report("Inspecting the cluster profile and consumer project metadata.")
        profile = self.catalog.get(cluster)
        project_root = self._project_root(project)
        native_specification = NativeEnvironmentSpecification.discover(project_root)
        requirements = self._requirements(project_root, native_specification)
        assert profile.storage is not None
        if wheelhouse is not None:
            profile = replace(profile, wheelhouse=str(Path(wheelhouse).expanduser().resolve()))
        if profile.wheelhouse is not None:
            wheelhouse_path = Path(profile.wheelhouse).expanduser().resolve()
            if not wheelhouse_path.is_dir():
                raise FileNotFoundError(f"Configured wheelhouse does not exist: {wheelhouse_path}")
            profile = replace(profile, wheelhouse=str(wheelhouse_path))
        assert profile.storage is not None
        storage = profile.storage
        transport = self.factory.transport(profile)
        if dry_run:
            report("Resolving the read-only runtime and environment plan.")
            if profile.environment == "existing":
                report("Dry-run complete; the configured existing environment would be used.")
                return ClusterBootstrapResult(
                    cluster=cluster,
                    environment_id="existing",
                    python=profile.python,
                    reused=True,
                    runtime=None,
                    native_environment=(
                        native_specification.planning_summary()
                        if native_specification is not None
                        else None
                    ),
                    planned=True,
                )
            planned_runtime = self.runtime_resolver.resolve(
                profile,
                transport,
                requirements=requirements,
                dry_run=True,
            )
            pytorch: dict[str, object]
            if planned_runtime.ready:
                pytorch = self.cuda_resolver.resolve(
                    profile, transport, python_executable=planned_runtime.executable
                ).to_dict()
            else:
                pytorch = {
                    "status": "pending-runtime",
                    "reason": "Exact wheel resolution follows managed Python provisioning.",
                }
            native_summary = None
            if native_specification is not None:
                report("Inspecting the cached native-package solve, if available.")
                native_summary = native_specification.planning_summary(
                    platform=PythonRuntimeResolver.platform_tag(
                        planned_runtime.system, planned_runtime.architecture
                    )
                )
                if planned_runtime.ready:
                    cached_native = self.native_planner.cached_plan(
                        native_specification,
                        planned_runtime,
                    )
                    if cached_native is not None:
                        native_summary = {
                            **cached_native.to_dict(),
                            "status": "exact solve cached; environment reuse is content-based",
                        }
            planned = ClusterBootstrapResult(
                cluster=cluster,
                environment_id=f"planned-for-{planned_runtime.runtime_id}",
                python=planned_runtime.executable,
                reused=planned_runtime.action == "reuse",
                pytorch=pytorch,
                runtime=planned_runtime.to_dict(),
                native_environment=native_summary,
                planned=True,
            )
            report("Dry-run complete; no remote state was changed.")
            return planned
        runtime: PythonRuntime | None = None
        torch_plan = None
        effective_profile = profile
        if profile.environment == "managed":
            report("Resolving a compatible remote Python and PyTorch/CUDA runtime.")
            rejected: list[str] = []
            while True:
                try:
                    runtime = self.runtime_resolver.resolve(
                        profile,
                        transport,
                        requirements=requirements,
                        excluded_runtime_ids=rejected,
                    )
                except NoCompatiblePythonRuntimeError as error:
                    if rejected:
                        raise NoCompatibleTorchWheelError(
                            "No Python runtime satisfies the combined LambdaForge, consumer "
                            "project and official PyTorch wheel constraints. Candidate runtimes "
                            f"rejected by PyTorch: {tuple(rejected)}."
                        ) from error
                    raise
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
        report("Preparing the LambdaForge workspace on the target.")
        created = transport.run(
            (
                "mkdir",
                "-p",
                storage.state_root,
                storage.bundle_root,
                storage.environment_root,
                storage.job_root,
                str(PurePosixPath(storage.cache_root) / "pip"),
            )
        )
        if created.returncode:
            raise RuntimeError(f"Could not create cluster workspace: {created.stderr.strip()}")
        if profile.environment == "existing":
            report("Verifying the configured existing Python environment.")
            probe = transport.run(
                (*profile.command_prefix, profile.python, "-c", "import lambdaforge, torch")
            )
            if probe.returncode:
                raise RuntimeError(
                    "The existing environment is not ready. Install the pinned LambdaForge "
                    f"release and project dependencies: {probe.stderr.strip()}"
                )
            report("Bootstrap complete; the existing environment is ready.")
            return ClusterBootstrapResult(cluster, "existing", profile.python, True)

        assert torch_plan is not None
        assert runtime is not None
        native_plan: NativeEnvironmentPlan | None = None
        if native_specification is not None:
            report("Resolving the exact native Conda package inventory.")
            native_plan = self.native_planner.plan(
                native_specification,
                profile,
                transport,
                runtime,
            )
        dependency_policy = {
            "python_runtime": runtime.to_dict(),
            "pytorch": torch_plan.to_dict(),
            **({"native_environment": native_plan.to_dict()} if native_plan is not None else {}),
        }
        report("Building reproducible framework and consumer wheels.")
        wheel = self.wheel_builder.build_installed(
            "lambdaforge", source_hint=Path(__file__).resolve().parents[3]
        )
        package_wheels = [wheel]
        framework_root = Path(__file__).resolve().parents[3]
        if project_root is not None and project_root != framework_root:
            consumer_wheel = self.wheel_builder.build(project_root)
            self.wheel_builder.validate_framework_dependency(
                consumer_wheel,
                LambdaForgeVersion.CURRENT,
                project_root=project_root,
            )
            package_wheels.append(consumer_wheel)
        descriptors = [
            {
                "name": selected.name,
                "sha256": f"sha256:{hashlib.sha256(selected.read_bytes()).hexdigest()}",
                "size_bytes": selected.stat().st_size,
            }
            for selected in package_wheels
        ]
        if profile.wheelhouse is not None:
            for dependency in sorted(Path(profile.wheelhouse).expanduser().glob("*.whl")):
                descriptors.append(
                    {
                        "name": f"wheelhouse/{dependency.name}",
                        "sha256": f"sha256:{hashlib.sha256(dependency.read_bytes()).hexdigest()}",
                        "size_bytes": dependency.stat().st_size,
                    }
                )
        identity = EnvironmentIdentity.create(
            descriptors,
            python_requirement=f"=={'.'.join(runtime.version.split('.')[:2])}.*",
            offline=profile.wheelhouse is not None,
            dependency_policy=dependency_policy,
        )
        directory = self.cache_root / "bootstrap" / identity.environment_id
        packages = directory / "packages"
        packages.mkdir(parents=True, exist_ok=True)
        for selected in package_wheels:
            shutil.copy2(selected, packages / selected.name)
        wheelhouse_directory = directory / "wheelhouse"
        wheelhouse_directory.mkdir(parents=True, exist_ok=True)
        if profile.wheelhouse is not None:
            for dependency in sorted(Path(profile.wheelhouse).expanduser().glob("*.whl")):
                shutil.copy2(dependency, wheelhouse_directory / dependency.name)
        if native_specification is not None:
            native_directory = directory / "native"
            native_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                native_specification.source,
                native_directory / native_specification.source.name,
            )
        manifest = directory / "manifest.json"
        identity_payload = identity.to_dict()
        manifest.write_text(
            json.dumps(identity_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        bundle = ExecutionBundle(
            f"bootstrap-{identity.environment_id}",
            directory,
            manifest,
            manifest,
            sum(item.stat().st_size for item in directory.rglob("*") if item.is_file()),
            environment_id=identity.environment_id,
            package_names=tuple(selected.name for selected in package_wheels),
            offline=identity.offline,
            environment_policy=identity.dependency_policy,
        )
        remote = PurePosixPath(storage.cache_root) / "bootstrap" / identity.environment_id
        report("Staging the content-addressed bootstrap bundle.")
        cached = transport.run(("test", "-f", str(remote / "manifest.json")))
        if cached.returncode != 0:
            parent = transport.run(("mkdir", "-p", str(remote.parent)))
            if parent.returncode:
                raise RuntimeError(f"Could not create bootstrap cache: {parent.stderr.strip()}")
            transport.put(directory, str(remote))
        report("Creating or verifying the immutable managed environment.")
        prepared = self.factory.environment_provider(effective_profile).prepare(
            effective_profile, transport, bundle, remote_bundle_dir=str(remote)
        )
        report("Activating the verified runtime and pruning superseded environments.")
        self.runtime_resolver.activate(profile, transport, runtime)
        cleanup = StorageService(self.catalog, self.factory).prune_environments(
            cluster,
            keep=(prepared.environment_id,),
            apply=True,
        )
        result = ClusterBootstrapResult(
            cluster=cluster,
            environment_id=prepared.environment_id,
            python=prepared.python,
            reused=prepared.reused,
            pytorch=torch_plan.to_dict(),
            runtime=runtime.to_dict(),
            native_environment=native_plan.to_dict() if native_plan is not None else None,
            planned=False,
            pruned_environments=tuple(str(value) for value in cleanup.get("pruned", ())),
            cleanup_blocked_reason=(
                str(cleanup["blocked_reason"]) if cleanup.get("blocked_reason") else None
            ),
        )
        report("Bootstrap complete; the managed environment is ready.")
        return result

    @staticmethod
    def _requirements(
        project_root: Path | None,
        native: NativeEnvironmentSpecification | None,
    ) -> tuple[str, ...]:
        """Combine standard project and declarative native Python requirements."""
        values = (
            PythonRuntimeRequirements.project(project_root),
            native.python_requirement if native is not None else None,
        )
        return tuple(value for value in values if value)

    @staticmethod
    def _project_root(value: str | Path | None) -> Path | None:
        """Resolve an explicit project or discover the nearest current project."""
        start = Path(value).expanduser().resolve() if value is not None else Path.cwd().resolve()
        root = next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            None,
        )
        if value is not None and root is None:
            raise FileNotFoundError(f"No pyproject.toml was found from project path {start}.")
        return root
