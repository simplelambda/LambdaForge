"""Application service for cluster workspace and environment preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
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
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder
from lambdaforge.controlplane.python_runtime import (
    NoCompatiblePythonRuntimeError,
    PythonRuntime,
    PythonRuntimePolicy,
    PythonRuntimeRequirements,
)
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver


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
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.cache_root = Path(cache_root).resolve()
        self.cuda_resolver = cuda_resolver or CudaCompatibilityResolver()
        self.wheel_builder = wheel_builder or ProjectWheelBuilder(self.cache_root / "wheels")
        self.runtime_resolver = runtime_resolver or PythonRuntimeResolver(
            MicromambaArtifactStore(self.cache_root / "runtime-installers")
        )

    def bootstrap(
        self,
        cluster: str,
        *,
        wheelhouse: str | Path | None = None,
        dry_run: bool = False,
    ) -> ClusterBootstrapResult:
        """Create workspace and idempotently verify/install the configured environment."""
        profile = self.catalog.get(cluster)
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
            if profile.environment == "existing":
                return ClusterBootstrapResult(
                    cluster, "existing", profile.python, True, runtime=None, planned=True
                )
            planned_runtime = self.runtime_resolver.resolve(
                profile,
                transport,
                requirements=self._requirements(),
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
            return ClusterBootstrapResult(
                cluster,
                f"planned-for-{planned_runtime.runtime_id}",
                planned_runtime.executable,
                planned_runtime.action == "reuse",
                pytorch,
                planned_runtime.to_dict(),
                True,
            )
        runtime: PythonRuntime | None = None
        torch_plan = None
        effective_profile = profile
        if profile.environment == "managed":
            rejected: list[str] = []
            while True:
                try:
                    runtime = self.runtime_resolver.resolve(
                        profile,
                        transport,
                        requirements=self._requirements(),
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
            probe = transport.run(
                (*profile.command_prefix, profile.python, "-c", "import lambdaforge, torch")
            )
            if probe.returncode:
                raise RuntimeError(
                    "The existing environment is not ready. Install the pinned LambdaForge "
                    f"release and project dependencies: {probe.stderr.strip()}"
                )
            return ClusterBootstrapResult(cluster, "existing", profile.python, True)

        assert torch_plan is not None
        assert runtime is not None
        dependency_policy = {
            "python_runtime": runtime.to_dict(),
            "pytorch": torch_plan.to_dict(),
        }
        wheel = self.wheel_builder.build_installed(
            "lambdaforge", source_hint=Path(__file__).resolve().parents[3]
        )
        descriptors = [
            {
                "name": wheel.name,
                "sha256": f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
                "size_bytes": wheel.stat().st_size,
            }
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
        shutil.copy2(wheel, packages / wheel.name)
        wheelhouse_directory = directory / "wheelhouse"
        wheelhouse_directory.mkdir(parents=True, exist_ok=True)
        if profile.wheelhouse is not None:
            for dependency in sorted(Path(profile.wheelhouse).expanduser().glob("*.whl")):
                shutil.copy2(dependency, wheelhouse_directory / dependency.name)
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
            package_names=(wheel.name,),
            offline=identity.offline,
            environment_policy=identity.dependency_policy,
        )
        remote = PurePosixPath(storage.cache_root) / "bootstrap" / identity.environment_id
        cached = transport.run(("test", "-f", str(remote / "manifest.json")))
        if cached.returncode != 0:
            parent = transport.run(("mkdir", "-p", str(remote.parent)))
            if parent.returncode:
                raise RuntimeError(f"Could not create bootstrap cache: {parent.stderr.strip()}")
            transport.put(directory, str(remote))
        prepared = self.factory.environment_provider(effective_profile).prepare(
            effective_profile, transport, bundle, remote_bundle_dir=str(remote)
        )
        self.runtime_resolver.activate(profile, transport, runtime)
        return ClusterBootstrapResult(
            cluster,
            prepared.environment_id,
            prepared.python,
            prepared.reused,
            torch_plan.to_dict(),
            runtime.to_dict(),
        )

    @staticmethod
    def _requirements() -> tuple[str, ...]:
        """Include a nearby consumer project constraint when bootstrap runs inside one."""
        current = Path.cwd().resolve()
        root = next(
            (
                candidate
                for candidate in (current, *current.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            None,
        )
        requirement = PythonRuntimeRequirements.project(root)
        return (requirement,) if requirement else ()
