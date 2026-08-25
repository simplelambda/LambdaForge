"""Local coordinator for materialization, staging and job submission."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.CudaCompatibilityResolver import (
    CudaCompatibilityResolver,
    NoCompatibleTorchWheelError,
)
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.jobs import JobHandle
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.NativeEnvironment import (
    NativeEnvironmentPlan,
    NativeEnvironmentPlanner,
    NativeEnvironmentSpecification,
)
from lambdaforge.controlplane.python_runtime import (
    NoCompatiblePythonRuntimeError,
    PythonRuntime,
    PythonRuntimePolicy,
    PythonRuntimeRequirements,
)
from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver
from lambdaforge.controlplane.TlsTrust import TlsTrust
from lambdaforge.controlplane.Transport import Transport
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
        native_planner: NativeEnvironmentPlanner | None = None,
    ) -> None:
        self.catalog = catalog or ClusterCatalog.load()
        self.factory = factory or ControlPlaneFactory()
        self.jobs = jobs or JobService(self.catalog, factory=self.factory)
        self.bundles = bundles or ExecutionBundleBuilder()
        self.cuda_resolver = cuda_resolver or CudaCompatibilityResolver()
        self.runtime_resolver = runtime_resolver or PythonRuntimeResolver()
        self.native_planner = native_planner or NativeEnvironmentPlanner(
            runtime_resolver=self.runtime_resolver
        )

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
        request = resources or ConfigurationResourceResolver.resolve(config_path)
        if not dry_run and not allow_duplicate:
            self.jobs.refuse_active_execution(
                descriptor.scientific_identity,
                cluster,
                name=descriptor.name,
                source=descriptor.source,
                exclude_job_id=reserved_job_id,
            )
        transport = self.factory.transport(profile) if cluster != "local" else None
        runtime: PythonRuntime | None = None
        effective_profile = profile
        torch_plan = None
        native_plan: NativeEnvironmentPlan | None = None
        project = self._project_root(Path(config_path).resolve().parent)
        native_specification = NativeEnvironmentSpecification.discover(project)
        if transport is not None and profile.environment == "managed":
            notify("runtime")
            requirement = PythonRuntimeRequirements.project(project)
            requirements = tuple(
                value
                for value in (
                    requirement,
                    (
                        native_specification.python_requirement
                        if native_specification is not None
                        else None
                    ),
                )
                if value
            )
            rejected: list[str] = []
            while True:
                try:
                    runtime = self.runtime_resolver.resolve(
                        profile,
                        transport,
                        requirements=requirements,
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
            if native_specification is not None:
                if dry_run:
                    native_plan = self.native_planner.cached_plan(
                        native_specification,
                        runtime,
                    )
                    if native_plan is None:
                        raise RuntimeError(
                            "This read-only run plan needs an exact native package solve that has "
                            f"not been prepared. Run 'lf clusters bootstrap {cluster} --project "
                            f"{project} --dry-run', then bootstrap without --dry-run."
                        )
                else:
                    native_plan = self.native_planner.plan(
                        native_specification,
                        profile,
                        transport,
                        runtime,
                    )
        notify("bundle")
        bundle = self.bundles.build(
            config_path,
            effective_profile,
            dependency_policy=(
                {
                    "python_runtime": runtime.to_dict(),
                    "pytorch": torch_plan.to_dict(),
                    **(
                        {"native_environment": native_plan.to_dict()}
                        if native_plan is not None
                        else {}
                    ),
                }
                if torch_plan is not None and runtime is not None
                else None
            ),
        )
        if transport is not None and bundle.shared_inputs:
            notify("inputs")
            self._verify_shared_inputs(transport, effective_profile, bundle.shared_inputs)
        work_dir: str | Path
        if cluster == "local":
            work_dir = Path(config_path).resolve().parent
            command = self._command(
                profile.command_prefix,
                profile.python,
                str(Path(config_path).resolve()),
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
            if not dry_run:
                reserved_job_id = reserved_job_id or JobService.new_id()
                work_dir = str(PurePosixPath(storage.job_root) / reserved_job_id / "work")
                staged = transport.run(("mkdir", "-p", str(work_dir)))
                if staged.returncode:
                    raise RuntimeError(f"Could not create job workspace: {staged.stderr}")
                # Copy-on-write is safe for mutable job workspaces and avoids physically
                # copying an unchanged cached bundle on filesystems that support reflinks.
                # Portable clusters fall back to the established recursive copy.
                copied = transport.run(
                    ("cp", "-a", "--reflink=auto", f"{remote_dir}/.", str(work_dir))
                )
                if copied.returncode:
                    copied = transport.run(("cp", "-a", f"{remote_dir}/.", str(work_dir)))
                if copied.returncode:
                    raise RuntimeError(f"Could not stage job workspace: {copied.stderr}")
                config = str(PurePosixPath(str(work_dir)) / "config.yaml")
            environment_assignments: list[str] = []
            trust = runtime.tls_trust if runtime is not None else None
            if isinstance(trust, TlsTrust):
                environment_assignments.extend(trust.assignments())
            if profile.environment == "managed":
                inherited_path = transport.run(
                    (*profile.command_prefix, "printenv", "PATH")
                )
                fallback_path = "/usr/local/bin:/usr/bin:/bin"
                path_value = inherited_path.stdout.strip().splitlines()
                selected_path = (
                    path_value[0]
                    if inherited_path.returncode == 0 and path_value
                    else fallback_path
                )
                environment_assignments.append(
                    f"PATH={PurePosixPath(remote_python).parent}:{selected_path}"
                )
            environment_assignments.extend(
                (
                    "LAMBDAFORGE_DATASET_REGISTRY="
                    f"{PurePosixPath(storage.state_root) / 'datasets.json'}",
                    f"LAMBDAFORGE_CACHE_ROOT={storage.cache_root}",
                    f"LAMBDAFORGE_CLUSTER={cluster}",
                    "LAMBDAFORGE_BUNDLE=1",
                    "LAMBDAFORGE_EXECUTION_MODE=worker",
                    f"LAMBDAFORGE_JOB_ID={reserved_job_id}" if reserved_job_id else "",
                    (
                        "LAMBDAFORGE_PROGRESS_PATH="
                        f"{PurePosixPath(str(work_dir)).parent / 'progress.json'}"
                        if reserved_job_id
                        else ""
                    ),
                    (
                        "LAMBDAFORGE_JOB_RESULT_PATH="
                        f"{PurePosixPath(str(work_dir)).parent / 'result.json'}"
                        if reserved_job_id
                        else ""
                    ),
                )
            )
            environment_assignments = [value for value in environment_assignments if value]
            if storage.dataset_root is not None:
                environment_assignments.append(f"LAMBDAFORGE_DATASET_ROOT={storage.dataset_root}")
            environment_prefix = (
                ("env", *environment_assignments) if environment_assignments else ()
            )
            command = self._command(
                (*profile.command_prefix, *environment_prefix),
                remote_python,
                config,
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
                "native_environment": (native_plan.to_dict() if native_plan is not None else None),
            },
            job_id=reserved_job_id,
            group_id=group_id,
            job_type=self._configuration_type(config_path),
        )
        return handle, bundle

    @staticmethod
    def _verify_shared_inputs(
        transport: Transport,
        profile: ClusterProfile,
        inputs: Sequence[Mapping[str, Any]],
    ) -> None:
        """Require an exact remote counterpart for every non-transferred project input."""
        probe = r"""
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
        raise ValueError("path is missing, symbolic, or not a regular file/directory")
    digest = hashlib.sha256()
    size = 0
    entries = (path,) if path.is_file() else tuple(sorted(path.rglob("*")))
    for item in entries:
        if item.is_symlink():
            raise ValueError(f"content contains symbolic link: {item}")
        if not item.is_file():
            continue
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    print(json.dumps({
        "kind": "file" if path.is_file() else "directory",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }, sort_keys=True))
except Exception as error:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(2)
"""
        for expected in inputs:
            remote_path = str(expected["remote_path"])
            result = transport.run(
                (*profile.command_prefix, profile.python, "-c", probe, remote_path),
                timeout=None,
            )
            if result.returncode:
                reason = (
                    result.stderr.strip().splitlines()[-1]
                    if result.stderr.strip()
                    else "probe failed"
                )
                raise ValueError(
                    f"Shared project input {expected['configured']!r} is not usable at "
                    f"{remote_path}: {reason}. Synchronize that project-relative path to the "
                    "cluster project_root, or use a managed dataset."
                )
            try:
                observed = json.loads(result.stdout)
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Could not read the remote identity of shared input {remote_path}."
                ) from error
            differences = [
                f"{field} expected={expected[field]!r} observed={observed.get(field)!r}"
                for field in ("kind", "sha256", "size_bytes")
                if observed.get(field) != expected[field]
            ]
            if differences:
                raise ValueError(
                    f"Shared project input {expected['configured']!r} differs at {remote_path}: "
                    f"{'; '.join(differences)}. Synchronize the remote mirror before retrying; "
                    "LambdaForge will not run against stale or partial input data."
                )

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
        arguments: Sequence[str],
    ) -> tuple[str, ...]:
        return (
            *prefix,
            python,
            "-m",
            "lambdaforge",
            "run",
            config,
            *tuple(arguments),
        )
