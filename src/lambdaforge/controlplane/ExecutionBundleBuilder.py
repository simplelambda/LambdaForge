"""Build immutable bundles for current Work YAML and explicitly typed small files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.NativeEnvironment import NativeEnvironmentSpecification
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
from lambdaforge.work import WorkConfig
from lambdaforge.work.managed import CANONICAL_FINGERPRINT_ALGORITHM, canonical_fingerprint


class ExecutionBundleBuilder:
    """Stage validated Work YAML, package wheels and bounded explicit file inputs."""

    def __init__(
        self,
        root: str | Path = ".lambdaforge/control/bundles",
        *,
        max_inline_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        if max_inline_bytes < 0:
            raise ValueError("Execution bundle inline limit cannot be negative.")
        self.root = Path(root).resolve()
        self.max_inline_bytes = max_inline_bytes

    def build(
        self,
        config_path: str | Path,
        profile: ClusterProfile,
        *,
        dependency_policy: dict[str, object] | None = None,
    ) -> ExecutionBundle:
        """Create or reuse a content-addressed bundle without legacy materialization."""
        source = Path(config_path).expanduser().resolve()
        config = WorkConfig.from_yaml(source)
        values = config.to_dict()
        staged: list[tuple[Path, str]] = []
        shared_inputs: list[dict[str, Any]] = []
        project_root = self._project_root(source.parent) or source.parent
        path_context: dict[str, Any] | None = None
        if profile.name != "local":
            values = self._stage_files(
                values,
                source.parent,
                staged,
                project_root=project_root,
                remote_project_root=profile.project_root,
                shared_inputs=shared_inputs,
            )
            if profile.project_root is not None:
                path_context = {
                    "path_context_version": 1,
                    "project_root": profile.project_root,
                    "source_relative": source.parent.relative_to(project_root).as_posix(),
                }
        environment = self._prepare_environment(
            source, profile, staged, dependency_policy=dependency_policy
        )
        code_identity = CodeIdentity.capture(project_root).to_dict()
        identity_payload = {
            "bundle_version": 4,
            "lambdaforge_version": LambdaForgeVersion.CURRENT,
            "cluster": profile.name,
            "config": values,
            "environment": environment.to_dict() if environment else None,
            "code_identity": code_identity,
            "staged": [(relative, self._fingerprint(path)) for path, relative in staged],
            "shared_inputs": shared_inputs,
            "path_context": path_context,
        }
        digest = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        bundle_id = f"bundle-{digest[:20]}"
        directory = self.root / bundle_id
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            temporary = self.root / f".{bundle_id}.{os.getpid()}.{uuid4().hex}.tmp"
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                for item, relative in staged:
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(item, destination) if item.is_dir() else shutil.copy2(
                        item, destination
                    )
                (temporary / "config.yaml").write_text(
                    yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
                )
                (temporary / "code-identity.json").write_text(
                    json.dumps(code_identity, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if path_context is not None:
                    (temporary / ".lambdaforge-paths.json").write_text(
                        json.dumps(path_context, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if shared_inputs:
                    (temporary / ".lambdaforge-shared-inputs.json").write_text(
                        json.dumps(shared_inputs, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                (temporary / "manifest.json").write_text(
                    json.dumps(identity_payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
                directory.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        size = sum(
            item.stat().st_size
            for item in directory.rglob("*")
            if item.is_file() and not item.is_symlink()
        )
        packages = tuple(sorted(path.name for path in (directory / "packages").glob("*.whl")))
        return ExecutionBundle(
            bundle_id,
            directory,
            directory / "config.yaml",
            manifest,
            size,
            environment_id=environment.environment_id if environment else None,
            package_names=packages,
            offline=environment.offline if environment else False,
            environment_policy=environment.dependency_policy if environment else None,
            shared_inputs=tuple(shared_inputs),
        )

    def _stage_files(
        self,
        values: Mapping[str, Any],
        source_dir: Path,
        staged: list[tuple[Path, str]],
        *,
        project_root: Path,
        remote_project_root: str | None,
        shared_inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        counter = 0

        def visit(item: Any) -> Any:
            nonlocal counter
            if isinstance(item, Mapping):
                if set(item) == {"file"}:
                    configured = Path(str(item["file"]))
                    local = (
                        configured.resolve()
                        if configured.is_absolute()
                        else (source_dir / configured).resolve()
                    )
                    size = self._size(local)
                    if size > self.max_inline_bytes:
                        if remote_project_root is None or not local.is_relative_to(project_root):
                            raise ValueError(
                                f"Typed file input {local} is {size} bytes; the automatic "
                                f"transfer limit is {self.max_inline_bytes}. Configure the "
                                "cluster's project_root when this path belongs to the project "
                                "mirror, or publish/materialize it as a managed dataset."
                            )
                        relative_to_project = local.relative_to(project_root)
                        remote = str(
                            PurePosixPath(remote_project_root)
                            / PurePosixPath(relative_to_project.as_posix())
                        )
                        digest, verified_size = canonical_fingerprint(local)
                        shared_inputs.append(
                            {
                                "configured": str(item["file"]),
                                "project_relative": relative_to_project.as_posix(),
                                "remote_path": remote,
                                "kind": "directory" if local.is_dir() else "file",
                                "fingerprint_algorithm": CANONICAL_FINGERPRINT_ALGORITHM,
                                "sha256": digest,
                                "size_bytes": verified_size,
                            }
                        )
                        return {"file": remote}
                    relative = f"inputs/{counter:04d}-{local.name}"
                    counter += 1
                    staged.append((local, relative))
                    return {"file": relative}
                return {str(key): visit(value) for key, value in item.items()}
            if isinstance(item, list):
                return [visit(value) for value in item]
            return item

        return visit(values)

    def _prepare_environment(
        self,
        source: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
        *,
        dependency_policy: dict[str, object] | None,
    ) -> EnvironmentIdentity | None:
        if profile.environment == "existing":
            return None
        builder = ProjectWheelBuilder(self.root.parent / "wheels")
        installed = distribution("lambdaforge")
        hint = Path(__file__).resolve().parents[3]
        framework_root = builder.installed_project_root(installed, source_hint=hint)
        wheels = [builder.build_installed("lambdaforge", source_hint=hint)]
        consumer = self._project_root(source.parent)
        if consumer is not None and consumer != framework_root:
            consumer_wheel = builder.build(consumer)
            builder.validate_framework_dependency(
                consumer_wheel, LambdaForgeVersion.CURRENT, project_root=consumer
            )
            wheels.append(consumer_wheel)
        descriptors: list[dict[str, Any]] = []
        for wheel in wheels:
            relative = f"packages/{wheel.name}"
            staged.append((wheel, relative))
            descriptors.append(
                {
                    "name": wheel.name,
                    "sha256": self._fingerprint(wheel),
                    "size_bytes": wheel.stat().st_size,
                }
            )
        if profile.wheelhouse is not None:
            wheelhouse = Path(profile.wheelhouse).expanduser().resolve()
            if not wheelhouse.is_dir():
                raise FileNotFoundError(f"Configured wheelhouse does not exist: {wheelhouse}")
            for wheel in sorted(wheelhouse.glob("*.whl")):
                staged.append((wheel, f"wheelhouse/{wheel.name}"))
                descriptors.append(
                    {
                        "name": f"wheelhouse/{wheel.name}",
                        "sha256": self._fingerprint(wheel),
                        "size_bytes": wheel.stat().st_size,
                    }
                )
        native = NativeEnvironmentSpecification.discover(consumer)
        native_policy = (dependency_policy or {}).get("native_environment")
        if native is not None:
            if not isinstance(native_policy, Mapping):
                raise RuntimeError(
                    "The project declares native dependencies, but no exact native package plan "
                    "was resolved before bundle construction."
                )
            if native_policy.get("specification_sha256") != native.sha256:
                raise RuntimeError(
                    "The native environment changed after planning; retry to build a coherent "
                    "content-addressed environment."
                )
            staged.append((native.source, f"native/{native.source.name}"))
        torch_policy = (dependency_policy or {}).get("pytorch", {})
        remote_python = (
            torch_policy.get("python_version") if isinstance(torch_policy, dict) else None
        )
        return EnvironmentIdentity.create(
            descriptors,
            python_requirement=(
                f"=={remote_python}.*"
                if remote_python
                else f">={sys.version_info.major}.{sys.version_info.minor}"
            ),
            offline=profile.wheelhouse is not None,
            dependency_policy=dependency_policy,
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
    def _size(path: Path) -> int:
        if not path.exists() or path.is_symlink():
            raise FileNotFoundError(f"Typed file input is missing or symbolic: {path}")
        if path.is_file():
            return path.stat().st_size
        size = 0
        for item in path.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"Typed file input contains a symbolic link: {item}")
            if item.is_file():
                size += item.stat().st_size
        return size

    @classmethod
    def _fingerprint(cls, path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            for item in sorted(path.rglob("*")):
                if item.is_file() and not item.is_symlink():
                    digest.update(item.relative_to(path).as_posix().encode())
                    digest.update(item.read_bytes())
        return f"sha256:{digest.hexdigest()}"
