"""Build cached, inspectable control-plane execution bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path
from uuid import uuid4

import yaml

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DatasetResolver import DatasetResolver
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion


class ExecutionBundleBuilder:
    """Materialize config and only auto-stage explicitly bounded small inputs."""

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
        """Create or reuse one content-addressed, redaction-safe bundle."""
        source = Path(config_path).resolve()
        materialized = AuthoringConfig.from_yaml(source).materialize()
        values = materialized.to_dict()
        staged: list[tuple[Path, str]] = []
        if profile.name != "local":
            if materialized.kind is ConfigurationKind.TASK:
                self._prepare_task_inputs(values, source.parent, profile, staged)
            elif materialized.kind is ConfigurationKind.EXPERIMENT:
                self._prepare_experiment_data(values, source.parent, profile, staged)
            elif materialized.kind is ConfigurationKind.DATASET:
                self._prepare_dataset_recipe(values, source, profile, staged)
        environment = self._prepare_environment(
            source, profile, staged, dependency_policy=dependency_policy
        )
        identity_payload = {
            "bundle_version": 2,
            "lambdaforge_version": LambdaForgeVersion.CURRENT,
            "cluster": profile.name,
            "config": values,
            "environment": environment.to_dict() if environment is not None else None,
            "staged": [(relative, self._fingerprint(path)) for path, relative in staged],
        }
        digest = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        bundle_id = f"bundle-{digest[:20]}"
        directory = self.root / bundle_id
        config_output = directory / "config.yaml"
        manifest_output = directory / "manifest.json"
        if not manifest_output.is_file():
            temporary = self.root / f".{bundle_id}.{os.getpid()}.{uuid4().hex}.tmp"
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                for item, relative in staged:
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if item.is_dir():
                        shutil.copytree(item, destination)
                    else:
                        shutil.copy2(item, destination)
                (temporary / "config.yaml").write_text(
                    yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8"
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
        for path, _ in staged:
            if path.parent == self.root and path.name.startswith((".catalog-", ".dataset-stage-")):
                path.unlink(missing_ok=True)
        packages = tuple(sorted(path.name for path in (directory / "packages").glob("*.whl")))
        return ExecutionBundle(
            bundle_id,
            directory,
            config_output,
            manifest_output,
            size,
            environment_id=environment.environment_id if environment is not None else None,
            package_names=packages,
            offline=environment.offline if environment is not None else False,
            environment_policy=(environment.dependency_policy if environment is not None else None),
        )

    def _prepare_environment(
        self,
        source: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
        *,
        dependency_policy: dict[str, object] | None = None,
    ) -> EnvironmentIdentity | None:
        if profile.environment == "existing":
            return None
        builder = ProjectWheelBuilder(self.root.parent / "wheels")
        framework_hint = Path(__file__).resolve().parents[3]
        installed = distribution("lambdaforge")
        framework_root = builder.installed_project_root(installed, source_hint=framework_hint)
        wheels = [builder.build_installed("lambdaforge", source_hint=framework_hint)]
        consumer = self._project_root(source.parent)
        if consumer is not None and consumer != framework_root:
            wheels.append(builder.build(consumer))
        descriptors = []
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
        offline = profile.wheelhouse is not None
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
            offline=offline,
            dependency_policy=dependency_policy,
        )

    @staticmethod
    def _project_root(start: Path) -> Path | None:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
        return None

    def _prepare_task_inputs(
        self,
        values: dict[str, object],
        source_dir: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
        prefix: str = "",
    ) -> None:
        inputs = values.get("inputs", [])
        if not isinstance(inputs, list):
            return
        authoring = self._authoring(values)
        catalog_path = authoring.get("data_catalog")
        catalog = None
        remote_datasets: dict[str, object] = {}
        remote_catalog: dict[str, object] = {"datasets": remote_datasets}
        if catalog_path is not None:
            path = Path(str(catalog_path))
            path = path if path.is_absolute() else (source_dir / path).resolve()
            catalog = DataCatalog.from_yaml(path)
        for index, raw in enumerate(inputs):
            if not isinstance(raw, dict):
                continue
            if "dataset" in raw:
                reference = DatasetReference.parse(str(raw["dataset"]))
                environment = profile.data_environment or profile.name
                resolution = DatasetResolver(
                    catalog=catalog,
                    environment=environment,
                    managed_environment=profile.name,
                    source_dir=source_dir,
                ).resolve(reference)
                descriptor = dict(resolution.descriptor)
                descriptor["identity"] = dict(resolution.identity)
                descriptor["version"] = (
                    resolution.record.version if resolution.record else reference.version
                )
                descriptor["dataset_id"] = resolution.identity.get("dataset_id")
                descriptor["locations"] = {environment: resolution.location.to_dict()}
                remote_datasets[reference.selector] = descriptor
                remote_datasets.setdefault(reference.name, descriptor)
                authoring["environment"] = environment
                continue
            if "path" not in raw:
                continue
            configured = Path(str(raw["path"]))
            local = configured if configured.is_absolute() else (source_dir / configured).resolve()
            size = self._size(local)
            if size > self.max_inline_bytes:
                raise ValueError(
                    f"Input {local} is {size} bytes and will not be transferred implicitly. "
                    "Register it in data_catalog with a location for the target cluster."
                )
            relative = f"{prefix}inputs/{index}-{local.name}"
            raw["path"] = relative
            staged.append((local, relative))
        if remote_catalog["datasets"]:
            relative = f"{prefix}data-catalog.yaml"
            catalog_file = self.root / f".catalog-{uuid4().hex}.yaml"
            catalog_file.parent.mkdir(parents=True, exist_ok=True)
            catalog_file.write_text(
                yaml.safe_dump(remote_catalog, sort_keys=False), encoding="utf-8"
            )
            staged.append((catalog_file, relative))
            authoring["data_catalog"] = relative

    def _prepare_dataset_recipe(
        self,
        values: dict[str, object],
        source: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
    ) -> None:
        """Stage recipe task documents and their bounded inputs into one durable bundle."""
        stages = values.get("stages")
        if not isinstance(stages, dict):
            raise TypeError("Dataset recipe stages must be a mapping.")
        for name, descriptor in stages.items():
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("task"), str):
                continue
            configured = Path(str(descriptor["task"]))
            task_path = (
                configured if configured.is_absolute() else (source.parent / configured).resolve()
            )
            stage_values = AuthoringConfig.from_yaml(task_path).materialize().to_dict()
            prefix = f"stage-data/{name}/"
            self._prepare_task_inputs(stage_values, task_path.parent, profile, staged, prefix)
            temporary = self.root / f".dataset-stage-{name}-{uuid4().hex}.yaml"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                yaml.safe_dump(stage_values, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            # Keep staged task documents at the bundle root.  Every rewritten input/catalog
            # path above is relative to that root; nesting the YAML under ``stages/`` would
            # make an otherwise valid relocatable bundle point at ``stages/stage-data``.
            relative = f"dataset-stage-{name}.yaml"
            staged.append((temporary, relative))
            descriptor["task"] = relative

    def _prepare_experiment_data(
        self,
        values: dict[str, object],
        source_dir: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
    ) -> None:
        extensions = values.setdefault("extensions", {})
        if not isinstance(extensions, dict):
            raise TypeError("Experiment extensions must be a mapping.")
        authoring = extensions.setdefault("authoring", {})
        if not isinstance(authoring, dict):
            raise TypeError("Experiment authoring extensions must be a mapping.")
        catalog_value = authoring.get("data_catalog")
        catalog = None
        if catalog_value is not None:
            catalog_path = Path(str(catalog_value))
            catalog_path = (
                catalog_path
                if catalog_path.is_absolute()
                else (source_dir / catalog_path).resolve()
            )
            catalog = DataCatalog.from_yaml(catalog_path)
        environment = profile.data_environment or profile.name
        referenced = self._experiment_dataset_references(values.get("data", {}))
        remote_datasets: dict[str, object] = {}
        remote_catalog: dict[str, object] = {"datasets": remote_datasets}
        resolver = DatasetResolver(
            catalog=catalog,
            environment=environment,
            managed_environment=profile.name,
            source_dir=source_dir,
        )
        for reference in referenced:
            resolution = resolver.resolve(reference)
            descriptor = dict(resolution.descriptor)
            descriptor["identity"] = dict(resolution.identity)
            descriptor["version"] = (
                resolution.record.version if resolution.record else reference.version
            )
            descriptor["dataset_id"] = resolution.identity.get("dataset_id")
            descriptor["locations"] = {environment: resolution.location.to_dict()}
            remote_datasets[reference.selector] = descriptor
            remote_datasets.setdefault(reference.name, descriptor)
        if referenced:
            catalog_file = self.root / f".catalog-{uuid4().hex}.yaml"
            catalog_file.parent.mkdir(parents=True, exist_ok=True)
            catalog_file.write_text(
                yaml.safe_dump(remote_catalog, sort_keys=False), encoding="utf-8"
            )
            staged.append((catalog_file, "data-catalog.yaml"))
            authoring["data_catalog"] = "data-catalog.yaml"
            authoring["environment"] = environment

    @classmethod
    def _experiment_dataset_references(cls, value: object) -> tuple[DatasetReference, ...]:
        references: set[DatasetReference] = set()
        if isinstance(value, str) and value.startswith("dataset:"):
            references.add(DatasetReference.parse(value))
        elif isinstance(value, dict):
            if "dataset" in value:
                raw = str(value["dataset"])
                reference = (
                    DatasetReference.parse(raw)
                    if raw.startswith("dataset:")
                    else DatasetReference(raw)
                )
                if value.get("subpath") is not None:
                    reference = DatasetReference(
                        reference.name,
                        str(value["subpath"]),
                        reference.version,
                        reference.content_id,
                    )
                references.add(reference)
            for item in value.values():
                references.update(cls._experiment_dataset_references(item))
        elif isinstance(value, list):
            for item in value:
                references.update(cls._experiment_dataset_references(item))
        return tuple(sorted(references, key=str))

    @staticmethod
    def _authoring(values: dict[str, object]) -> dict[str, object]:
        extensions = values.setdefault("extensions", {})
        if not isinstance(extensions, dict):
            raise TypeError("Task extensions must be a mapping.")
        authoring = extensions.setdefault("authoring", {})
        if not isinstance(authoring, dict):
            raise TypeError("Task authoring extensions must be a mapping.")
        return authoring

    @staticmethod
    def _size(path: Path) -> int:
        if not path.exists() or path.is_symlink():
            raise FileNotFoundError(f"Bundle input is missing or symbolic: {path}")
        if path.is_file():
            return path.stat().st_size
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        )

    @staticmethod
    def _fingerprint(path: Path) -> str:
        hasher = hashlib.sha256()
        if path.is_file():
            hasher.update(path.read_bytes())
        else:
            for item in sorted(path.rglob("*")):
                if item.is_file() and not item.is_symlink():
                    hasher.update(item.relative_to(path).as_posix().encode("utf-8"))
                    hasher.update(item.read_bytes())
        return f"sha256:{hasher.hexdigest()}"
