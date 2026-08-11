"""Build cached, inspectable control-plane execution bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import yaml

from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DatasetReference import DatasetReference
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

    def build(self, config_path: str | Path, profile: ClusterProfile) -> ExecutionBundle:
        """Create or reuse one content-addressed, redaction-safe bundle."""
        source = Path(config_path).resolve()
        materialized = AuthoringConfig.from_yaml(source).materialize()
        values = materialized.to_dict()
        staged: list[tuple[Path, str]] = []
        if materialized.kind is ConfigurationKind.TASK and profile.name != "local":
            self._prepare_task_inputs(values, source.parent, profile, staged)
        identity_payload = {
            "bundle_version": 1,
            "lambdaforge_version": LambdaForgeVersion.CURRENT,
            "cluster": profile.name,
            "config": values,
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
            if path.parent == self.root and path.name.startswith(".catalog-"):
                path.unlink(missing_ok=True)
        return ExecutionBundle(bundle_id, directory, config_output, manifest_output, size)

    def _prepare_task_inputs(
        self,
        values: dict[str, object],
        source_dir: Path,
        profile: ClusterProfile,
        staged: list[tuple[Path, str]],
    ) -> None:
        inputs = values.get("inputs", [])
        if not isinstance(inputs, list):
            return
        authoring = self._authoring(values)
        catalog_path = authoring.get("data_catalog")
        catalog = None
        remote_catalog: dict[str, object] = {"datasets": {}}
        if catalog_path is not None:
            path = Path(str(catalog_path))
            path = path if path.is_absolute() else (source_dir / path).resolve()
            catalog = DataCatalog.from_yaml(path)
        for index, raw in enumerate(inputs):
            if not isinstance(raw, dict):
                continue
            if "dataset" in raw:
                if catalog is None:
                    raise ValueError(f"Remote input {raw['dataset']!r} requires data_catalog.")
                reference = DatasetReference.parse(str(raw["dataset"]))
                descriptor = catalog.descriptor(reference)
                environment = profile.data_environment or profile.name
                location = catalog.resolve(reference, environment=environment)
                descriptor["locations"] = {environment: location.to_dict()}
                remote_catalog["datasets"][reference.name] = descriptor  # type: ignore[index]
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
            relative = f"inputs/{index}-{local.name}"
            raw["path"] = relative
            staged.append((local, relative))
        if remote_catalog["datasets"]:
            relative = "data-catalog.yaml"
            catalog_file = self.root / f".catalog-{uuid4().hex}.yaml"
            catalog_file.parent.mkdir(parents=True, exist_ok=True)
            catalog_file.write_text(
                yaml.safe_dump(remote_catalog, sort_keys=False), encoding="utf-8"
            )
            staged.append((catalog_file, relative))
            authoring["data_catalog"] = relative

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
