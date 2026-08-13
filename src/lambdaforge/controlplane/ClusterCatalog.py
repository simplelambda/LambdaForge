"""Layered loader for named cluster profiles."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionProfile import ExecutionProfile


class ClusterCatalog:
    """Merge user, project and explicit catalogs with auditable precedence."""

    def __init__(
        self,
        profiles: Mapping[str, ClusterProfile],
        execution_profiles: Mapping[str, ExecutionProfile] | None = None,
        *,
        sources: Mapping[str, Path | None] | None = None,
        shadowed_sources: Mapping[str, tuple[Path, ...]] | None = None,
    ) -> None:
        self._profiles = dict(profiles)
        self._execution_profiles = dict(execution_profiles or {})
        self._sources = dict(sources or {})
        self._shadowed_sources = dict(shadowed_sources or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClusterCatalog:
        """Merge user < project < explicit catalogs and always provide local."""
        explicit = path
        if explicit is None and os.environ.get("LAMBDAFORGE_CLUSTERS"):
            explicit = os.environ["LAMBDAFORGE_CLUSTERS"]
        candidates = [cls.user_path(), cls.project_path()]
        if explicit is not None:
            candidates.append(Path(explicit).expanduser().resolve())
        unique: list[Path] = []
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved not in unique:
                unique.append(resolved)
        profiles: dict[str, ClusterProfile] = {
            "local": ClusterProfile("local", python=sys.executable)
        }
        execution_profiles: dict[str, ExecutionProfile] = {}
        sources: dict[str, Path | None] = {"local": None}
        shadowed: dict[str, list[Path]] = {}
        for source in unique:
            if not source.is_file():
                continue
            value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            raw = value.get("clusters") if isinstance(value, Mapping) else None
            if not isinstance(raw, Mapping):
                raise ValueError(f"Cluster catalog {source} requires a top-level clusters mapping.")
            for name, descriptor in raw.items():
                if not isinstance(descriptor, Mapping):
                    raise TypeError(f"Cluster profile {name!r} in {source} must be a mapping.")
                key = str(name)
                previous = sources.get(key)
                if previous is not None:
                    shadowed.setdefault(key, []).append(previous)
                profiles[key] = ClusterProfile.from_mapping(key, descriptor)
                sources[key] = source
            raw_execution = value.get("profiles", {})
            if not isinstance(raw_execution, Mapping):
                raise TypeError(f"Cluster catalog profiles in {source} must be a mapping.")
            for name, descriptor in raw_execution.items():
                if not isinstance(descriptor, Mapping):
                    raise TypeError(f"Execution profile {name!r} in {source} must be a mapping.")
                execution_profiles[str(name)] = ExecutionProfile.from_mapping(str(name), descriptor)
        return cls(
            profiles,
            execution_profiles,
            sources=sources,
            shadowed_sources={key: tuple(values) for key, values in shadowed.items()},
        )

    @staticmethod
    def user_path() -> Path:
        """Return the XDG-aware default user catalog path."""
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return (root / "lambdaforge" / "clusters.yaml").expanduser().resolve()

    @staticmethod
    def project_path() -> Path:
        """Return the nearest project catalog/root instead of depending on one subdirectory."""
        current = Path.cwd().resolve()
        parents = (current, *current.parents)
        for directory in parents:
            candidate = directory / "lambdaforge.clusters.yaml"
            if candidate.is_file():
                return candidate
        for directory in parents:
            if (directory / "pyproject.toml").is_file():
                return directory / "lambdaforge.clusters.yaml"
        return current / "lambdaforge.clusters.yaml"

    def names(self) -> tuple[str, ...]:
        """Return configured cluster names in stable order."""
        return tuple(sorted(self._profiles))

    def get(self, name: str) -> ClusterProfile:
        """Return one named profile or raise a useful error."""
        try:
            return self._profiles[name]
        except KeyError as error:
            raise KeyError(f"Unknown cluster {name!r}; configured: {self.names()}.") from error

    def source(self, name: str) -> Path | None:
        """Return the winning catalog path, or ``None`` for built-in local."""
        self.get(name)
        return self._sources.get(name)

    def inspect(self, name: str) -> dict[str, Any]:
        """Return a redaction-safe profile with precedence and auth status."""
        profile = self.get(name)
        reference = profile.auth.credential
        auth_status = (
            "openssh-managed"
            if profile.auth.mode == "openssh"
            else "interactive"
            if reference is None
            else "environment-reference"
            if reference.startswith("env:")
            else "keyring-reference"
        )
        return {
            "profile": profile.to_dict(),
            "source": str(self.source(name)) if self.source(name) is not None else "built-in",
            "shadowed_sources": [str(item) for item in self._shadowed_sources.get(name, ())],
            "authentication_status": auth_status,
        }

    def execution_profile(self, name: str) -> ExecutionProfile:
        """Return one reusable placement/resource preset."""
        try:
            return self._execution_profiles[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown execution profile {name!r}; configured: "
                f"{tuple(sorted(self._execution_profiles))}."
            ) from error

    def execution_profile_names(self) -> tuple[str, ...]:
        """Return configured execution profile names."""
        return tuple(sorted(self._execution_profiles))

    def for_data_environment(self, environment: str) -> ClusterProfile:
        """Resolve a transfer destination by profile name or data environment."""
        if environment in self._profiles:
            return self._profiles[environment]
        matches = tuple(
            profile
            for profile in self._profiles.values()
            if profile.data_environment == environment
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Data environment {environment!r} is ambiguous across "
                f"{tuple(profile.name for profile in matches)}."
            )
        raise KeyError(f"No cluster profile owns data environment {environment!r}.")

    @staticmethod
    def add(path: str | Path, profile: ClusterProfile) -> Path:
        """Atomically add/update one redaction-safe profile descriptor."""
        destination = Path(path).expanduser().resolve()
        value: dict[str, object] = {}
        if destination.is_file():
            loaded = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, Mapping):
                raise TypeError("Cluster catalogue must contain a mapping.")
            value = dict(loaded)
        clusters = value.setdefault("clusters", {})
        if not isinstance(clusters, dict):
            raise TypeError("Cluster catalogue clusters must be a mapping.")
        descriptor = profile.to_dict(include_defaults=False)
        descriptor.pop("name", None)
        clusters[profile.name] = descriptor
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor_handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor_handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def export(path: str | Path, profile: ClusterProfile) -> Path:
        """Export one portable profile, retaining only its non-secret reference."""
        return ClusterCatalog.add(path, profile)

    @staticmethod
    def remove(path: str | Path, name: str) -> Path:
        """Atomically remove one profile from one explicit writable catalogue."""
        destination = Path(path).expanduser().resolve()
        loaded = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError("Cluster catalogue must contain a mapping.")
        value = dict(loaded)
        clusters = value.get("clusters", {})
        if not isinstance(clusters, dict) or name not in clusters:
            raise KeyError(f"Cluster {name!r} is not defined in {destination}.")
        del clusters[name]
        return ClusterCatalog._write(destination, value)

    @staticmethod
    def _write(destination: Path, value: Mapping[str, object]) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor_handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor_handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
