"""Loader for named cluster profiles."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionProfile import ExecutionProfile


class ClusterCatalog:
    """Resolve cluster names from one explicit, project or user YAML catalogue."""

    def __init__(
        self,
        profiles: Mapping[str, ClusterProfile],
        execution_profiles: Mapping[str, ExecutionProfile] | None = None,
    ) -> None:
        self._profiles = dict(profiles)
        self._execution_profiles = dict(execution_profiles or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClusterCatalog:
        """Load configured clusters; always provide a safe local profile."""
        candidates = []
        if path is not None:
            candidates.append(Path(path))
        elif os.environ.get("LAMBDAFORGE_CLUSTERS"):
            candidates.append(Path(os.environ["LAMBDAFORGE_CLUSTERS"]))
        else:
            candidates.extend(
                (
                    Path("lambdaforge.clusters.yaml"),
                    Path.home() / ".config/lambdaforge/clusters.yaml",
                )
            )
        selected = next(
            (item.expanduser().resolve() for item in candidates if item.expanduser().is_file()),
            None,
        )
        profiles: dict[str, ClusterProfile] = {
            "local": ClusterProfile("local", python=sys.executable)
        }
        execution_profiles: dict[str, ExecutionProfile] = {}
        if selected is not None:
            value = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
            raw = value.get("clusters") if isinstance(value, Mapping) else None
            if not isinstance(raw, Mapping):
                raise ValueError("Cluster catalog requires a top-level clusters mapping.")
            for name, descriptor in raw.items():
                if not isinstance(descriptor, Mapping):
                    raise TypeError(f"Cluster profile {name!r} must be a mapping.")
                profiles[str(name)] = ClusterProfile.from_mapping(str(name), descriptor)
            raw_execution = value.get("profiles", {})
            if not isinstance(raw_execution, Mapping):
                raise TypeError("Cluster catalog profiles must be a mapping.")
            for name, descriptor in raw_execution.items():
                if not isinstance(descriptor, Mapping):
                    raise TypeError(f"Execution profile {name!r} must be a mapping.")
                execution_profiles[str(name)] = ExecutionProfile.from_mapping(str(name), descriptor)
        return cls(profiles, execution_profiles)

    def names(self) -> tuple[str, ...]:
        """Return configured cluster names in stable order."""
        return tuple(sorted(self._profiles))

    def get(self, name: str) -> ClusterProfile:
        """Return one named profile or raise a useful error."""
        try:
            return self._profiles[name]
        except KeyError as error:
            raise KeyError(f"Unknown cluster {name!r}; configured: {self.names()}.") from error

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
        """Resolve a transfer destination by profile name or declared data environment."""
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
