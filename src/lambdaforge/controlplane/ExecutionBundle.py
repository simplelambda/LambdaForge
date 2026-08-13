"""Immutable small control bundle staged to an execution environment."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ExecutionBundle:
    """Identify a cached materialized config and its small staged inputs."""

    bundle_id: str
    directory: Path
    config_path: Path
    manifest_path: Path
    size_bytes: int
    environment_id: str | None = None
    package_names: tuple[str, ...] = ()
    offline: bool = False
    environment_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.environment_policy is not None:
            object.__setattr__(
                self,
                "environment_policy",
                FrozenJsonMapping(self.environment_policy),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a portable bundle description."""
        return {
            "bundle_id": self.bundle_id,
            "directory": str(self.directory),
            "config_path": str(self.config_path),
            "manifest_path": str(self.manifest_path),
            "size_bytes": self.size_bytes,
            "environment_id": self.environment_id,
            "package_names": list(self.package_names),
            "offline": self.offline,
            "environment_policy": copy.deepcopy(self.environment_policy or {}),
        }
