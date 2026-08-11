"""Immutable small control bundle staged to an execution environment."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExecutionBundle:
    """Identify a cached materialized config and its small staged inputs."""

    bundle_id: str
    directory: Path
    config_path: Path
    manifest_path: Path
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a portable bundle description."""
        return {
            "bundle_id": self.bundle_id,
            "directory": str(self.directory),
            "config_path": str(self.config_path),
            "manifest_path": str(self.manifest_path),
            "size_bytes": self.size_bytes,
        }
