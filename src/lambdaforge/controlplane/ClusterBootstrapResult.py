"""Cluster bootstrap result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterBootstrapResult:
    """Describe an idempotent workspace/environment preparation."""

    cluster: str
    environment_id: str
    python: str
    reused: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Return a machine-readable result."""
        return {
            "cluster": self.cluster,
            "environment_id": self.environment_id,
            "python": self.python,
            "reused": self.reused,
        }
