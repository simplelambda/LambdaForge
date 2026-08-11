"""Prepared execution-environment result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    """Return the exact interpreter and whether preparation reused cache."""

    environment_id: str
    python: str
    reused: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Return a machine-readable preparation result."""
        return {
            "environment_id": self.environment_id,
            "python": self.python,
            "reused": self.reused,
        }
