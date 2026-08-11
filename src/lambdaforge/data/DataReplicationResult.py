"""Explicit dataset replication plan/result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataReplicationResult:
    """Report an exact source, destination and whether bytes were transferred."""

    dataset: str
    source: str
    destination: str
    applied: bool
    returncode: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, str | bool | int]:
        """Return a stable CLI/API payload."""
        return {
            "dataset": self.dataset,
            "source": self.source,
            "destination": self.destination,
            "applied": self.applied,
            "returncode": self.returncode,
            "message": self.message,
        }
