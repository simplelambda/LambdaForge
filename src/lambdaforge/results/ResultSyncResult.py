"""Lightweight remote result synchronization result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResultSyncResult:
    """Describe exactly which small result files were retrieved."""

    job_id: str
    destination: str
    files: tuple[str, ...]
    bytes_transferred: int

    def to_dict(self) -> dict[str, object]:
        """Return machine-readable sync evidence."""
        return {
            "job_id": self.job_id,
            "destination": self.destination,
            "files": list(self.files),
            "bytes_transferred": self.bytes_transferred,
        }
