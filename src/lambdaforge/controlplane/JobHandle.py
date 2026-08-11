"""Small serializable handle returned after submission."""

from dataclasses import dataclass

from lambdaforge.controlplane.JobState import JobState


@dataclass(frozen=True, slots=True)
class JobHandle:
    """Identify a locally tracked job and its provider scheduler id."""

    job_id: str
    cluster: str
    state: JobState
    scheduler_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a concise CLI/API payload."""
        return {
            "job_id": self.job_id,
            "cluster": self.cluster,
            "state": self.state.value,
            "scheduler_id": self.scheduler_id,
        }
