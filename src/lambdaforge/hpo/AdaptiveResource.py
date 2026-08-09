"""One logical resource available to adaptive scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveResource:
    """Describe a CPU lane or logical visible GPU without physical-ID assumptions."""

    name: str
    device: int | None
    memory_capacity_bytes: int = 0
    cpu_cores: int = 1
    max_jobs: int = 1

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.memory_capacity_bytes < 0
            or self.cpu_cores < 1
            or self.max_jobs < 1
        ):
            raise ValueError("Invalid adaptive resource declaration.")
        if self.device is not None and self.device < 0:
            raise ValueError("Adaptive GPU device indices must be non-negative logical indices.")
