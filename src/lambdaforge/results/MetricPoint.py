"""One normalized metric observation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """Represent one scalar metric at an explicit run/seed/step."""

    run: str
    seed: int | None
    variant: str
    split: str | None
    metric: str
    step: float
    value: float
    timestamp: str | None = None

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return one tabular row."""
        return {
            "run": self.run,
            "seed": self.seed,
            "variant": self.variant,
            "split": self.split,
            "metric": self.metric,
            "step": self.step,
            "value": self.value,
            "timestamp": self.timestamp,
        }
