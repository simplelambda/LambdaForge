"""Example project-defined metric used by YAML extension tests."""

from collections.abc import Mapping
from typing import Any

from lambdaforge.metrics.Metric import Metric


class UserMetric(Metric):
    """Accumulate binary accuracy from project-specific mapping keys."""

    def __init__(self, name: str = "user_accuracy") -> None:
        super().__init__(name=name, higher_is_better=True)
        self.reset()

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Accumulate correct and total predictions."""
        del context
        prediction = outputs["user_logits"].detach().view(-1) >= 0
        target = batch["target"].detach().view(-1).bool()
        self.correct += int((prediction == target).sum().item())
        self.total += int(target.numel())

    def compute(self) -> float:
        """Return accumulated accuracy or NaN for empty state."""
        return self.correct / self.total if self.total else float("nan")

    def reset(self) -> None:
        """Clear accumulated counts."""
        self.correct = 0
        self.total = 0

    def distributed_state(self) -> dict[str, int]:
        """Return mergeable count state."""
        return {"correct": self.correct, "total": self.total}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Merge another rank's counts."""
        self.correct += int(state["correct"])
        self.total += int(state["total"])
