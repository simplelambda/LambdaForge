"""Side-effect-free plan for one dataset recipe build."""

from __future__ import annotations

from dataclasses import dataclass

from lambdaforge.data.DatasetStagePlan import DatasetStagePlan


@dataclass(frozen=True, slots=True)
class DatasetBuildPlan:
    """Describe stage reuse and final publication before executing project code."""

    dataset: str
    recipe_fingerprint: str
    target_cluster: str
    stages: tuple[DatasetStagePlan, ...]
    publish_action: str
    publish_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable build-plan envelope."""
        return {
            "kind": "dataset-build-plan",
            "dataset": self.dataset,
            "recipe_fingerprint": self.recipe_fingerprint,
            "target_cluster": self.target_cluster,
            "stages": [stage.to_dict() for stage in self.stages],
            "publish": {"action": self.publish_action, "reason": self.publish_reason},
        }
