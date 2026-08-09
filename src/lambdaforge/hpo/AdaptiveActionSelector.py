"""Cost-aware constrained acquisition over heterogeneous adaptive actions."""

from __future__ import annotations

from collections.abc import Sequence

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.EmpiricalCostModel import EmpiricalCostModel
from lambdaforge.hpo.FeatureAwareMemoryModel import FeatureAwareMemoryModel
from lambdaforge.hpo.GaussianValueOfInformation import GaussianValueOfInformation
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.ResourceAdmissionController import ResourceAdmissionController


class AdaptiveActionSelector:
    """Score expected information per cost times probability of feasibility."""

    def __init__(
        self,
        *,
        direction: str,
        max_budget: int,
        exploration_weight: float = 1.0,
        risk_type: str = "mean",
        risk_lambda: float = 0.0,
    ) -> None:
        self.direction = direction
        self.max_budget = max_budget
        self.exploration_weight = exploration_weight
        self.risk_type = risk_type
        self.risk_lambda = risk_lambda
        self.value_model = GaussianValueOfInformation(
            max_budget=max_budget,
            exploration_weight=exploration_weight,
        )

    def rank(
        self,
        actions: Sequence[AdaptiveAction],
        state: AdaptiveOptimizerState,
        *,
        learning_model: LearningCurveModel,
        cost_model: EmpiricalCostModel,
        memory_model: FeatureAwareMemoryModel,
        admission: ResourceAdmissionController,
        available_bytes: int | MemoryCapacity | None,
    ) -> tuple[AdaptiveAction, ...]:
        """Return admitted actions in deterministic descending utility order."""
        estimates = {
            config_id: learning_model.predict_configuration(
                state, config_id, max_budget=self.max_budget
            )
            for config_id in state.configurations
        }

        scored: list[AdaptiveAction] = []
        for action in actions:
            prediction = learning_model.predict_configuration(
                state, action.config_id, max_budget=self.max_budget
            )
            estimates.setdefault(action.config_id, prediction)
            information_gain = self.value_model.estimate(
                action,
                state,
                learning_model,
                estimates,
                direction=self.direction,
                risk_type=self.risk_type,
                risk_lambda=self.risk_lambda,
            )
            cost = cost_model.predict(action, state)
            admitted, probability, reservation, memory = admission.assess(
                action,
                state,
                memory_model,
                available_bytes=available_bytes,
            )
            if not admitted:
                continue
            scored.append(
                action.with_scores(
                    information_gain=max(information_gain, 1e-12),
                    predicted_cost=max(cost.mean, 1e-12),
                    feasibility_probability=probability,
                    memory_reservation_bytes=reservation,
                    reasons={
                        "predicted_final": prediction.to_dict(),
                        "predicted_cost": cost.to_dict(),
                        "predicted_memory": memory.to_dict(),
                        "acquisition": "gaussian_one_step_kg/cost*feasibility",
                        "value_of_information": "gaussian_moment_knowledge_gradient",
                    },
                )
            )
        return tuple(sorted(scored, key=lambda action: (-action.utility, action.action_id)))
