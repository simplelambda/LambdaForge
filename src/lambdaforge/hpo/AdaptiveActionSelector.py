"""Cost-aware constrained acquisition over heterogeneous adaptive actions."""

from __future__ import annotations

from collections.abc import Sequence

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.EmpiricalCostModel import EmpiricalCostModel
from lambdaforge.hpo.EmpiricalMemoryModel import EmpiricalMemoryModel
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
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

    def rank(
        self,
        actions: Sequence[AdaptiveAction],
        state: AdaptiveOptimizerState,
        *,
        learning_model: LearningCurveModel,
        cost_model: EmpiricalCostModel,
        memory_model: EmpiricalMemoryModel,
        admission: ResourceAdmissionController,
        available_bytes: int,
    ) -> tuple[AdaptiveAction, ...]:
        """Return admitted actions in deterministic descending utility order."""
        estimates = {
            config_id: learning_model.predict_configuration(
                state, config_id, max_budget=self.max_budget
            )
            for config_id in state.configurations
        }

        def oriented(mean: float, standard_deviation: float) -> float:
            value = mean if self.direction == "maximize" else -mean
            return value - (
                self.risk_lambda * standard_deviation if self.risk_type == "mean_minus_std" else 0.0
            )

        incumbent = max(
            (oriented(value.mean, value.standard_deviation) for value in estimates.values()),
            default=0.0,
        )
        scored: list[AdaptiveAction] = []
        for action in actions:
            prediction = learning_model.predict_configuration(
                state, action.config_id, max_budget=self.max_budget
            )
            improvement = max(
                0.0,
                oriented(prediction.mean, prediction.standard_deviation) - incumbent,
            )
            uncertainty = prediction.standard_deviation * self.exploration_weight
            if action.kind is AdaptiveActionKind.ADD_SEED:
                seed_count = len({item.seed for item in state.observations_for(action.config_id)})
                information_gain = uncertainty / ((seed_count + 1) ** 0.5)
            elif action.kind is AdaptiveActionKind.RESUME:
                fraction = (action.target_budget - action.current_budget) / self.max_budget
                information_gain = improvement + uncertainty * max(fraction, 0.1)
            elif action.kind is AdaptiveActionKind.CONFIRM:
                information_gain = uncertainty * 1.25 + improvement
            else:
                information_gain = improvement + uncertainty
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
                        "incumbent_oriented_mean": incumbent,
                        "acquisition": "information_gain/cost*feasibility",
                    },
                )
            )
        return tuple(sorted(scored, key=lambda action: (-action.utility, action.action_id)))
