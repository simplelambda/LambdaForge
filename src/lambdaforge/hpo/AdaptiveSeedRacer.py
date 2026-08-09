"""Shared-seed uncertainty reduction policy."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel


class AdaptiveSeedRacer:
    """Request the next shared seed only for configurations not clearly dominated."""

    def __init__(self, config: AdaptiveOptimizerConfig) -> None:
        self.config = config

    def candidates(
        self,
        state: AdaptiveOptimizerState,
        model: LearningCurveModel,
    ) -> tuple[AdaptiveAction, ...]:
        """Return uncertainty-reducing seed actions in configured shared order."""
        if self.config.seed_strategy != "adaptive_racing" or not state.configurations:
            return ()
        estimates = {
            config_id: model.predict_configuration(
                state, config_id, max_budget=self.config.max_budget
            )
            for config_id in state.configurations
            if config_id not in state.dropped_configurations
        }
        if not estimates:
            return ()
        incumbent_id = (
            max(estimates, key=lambda key: estimates[key].mean)
            if self.config.direction == "maximize"
            else min(estimates, key=lambda key: estimates[key].mean)
        )
        output: list[AdaptiveAction] = []
        for config_id, parameters in sorted(state.configurations.items()):
            if config_id not in estimates:
                continue
            completed_seeds = {
                item.seed
                for item in state.observations_for(config_id)
                if item.budget >= self.config.max_budget and item.score is not None
            }
            if not completed_seeds or len(completed_seeds) >= self.config.max_search_seeds:
                continue
            probability = model.probability_configuration_competitive(
                state,
                config_id,
                incumbent_id,
                max_budget=self.config.max_budget,
                margin=self.config.equivalence_margin,
                direction=self.config.direction,
            )
            if config_id != incumbent_id and probability < self.config.seed_probability_threshold:
                continue
            next_seed = next(
                (seed for seed in self.config.search_seeds if seed not in completed_seeds), None
            )
            if next_seed is None:
                continue
            output.append(
                AdaptiveAction(
                    f"candidate-{state.decision_index + 1:06d}-seed-{config_id}-{next_seed}",
                    AdaptiveActionKind.ADD_SEED,
                    config_id,
                    parameters,
                    next_seed,
                    0,
                    self.config.min_budget,
                    reasons={"competitive_probability": probability},
                )
            )
        return tuple(output)
