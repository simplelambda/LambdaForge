"""Learning-curve-aware fidelity action policy."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel


class AdaptiveFidelityPolicy:
    """Propose real checkpoint continuations and conservative drops."""

    def __init__(self, config: AdaptiveOptimizerConfig) -> None:
        self.config = config

    def resume_candidates(self, state: AdaptiveOptimizerState) -> tuple[AdaptiveAction, ...]:
        """Continue each viable partial configuration/seed by one budget increment."""
        candidates: list[AdaptiveAction] = []
        for config_id, parameters in sorted(state.configurations.items()):
            if config_id in state.dropped_configurations:
                continue
            by_seed: dict[int, int] = {}
            for observation in state.observations_for(config_id):
                if observation.status.value in {"paused", "completed", "cancelled"}:
                    by_seed[observation.seed] = max(
                        by_seed.get(observation.seed, 0), observation.budget
                    )
            for seed, current in sorted(by_seed.items()):
                if current >= self.config.max_budget:
                    continue
                target = min(current + self.config.budget_step, self.config.max_budget)
                candidates.append(
                    AdaptiveAction(
                        action_id=(
                            f"candidate-{state.decision_index + 1:06d}-resume-"
                            f"{config_id}-{seed}-{target}"
                        ),
                        kind=AdaptiveActionKind.RESUME,
                        config_id=config_id,
                        parameters=parameters,
                        seed=seed,
                        current_budget=current,
                        target_budget=target,
                    )
                )
        return tuple(candidates)

    def dominated(
        self,
        state: AdaptiveOptimizerState,
        model: LearningCurveModel,
    ) -> tuple[tuple[str, float], ...]:
        """Return configurations whose probability of competitiveness is below threshold."""
        if not self.config.pruning_enabled or len(state.configurations) < 2:
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
        incumbent = estimates[incumbent_id]
        output: list[tuple[str, float]] = []
        for config_id, estimate in estimates.items():
            if config_id == incumbent_id:
                continue
            maximum_budget = max(
                (item.budget for item in state.observations_for(config_id)), default=0
            )
            if maximum_budget < self.config.min_budget_before_drop:
                continue
            probability = model.probability_competitive(
                estimate,
                incumbent,
                margin=self.config.equivalence_margin,
                direction=self.config.direction,
            )
            if probability < self.config.drop_probability_threshold:
                output.append((config_id, probability))
        return tuple(sorted(output))
