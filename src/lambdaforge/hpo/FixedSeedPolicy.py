"""Fixed-seed baseline policy."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel


class FixedSeedPolicy:
    """Schedule every declared seed for every configuration."""

    def __init__(self, config: AdaptiveOptimizerConfig) -> None:
        self.config = config

    def candidates(
        self,
        state: AdaptiveOptimizerState,
        model: LearningCurveModel | None = None,
    ) -> tuple[AdaptiveAction, ...]:
        """Return missing fixed-seed actions in shared declaration order."""
        del model
        output: list[AdaptiveAction] = []
        for config_id, parameters in sorted(state.configurations.items()):
            observed = {item.seed for item in state.observations_for(config_id)}
            pending = {
                item.seed for item in state.pending_actions.values() if item.config_id == config_id
            }
            for seed in self.config.search_seeds:
                if seed not in observed | pending:
                    output.append(
                        AdaptiveAction(
                            (
                                f"candidate-{state.decision_index + 1:06d}-fixed-seed-"
                                f"{config_id}-{seed}"
                            ),
                            AdaptiveActionKind.ADD_SEED,
                            config_id,
                            parameters,
                            seed,
                            0,
                            (
                                self.config.max_budget
                                if self.config.fidelity_strategy == "fixed"
                                else self.config.min_budget
                            ),
                        )
                    )
        return tuple(output)
