"""Full-budget baseline fidelity policy."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState


class FixedBudgetFidelityPolicy:
    """Promote every partial trial directly to the declared maximum budget."""

    def __init__(self, config: AdaptiveOptimizerConfig) -> None:
        self.config = config

    def resume_candidates(self, state: AdaptiveOptimizerState) -> tuple[AdaptiveAction, ...]:
        """Return deterministic full-budget continuation actions."""
        output: list[AdaptiveAction] = []
        for config_id, parameters in sorted(state.configurations.items()):
            for seed in self.config.search_seeds:
                current = max(
                    (
                        item.budget
                        for item in state.observations_for(config_id, seed=seed)
                        if item.status.value in {"paused", "completed", "cancelled"}
                    ),
                    default=0,
                )
                if 0 < current < self.config.max_budget:
                    output.append(
                        AdaptiveAction(
                            f"candidate-{state.decision_index + 1:06d}-fixed-{config_id}-{seed}",
                            AdaptiveActionKind.RESUME,
                            config_id,
                            parameters,
                            seed,
                            current,
                            self.config.max_budget,
                        )
                    )
        return tuple(output)
