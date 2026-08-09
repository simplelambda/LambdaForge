"""Deterministic successive-halving baseline."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState


class SuccessiveHalvingFidelityPolicy:
    """Promote the best half at each rung for an explicit baseline comparison."""

    def __init__(self, config: AdaptiveOptimizerConfig) -> None:
        self.config = config

    def resume_candidates(self, state: AdaptiveOptimizerState) -> tuple[AdaptiveAction, ...]:
        """Return next-rung actions for the top half by observed score."""
        latest: list[tuple[str, int, int, float]] = []
        for config_id in state.configurations:
            for seed in self.config.search_seeds:
                observations = [
                    item
                    for item in state.observations_for(config_id, seed=seed)
                    if item.score is not None
                    and not item.oom
                    and item.status.value != "early_stopped"
                ]
                if observations:
                    observation = max(observations, key=lambda item: item.budget)
                    assert observation.score is not None
                    latest.append((config_id, seed, observation.budget, float(observation.score)))
        reverse = self.config.direction == "maximize"
        ordered = sorted(latest, key=lambda item: item[3], reverse=reverse)
        promoted = ordered[: max(1, (len(ordered) + 1) // 2)]
        output: list[AdaptiveAction] = []
        for config_id, seed, current, _ in promoted:
            if current >= self.config.max_budget:
                continue
            target = min(max(current * 2, current + 1), self.config.max_budget)
            output.append(
                AdaptiveAction(
                    f"candidate-{state.decision_index + 1:06d}-halving-{config_id}-{seed}-{target}",
                    AdaptiveActionKind.RESUME,
                    config_id,
                    state.configurations[config_id],
                    seed,
                    current,
                    target,
                )
            )
        return tuple(output)
