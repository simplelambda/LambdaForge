"""Dependency-free deterministic adaptive-search baseline."""

from __future__ import annotations

import random

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveSearcher import AdaptiveSearcher
from lambdaforge.hpo.SearchSpace import SearchSpace


class RandomAdaptiveSearcher(AdaptiveSearcher):
    """Sample unseen unit vectors with a controller-state-derived private RNG."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def propose(
        self,
        space: SearchSpace,
        state: AdaptiveOptimizerState,
        *,
        count: int = 1,
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic unseen random mappings."""
        if count < 1:
            raise ValueError("Random candidate count must be positive.")
        rng = random.Random((self.seed << 32) ^ state.decision_index)
        existing = set(state.configurations)
        existing.update(action.config_id for action in state.pending_actions.values())
        output: list[dict[str, object]] = []
        for _ in range(max(count * 100, 100)):
            values = space.decode([rng.random() for _ in range(space.dimension)])
            identifier = space.identifier(values)
            if identifier not in existing:
                existing.add(identifier)
                output.append(values)
                if len(output) == count:
                    break
        return tuple(output)
