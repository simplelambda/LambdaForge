"""Reproducible quasi-random Sobol initial design."""

from __future__ import annotations

import torch

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveSearcher import AdaptiveSearcher
from lambdaforge.hpo.SearchSpace import SearchSpace


class SobolSearcher(AdaptiveSearcher):
    """Cover mixed spaces by decoding a scrambled Sobol unit cube."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def propose(
        self,
        space: SearchSpace,
        state: AdaptiveOptimizerState,
        *,
        count: int = 1,
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic unseen points independent of process timing."""
        if count < 1:
            raise ValueError("Sobol candidate count must be positive.")
        engine = torch.quasirandom.SobolEngine(space.dimension, scramble=True, seed=self.seed)
        engine.fast_forward(max(0, state.decision_index))
        existing = set(state.configurations)
        existing.update(action.config_id for action in state.pending_actions.values())
        output: list[dict[str, object]] = []
        for vector in engine.draw(max(count * 16, 32)).tolist():
            values = space.decode(vector)
            identifier = space.identifier(values)
            if identifier not in existing:
                existing.add(identifier)
                output.append(values)
                if len(output) == count:
                    break
        return tuple(output)
