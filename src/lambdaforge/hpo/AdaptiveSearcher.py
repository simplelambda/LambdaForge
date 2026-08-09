"""Search-model contract for adaptive optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.SearchSpace import SearchSpace


class AdaptiveSearcher(ABC):
    """Propose non-duplicate parameter mappings from current optimizer knowledge."""

    @abstractmethod
    def propose(
        self,
        space: SearchSpace,
        state: AdaptiveOptimizerState,
        *,
        count: int = 1,
    ) -> tuple[dict[str, object], ...]:
        """Return up to ``count`` candidate mappings."""
        raise NotImplementedError
