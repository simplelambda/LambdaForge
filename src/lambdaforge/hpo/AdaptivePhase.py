"""Scientific phases of adaptive optimization."""

from enum import Enum


class AdaptivePhase(str, Enum):
    """Keep resource-efficient search separate from final confirmation."""

    SEARCH = "search"
    CONFIRMATION = "confirmation"
    FINISHED = "finished"
