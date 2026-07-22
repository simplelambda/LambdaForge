"""Alternative hypotheses supported by paired comparison tests."""

from enum import Enum
from typing import TypeVar

_Number = TypeVar("_Number")


class PairedAlternative(str, Enum):
    """Declare the hypothesis used for the selected comparison p-value."""

    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"
    OBSERVED_DIRECTION = "observed_direction"

    def select(
        self,
        *,
        two_sided: _Number,
        better: _Number,
        worse: _Number,
        observed_mean: float,
    ) -> _Number:
        """Select the value associated with this declared alternative."""
        if self is PairedAlternative.TWO_SIDED:
            return two_sided
        if self is PairedAlternative.GREATER:
            return better
        if self is PairedAlternative.LESS:
            return worse
        return better if observed_mean >= 0.0 else worse
