"""Deterministic slow-start and plateau curves."""


class SyntheticLearningCurves:
    """Provide exact curves used by pruning tests."""

    def slow_starter(self, budget: int) -> float:
        """Improve steadily to the best terminal value."""
        return 0.1 + 0.09 * budget

    def early_plateau(self, budget: int) -> float:
        """Start well but plateau below the slow starter."""
        return min(0.62, 0.54 + 0.04 * budget)
