"""Deterministic cumulative-budget HPO backend."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveObservation import AdaptiveObservation
from lambdaforge.hpo.AdaptiveTrialStatus import AdaptiveTrialStatus
from tests.hpo.synthetic.SyntheticObjective import SyntheticObjective


class FakeTrainingBackend:
    """Execute only missing budgets against a synthetic objective."""

    def __init__(self, objective: SyntheticObjective) -> None:
        self.objective = objective
        self.executed_budgets: dict[tuple[str, int], int] = {}

    def execute(self, action: AdaptiveAction) -> AdaptiveObservation:
        """Return a cumulative curve and account only the action increment."""
        key = (action.config_id, action.seed)
        previous = self.executed_budgets.get(key, 0)
        if previous != action.current_budget:
            raise ValueError("Synthetic resume current budget does not match backend state.")
        self.executed_budgets[key] = action.target_budget
        value = float(action.parameters["x"])
        curve = tuple(
            (budget, self.objective.evaluate(value, budget))
            for budget in range(1, action.target_budget + 1)
        )
        return AdaptiveObservation(
            action.action_id,
            action.config_id,
            action.parameters,
            action.seed,
            action.target_budget,
            curve[-1][1],
            curve,
            AdaptiveTrialStatus.COMPLETED,
        )

    @property
    def total_executed_budget(self) -> int:
        """Return real cumulative work, not the sum of target fidelities."""
        return sum(self.executed_budgets.values())
