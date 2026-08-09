"""Exact synthetic contracts for adaptive HPO hardening."""

from __future__ import annotations

import pytest

from lambdaforge.hpo import (
    AdaptiveAction,
    AdaptiveActionKind,
    AdaptiveObservation,
    AdaptiveOptimizerState,
    AdaptiveTrialStatus,
    SearchSpace,
)
from lambdaforge.hpo.ResourceAdmissionController import ResourceAdmissionController
from lambdaforge.hpo.SobolSearcher import SobolSearcher
from tests.hpo.synthetic.FakeTrainingBackend import FakeTrainingBackend
from tests.hpo.synthetic.SyntheticCostModel import SyntheticCostModel
from tests.hpo.synthetic.SyntheticLearningCurves import SyntheticLearningCurves
from tests.hpo.synthetic.SyntheticMemoryModel import SyntheticMemoryModel
from tests.hpo.synthetic.SyntheticObjective import SyntheticObjective


def test_fake_backend_pause_resume_executes_cumulative_budget_once() -> None:
    objective = SyntheticObjective()
    resumed = FakeTrainingBackend(objective)
    first = AdaptiveAction("first", AdaptiveActionKind.START_NEW, "c", {"x": 0.65}, 7, 0, 10)
    second = AdaptiveAction("second", AdaptiveActionKind.RESUME, "c", {"x": 0.65}, 7, 10, 30)
    resumed.execute(first)
    resumed_result = resumed.execute(second)

    uninterrupted = FakeTrainingBackend(objective)
    full_result = uninterrupted.execute(
        AdaptiveAction("full", AdaptiveActionKind.START_NEW, "c", {"x": 0.65}, 7, 0, 30)
    )

    assert resumed_result.score == pytest.approx(full_result.score)
    assert resumed.total_executed_budget == uninterrupted.total_executed_budget == 30


def test_low_fidelity_objective_locates_optimum_below_fixed_full_equivalents() -> None:
    objective = SyntheticObjective(optimum=0.65, transient=0.03, rate=0.5)
    candidates = [0.05, 0.25, 0.45, 0.65, 0.85]
    low_budget = 3
    full_budget = 30
    selected = max(candidates, key=lambda value: objective.evaluate(value, low_budget))

    assert selected == 0.65
    adaptive_equivalents = (len(candidates) * low_budget + full_budget) / full_budget
    fixed_equivalents = len(candidates)
    assert adaptive_equivalents < fixed_equivalents


def test_slow_starter_is_not_dominated_at_conservative_minimum_budget() -> None:
    curves = SyntheticLearningCurves()
    assert curves.slow_starter(3) < curves.early_plateau(3)
    assert curves.slow_starter(10) > curves.early_plateau(10)


def test_synthetic_cost_and_memory_models_are_candidate_aware() -> None:
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    action = AdaptiveAction(
        "candidate",
        AdaptiveActionKind.START_NEW,
        "c",
        {"x": 1},
        7,
        0,
        1,
        resource_features={"size": 8},
    )
    cost = SyntheticCostModel({"candidate": 2.0}).predict(action, state)
    memory = SyntheticMemoryModel(bytes_per_unit=10).predict(action, state)
    admitted = ResourceAdmissionController().assess(
        action,
        state,
        SyntheticMemoryModel(bytes_per_unit=10),  # type: ignore[arg-type]
        available_bytes=100,
    )

    assert cost.mean == 2.0
    assert memory.mean == 80.0
    assert admitted[0]


def test_twenty_decisions_equal_ten_save_reload_ten(tmp_path) -> None:
    space = SearchSpace.from_mapping({"x": {"type": "float", "low": 0.0, "high": 1.0}})
    searcher = SobolSearcher(seed=19)

    def advance(state: AdaptiveOptimizerState, count: int) -> None:
        for _ in range(count):
            parameters = searcher.propose(space, state, count=1)[0]
            config_id = space.identifier(parameters)
            action_id = state.next_action_id()
            state.configurations[config_id] = parameters
            state.observations.append(
                AdaptiveObservation(
                    action_id,
                    config_id,
                    parameters,
                    7,
                    1,
                    float(parameters["x"]),
                    ((1, float(parameters["x"])),),
                    AdaptiveTrialStatus.COMPLETED,
                    observed_at_utc="2026-01-01T00:00:00+00:00",
                )
            )

    uninterrupted = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=19)
    advance(uninterrupted, 20)
    resumed = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=19)
    advance(resumed, 10)
    resumed = AdaptiveOptimizerState.load(resumed.save(tmp_path / "state.json"))
    advance(resumed, 10)

    assert resumed.to_dict() == uninterrupted.to_dict()
