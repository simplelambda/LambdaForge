"""Optional provider compatibility contract."""

from __future__ import annotations

import pytest

from lambdaforge.hpo import (
    AdaptiveObservation,
    AdaptiveOptimizerState,
    AdaptiveTrialStatus,
    SearchSpace,
)
from lambdaforge.hpo.BoTorchSearcher import BoTorchSearcher


@pytest.mark.optional_hpo
def test_botorch_searcher_fits_and_proposes_a_novel_point() -> None:
    pytest.importorskip("botorch")
    space = SearchSpace.from_mapping(
        {"optimizer.params.lr": {"type": "float", "low": 0.0, "high": 1.0}}
    )
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=9)
    for index, value in enumerate((0.1, 0.3, 0.7, 0.9)):
        parameters = {"optimizer.params.lr": value}
        config_id = space.identifier(parameters)
        score = -((value - 0.6) ** 2)
        state.configurations[config_id] = parameters
        state.observations.append(
            AdaptiveObservation(
                str(index),
                config_id,
                parameters,
                7,
                1,
                score,
                ((1, score),),
                AdaptiveTrialStatus.COMPLETED,
            )
        )
    proposed = BoTorchSearcher(
        seed=9,
        acquisition="expected_improvement",
        raw_samples=32,
        num_restarts=3,
    ).propose(space, state)
    assert len(proposed) == 1
    assert space.identifier(proposed[0]) not in state.configurations


@pytest.mark.optional_hpo
def test_botorch_refresh_interval_reuses_then_refits_the_surrogate() -> None:
    pytest.importorskip("botorch")
    space = SearchSpace.from_mapping({"x": {"type": "float", "low": 0.0, "high": 1.0}})
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=3)

    def observe(index: int, value: float) -> None:
        parameters = {"x": value}
        config_id = space.identifier(parameters)
        score = -((value - 0.6) ** 2)
        state.configurations[config_id] = parameters
        state.observations.append(
            AdaptiveObservation(
                str(index),
                config_id,
                parameters,
                7,
                1,
                score,
                ((1, score),),
                AdaptiveTrialStatus.COMPLETED,
            )
        )

    for index, value in enumerate((0.1, 0.3, 0.7, 0.9)):
        observe(index, value)
    searcher = BoTorchSearcher(
        seed=3,
        acquisition="expected_improvement",
        raw_samples=16,
        num_restarts=2,
        refresh_interval=4,
    )
    searcher.propose(space, state)
    first_model = searcher._cached_model

    observe(4, 0.2)
    state.decision_index += 1
    searcher.propose(space, state)
    assert searcher._cached_model is first_model

    for index, value in enumerate((0.4, 0.6, 0.8), start=5):
        observe(index, value)
    state.decision_index += 1
    searcher.propose(space, state)
    assert searcher._cached_model is not first_model
