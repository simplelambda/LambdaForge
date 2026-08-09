"""Optional provider compatibility contract."""

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


@pytest.mark.optional_hpo
def test_botorch_uses_mixed_geometry_all_fidelities_and_pending_points() -> None:
    pytest.importorskip("botorch")
    space = SearchSpace.from_mapping(
        {
            "optimizer": {
                "type": "categorical",
                "choices": ["Lion", "AdamW", "SGD"],
            },
            "x": {"type": "float", "low": 0.0, "high": 1.0},
        }
    )
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=5)
    values = (
        ("AdamW", 0.1),
        ("SGD", 0.3),
        ("Lion", 0.6),
        ("AdamW", 0.9),
    )
    for index, (optimizer, x) in enumerate(values):
        parameters = {"optimizer": optimizer, "x": x}
        config_id = space.identifier(parameters)
        final = -((x - 0.55) ** 2) + (0.02 if optimizer == "Lion" else 0.0)
        state.configurations[config_id] = parameters
        state.observations.append(
            AdaptiveObservation(
                str(index),
                config_id,
                parameters,
                7,
                3,
                final,
                ((1, final - 0.2), (2, final - 0.05), (3, final)),
                AdaptiveTrialStatus.COMPLETED,
            )
        )
    pending_parameters = {"optimizer": "SGD", "x": 0.55}
    pending = AdaptiveAction(
        "pending",
        AdaptiveActionKind.START_NEW,
        space.identifier(pending_parameters),
        pending_parameters,
        7,
        0,
        1,
    )
    state.register_pending(pending)
    searcher = BoTorchSearcher(
        seed=5,
        acquisition="expected_improvement",
        raw_samples=24,
        num_restarts=2,
        min_budget=1,
        max_budget=3,
        budget_step=1,
    )

    proposed = searcher.propose(space, state)

    assert searcher._cached_model.__class__.__name__ == "MixedSingleTaskGP"
    train_x = searcher._cached_model.train_inputs[0]
    assert sorted(set(train_x[:, -1].tolist())) == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert space.identifier(proposed[0]) not in set(state.configurations) | {pending.config_id}
