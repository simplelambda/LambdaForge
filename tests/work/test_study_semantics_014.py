from __future__ import annotations

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility
from lambdaforge.work.runner import (
    _adaptive_parallelism,
    _adaptive_startup_trial_count,
    _admissible_gpu_slots,
    _gpu_admission_diagnostics,
)


def test_incomplete_composite_objective_explains_missing_component() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "quality": {"mode": "max", "weight": 0.5, "range": [0, 1]},
                "stability": {"mode": "max", "weight": 0.5, "range": [0, 1]},
            }
        }
    )
    evaluator = ObjectiveUtility(objective)
    records = [{"name": "quality", "value": 0.8, "step": 7}]
    assert evaluator.observation(records, fallback={}) is None
    assert evaluator.status(records, fallback={}) == {
        "status": "incomplete",
        "missing_components": ["stability"],
        "latest_step": 7,
        "latest_complete_step": None,
    }


def test_five_runs_per_each_of_two_gpus_has_no_artificial_three_run_limit() -> None:
    gib = 1024**3
    resources = ResourceRequest.from_mapping(
        {"cpu": 20, "memory": "64GiB", "gpu": 2, "gpu_memory": "10GiB"}
    )
    policy = AdaptiveSearchPolicy(runs_per_gpu=5, max_parallel=10)
    assert _adaptive_parallelism(resources, policy) == 10
    slots = _admissible_gpu_slots(
        ((40 * gib, 80 * gib), (40 * gib, 80 * gib)),
        active=(4, 4),
        last_launch=(0.0, 0.0),
        required_bytes=10 * gib,
        runs_per_gpu=5,
        now=100.0,
        launch_stagger_seconds=1.0,
        usable=(0, 1),
    )
    assert slots == (0, 1)


def test_startup_count_helper_requires_scientific_geometry_not_parallelism() -> None:
    policy = AdaptiveSearchPolicy(
        runs_per_gpu=10,
        max_parallel=20,
        candidate_budget=500,
        proposal_pool_size=500,
    )

    assert policy.startup_trials is None
    assert (
        _adaptive_startup_trial_count(
            policy,
            parallelism=20,
            candidate_budget=500,
            geometry_required=13,
        )
        == 13
    )
    assert (
        _adaptive_startup_trial_count(
            policy,
            parallelism=2,
            candidate_budget=500,
            geometry_required=13,
        )
        == 13
    )
    explicit = AdaptiveSearchPolicy(
        startup_trials=6,
        candidate_budget=500,
        proposal_pool_size=500,
    )
    assert _adaptive_startup_trial_count(explicit, parallelism=20, candidate_budget=500) == 6


def test_gpu_admission_prefers_the_least_loaded_device_instead_of_index_zero() -> None:
    gib = 1024**3

    slots = _admissible_gpu_slots(
        ((80 * gib, 96 * gib), (80 * gib, 96 * gib)),
        active=(4, 2),
        last_launch=(10.0, 20.0),
        required_bytes=20 * gib,
        runs_per_gpu=10,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0, 1),
    )

    assert slots == (1, 0)


def test_oom_backoff_is_reported_per_device_without_reserving_threshold_per_run() -> None:
    gib = 1024**3
    diagnostics = _gpu_admission_diagnostics(
        ((40 * gib, 96 * gib), (40 * gib, 96 * gib)),
        active=(3, 5),
        required_bytes=20 * gib,
        runs_per_gpu=(3, 10),
        configured_runs_per_gpu=10,
        max_parallel=20,
        pending=4,
        visible_gpus=("grant-a", "grant-b"),
        usable=(0, 1),
        admissible=(1,),
    )

    assert diagnostics["devices"][0]["reason"] == "oom_backoff"
    assert diagnostics["devices"][1]["runs_per_gpu"] == 10
    assert diagnostics["effective_runs_per_gpu_by_device"] == {
        "grant-a": 3,
        "grant-b": 10,
    }
    assert diagnostics["gpu_memory_semantics"].endswith("not-a-reservation")


def test_vram_threshold_wait_is_structured_and_retryable() -> None:
    gib = 1024**3
    memory = ((19 * gib, 80 * gib), (45 * gib, 80 * gib))
    slots = _admissible_gpu_slots(
        memory,
        active=(3, 3),
        last_launch=(0.0, 0.0),
        required_bytes=20 * gib,
        runs_per_gpu=5,
        now=100.0,
        launch_stagger_seconds=1.0,
        usable=(0, 1),
    )
    diagnostics = _gpu_admission_diagnostics(
        memory,
        active=(3, 3),
        required_bytes=20 * gib,
        runs_per_gpu=5,
        max_parallel=10,
        pending=4,
        visible_gpus=("0", "1"),
        usable=(0, 1),
        admissible=slots,
    )
    assert slots == (1,)
    assert diagnostics["devices"][0]["reason"] == "insufficient_free_vram"
    assert diagnostics["devices"][0]["retryable"] is True
    assert diagnostics["devices"][0]["required"] == 20 * gib
