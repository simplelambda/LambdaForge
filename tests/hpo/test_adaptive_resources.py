"""CPU-only decision tests for adaptive resource intelligence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    BoundedResourceTrajectory,
    CandidateResourceAction,
    ExplorationEvaluation,
    GPUPlacementPlanner,
    GPUResourceState,
    PlacementOOMEvidence,
    ResourceDemandModel,
    ResourceHistoryStore,
    ResourcePrediction,
    ResourceProfileObservation,
    ResourceTrajectoryAnalyzer,
    ResourceTrajectorySample,
    WaitRegretTracker,
    oom_evidence,
    resource_state,
)
from lambdaforge.work.models import WorkResources, WorkResult

GIB = 1024**3


def observation(
    key: str,
    peak: int,
    *,
    compatibility: str = "compatible",
    state: str = "completed",
    exact: bool = True,
    lower: int = 0,
    parameters: dict[str, Any] | None = None,
    co_runners: int = 0,
    throughput: float | None = None,
    hardware: str = "H100-80",
) -> ResourceProfileObservation:
    return ResourceProfileObservation(
        candidate_key=key,
        compatibility_key=compatibility,
        work_class="project.Train",
        parameters=parameters or {},
        fixed_arguments={"batch_size": 32},
        fidelity={"target": 100, "maximum": 100},
        hardware=hardware,
        total_bytes=80 * GIB,
        state=state,  # type: ignore[arg-type]
        observed_peak_bytes=peak,
        peak_is_exact=exact,
        duration_seconds=100.0,
        time_to_peak_seconds=20.0,
        lower_bound_bytes=lower,
        co_runners=co_runners,
        throughput=throughput,
    )


def prediction(
    key: str,
    peak: int,
    *,
    duration: float = 10.0,
    lower_bound: int = 0,
    history: int = 4,
) -> ResourcePrediction:
    return ResourcePrediction(
        candidate_key=key,
        predicted_peak_bytes=peak,
        lower_bytes=max(lower_bound, peak),
        upper_bytes=max(lower_bound, peak),
        known_lower_bound_bytes=lower_bound,
        predicted_time_to_envelope_seconds=2.0,
        predicted_duration_seconds=duration,
        compatible_history_count=history,
        exact_history_count=history,
        support="exact-candidate",
        calibration="good",
        backend="test",
        samples=(peak, peak, peak),
    )


def action(key: str, peak: int, value: float, *, duration: float = 10.0) -> CandidateResourceAction:
    return CandidateResourceAction(
        key,
        {"trial_index": int(key.removeprefix("c") or 0), "resource_compatibility_key": "c"},
        value,
        prediction(key, peak, duration=duration),
    )


def gpu(
    index: int = 0,
    *,
    free: int = 80,
    external: int = 0,
    active: tuple[ActiveResourceCommitment, ...] = (),
    cap: int = 10,
) -> GPUResourceState:
    return GPUResourceState(
        index,
        str(index),
        "H100-80",
        80 * GIB,
        free * GIB,
        external * GIB,
        active,
        cap,
    )


def test_candidate_specific_model_packs_small_and_separates_large_runs() -> None:
    observations = [observation(f"c{index}", 10 * GIB) for index in range(1, 5)]
    model = ResourceDemandModel(observations)
    actions = [
        CandidateResourceAction(
            f"c{index}",
            {"trial_index": index, "resource_compatibility_key": "compatible"},
            1.0 - index / 100,
            model.predict(
                candidate_key=f"c{index}",
                compatibility_key="compatible",
                parameters={},
                hardware="H100-80",
                total_bytes=80 * GIB,
            ),
        )
        for index in range(1, 5)
    ]
    admitted, blocked = GPUPlacementPlanner(model).place(actions, (gpu(cap=4),), max_launches=4)
    assert len(admitted) == 4
    assert not blocked

    large = action("c1", 48 * GIB, 1.0)
    small = action("c2", 10 * GIB, 0.9)
    admitted, _ = GPUPlacementPlanner(ResourceDemandModel()).place(
        (large, small), (gpu(cap=2),), max_launches=2
    )
    assert len(admitted) == 2
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (large, action("c2", 48 * GIB, 0.9)), (gpu(cap=2),), max_launches=2
    )
    assert len(admitted) == 1
    assert blocked[0].state == "RESOURCE_BLOCKED"


def test_cold_start_is_one_run_per_gpu_until_evidence_exists() -> None:
    model = ResourceDemandModel()
    unknown = [
        CandidateResourceAction(
            f"unknown-{index}",
            {"trial_index": index},
            1.0,
            model.predict(
                candidate_key=f"unknown-{index}",
                compatibility_key="new",
                parameters={"width": index},
                hardware="H100-80",
                total_bytes=80 * GIB,
            ),
        )
        for index in range(10)
    ]
    admitted, _ = GPUPlacementPlanner(model).place(unknown, (gpu(0), gpu(1)), max_launches=10)
    assert len(admitted) == 2
    assert {value.target_gpu for value in admitted} == {0, 1}


def test_one_neighbour_does_not_collapse_between_candidate_uncertainty() -> None:
    model = ResourceDemandModel([observation("known", 10 * GIB, parameters={"width": 64})])
    unseen = model.predict(
        candidate_key="unseen",
        compatibility_key="compatible",
        parameters={"width": 256},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )

    assert unseen.predicted_peak_bytes == 10 * GIB
    assert unseen.commitment_bytes == 80 * GIB
    assert unseen.support == "sparse-near-compatible"


def test_early_low_memory_uses_future_commitment_not_current_usage() -> None:
    ramping = ActiveResourceCommitment("large", 45 * GIB, current_bytes=4 * GIB)
    candidate = action("c2", 20 * GIB, 0.8)
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), (gpu(free=60, external=16, active=(ramping,)),), max_launches=1
    )
    assert not admitted
    assert blocked[0].reason == "predicted future GPU-memory envelope does not fit now"


def test_new_external_pressure_cannot_be_hidden_by_future_attribution() -> None:
    active = ActiveResourceCommitment(
        "running",
        40 * GIB,
        current_bytes=10 * GIB,
        remaining_seconds=20.0,
    )
    # The learned future ledger alone leaves 40 GiB, but physical free VRAM has fallen to 30 GiB
    # since the prior baseline. Admission must honor the stricter live physical observation.
    pressured = gpu(free=30, external=0, active=(active,))
    candidate = action("c2", 35 * GIB, 0.8)

    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), (pressured,), max_launches=1
    )

    assert pressured.predicted_headroom_bytes == 40 * GIB
    assert pressured.admission_headroom_bytes == 30 * GIB
    assert not admitted
    assert blocked[0].devices[0]["admission_headroom_bytes"] == 30 * GIB


def test_oom_lower_bound_blocks_same_or_less_headroom_and_allows_more() -> None:
    evidence = oom_evidence(
        "CUDA out of memory. Tried to allocate 2 GiB",
        candidate_key="heavy",
        compatibility_key="compatible",
        hardware="H100-80",
        total_bytes=80 * GIB,
        headroom_bytes=26 * GIB,
        resident_bytes=24 * GIB,
        physical_free_bytes=2 * GIB,
        external_bytes=0,
        active_co_runs=1,
    )
    assert evidence.lower_bound_bytes == 26 * GIB
    model = ResourceDemandModel(
        [observation("heavy", 24 * GIB, state="oom", exact=False, lower=26 * GIB)]
    )
    predicted = model.predict(
        candidate_key="heavy",
        compatibility_key="compatible",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    candidate = CandidateResourceAction("retry", {"trial_index": 1}, 1.0, predicted)
    admitted, blocked = GPUPlacementPlanner(model).place(
        (candidate,), (gpu(0, free=25, external=55), gpu(1, free=18, external=62)), max_launches=1
    )
    assert not admitted
    assert all(
        device["reason"] == "dominated-by-known-oom-lower-bound" for device in blocked[0].devices
    )
    admitted, _ = GPUPlacementPlanner(model).place(
        (candidate,), (gpu(free=40, external=40),), max_launches=1
    )
    assert len(admitted) == 1


def test_infeasible_device_type_is_not_a_scientific_failure() -> None:
    candidate = CandidateResourceAction(
        "huge",
        {"trial_index": 7},
        0.99,
        prediction("huge", 83 * GIB, lower_bound=82 * GIB),
    )
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), (gpu(),), max_launches=1
    )
    assert not admitted
    assert blocked[0].state == "RESOURCE_INFEASIBLE_ON_DEVICE_TYPE"
    assert "objective" not in blocked[0].to_dict()


def test_best_action_stays_blocked_while_next_feasible_action_backfills() -> None:
    active = ActiveResourceCommitment("running", 50 * GIB, remaining_seconds=20.0)
    heavy = action("c1", 42 * GIB, 0.91)
    light = action("c2", 14 * GIB, 0.78, duration=5.0)
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (heavy, light), (gpu(free=30, active=(active,)),), max_launches=1
    )
    assert admitted[0].candidate_key == "c2"
    assert admitted[0].backfill is True
    assert blocked[0].candidate_key == "c1"
    assert blocked[0].earliest_opportunity_seconds == 20.0


def test_best_fit_preserves_large_gpu_for_heavy_action() -> None:
    devices = (gpu(0, free=50, external=30), gpu(1, free=25, external=55))
    small = action("c1", 20 * GIB, 1.0)
    heavy = action("c2", 45 * GIB, 0.9)
    admitted, _ = GPUPlacementPlanner(ResourceDemandModel()).place(
        (small, heavy), devices, max_launches=2
    )
    assert [(value.candidate_key, value.target_gpu) for value in admitted] == [
        ("c1", 1),
        ("c2", 0),
    ]


def test_heterogeneous_devices_use_their_own_conditioned_prediction() -> None:
    smaller = GPUResourceState(0, "a100", "A100-40", 40 * GIB, 40 * GIB, 0, (), 2)
    larger = GPUResourceState(1, "h100", "H100-80", 80 * GIB, 80 * GIB, 0, (), 2)
    candidate = CandidateResourceAction(
        "mixed",
        {"trial_index": 1},
        1.0,
        prediction("mixed-h100", 30 * GIB),
        {
            0: prediction("mixed-a100", 45 * GIB, lower_bound=45 * GIB),
            1: prediction("mixed-h100", 30 * GIB),
        },
    )

    admitted, _ = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), (smaller, larger), max_launches=1
    )

    assert admitted[0].target_gpu == 1
    assert admitted[0].prediction["candidate_key"] == "mixed-h100"


def test_backfill_reservation_rejects_a_run_that_would_starve_heavy_work() -> None:
    active = ActiveResourceCommitment("active", 60 * GIB, remaining_seconds=10.0)
    device = gpu(free=20, active=(active,))
    heavy = action("c1", 70 * GIB, 1.0, duration=30.0)
    short = action("c2", 10 * GIB, 0.8, duration=5.0)
    long = action("c3", 10 * GIB, 0.7, duration=20.0)
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (heavy, short, long), (device,), max_launches=2
    )
    assert [value.candidate_key for value in admitted] == ["c2"]
    assert {value.candidate_key for value in blocked} == {"c1", "c3"}


def test_measured_interference_can_stop_unproductive_extra_packing() -> None:
    history = [
        observation("one", 10 * GIB, co_runners=0, throughput=100.0),
        observation("two", 10 * GIB, co_runners=1, throughput=75.0),
        observation("three", 10 * GIB, co_runners=2, throughput=50.0),
    ]
    model = ResourceDemandModel(history)
    active = (
        ActiveResourceCommitment("a", 10 * GIB),
        ActiveResourceCommitment("b", 10 * GIB),
    )
    third = CandidateResourceAction(
        "third",
        {"trial_index": 3, "resource_compatibility_key": "compatible"},
        0.8,
        prediction("third", 10 * GIB),
    )
    admitted, blocked = GPUPlacementPlanner(model).place(
        (third,), (gpu(free=60, active=active),), max_launches=1
    )
    assert not admitted
    assert blocked[0].devices[0]["reason"] == "predicted-aggregate-throughput-would-not-improve"


def test_duration_prediction_is_conditioned_on_observed_colocation() -> None:
    solo = observation("solo", 10 * GIB, co_runners=0)
    packed = ResourceProfileObservation(
        **{
            **observation("packed", 10 * GIB, co_runners=1).to_dict(),
            "duration_seconds": 175.0,
        }
    )
    model = ResourceDemandModel((solo, packed))

    assert model.adjusted_duration("compatible", 20.0, 1) == 20.0
    assert model.adjusted_duration("compatible", 20.0, 2) == 35.0
    assert model.adjusted_duration("unknown", 20.0, 2) == 20.0


def test_late_validation_peak_and_pruned_resource_evidence_remain_conservative() -> None:
    history = [
        ResourceProfileObservation(
            **{
                **observation("candidate", 28 * GIB).to_dict(),
                "peak_phase": "validation",
            }
        ),
        observation("pruned", 12 * GIB, state="pruned", exact=False, lower=12 * GIB),
    ]
    model = ResourceDemandModel(history)
    live = model.predict(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
        observed_prefix_peak_bytes=20 * GIB,
        phase="training",
    )
    assert live.commitment_bytes >= 28 * GIB
    censored = model.predict(
        candidate_key="pruned",
        compatibility_key="compatible",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert censored.exact_history_count == 0
    assert censored.commitment_bytes == 80 * GIB


def test_resource_stability_depends_on_decision_uncertainty_not_a_timer() -> None:
    stable = prediction("known", 10 * GIB, history=8)
    uncertain = ResourcePrediction(
        candidate_key="known",
        predicted_peak_bytes=10 * GIB,
        lower_bytes=8 * GIB,
        upper_bytes=20 * GIB,
        known_lower_bound_bytes=0,
        predicted_time_to_envelope_seconds=2.0,
        predicted_duration_seconds=10.0,
        compatible_history_count=8,
        exact_history_count=8,
        support="exact-candidate",
        calibration="good",
        backend="test",
        samples=(8 * GIB, 20 * GIB),
    )
    assert (
        resource_state(stable, observed_peak_bytes=10 * GIB, phase="validation")
        == "RESOURCE_STABLE"
    )
    assert resource_state(uncertain, observed_peak_bytes=8 * GIB, phase="training") == "RAMPING"


def test_history_is_persistent_compatible_and_deterministic(tmp_path: Path) -> None:
    study = tmp_path / "study"
    shared = tmp_path / "shared"
    store = ResourceHistoryStore(study, shared)
    store.append(observation("candidate", 11 * GIB))
    loaded = ResourceHistoryStore(tmp_path / "other", shared).load()
    assert len(loaded) == 1
    model = ResourceDemandModel(loaded)
    first = model.predict(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    second = model.predict(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert first == second
    incompatible = model.predict(
        candidate_key="candidate",
        compatibility_key="different-code",
        parameters={},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert incompatible.support == "cold-start"


def test_active_snapshot_survives_restart_only_as_provisional_evidence(tmp_path: Path) -> None:
    store = ResourceHistoryStore(tmp_path / "study")
    live = ActiveResourceEvidence(
        "candidate",
        "compatible",
        {"width": 64},
        "H100-80",
        80 * GIB,
        11 * GIB,
        12 * GIB,
        (12 * GIB, 16 * GIB, 80 * GIB),
        "PLATEAU_UNCONFIRMED",
        30.0,
        phase="training",
        step=3,
        trajectory=(ResourceTrajectorySample(30.0, 12 * GIB, step=3, phase="training"),),
    )
    store.persist_active((live,))

    restored = ResourceHistoryStore(tmp_path / "study").load_active()
    assert restored == (live,)
    predicted = ResourceDemandModel(active_evidence=restored).predict(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert predicted.is_provisional is True
    assert predicted.exact_history_count == 0

    store.persist_active(())
    assert ResourceHistoryStore(tmp_path / "study").load_active() == ()


def test_bounded_trajectory_preserves_the_true_peak_and_transitions() -> None:
    trajectory = BoundedResourceTrajectory()
    for index in range(500):
        trajectory.append(
            ResourceTrajectorySample(
                float(index),
                (999 if index == 177 else index % 50) * GIB,
                phase="validation" if 200 <= index < 220 else "training",
                step=index // 5,
            )
        )
    assert len(trajectory.values) <= 192
    assert max(value.physical_bytes for value in trajectory.values) == 999 * GIB
    assert {value.phase for value in trajectory.values} == {"training", "validation"}


def test_live_plateau_enables_one_protected_checkpoint_aware_cold_start_probe() -> None:
    live = ActiveResourceEvidence(
        candidate_key="resident-a",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
        current_bytes=12 * GIB,
        running_peak_bytes=12 * GIB,
        future_peak_samples=(12 * GIB, 15 * GIB, 80 * GIB),
        resource_state="PROVISIONALLY_STABLE",
        elapsed_seconds=120.0,
        step=6,
        checkpoint_step=6,
        checkpoint_elapsed_seconds=115.0,
        checkpoint_resumable=True,
    )
    model = ResourceDemandModel(active_evidence=(live,))
    queued = model.predict(
        candidate_key="queued-c",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
        user_minimum_bytes=20 * GIB,
    )
    action_c = CandidateResourceAction(
        "c",
        {"trial_index": 3, "resource_compatibility_key": "compatible"},
        1.0,
        queued,
    )
    resident_a = ActiveResourceCommitment(
        "resident-a",
        80 * GIB,
        current_bytes=12 * GIB,
        running_peak_bytes=12 * GIB,
        future_peak_samples=(12 * GIB, 15 * GIB, 80 * GIB),
        remaining_seconds=500.0,
        resource_state="PROVISIONALLY_STABLE",
        checkpoint_step=6,
        checkpoint_elapsed_seconds=115.0,
        checkpoint_resumable=True,
        elapsed_seconds=120.0,
    )
    resident_b = ActiveResourceCommitment(
        "resident-b",
        80 * GIB,
        current_bytes=40 * GIB,
        running_peak_bytes=40 * GIB,
        future_peak_samples=(40 * GIB, 80 * GIB),
        resource_state="RAMPING",
        elapsed_seconds=120.0,
    )

    admitted, _ = GPUPlacementPlanner(model).place(
        (action_c,),
        (
            gpu(0, free=68, active=(resident_a,), cap=5),
            gpu(1, free=40, active=(resident_b,), cap=5),
        ),
        max_launches=1,
    )

    assert len(admitted) == 1
    assert admitted[0].target_gpu == 0
    assert admitted[0].admission_mode == "EXPLORATORY_ADMISSION"
    # Unknown checkpoint cadence is not misread as the five-second checkpoint age. The next
    # evidence event therefore comes from the real duration estimate (500 s), not a tiny horizon.
    assert admitted[0].rollback_cost_seconds == 250.75


def test_concurrency_ladder_allows_only_one_uncharacterized_increment() -> None:
    stable = ActiveResourceCommitment(
        "a",
        80 * GIB,
        current_bytes=10 * GIB,
        running_peak_bytes=10 * GIB,
        future_peak_samples=(10 * GIB, 12 * GIB, 80 * GIB),
        resource_state="PROVISIONALLY_STABLE",
        checkpoint_resumable=True,
    )
    probe = ActiveResourceCommitment(
        "b",
        80 * GIB,
        current_bytes=10 * GIB,
        running_peak_bytes=10 * GIB,
        future_peak_samples=(10 * GIB, 15 * GIB, 80 * GIB),
        resource_state="PLATEAU_UNCONFIRMED",
        admission_mode="EXPLORATORY_ADMISSION",
    )
    candidate = CandidateResourceAction(
        "c",
        {"trial_index": 3, "resource_compatibility_key": "compatible"},
        1.0,
        ResourcePrediction(
            "c",
            10 * GIB,
            10 * GIB,
            80 * GIB,
            10 * GIB,
            None,
            100.0,
            0,
            0,
            "cold-start",
            "poor",
            "test",
            (10 * GIB, 20 * GIB, 80 * GIB),
        ),
    )
    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), (gpu(free=60, active=(stable, probe), cap=5),), max_launches=1
    )
    assert not admitted
    assert blocked


def test_live_evidence_transfers_across_seed_without_becoming_exact() -> None:
    live = ActiveResourceEvidence(
        "same-candidate",
        "compatible",
        {"width": 64},
        "H100-80",
        80 * GIB,
        11 * GIB,
        12 * GIB,
        (12 * GIB, 14 * GIB, 80 * GIB),
        "PLATEAU_UNCONFIRMED",
        50.0,
    )
    predicted = ResourceDemandModel(active_evidence=(live,)).predict(
        candidate_key="same-candidate",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert predicted.support == "active-same-candidate"
    assert predicted.is_provisional is True
    assert predicted.exact_history_count == 0
    assert predicted.provisional_lower_bound_bytes == 12 * GIB


def test_active_neighbour_widens_distribution_without_imposing_intrinsic_lower_bound() -> None:
    live = ActiveResourceEvidence(
        "different-candidate",
        "compatible",
        {"width": 64},
        "H100-80",
        80 * GIB,
        40 * GIB,
        40 * GIB,
        (40 * GIB, 55 * GIB),
        "RAMPING",
        30.0,
        future_peak_weights=(0.8, 0.2),
    )
    predicted = ResourceDemandModel(
        active_evidence=(live,),
        parameter_schema={"width": {"range": [32, 256]}},
    ).predict(
        candidate_key="queued",
        compatibility_key="compatible",
        parameters={"width": 96},
        hardware="H100-80",
        total_bytes=80 * GIB,
        user_minimum_bytes=8 * GIB,
    )

    assert predicted.provisional_lower_bound_bytes == 8 * GIB
    assert predicted.upper_bytes >= 40 * GIB


def test_placement_oom_does_not_create_intrinsic_bound_or_global_backoff() -> None:
    extracted = oom_evidence(
        "CUDA out of memory. Tried to allocate 2 GiB",
        candidate_key="heavy",
        compatibility_key="compatible",
        hardware="H100-80",
        total_bytes=80 * GIB,
        headroom_bytes=30 * GIB,
        resident_bytes=None,
        physical_free_bytes=1 * GIB,
        external_bytes=0,
        active_co_runs=1,
    )
    assert extracted.lower_bound_bytes == 0
    failed = PlacementOOMEvidence(
        "heavy",
        "compatible",
        "H100-80",
        ("heavy-resident",),
        30 * GIB,
        1 * GIB,
        79 * GIB,
        2 * GIB,
        None,
        "lower-bound-only",
        "failed-heavy-heavy",
    )
    model = ResourceDemandModel(placement_failures=(failed,))
    light = ActiveResourceCommitment(
        "light-resident",
        10 * GIB,
        current_bytes=10 * GIB,
        running_peak_bytes=10 * GIB,
        future_peak_samples=(10 * GIB,),
        resource_state="RESOURCE_STABLE",
        checkpoint_resumable=True,
    )
    candidate = CandidateResourceAction(
        "heavy-action",
        {"trial_index": 2, "resource_compatibility_key": "compatible"},
        1.0,
        prediction("heavy", 48 * GIB),
    )
    admitted, _ = GPUPlacementPlanner(model).place(
        (candidate,), (gpu(free=70, active=(light,), cap=5),), max_launches=1
    )
    assert admitted


def test_schema_aware_distance_handles_log_scale_and_inactive_conditionals() -> None:
    from lambdaforge.hpo.AdaptiveResources import _mixed_distance

    schema = {
        "learning_rate": {"range": [1e-5, 1e-3], "scale": "log"},
        "depth": {"values": [1, 2], "when": {"enabled": True}},
        "enabled": {"values": [True, False]},
    }
    first = _mixed_distance(
        {"learning_rate": 1e-5, "enabled": False},
        {"learning_rate": 1e-4, "enabled": False},
        schema,
    )
    second = _mixed_distance(
        {"learning_rate": 1e-4, "enabled": False},
        {"learning_rate": 1e-3, "enabled": False},
        schema,
    )
    assert math.isclose(first, second)
    assert (
        _mixed_distance(
            {"learning_rate": 1e-4, "enabled": False, "depth": 1},
            {"learning_rate": 1e-4, "enabled": False, "depth": 2},
            schema,
        )
        == 0.0
    )


def test_live_update_prefers_per_process_vram_over_predicted_proportions(
    monkeypatch: Any,
) -> None:
    from lambdaforge.work.runner import _update_active_resource_commitments

    first, second = object(), object()
    specification = {
        "trial_parameters": {},
        "definition": {"resources": {"gpu_memory_bytes": 0}},
        "hpo_dispatched_monotonic": 0.0,
    }
    pending = {
        first: (dict(specification), 0, object()),
        second: (dict(specification), 0, object()),
    }
    commitments = {
        first: ActiveResourceCommitment("a", 25 * GIB),
        second: ActiveResourceCommitment("b", 15 * GIB),
    }
    metadata = {
        first: {"candidate_key": "a", "compatibility_key": "c", "hardware": "H100-80"},
        second: {"candidate_key": "b", "compatibility_key": "c", "hardware": "H100-80"},
    }
    monkeypatch.setattr(
        "lambdaforge.work.runner._read_live_resource_snapshot",
        lambda _: {"phase": "training", "step": 2},
    )
    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_process_memory",
        lambda *_: ({first: 10 * GIB, second: 30 * GIB}, "nvml-process-exact"),
    )

    evidence = _update_active_resource_commitments(
        ((40 * GIB, 80 * GIB),),
        pending=pending,  # type: ignore[arg-type]
        commitments=commitments,
        trajectories={},
        metadata=metadata,
        model=ResourceDemandModel(),
        now=10.0,
    )

    assert commitments[first].current_bytes == 10 * GIB
    assert commitments[second].current_bytes == 30 * GIB
    assert {value.measurement_provenance for value in evidence} == {"nvml-process-exact"}


def test_known_failed_placement_blocks_only_the_dominated_resident_set() -> None:
    resident = ActiveResourceCommitment(
        "resident",
        20 * GIB,
        current_bytes=20 * GIB,
        running_peak_bytes=20 * GIB,
        future_peak_samples=(20 * GIB,),
        resource_state="RESOURCE_STABLE",
    )
    failed = PlacementOOMEvidence(
        "candidate",
        "compatible",
        "H100-80",
        ("resident",),
        60 * GIB,
        1 * GIB,
        79 * GIB,
        2 * GIB,
        58 * GIB,
        "sampled-physical",
        "known-failure",
    )
    candidate = CandidateResourceAction(
        "candidate-action",
        {"trial_index": 2, "resource_compatibility_key": "compatible"},
        1.0,
        prediction("candidate", 50 * GIB),
    )

    admitted, blocked = GPUPlacementPlanner(
        ResourceDemandModel(placement_failures=(failed,))
    ).place((candidate,), (gpu(free=60, active=(resident,), cap=5),), max_launches=1)

    assert not admitted
    assert blocked[0].devices[0]["reason"] == "known-failed-placement-dominates-current-condition"


def test_distinct_experiments_may_use_large_gpu_group_but_preserve_one_lane() -> None:
    existing_probe = ActiveResourceCommitment(
        "probe-a",
        20 * GIB,
        current_bytes=10 * GIB,
        running_peak_bytes=10 * GIB,
        future_peak_samples=(10 * GIB, 20 * GIB, 80 * GIB),
        resource_state="PLATEAU_UNCONFIRMED",
        admission_mode="EXPLORATORY_ADMISSION",
        exploration_signature="different-question",
    )
    stable = ActiveResourceCommitment(
        "resident",
        80 * GIB,
        current_bytes=10 * GIB,
        running_peak_bytes=10 * GIB,
        future_peak_samples=(10 * GIB, 20 * GIB, 80 * GIB),
        resource_state="PROVISIONALLY_STABLE",
        checkpoint_resumable=True,
    )
    candidate = CandidateResourceAction(
        "candidate-action",
        {"trial_index": 2, "resource_compatibility_key": "compatible"},
        1.0,
        ResourcePrediction(
            "candidate",
            10 * GIB,
            10 * GIB,
            80 * GIB,
            10 * GIB,
            None,
            100.0,
            0,
            0,
            "active-near-compatible",
            "poor",
            "test",
            (10 * GIB, 20 * GIB, 80 * GIB),
            is_provisional=True,
        ),
    )
    devices = (
        gpu(0, free=60, active=(stable, existing_probe), cap=5),
        gpu(1, free=70, active=(stable,), cap=5),
        gpu(2, free=70, active=(stable,), cap=5),
        gpu(3, free=70, active=(stable,), cap=5),
    )

    admitted, _ = GPUPlacementPlanner(ResourceDemandModel()).place(
        (candidate,), devices, max_launches=1
    )

    assert admitted
    assert admitted[0].target_gpu in {1, 2, 3}
    assert admitted[0].admission_mode == "EXPLORATORY_ADMISSION"


def test_nearby_censored_oom_widens_prediction_without_becoming_exact() -> None:
    exact = (
        observation("small-a", 10 * GIB, parameters={"width": 64}),
        observation("small-b", 12 * GIB, parameters={"width": 96}),
        observation(
            "oom-near",
            44 * GIB,
            state="oom",
            exact=False,
            lower=44 * GIB,
            parameters={"width": 128},
        ),
    )
    predicted = ResourceDemandModel(
        exact,
        parameter_schema={"width": {"range": [64, 256]}},
    ).predict(
        candidate_key="query",
        compatibility_key="compatible",
        parameters={"width": 144},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )

    assert predicted.upper_bytes > 30 * GIB
    assert predicted.known_lower_bound_bytes == 0
    assert predicted.exact_history_count == 0


def test_live_throughput_can_stop_a_counterproductive_ladder_increment() -> None:
    live = (
        ActiveResourceEvidence(
            "solo",
            "compatible",
            {},
            "H100-80",
            80 * GIB,
            10 * GIB,
            10 * GIB,
            (10 * GIB,),
            "RESOURCE_STABLE",
            10.0,
            throughput=100.0,
            co_runners=0,
        ),
        ActiveResourceEvidence(
            "packed-a",
            "compatible",
            {},
            "H100-80",
            80 * GIB,
            10 * GIB,
            10 * GIB,
            (10 * GIB,),
            "RESOURCE_STABLE",
            10.0,
            throughput=45.0,
            co_runners=1,
        ),
    )
    residents = (
        ActiveResourceCommitment("a", 10 * GIB, current_bytes=10 * GIB),
        ActiveResourceCommitment("b", 10 * GIB, current_bytes=10 * GIB),
    )
    candidate = CandidateResourceAction(
        "candidate-action",
        {"trial_index": 3, "resource_compatibility_key": "compatible"},
        1.0,
        prediction("candidate", 10 * GIB),
    )

    admitted, blocked = GPUPlacementPlanner(ResourceDemandModel(active_evidence=live)).place(
        (candidate,), (gpu(free=60, active=residents, cap=5),), max_launches=1
    )

    assert not admitted
    assert blocked[0].devices[0]["reason"] == "predicted-aggregate-throughput-would-not-improve"


def test_tiny_allocator_drift_contracts_hazard_but_true_ramp_does_not() -> None:
    phases = (
        "training",
        "train-forward",
        "backward",
        "optimizer-step",
        "training",
        "validation",
        "checkpoint",
    )
    light = tuple(
        ResourceTrajectorySample(
            index * 10.0,
            int(value * GIB),
            cuda_reserved_bytes=int(value * GIB),
            step=index,
            phase=phases[index],
        )
        for index, value in enumerate((8.0, 9.4, 9.9, 10.1, 10.15, 10.18, 10.22))
    )
    stable = ResourceTrajectoryAnalyzer.analyze(light, total_bytes=80 * GIB)

    ramp = tuple(
        ResourceTrajectorySample(index * 10.0, value * GIB, step=index, phase="training")
        for index, value in enumerate((10, 18, 29, 41, 55))
    )
    growing = ResourceTrajectoryAnalyzer.analyze(ramp, total_bytes=80 * GIB)

    assert stable.state == "PROVISIONALLY_STABLE"
    assert stable.growth_hazard < 0.1
    assert stable.residual_weights[-1] < 0.02
    assert growing.state == "RAMPING"
    assert growing.growth_hazard > 0.8


def test_real_live_telemetry_path_admits_second_run_before_completion(
    monkeypatch: Any,
) -> None:
    """Exercise telemetry -> analyzer -> evidence -> model -> planner, without hand states."""
    from lambdaforge.work.runner import _update_active_resource_commitments

    resident = object()
    specification = {
        "trial_parameters": {"width": 64},
        "definition": {"resources": {"gpu_memory_bytes": 8 * GIB}},
        "hpo_dispatched_monotonic": 0.0,
        "hpo_fidelity": {"current": 0, "target": 100, "maximum": 100},
    }
    pending = {resident: (specification, 0, object())}
    commitments = {resident: ActiveResourceCommitment("resident", 80 * GIB)}
    metadata = {
        resident: {
            "candidate_key": "resident",
            "compatibility_key": "compatible",
            "hardware": "H100-80",
        }
    }
    trajectories: dict[Any, BoundedResourceTrajectory] = {}
    current: dict[str, Any] = {}
    monkeypatch.setattr(
        "lambdaforge.work.runner._read_live_resource_snapshot", lambda _value: dict(current)
    )
    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_process_memory",
        lambda *_args: ({resident: int(current["bytes"])}, "nvml-process-exact"),
    )
    planner = GPUPlacementPlanner(ResourceDemandModel(), wait_regret=WaitRegretTracker())
    admitted = ()
    sequence = (
        (8.0, "training"),
        (9.4, "train-forward"),
        (9.9, "backward"),
        (10.1, "optimizer-step"),
        (10.15, "training"),
        (10.18, "validation"),
        (10.22, "checkpoint"),
    )
    for step, (amount, phase) in enumerate(sequence, start=1):
        current.update(
            {
                "bytes": int(amount * GIB),
                "cuda_reserved_bytes": int(amount * GIB),
                "cuda_allocated_bytes": int(amount * GIB),
                "phase": phase,
                "step": step,
            }
        )
        free = int((80.0 - amount) * GIB)
        model = ResourceDemandModel()
        live = _update_active_resource_commitments(
            ((free, 80 * GIB),),
            pending=pending,  # type: ignore[arg-type]
            commitments=commitments,
            trajectories=trajectories,
            metadata=metadata,
            model=model,
            now=float(step * 10),
        )
        model = ResourceDemandModel(
            active_evidence=live,
            parameter_schema={"width": {"values": [64, 96]}},
        )
        queued = model.predict(
            candidate_key="queued",
            compatibility_key="compatible",
            parameters={"width": 96},
            hardware="H100-80",
            total_bytes=80 * GIB,
            user_minimum_bytes=8 * GIB,
        )
        device = gpu(free=int(80.0 - amount), active=(commitments[resident],), cap=5)
        planner.update_model(model)
        admitted, _blocked = planner.place(
            (
                CandidateResourceAction(
                    "queued",
                    {"trial_index": 2, "resource_compatibility_key": "compatible"},
                    0.00001,
                    queued,
                ),
            ),
            (device,),
            max_launches=1,
            now=float(step * 10),
        )
        if admitted and commitments[resident].growth_hazard < 0.2:
            break

    assert admitted
    assert admitted[0].admission_mode == "EXPLORATORY_ADMISSION"
    assert step < 100
    assert commitments[resident].growth_hazard < 0.2


def test_wait_regret_eventually_beats_wait_without_overriding_hard_evidence() -> None:
    uncertain = ResourcePrediction(
        "candidate",
        10 * GIB,
        10 * GIB,
        80 * GIB,
        8 * GIB,
        None,
        None,
        0,
        0,
        "cold-start",
        "poor",
        "test",
        (10 * GIB, 80 * GIB),
        sample_weights=(0.7, 0.3),
    )
    queued = CandidateResourceAction(
        "candidate",
        {"trial_index": 1, "resource_compatibility_key": "compatible"},
        1e-12,
        uncertain,
    )
    resident = ActiveResourceCommitment(
        "resident",
        15 * GIB,
        current_bytes=15 * GIB,
        future_peak_samples=(15 * GIB, 30 * GIB),
        future_peak_weights=(0.8, 0.2),
        remaining_seconds=200.0,
        next_decision_seconds=20.0,
        elapsed_seconds=10_000.0,
        growth_hazard=0.2,
        resource_state="PLATEAU_UNCONFIRMED",
    )
    device = gpu(free=65, active=(resident,), cap=5)
    planner = GPUPlacementPlanner(ResourceDemandModel())

    first, _ = planner.place((queued,), (device,), max_launches=1, now=0.0)
    later, _ = planner.place((queued,), (device,), max_launches=1, now=4_000.0)

    assert not first
    assert later
    assert later[0].admission_mode == "EXPLORATORY_ADMISSION"

    impossible = CandidateResourceAction(
        "impossible",
        {"trial_index": 2},
        1.0,
        prediction("impossible", 90 * GIB, lower_bound=90 * GIB),
    )
    denied, _ = planner.place((impossible,), (device,), max_launches=1, now=1_000_000.0)
    assert not denied


def test_checkpoint_then_explore_is_selected_when_it_removes_dominant_rollback() -> None:
    uncertain = ResourcePrediction(
        "candidate",
        10 * GIB,
        10 * GIB,
        80 * GIB,
        8 * GIB,
        None,
        None,
        0,
        0,
        "cold-start",
        "poor",
        "test",
        (10 * GIB, 80 * GIB),
        sample_weights=(0.7, 0.3),
    )
    queued = CandidateResourceAction(
        "candidate",
        {"trial_index": 1, "resource_compatibility_key": "compatible"},
        0.1,
        uncertain,
    )
    resident = ActiveResourceCommitment(
        "resident",
        15 * GIB,
        current_bytes=15 * GIB,
        future_peak_samples=(15 * GIB, 30 * GIB),
        future_peak_weights=(0.8, 0.2),
        remaining_seconds=200.0,
        next_decision_seconds=20.0,
        elapsed_seconds=1_000.0,
        growth_hazard=0.2,
        resource_state="PLATEAU_UNCONFIRMED",
        checkpoint_request_path="/owned/checkpoint.request",
        checkpoint_duration_seconds=2.0,
    )
    planner = GPUPlacementPlanner(ResourceDemandModel())

    admitted, _ = planner.place(
        (queued,), (gpu(free=65, active=(resident,), cap=5),), max_launches=1, now=0.0
    )

    assert not admitted
    evaluation = planner.last_exploration_evaluations[0]
    assert evaluation.plan == "CHECKPOINT_THEN_EXPLORE"
    assert evaluation.rejection_reason == "CHECKPOINT_REQUESTED"
    assert evaluation.final_delta_value > 0


def test_scientifically_pruned_run_can_publish_phase_complete_resource_evidence(
    tmp_path: Path,
) -> None:
    from lambdaforge.work.runner import _resource_observation_for_result

    metrics = tmp_path / "training-metrics.jsonl"
    metrics.write_text(
        '{"name":"gpu_mem_mb","value":10240,"step":8}\n', encoding="utf-8"
    )
    trajectory = BoundedResourceTrajectory(
        tuple(
            ResourceTrajectorySample(
                float(index),
                10 * GIB,
                step=index,
                phase=phase,
            )
            for index, phase in enumerate(
                ("train-forward", "backward", "optimizer-step", "validation")
            )
        )
    )
    result = WorkResult(
        name="study",
        work_class="project.Train",
        execution_id="execution-1",
        run_id="run-1",
        attempt_id="attempt-1",
        attempt_number=1,
        scientific_fingerprint="sha256:test",
        status="succeeded",
        run_dir=tmp_path,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:01:00+00:00",
        duration_seconds=60.0,
        seed=1,
        trial={"index": 1, "parameters": {"width": 64}},
        parameters={"width": 64},
        inputs=(),
        requested_resources=WorkResources(1, 0, 1, 0, None, 0, 1),
        pruned=True,
        termination_type="pruned",
    )
    specification = {
        "trial_parameters": {"width": 64},
        "parameters": {"width": 64},
        "definition": {"work_class": "project.Train"},
    }

    observed = _resource_observation_for_result(
        specification,
        result,
        metadata={
            "candidate_key": "candidate",
            "compatibility_key": "compatible",
            "hardware": "H100-80",
            "total_bytes": 80 * GIB,
            "measurement_provenance": "nvml-process-exact",
            "trajectory": trajectory,
        },
        memory_failure=False,
    )

    assert observed.state == "pruned"
    assert observed.scientific_termination == "pruned"
    assert observed.resource_profile_quality == "PHASE_COMPLETE"
    assert observed.peak_is_exact is True
    predicted = ResourceDemandModel((observed,)).predict(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )
    assert predicted.exact_history_count == 1


def test_many_phase_complete_pruned_runs_end_permanent_cold_start() -> None:
    history = tuple(
        observation(
            f"pruned-{index}",
            (10 + index) * GIB,
            state="pruned",
            exact=True,
            parameters={"width": 64 + index * 16},
        )
        for index in range(6)
    )
    predicted = ResourceDemandModel(
        history,
        parameter_schema={"width": {"range": [64, 256]}},
    ).predict(
        candidate_key="new-candidate",
        compatibility_key="compatible",
        parameters={"width": 120},
        hardware="H100-80",
        total_bytes=80 * GIB,
    )

    assert predicted.support == "near-compatible"
    assert predicted.compatible_history_count == 6
    assert predicted.commitment_bytes < 80 * GIB


def test_why_wait_history_ignores_poll_noise_but_records_decision_changes(
    tmp_path: Path,
) -> None:
    store = ResourceHistoryStore(tmp_path)

    def waiting(*, regret: float, reason: str = "WAIT_CURRENTLY_BETTER") -> ExplorationEvaluation:
        return ExplorationEvaluation(
            candidate="candidate",
            gpu=0,
            physical_free_bytes=60 * GIB,
            provisional_headroom_bytes=55 * GIB,
            resource_states=("PLATEAU_UNCONFIRMED",),
            peak_hazard=0.22,
            fit_probability=0.74,
            scientific_value=0.81,
            normalized_scientific_value=1.0,
            scientific_value_rate=0.01,
            resource_information_value=0.2,
            rollback_seconds=12.0,
            rollback_value=0.12,
            interference_cost=0.01,
            wait_regret=regret,
            final_delta_value=-0.02,
            accepted=False,
            rejection_reason=reason,
            plan="WAIT",
        )

    store.record_exploration_evaluations((waiting(regret=0.01),))
    store.record_exploration_evaluations((waiting(regret=0.02),))

    evaluations = (tmp_path / "exploration-evaluations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    wait_events = (tmp_path / "resource-events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(evaluations) == 1
    assert len(wait_events) == 1

    store.record_exploration_evaluations(
        (waiting(regret=0.02, reason="PROTECTED_LANE"),)
    )
    assert len(
        (tmp_path / "exploration-evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 2
    assert len(
        (tmp_path / "resource-events.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 2
