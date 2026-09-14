"""ARI v3.1 invariants that are easy to regress during scheduler refinement."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    BoundedResourceTrajectory,
    CandidateResourceAction,
    GPUPlacementPlanner,
    GPUResourceState,
    ResourceDemandModel,
    ResourceHistoryStore,
    ResourcePhaseModel,
    ResourcePrediction,
    ResourceProfileObservation,
    ResourceTrajectoryAnalyzer,
    ResourceTrajectorySample,
    ResourceTrajectoryStatistics,
    WaitRegretTracker,
    _joint_fit_probability,
    _next_decision_horizon,
)

GIB = 1024**3


def _prediction(
    samples: tuple[int, ...] = (10 * GIB,),
    weights: tuple[float, ...] = (),
    *,
    duration: float | None = 100.0,
) -> ResourcePrediction:
    return ResourcePrediction(
        "candidate",
        samples[0],
        min(samples),
        max(samples),
        min(samples),
        None,
        duration,
        3,
        3,
        "test",
        "test",
        "test",
        samples,
        sample_weights=weights,
    )


def _action(key: str = "candidate") -> CandidateResourceAction:
    return CandidateResourceAction(
        key,
        {"trial_index": 1, "resource_compatibility_key": "compatible"},
        1.0,
        _prediction(),
    )


def _device(
    *, active: tuple[ActiveResourceCommitment, ...] = (), free: int = 60
) -> GPUResourceState:
    return GPUResourceState(0, "0", "H100-80", 80 * GIB, free * GIB, 0, active, 8)


def test_hazard_is_invariant_to_duplicate_polling() -> None:
    sparse = tuple(
        ResourceTrajectorySample(float(step * 20), 10 * GIB, step=step, phase="training")
        for step in range(12)
    )
    dense: list[ResourceTrajectorySample] = []
    for sample in sparse:
        dense.extend(
            ResourceTrajectorySample(
                sample.elapsed_seconds + offset,
                sample.physical_bytes,
                step=sample.step,
                phase=sample.phase,
            )
            for offset in range(10)
        )

    sparse_result = ResourceTrajectoryAnalyzer.analyze(sparse, total_bytes=80 * GIB)
    dense_result = ResourceTrajectoryAnalyzer.analyze(tuple(dense), total_bytes=80 * GIB)
    assert dense_result.evidence_cycles == sparse_result.evidence_cycles
    assert dense_result.growth_hazard == sparse_result.growth_hazard
    assert dense_result.residual_samples == sparse_result.residual_samples
    assert dense_result.residual_weights == sparse_result.residual_weights


def test_long_run_keeps_incremental_evidence_beyond_bounded_curve() -> None:
    trajectory = BoundedResourceTrajectory()
    for step in range(2_000):
        trajectory.append(
            ResourceTrajectorySample(float(step), 12 * GIB, step=step, phase="training")
        )
    analysis = ResourceTrajectoryAnalyzer.analyze(
        trajectory.values,
        total_bytes=80 * GIB,
        trajectory_statistics=trajectory.statistics,
    )
    assert len(trajectory.values) <= 192
    assert trajectory.statistics.progress_cycles_seen == 1_999
    assert analysis.evidence_cycles == 1_999
    assert analysis.growth_hazard <= 0.0051

    restored = ResourceTrajectoryStatistics.from_mapping(trajectory.statistics.to_dict())
    assert restored.progress_cycles_seen == 1_999
    assert ResourceTrajectoryAnalyzer.analyze(
        (), total_bytes=80 * GIB, trajectory_statistics=restored
    ).growth_hazard == analysis.growth_hazard


def test_monotonic_growth_is_not_confused_with_jitter() -> None:
    slow_ramp = tuple(
        ResourceTrajectorySample(
            float(step),
            int((10.0 + 0.05 * step) * GIB),
            cuda_allocated_bytes=int((10.0 + 0.05 * step) * GIB),
            cuda_reserved_bytes=int((10.0 + 0.05 * step) * GIB),
            step=step,
            phase="forward",
        )
        for step in range(8)
    )
    jitter = tuple(
        ResourceTrajectorySample(
            float(step), int(value * GIB), step=step, phase="forward"
        )
        for step, value in enumerate((10.00, 10.05, 10.00, 10.04, 10.01, 10.03))
    )
    same_step = tuple(
        ResourceTrajectorySample(
            float(index),
            value * GIB,
            cuda_allocated_bytes=value * GIB,
            cuda_reserved_bytes=value * GIB,
            step=0,
            phase="forward",
        )
        for index, value in enumerate((2, 4, 8, 14))
    )
    assert ResourceTrajectoryAnalyzer.analyze(slow_ramp, total_bytes=80 * GIB).state == "RAMPING"
    assert ResourceTrajectoryAnalyzer.analyze(jitter, total_bytes=80 * GIB).state != "RAMPING"
    assert ResourceTrajectoryAnalyzer.analyze(same_step, total_bytes=80 * GIB).state == "RAMPING"


def test_wait_regret_closes_segments_and_survives_restart(tmp_path: Path) -> None:
    tracker = WaitRegretTracker()
    device = _device(active=(ActiveResourceCommitment("resident", 20 * GIB),))
    assert tracker.update(device, (_action("a"),), {"a": 1.0}, now=0.0) == 0.0
    first = tracker.update(device, (_action("b"),), {"b": 1.0}, now=10.0)
    second = tracker.update(device, (_action("c"),), {"c": 1.0}, now=20.0)
    assert first > 0
    assert math.isclose(second, first * 2.0)

    store = ResourceHistoryStore(tmp_path)
    store.persist_wait_regret(tracker)
    restored = WaitRegretTracker(ResourceHistoryStore(tmp_path).load_wait_regret())
    assert restored.update(device, (_action("d"),), {"d": 1.0}, now=30.0) == second
    assert restored.update(device, (_action("e"),), {"e": 1.0}, now=40.0) > second


def test_rare_and_multiple_tails_remain_in_joint_fit_probability() -> None:
    rare = _prediction((10 * GIB, 80 * GIB), (0.995, 0.005))
    assert math.isclose(_joint_fit_probability(rare, _device(free=70)), 0.995)

    resident = ActiveResourceCommitment(
        "resident",
        10 * GIB,
        current_bytes=10 * GIB,
        future_peak_samples=(10 * GIB, 80 * GIB),
        future_peak_weights=(0.99, 0.01),
    )
    combined = _joint_fit_probability(
        _prediction(), _device(active=(resident, resident), free=60)
    )
    # Equal ensemble of independent residual tails (0.99²) and shared epistemic tail (0.99).
    assert math.isclose(combined, (0.99**2 + 0.99) / 2.0)


def test_checkpoint_horizon_and_per_resident_rollback_are_dimensionally_correct() -> None:
    cadence = ActiveResourceCommitment(
        "resident",
        10 * GIB,
        elapsed_seconds=250.0,
        checkpoint_elapsed_seconds=200.0,
        checkpoint_cadence_seconds=100.0,
    )
    assert _next_decision_horizon(
        _prediction(duration=None), _device(active=(cadence,))
    ) == 50.0

    residents = (
        ActiveResourceCommitment(
            "a",
            10 * GIB,
            current_bytes=10 * GIB,
            elapsed_seconds=100.0,
            checkpoint_elapsed_seconds=90.0,
            checkpoint_resumable=True,
            checkpoint_request_path="/not-created/a",
            checkpoint_duration_seconds=2.0,
        ),
        ActiveResourceCommitment(
            "b", 10 * GIB, current_bytes=10 * GIB, elapsed_seconds=50.0
        ),
        ActiveResourceCommitment(
            "c",
            10 * GIB,
            current_bytes=10 * GIB,
            elapsed_seconds=80.0,
            checkpoint_elapsed_seconds=60.0,
            checkpoint_resumable=True,
            checkpoint_request_path="/not-created/c",
            checkpoint_duration_seconds=2.0,
        ),
    )
    planner = GPUPlacementPlanner(ResourceDemandModel())
    device = _device(active=residents, free=50)
    candidate = CandidateResourceAction(
        "candidate",
        {"trial_index": 1, "resource_compatibility_key": "compatible"},
        1.0,
        _prediction(duration=10.0),
    )
    evaluation = planner._exploration_value(
        candidate, device, (device,), normalized_scientific_value=1.0, wait_regret=0.0
    )
    # A/C become durable at the new checkpoint; only B retains its 50 seconds of progress.
    # The new candidate's expected failed progress is accounted once in checkpoint_delta.
    assert evaluation.rollback_after_checkpoint_seconds == 50.0
    assert evaluation.checkpoint_cost_seconds == 2.0


def test_checkpoint_rollback_handles_all_and_no_checkpointable_residents() -> None:
    planner = GPUPlacementPlanner(ResourceDemandModel())
    candidate = CandidateResourceAction(
        "candidate",
        {"trial_index": 1, "resource_compatibility_key": "compatible"},
        1.0,
        _prediction(duration=10.0),
    )
    all_checkpointable = tuple(
        ActiveResourceCommitment(
            name,
            10 * GIB,
            current_bytes=10 * GIB,
            elapsed_seconds=elapsed,
            checkpoint_elapsed_seconds=elapsed - 10.0,
            checkpoint_resumable=True,
            checkpoint_request_path=f"/not-created/{name}",
            checkpoint_duration_seconds=2.0,
        )
        for name, elapsed in (("a", 100.0), ("b", 70.0))
    )
    all_result = planner._exploration_value(
        candidate,
        _device(active=all_checkpointable, free=55),
        (_device(active=all_checkpointable, free=55),),
        normalized_scientific_value=1.0,
        wait_regret=0.0,
    )
    assert all_result.rollback_after_checkpoint_seconds == 0.0

    none_checkpointable = (
        ActiveResourceCommitment(
            "a", 10 * GIB, current_bytes=10 * GIB, elapsed_seconds=100.0
        ),
        ActiveResourceCommitment(
            "b", 10 * GIB, current_bytes=10 * GIB, elapsed_seconds=70.0
        ),
    )
    none_result = planner._exploration_value(
        candidate,
        _device(active=none_checkpointable, free=55),
        (_device(active=none_checkpointable, free=55),),
        normalized_scientific_value=1.0,
        wait_regret=0.0,
    )
    assert none_result.rollback_after_checkpoint_seconds == 170.0
    assert none_result.checkpoint_cost_seconds is None


def test_dynamic_frontier_does_not_prevent_wait_regret_liveness() -> None:
    uncertain = _prediction((10 * GIB, 80 * GIB), (0.7, 0.3), duration=100.0)
    planner = GPUPlacementPlanner(ResourceDemandModel())
    admitted = ()
    for cycle in range(12):
        action = CandidateResourceAction(
            f"candidate-{cycle}",
            {"trial_index": cycle, "resource_compatibility_key": "compatible"},
            1e-9,
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
            elapsed_seconds=10_000.0 + cycle * 100.0,
            checkpoint_step=cycle,
            checkpoint_elapsed_seconds=9_990.0 + cycle * 100.0,
            growth_hazard=max(0.02, 0.2 - cycle * 0.01),
            resource_state="PLATEAU_UNCONFIRMED",
        )
        device = _device(active=(resident,), free=65)
        admitted, _ = planner.place(
            (action,), (device,), max_launches=1, now=float(cycle * 2_000)
        )
        if admitted:
            break
    assert admitted
    assert admitted[0].admission_mode == "EXPLORATORY_ADMISSION"


def test_active_snapshot_is_versioned_and_legacy_snapshots_remain_readable(
    tmp_path: Path,
) -> None:
    store = ResourceHistoryStore(tmp_path)
    active = ActiveResourceEvidence(
        "candidate",
        "compatible",
        {},
        "H100-80",
        80 * GIB,
        10 * GIB,
        10 * GIB,
        (10 * GIB,),
        "PROVISIONALLY_STABLE",
        20.0,
    )
    store.persist_active((active,))
    payload = (tmp_path / "active-evidence.json").read_text(encoding="utf-8")
    assert '"active_evidence_version": 2' in payload
    assert store.load_active()[0].candidate_key == "candidate"

    (tmp_path / "active-evidence.json").write_text(
        "[" + json.dumps(active.to_dict()) + "]", encoding="utf-8"
    )
    assert ResourceHistoryStore(tmp_path).load_active()[0].candidate_key == "candidate"


def test_phase_model_preserves_a_known_late_peak_without_universal_validation() -> None:
    history = tuple(
        ResourceProfileObservation(
            candidate_key=f"c{index}",
            compatibility_key="compatible",
            work_class="project.Work",
            parameters={},
            fixed_arguments={},
            fidelity={},
            hardware="H100-80",
            total_bytes=80 * GIB,
            state="completed",
            observed_peak_bytes=20 * GIB,
            peak_is_exact=True,
            duration_seconds=100.0,
            peak_phase="validation",
            phases_seen=("sampling", "validation"),
        )
        for index in range(3)
    )
    model = ResourcePhaseModel.from_observations(history)
    current = tuple(
        ResourceTrajectorySample(float(step), 10 * GIB, step=step, phase="sampling")
        for step in range(4)
    )
    incomplete = ResourceTrajectoryAnalyzer.analyze(
        current, total_bytes=80 * GIB, phase_model=model
    )
    generic = ResourceTrajectoryAnalyzer.analyze(current, total_bytes=80 * GIB)
    assert incomplete.state == "PLATEAU_UNCONFIRMED"
    assert incomplete.phase_hazards["validation"] > incomplete.phase_hazards["sampling"]
    assert generic.state == "PROVISIONALLY_STABLE"


def test_benign_historical_validation_is_represented_in_tail_not_required_phase() -> None:
    history = tuple(
        ResourceProfileObservation(
            candidate_key=f"c{index}",
            compatibility_key="compatible",
            work_class="project.Work",
            parameters={},
            fixed_arguments={},
            fidelity={},
            hardware="H100-80",
            total_bytes=80 * GIB,
            state="completed",
            observed_peak_bytes=20 * GIB,
            peak_is_exact=True,
            duration_seconds=100.0,
            peak_phase="forward",
            phases_seen=("forward", "validation"),
        )
        for index in range(3)
    )
    model = ResourcePhaseModel.from_observations(history)
    assert model.expected_phases == ("forward",)
    assert model.completeness(("forward",))[0]


def test_phase_statistics_capture_checkpoint_peaks_and_generic_training() -> None:
    samples = (
        ResourceTrajectorySample(0.0, 8 * GIB, step=0, phase="forward"),
        ResourceTrajectorySample(10.0, 9 * GIB, step=0, phase="backward"),
        ResourceTrajectorySample(20.0, 9 * GIB, step=1, phase="optimizer-step"),
        ResourceTrajectorySample(
            30.0, 18 * GIB, step=1, phase="checkpoint", checkpoint_step=1
        ),
        ResourceTrajectorySample(40.0, 10 * GIB, step=2, phase="forward"),
    )
    statistics_state = ResourceTrajectoryStatistics()
    for sample in samples:
        ResourceTrajectoryAnalyzer.update(
            statistics_state, sample, total_bytes=80 * GIB
        )
    assert statistics_state.phase_maxima["checkpoint"] == 18 * GIB
    assert statistics_state.last_material_peak_phase == "checkpoint"
    assert statistics_state.checkpoint_times == [30.0]
    assert statistics_state.distinct_steps_seen == 3
    assert statistics_state.optimizer_cycles_seen == 1

    no_validation = tuple(
        ResourceProfileObservation(
            candidate_key=f"train-{index}",
            compatibility_key="compatible",
            work_class="project.Train",
            parameters={},
            fixed_arguments={},
            fidelity={},
            hardware="H100-80",
            total_bytes=80 * GIB,
            state="completed",
            observed_peak_bytes=10 * GIB,
            peak_is_exact=True,
            duration_seconds=100.0,
            peak_phase="optimizer",
            phases_seen=("forward", "backward", "optimizer"),
        )
        for index in range(3)
    )
    phase_model = ResourcePhaseModel.from_observations(no_validation)
    assert "validation" not in phase_model.expected_phases
    assert phase_model.completeness(("forward", "backward", "optimizer"))[0]


def test_checkpoint_cadence_predicts_time_until_next_event() -> None:
    samples = (
        ResourceTrajectorySample(
            100.0, 10 * GIB, step=1, phase="checkpoint", checkpoint_step=1
        ),
        ResourceTrajectorySample(
            200.0, 10 * GIB, step=2, phase="checkpoint", checkpoint_step=2
        ),
        ResourceTrajectorySample(
            250.0, 10 * GIB, step=3, phase="forward", checkpoint_step=2
        ),
    )
    analysis = ResourceTrajectoryAnalyzer.analyze(samples, total_bytes=80 * GIB)
    assert analysis.checkpoint_cadence_seconds == 100.0
    assert analysis.next_decision_seconds == 50.0
    assert analysis.next_decision_uncertainty_seconds is None

    repeated = ResourceTrajectoryAnalyzer.analyze(
        (
            *samples[:2],
            ResourceTrajectorySample(
                300.0, 10 * GIB, step=3, phase="checkpoint", checkpoint_step=3
            ),
            ResourceTrajectorySample(
                350.0, 10 * GIB, step=4, phase="forward", checkpoint_step=3
            ),
        ),
        total_bytes=80 * GIB,
    )
    assert repeated.next_decision_seconds == 50.0
    assert repeated.next_decision_uncertainty_seconds == 0.0


def test_checkpoint_duration_is_learned_from_compatible_terminal_evidence() -> None:
    observations = tuple(
        ResourceProfileObservation(
            candidate_key=f"candidate-{index}",
            compatibility_key="compatible",
            work_class="project.Work",
            parameters={},
            fixed_arguments={},
            fidelity={},
            hardware="H100-80",
            total_bytes=80 * GIB,
            state="completed",
            observed_peak_bytes=10 * GIB,
            peak_is_exact=True,
            duration_seconds=100.0,
            checkpoint_duration_seconds=duration,
        )
        for index, duration in enumerate((2.0, 4.0, 9.0))
    )
    model = ResourceDemandModel(observations)

    assert model.checkpoint_duration("compatible", "H100-80") == 4.0
    assert model.checkpoint_duration("other", "H100-80") is None
