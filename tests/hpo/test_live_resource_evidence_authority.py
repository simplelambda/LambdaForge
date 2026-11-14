"""One physical trajectory is evidence once, not once per recursive forecast/poll."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    BoundedResourceTrajectory,
    CandidateResourceAction,
    GPUPlacementPlanner,
    GPUResourceState,
    ResourceDemandModel,
)
from lambdaforge.work import runner

GIB = 1024**3


def live(key: str = "resident", *, peak: int = 18 * GIB) -> ActiveResourceEvidence:
    return ActiveResourceEvidence(
        candidate_key=key,
        compatibility_key="work",
        parameters={"coefficient": 0.04},
        hardware="H200",
        total_bytes=140 * GIB,
        current_bytes=peak,
        running_peak_bytes=peak,
        future_peak_samples=(peak, 20 * GIB, 140 * GIB),
        future_peak_weights=(0.95, 0.04, 0.01),
        resource_state="PROVISIONALLY_STABLE",
        elapsed_seconds=3000.0,
        phase="work-progress",
        step=10,
        measurement_provenance="nvml-process-exact",
    )


def forecast(model: ResourceDemandModel, key: str = "next", floor: int = 0):
    return model.predict(
        candidate_key=key,
        compatibility_key="work",
        parameters={"coefficient": 0.04},
        hardware="H200",
        total_bytes=140 * GIB,
        user_minimum_bytes=floor,
    )


@pytest.mark.parametrize("state", ["STARTING", "PLATEAU_UNCONFIRMED"])
def test_zero_cuda_startup_is_not_a_gpu_demand_observation(state):
    resident = live()
    startup = replace(live("initializing", peak=0), resource_state=state)
    baseline = forecast(ResourceDemandModel(active_evidence=(resident,)))
    with_startup = forecast(ResourceDemandModel(active_evidence=(resident, startup)))
    assert with_startup == baseline
    assert baseline.predicted_peak_bytes == 18 * GIB
    assert baseline.known_lower_bound_bytes == 0  # not a hard neighbour bound
    assert baseline.upper_bytes == 140 * GIB  # unknown late allocations are not erased
    assert baseline.is_provisional and baseline.exact_history_count == 0


def test_zero_cuda_baseline_does_not_teach_a_zero_gpu_footprint():
    assert forecast(ResourceDemandModel(active_evidence=(live(peak=0),))) == forecast(
        ResourceDemandModel()
    )
    assert forecast(ResourceDemandModel(active_evidence=(live(),)), floor=25 * GIB).lower_bytes == (
        25 * GIB
    )


@pytest.mark.parametrize("feedback", [False, True])
def test_real_live_update_does_not_reingest_its_own_posterior(monkeypatch, feedback):
    run = object()
    spec: dict[str, Any] = {
        "trial_parameters": {"coefficient": 0.04},
        "definition": {"resources": {"gpu_memory_bytes": 0}},
        "hpo_dispatched_monotonic": 0.0,
    }
    step = {"value": 0}
    monkeypatch.setattr(
        runner,
        "_read_live_resource_snapshot",
        lambda _: {"step": step["value"], "phase": "work-progress"},
    )
    monkeypatch.setattr(
        runner, "_gpu_process_memory", lambda *_: ({run: 18 * GIB}, "nvml-process-exact")
    )
    commitments = {run: ActiveResourceCommitment("resident", 140 * GIB)}
    trajectories: dict[Any, BoundedResourceTrajectory] = {}
    metadata = {run: {"candidate_key": "resident", "compatibility_key": "work", "hardware": "H200"}}
    evidence = ()
    stable = None
    for poll in range(30):
        step["value"] = min(poll, 5)  # later polls contain no new allocation/progress
        evidence = runner._update_active_resource_commitments(
            ((122 * GIB, 140 * GIB),),
            pending={run: (spec, 0, object())},
            commitments=commitments,
            trajectories=trajectories,
            metadata=metadata,
            model=ResourceDemandModel(active_evidence=evidence if feedback else ()),
            now=float(poll * 10),
        )
        actual = evidence[0]
        if poll == 5:
            stable = actual
        if poll > 5:
            assert stable is not None
            assert actual.future_peak_samples == stable.future_peak_samples
            assert actual.future_peak_weights == stable.future_peak_weights
            assert actual.growth_hazard == stable.growth_hazard
    # Both feedback/no-feedback routes must produce this identical raw-trajectory posterior.
    assert actual.future_peak_samples == (18 * GIB, 140 * GIB)
    assert actual.future_peak_weights == pytest.approx((1 - 0.05 / 6, 0.05 / 6))
    assert actual.resource_state == "PROVISIONALLY_STABLE"
    assert actual.measurement_provenance == "nvml-process-exact"


def test_materialized_prefix_still_allows_bounded_exploration_without_terminal_results():
    model = ResourceDemandModel(active_evidence=(live(),))
    pred = forecast(model)
    resident = ActiveResourceCommitment(
        "resident",
        140 * GIB,
        current_bytes=18 * GIB,
        running_peak_bytes=18 * GIB,
        future_peak_samples=(18 * GIB, 140 * GIB),
        future_peak_weights=(0.99, 0.01),
        resource_state="PROVISIONALLY_STABLE",
        admission_mode="BASELINE_ADMISSION",
        growth_hazard=0.01,
    )
    devices = (
        GPUResourceState(0, "granted-1", "H200", 140 * GIB, 122 * GIB, 0, (resident,), 5),
        GPUResourceState(1, "granted-3", "H200", 140 * GIB, 122 * GIB, 0, (resident,), 5),
    )
    planner = GPUPlacementPlanner(model)
    action = CandidateResourceAction(
        "next", {"trial_index": 3, "resource_compatibility_key": "work"}, 1.0, pred
    )
    admitted, _ = planner.place((action,), devices, max_launches=1)
    assert len(admitted) == 1
    assert admitted[0].admission_mode == "EXPLORATORY_ADMISSION"
    # The arbitrary 140-GiB uncertainty endpoint isn't used as an intrinsic OOM bound;
    # the existing joint-fit/rollback/protected-lane policy remains the decision authority.
    probe = replace(resident, candidate_key="probe", admission_mode="EXPLORATORY_ADMISSION")
    busy = replace(devices[0], active=(resident, probe))
    admitted, _ = planner.place((action,), (busy, devices[1]), max_launches=1)
    assert not admitted
    assert {value.rejection_reason for value in planner.last_exploration_evaluations} == {
        "UNVALIDATED_LADDER_STEP",
        "PROTECTED_LANE",
    }


def test_admission_reports_unmaterialized_work_and_actual_planner_wait_reason():
    resident = ActiveResourceCommitment(
        "resident", 140 * GIB, current_bytes=18 * GIB, running_peak_bytes=18 * GIB
    )
    startup = ActiveResourceCommitment(
        "loading", 140 * GIB, elapsed_seconds=2700.0, admission_mode="EXPLORATORY_ADMISSION"
    )
    device = GPUResourceState(0, "granted", "H200", 140 * GIB, 122 * GIB, 0, (resident, startup), 5)
    planner = GPUPlacementPlanner(ResourceDemandModel(active_evidence=(live(),)))
    action = CandidateResourceAction("next", {"trial_index": 3}, 1.0, forecast(planner.model))
    admitted, blocked = planner.place((action,), (device,), max_launches=1)
    report = runner._resource_admission_diagnostics(
        (device,),
        admitted=admitted,
        blocked=blocked,
        pending=76,
        max_parallel=None,
        configured_runs_per_gpu=None,
        user_minimum_bytes=0,
        exploration_evaluations=planner.last_exploration_evaluations,
    )["devices"][0]
    assert report["active_runs"] == 2
    assert report["materialized_runs"] == 1
    assert report["unmaterialized_runs"] == 1
    assert report["capacity_wait_reason"] == "UNVALIDATED_LADDER_STEP"
    assert report["active"][1]["elapsed_seconds"] == 2700.0
    assert report["physical_free_bytes"] == 122 * GIB
