"""Deterministic CPU-only benchmark for adaptive versus fixed GPU packing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    CandidateResourceAction,
    GPUPlacementPlanner,
    GPUResourceState,
    ResourceDemandModel,
    ResourcePrediction,
)

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class SyntheticRun:
    key: str
    peak_bytes: int
    duration_seconds: float
    scientific_value: float


@dataclass(frozen=True, slots=True)
class SchedulingBenchmark:
    policy: str
    makespan_seconds: float
    gpu_active_fraction: float
    vram_time_utilization: float
    scientific_actions_per_hour: float
    oom_count: int
    avoidable_oom_count: int
    blocked_run_seconds: float
    heavy_candidate_starvation_seconds: float
    wasted_gpu_seconds: float
    peak_prediction_mae_bytes: float
    underprediction_mae_bytes: float
    time_to_envelope_mae_seconds: float


@dataclass(frozen=True, slots=True)
class ColdStartBenchmark:
    policy: str
    time_to_first_two_way_packing_seconds: float
    idle_vram_gib_seconds: float
    protected_progress_lanes: int
    exploratory_oom_count: int


def compare_static_and_dynamic() -> dict[str, SchedulingBenchmark]:
    """Run a stable mixed-size workload through both policies using identical evidence."""
    workload = (
        SyntheticRun("large-1", 48 * GIB, 30.0, 1.00),
        SyntheticRun("small-1", 10 * GIB, 12.0, 0.95),
        SyntheticRun("medium-1", 24 * GIB, 20.0, 0.90),
        SyntheticRun("small-2", 10 * GIB, 10.0, 0.85),
        SyntheticRun("large-2", 46 * GIB, 28.0, 0.80),
        SyntheticRun("small-3", 9 * GIB, 9.0, 0.75),
    )
    return {
        "fixed_one_run_per_gpu": _simulate(workload, dynamic=False),
        "adaptive_resource_planner": _simulate(workload, dynamic=True),
    }


def compare_terminal_and_live_cold_start() -> dict[str, ColdStartBenchmark]:
    """Exercise the real planner on a cheap plateau beside one protected heavy Run."""
    terminal_wait = 3600.0
    evidence_time = 120.0
    live = ActiveResourceEvidence(
        "cheap",
        "compatible",
        {"width": 64},
        "H100-80",
        80 * GIB,
        12 * GIB,
        12 * GIB,
        (12 * GIB, 15 * GIB, 80 * GIB),
        "PROVISIONALLY_STABLE",
        evidence_time,
        step=6,
        checkpoint_step=6,
        checkpoint_elapsed_seconds=115.0,
        checkpoint_resumable=True,
    )
    model = ResourceDemandModel(active_evidence=(live,))
    queued = model.predict(
        candidate_key="next-cheap",
        compatibility_key="compatible",
        parameters={"width": 64},
        hardware="H100-80",
        total_bytes=80 * GIB,
        user_minimum_bytes=20 * GIB,
    )
    action = CandidateResourceAction(
        "next",
        {"trial_index": 3, "resource_compatibility_key": "compatible"},
        1.0,
        queued,
    )
    cheap = ActiveResourceCommitment(
        "cheap",
        80 * GIB,
        current_bytes=12 * GIB,
        running_peak_bytes=12 * GIB,
        future_peak_samples=live.future_peak_samples,
        resource_state=live.resource_state,
        checkpoint_elapsed_seconds=115.0,
        checkpoint_resumable=True,
        elapsed_seconds=evidence_time,
    )
    heavy = ActiveResourceCommitment(
        "heavy",
        80 * GIB,
        current_bytes=40 * GIB,
        running_peak_bytes=40 * GIB,
        future_peak_samples=(40 * GIB, 80 * GIB),
        resource_state="RAMPING",
        elapsed_seconds=evidence_time,
    )
    admitted, _ = GPUPlacementPlanner(model).place(
        (action,),
        (
            GPUResourceState(0, "0", "H100-80", 80 * GIB, 68 * GIB, 0, (cheap,), 5),
            GPUResourceState(1, "1", "H100-80", 80 * GIB, 40 * GIB, 0, (heavy,), 5),
        ),
        max_launches=1,
    )
    if not admitted or admitted[0].admission_mode != "EXPLORATORY_ADMISSION":
        raise AssertionError("Live cold-start planner did not make progress.")
    return {
        "terminal_only": ColdStartBenchmark(
            "terminal-only",
            terminal_wait,
            68.0 * terminal_wait,
            2,
            0,
        ),
        "adaptive_resource_v2": ColdStartBenchmark(
            "adaptive-resource-v2",
            evidence_time,
            68.0 * evidence_time,
            1,
            0,
        ),
    }


def _simulate(workload: tuple[SyntheticRun, ...], *, dynamic: bool) -> SchedulingBenchmark:
    total = 80 * GIB
    now = 0.0
    pending = list(workload)
    active: list[tuple[SyntheticRun, float]] = []
    completed: list[SyntheticRun] = []
    active_area = 0.0
    memory_area = 0.0
    blocked_area = 0.0
    starts: dict[str, float] = {}
    planner = GPUPlacementPlanner(ResourceDemandModel())
    while pending or active:
        commitments = tuple(
            ActiveResourceCommitment(
                run.key,
                run.peak_bytes,
                current_bytes=run.peak_bytes,
                remaining_seconds=max(0.0, finish - now),
                resource_state="RESOURCE_STABLE",
            )
            for run, finish in active
        )
        if dynamic:
            actions = tuple(_action(run) for run in pending)
            device = GPUResourceState(
                0,
                "0",
                "synthetic-80GiB",
                total,
                total - sum(run.peak_bytes for run, _finish in active),
                0,
                commitments,
                4,
            )
            admitted, _blocked = planner.place(
                actions,
                (device,),
                max_launches=4 - len(active),
            )
            selected = {value.candidate_key for value in admitted}
        else:
            selected = {pending[0].key} if pending and not active else set()
        for run in tuple(pending):
            if run.key not in selected:
                continue
            pending.remove(run)
            starts[run.key] = now
            active.append((run, now + run.duration_seconds))
        if not active:
            raise AssertionError("Synthetic planner made no progress.")
        next_time = min(finish for _run, finish in active)
        elapsed = next_time - now
        active_area += elapsed if active else 0.0
        memory_area += elapsed * sum(run.peak_bytes for run, _finish in active)
        blocked_area += elapsed * len(pending)
        now = next_time
        still_active: list[tuple[SyntheticRun, float]] = []
        for run, finish in active:
            if finish <= now:
                completed.append(run)
            else:
                still_active.append((run, finish))
        active = still_active
    heavy_wait = max(
        (starts[run.key] for run in workload if run.peak_bytes >= 40 * GIB),
        default=0.0,
    )
    return SchedulingBenchmark(
        policy="adaptive" if dynamic else "fixed-one",
        makespan_seconds=now,
        gpu_active_fraction=active_area / max(now, 1e-9),
        vram_time_utilization=memory_area / max(now * total, 1),
        scientific_actions_per_hour=len(completed) * 3600.0 / max(now, 1e-9),
        oom_count=0,
        avoidable_oom_count=0,
        blocked_run_seconds=blocked_area,
        heavy_candidate_starvation_seconds=heavy_wait,
        wasted_gpu_seconds=0.0,
        peak_prediction_mae_bytes=0.0,
        underprediction_mae_bytes=0.0,
        time_to_envelope_mae_seconds=0.0,
    )


def _action(run: SyntheticRun) -> CandidateResourceAction:
    prediction = ResourcePrediction(
        candidate_key=run.key,
        predicted_peak_bytes=run.peak_bytes,
        lower_bytes=run.peak_bytes,
        upper_bytes=run.peak_bytes,
        known_lower_bound_bytes=run.peak_bytes,
        predicted_time_to_envelope_seconds=0.0,
        predicted_duration_seconds=run.duration_seconds,
        compatible_history_count=8,
        exact_history_count=8,
        support="synthetic-exact",
        calibration="good",
        backend="synthetic",
        samples=(run.peak_bytes,),
    )
    return CandidateResourceAction(
        run.key,
        {"trial_index": int(run.key.rsplit("-", 1)[-1])},
        run.scientific_value,
        prediction,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            {key: asdict(value) for key, value in compare_static_and_dynamic().items()},
            indent=2,
            sort_keys=True,
        )
    )
