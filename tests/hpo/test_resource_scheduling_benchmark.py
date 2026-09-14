"""Regression contract for the synthetic resource scheduling benchmark."""

import json
from pathlib import Path

from benchmarks.resource_scheduling import (
    compare_static_and_dynamic,
    compare_terminal_and_live_cold_start,
    replay_execution,
)


def test_dynamic_resource_planner_improves_useful_throughput_without_ooms() -> None:
    comparison = compare_static_and_dynamic()
    fixed = comparison["fixed_one_run_per_gpu"]
    dynamic = comparison["adaptive_resource_planner"]

    assert dynamic.scientific_actions_per_hour > fixed.scientific_actions_per_hour
    assert dynamic.vram_time_utilization > fixed.vram_time_utilization
    assert dynamic.makespan_seconds < fixed.makespan_seconds
    assert dynamic.oom_count == dynamic.avoidable_oom_count == 0
    assert dynamic.wasted_gpu_seconds == 0
    assert dynamic.heavy_candidate_starvation_seconds <= fixed.heavy_candidate_starvation_seconds


def test_live_resource_evidence_materially_shortens_safe_cold_start() -> None:
    comparison = compare_terminal_and_live_cold_start()
    old = comparison["adaptive_resource_v2"]
    new = comparison["adaptive_resource_v3"]
    assert new.time_to_first_two_way_packing_seconds < old.time_to_first_two_way_packing_seconds
    assert new.idle_vram_gib_seconds < old.idle_vram_gib_seconds
    assert new.protected_progress_lanes == 1
    assert new.exploratory_oom_count == 0
    assert new.mean_active_runs_per_gpu > old.mean_active_runs_per_gpu
    assert new.scientific_value_per_hour > old.scientific_value_per_hour
    assert new.useful_actions_per_hour > old.useful_actions_per_hour
    assert new.aggregate_throughput > old.aggregate_throughput
    assert new.heavy_candidate_starvation_seconds < old.heavy_candidate_starvation_seconds
    assert new.redundant_oom_count == new.false_safe_admissions == 0


def test_trace_benchmark_derives_metrics_from_recorded_events(tmp_path: Path) -> None:
    resource_root = tmp_path / "hpo-control" / "resources"
    resource_root.mkdir(parents=True)
    events = (
        {
            "trace_version": 1,
            "elapsed_seconds": 0.0,
            "devices": [
                {
                    "gpu": 0,
                    "total_bytes": 80 * 1024**3,
                    "physical_free_bytes": 70 * 1024**3,
                    "active_runs": 1,
                    "active_candidates": ["resident"],
                }
            ],
            "frontier": [{"candidate": "next", "value_contract": {"normalized_value": 1.0}}],
            "decisions": [{"candidate_key": "next", "state": "RESOURCE_BLOCKED"}],
        },
        {
            "trace_version": 1,
            "elapsed_seconds": 30.0,
            "devices": [
                {
                    "gpu": 0,
                    "total_bytes": 80 * 1024**3,
                    "physical_free_bytes": 60 * 1024**3,
                    "active_runs": 2,
                    "active_candidates": ["resident", "next"],
                }
            ],
            "frontier": [{"candidate": "next", "value_contract": {"normalized_value": 1.0}}],
            "decisions": [
                {
                    "candidate_key": "next",
                    "target_gpu": 0,
                    "state": "ADMITTED",
                    "admission_mode": "EXPLORATORY_ADMISSION",
                }
            ],
        },
    )
    (resource_root / "scheduler-trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    comparison = replay_execution(tmp_path)

    assert comparison["recorded"]["metrics"]["time_to_concurrency"]["0"]["2"] == 30.0
    assert comparison["ari-v2-compat"]["counterfactual_divergence"] is not None
    assert comparison["ari-v3.1"]["segments"]["post_divergence_is_observed"] is False
