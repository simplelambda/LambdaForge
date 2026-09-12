"""Regression contract for the synthetic resource scheduling benchmark."""

from benchmarks.resource_scheduling import (
    compare_static_and_dynamic,
    compare_terminal_and_live_cold_start,
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
    old = comparison["terminal_only"]
    new = comparison["adaptive_resource_v2"]
    assert new.time_to_first_two_way_packing_seconds < old.time_to_first_two_way_packing_seconds
    assert new.idle_vram_gib_seconds < old.idle_vram_gib_seconds
    assert new.protected_progress_lanes == 1
    assert new.exploratory_oom_count == 0
