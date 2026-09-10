"""Regression contract for the synthetic resource scheduling benchmark."""

from benchmarks.resource_scheduling import compare_static_and_dynamic


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
