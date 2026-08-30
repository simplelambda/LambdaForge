"""Focused adaptive study behavior without requiring a GPU or external provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import WorkResources, WorkResult
from lambdaforge.work.runner import _candidate_score, _gpu_worker_capacities, _request_early_stops


def test_adaptive_search_allocates_more_seeds_only_to_promising_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path = tmp_path / "live-study"
    monkeypatch.setenv("LAMBDAFORGE_STUDY_PATH", str(study_path))
    monkeypatch.setenv("LAMBDAFORGE_PROGRESS_PATH", str(tmp_path / "progress.json"))
    source = tmp_path / "study.yaml"
    config = WorkConfig.from_mapping(
        {
            "name": "adaptive-study",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1, 2, 3, 4],
            "search": {
                "strategy": "adaptive",
                "trials": 3,
                "max_parallel": 2,
                "min_seeds": 1,
                "reduction_factor": 2,
                "quality": {"values": [1.0, 2.0, 3.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=source,
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert len(result.runs) == 7
    assert result.summary["planned_runs"] == 12
    assert result.summary["completed_runs"] == 7
    assert result.summary["best"]["parameters"] == {"quality": 3.0}
    seeds_by_quality: dict[float, set[int | None]] = {}
    for run in result.runs:
        quality = float(run.trial["parameters"]["quality"])
        seeds_by_quality.setdefault(quality, set()).add(run.seed)
    assert seeds_by_quality[3.0] == {1, 2, 3, 4}
    assert len(seeds_by_quality[1.0]) == 1
    telemetry = json.loads((study_path / "summary.json").read_text(encoding="utf-8"))
    assert telemetry["strategy"] == "adaptive"
    assert telemetry["counts"]["scheduled_runs"] == 7
    assert telemetry["counts"]["completed_runs"] == 7
    assert len(telemetry["candidates"]) == 3
    observed_runs = [
        run for candidate in telemetry["candidates"] for run in candidate["runs"]
    ]
    assert all(run["log_path"] for run in observed_runs)


def test_gpu_packing_is_explicit_and_requires_a_per_run_memory_bound(tmp_path: Path) -> None:
    base = {
        "name": "gpu-study",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": [1, 2],
        "search": {
            "strategy": "adaptive",
            "runs_per_gpu": 4,
            "quality": {"values": [1.0, 2.0]},
        },
        "objective": {"metric": "score", "mode": "max"},
        "resources": {"gpu": 2},
    }
    try:
        WorkConfig.from_mapping(base, source=tmp_path / "study.yaml")
    except ValueError as error:
        assert "resources.gpu_memory" in str(error)
    else:
        raise AssertionError("Unsafe GPU packing was accepted without gpu_memory.")

    configured = {**base, "resources": {"gpu": 2, "gpu_memory": "16GiB"}}
    config = WorkConfig.from_mapping(configured, source=tmp_path / "study.yaml")
    policy = config.levels[0].runs[0].search_policy
    assert policy is not None
    assert policy.runs_per_gpu == 4
    assert config.resources.gpu_count == 2
    assert _gpu_worker_capacities(2, 8, 4) == (4, 4)
    assert _gpu_worker_capacities(6, 12, 2) == (2, 2, 2, 2, 2, 2)
    assert _gpu_worker_capacities(2, 3, 4) == (2, 1)


def test_early_stopping_requests_only_the_current_bottom_fraction(tmp_path: Path) -> None:
    specifications = []
    for index, value in enumerate((0.9, 0.8, 0.2, 0.1), 1):
        metrics = tmp_path / f"trial-{index}.jsonl"
        metrics.write_text(
            json.dumps({"name": "score", "value": value, "step": 5, "split": None}) + "\n",
            encoding="utf-8",
        )
        specifications.append(
            {"hpo_metrics_path": metrics, "hpo_stop_path": tmp_path / f"trial-{index}.stop"}
        )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        reduction_factor=2,
    )

    assert not Path(specifications[0]["hpo_stop_path"]).exists()
    assert not Path(specifications[1]["hpo_stop_path"]).exists()
    assert Path(specifications[2]["hpo_stop_path"]).is_file()
    assert Path(specifications[3]["hpo_stop_path"]).is_file()


def test_pruned_runs_are_not_promoted_by_a_partial_objective(tmp_path: Path) -> None:
    pruned = WorkResult(
        name="work",
        work_class="tests.work_cases.AdaptiveScoreWork",
        execution_id="execution-1",
        run_id="run-1",
        attempt_id="attempt-1",
        attempt_number=1,
        scientific_fingerprint="sha256:test",
        status="succeeded",
        run_dir=tmp_path,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        seed=1,
        trial={"index": 1, "parameters": {}},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 0, 0, None, 0, 1),
        metrics={"score": 0.99},
        pruned=True,
        prune_reason="adaptive-early-stopping",
    )

    assert _candidate_score((pruned,), "score", "max", 1.0) is None
