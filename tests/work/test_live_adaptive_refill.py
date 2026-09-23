"""Regressions for idle scientific queues and generic-Work resource progress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.work import WorkConfig, WorkRunner, runner
from lambdaforge.work.runtime import MetricCollection


def test_physical_refill_consumes_deferred_anchors_after_batched_completions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Checked(RuntimeError):
        pass

    def dispatch(scheduled: Any, **kwargs: Any) -> None:
        assert len(scheduled) == 4
        pending = scheduled[2:]
        queued: Any = []
        for specification in scheduled[:2]:
            result = runner._execute_run(specification)
            queued = kwargs["on_result"](result, queued, pending)
        assert len(queued) == 1
        pending = [*pending, *queued]
        refill = kwargs["on_resource_blocked"]([], pending)
        assert len(refill) == 1
        assert refill[0]["hpo_scheduler_action"] == "STARTUP_ANCHOR"
        assert refill[0]["seed"] == 1
        assert refill[0]["trial_index"] not in {value["trial_index"] for value in pending}
        # The exact Run is registered once, even if the same readiness is polled again.
        again = kwargs["on_resource_blocked"]([], [*pending, *refill])
        assert not again
        raise Checked("idle GPU received the waiting protected anchor")

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", dispatch)
    config = WorkConfig.from_mapping(
        {
            "name": "refill",
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "seeds": [1, 2, 3],
            "search": {
                "trials": 6,
                "startup_trials": 6,
                "max_parallel": 4,
                "min_seeds": 2,
                "confirmation_seeds": [],
                "early_stopping": False,
                "choice": {"values": list(range(6))},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 4},
        },
        source=tmp_path / "study.yaml",
    )
    with pytest.raises(Checked, match="waiting protected anchor"):
        WorkRunner().run(config)


def test_physical_refill_can_collect_authored_initial_seeds_before_a_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Checked(RuntimeError):
        pass

    def dispatch(scheduled: Any, **kwargs: Any) -> None:
        assert len(scheduled) == 2
        assert {value["seed"] for value in scheduled} == {1}
        pending = list(scheduled)
        for _ in range(2):
            refill = kwargs["on_resource_blocked"]([], pending)
            assert len(refill) == 1
            assert refill[0]["seed"] == 2
            assert (refill[0]["trial_index"], 2) not in {
                (value["trial_index"], value["seed"]) for value in pending
            }
            pending.extend(refill)
        assert not kwargs["on_resource_blocked"]([], pending)
        raise Checked("initial seeds fill spare capacity without waiting for completion")

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", dispatch)
    config = WorkConfig.from_mapping(
        {
            "name": "seed-refill",
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "seeds": [1, 2, 3],
            "search": {
                "trials": 2,
                "startup_trials": 2,
                "max_parallel": 4,
                "min_seeds": 2,
                "confirmation_seeds": [],
                "early_stopping": False,
                "choice": {"values": [0, 1]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 4},
        },
        source=tmp_path / "seeds.yaml",
    )
    with pytest.raises(Checked, match="initial seeds fill spare capacity"):
        WorkRunner().run(config)


def test_generic_metric_steps_reach_resource_admission_without_history_scan(tmp_path: Path) -> None:
    metrics = MetricCollection(tmp_path)
    manifest = tmp_path / "manifest.json"
    heartbeat = tmp_path / "resource-heartbeat.json"
    manifest.write_text(
        json.dumps({"run_dir": str(tmp_path), "resource_heartbeat": str(heartbeat)})
    )
    heartbeat.write_text(json.dumps({"phase": "startup"}))
    specification = {"hpo_checkpoint_manifest_path": str(manifest)}
    metrics.log("loss", 0.5, step=20)
    progress = tmp_path / "metric-progress.json"
    modified = progress.stat().st_mtime_ns
    for step in (20, 19, None):
        metrics.log("accuracy", 0.6, step=step)
    assert progress.stat().st_mtime_ns == modified
    # The append-only scientific history need not be present for the bounded read model.
    (tmp_path / "metrics.jsonl").unlink()
    snapshot = runner._read_live_resource_snapshot(specification)
    assert snapshot["step"] == 20
    assert snapshot["phase"] == "work-progress"
    heartbeat.write_text(json.dumps({"phase": "validation", "step": 21}))
    snapshot = runner._read_live_resource_snapshot(specification)
    assert snapshot["step"] == 21
    assert snapshot["phase"] == "validation"
    progress.write_text("[]")
    assert runner._read_live_resource_snapshot(specification)["phase"] == "validation"


def test_adaptive_child_caps_inherited_aggregate_thread_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(runner.multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(
        ProcessGuard,
        "configure_cpu_thread_limits",
        lambda self, **kwargs: seen.append(kwargs),
    )
    monkeypatch.setattr(runner, "_execute_run_inner", lambda value: "result")
    monkeypatch.setenv("OMP_NUM_THREADS", "36")
    assert (
        runner._execute_run(
            {
                "hpo_metrics_path": "unused.jsonl",
                "definition": {"resources": {"cpu_cores": 4}},
            }
        )
        == "result"
    )
    assert seen == [{"torch_threads": 4, "interop_threads": 1, "override_env": True}]
    monkeypatch.setattr(runner.multiprocessing, "parent_process", lambda: None)
    runner._execute_run({"hpo_metrics_path": "unused.jsonl"})
    assert len(seen) == 1  # Embedded/controller execution must not change native thread pools.
