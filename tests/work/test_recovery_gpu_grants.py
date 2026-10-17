"""Resume against current grants with native evidence/ARI and CPU-only hardware fixtures."""

from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.work import WorkConfig, WorkRunner


@pytest.mark.parametrize("design", ["adaptive", "repeated", "sweep"])
def test_resume_three_gpu_study_uses_only_current_single_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, design: str
) -> None:
    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setenv("LAMBDAFORGE_GPU_ACCESS_MODE", "command")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    monkeypatch.setenv("LAMBDAFORGE_CACHE_ROOT", str(tmp_path / "cache"))
    for name in (
        "LAMBDAFORGE_GPU_ADMISSION_ROOT",
        "LAMBDAFORGE_GPU_VISIBILITY_COMMAND",
        "LAMBDAFORGE_STUDY_PATH",
        "LAMBDAFORGE_JOB_RESULT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    # The scalar consumer fixtures use no RNG/CUDA. Replace only hardware/process boundaries;
    # actual design, recovery, dispatch, placement and immutable Run/Attempt records stay native.
    monkeypatch.setattr(runner, "_seed", lambda seed: None)
    monkeypatch.setattr(runner, "_release_unused_cuda_cache", lambda: None)
    monkeypatch.setattr(runner, "_GPU_ADMISSION_POLL_SECONDS", 0.0)
    monkeypatch.setattr(runner, "_GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    monkeypatch.setattr(runner, "_GPU_RESOURCE_SAMPLE_SECONDS", 0.0)
    monkeypatch.setattr(
        runner, "_gpu_compute_activity", lambda tokens: {token: False for token in tokens}
    )
    monkeypatch.setattr(runner, "_gpu_process_memory", lambda *args: ({}, "unavailable"))
    queries: list[tuple[bool, int]] = []
    launches: list[tuple[bool, str, int, int, int | None]] = []
    dispatches: list[tuple[bool, tuple[str, ...], int]] = []
    recovered = False
    native_gpu_dispatch = runner._execute_gpu_admitted_runs

    def dispatch(*args: Any, **kwargs: Any) -> None:
        dispatches.append((recovered, tuple(kwargs["visible_gpus"]), kwargs["parallelism"]))
        native_gpu_dispatch(*args, **kwargs)

    monkeypatch.setattr(runner, "_execute_gpu_admitted_runs", dispatch)

    def inventory(count: int) -> tuple[tuple[int, int], ...]:
        assert count == (1 if recovered else 3)
        queries.append((recovered, count))
        return ((80 * 1024**3, 80 * 1024**3),) * count

    monkeypatch.setattr(runner, "_gpu_memory_inventory", inventory)
    monkeypatch.setattr(
        runner, "_gpu_hardware_labels", lambda count, memory: ("synthetic",) * count
    )

    class FinishedPool:
        def shutdown(self, **kwargs: Any) -> None:
            pass

    def launch(value: dict[str, Any], *, token: str, slot: int, resources: Any, **kwargs: Any):
        tokens = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert token == tokens[slot]
        assert resources.gpu_count == len(tokens)
        assert value["gpu_index"] == slot and value["gpu_slot"] == token
        launches.append((recovered, token, slot, resources.gpu_count, value.get("seed")))
        future: Future[Any] = Future()
        future.set_result(runner._execute_run(value))
        return future, FinishedPool()

    monkeypatch.setattr(runner, "_launch_gpu_run", launch)
    raw: dict[str, Any] = {
        "name": "recover-current-grant",
        "run": "tests.work_cases.FixedRecoveryWork",
        "seeds": [4, 7, 54],
        "resources": {"cpu": 6, "gpu": 3},
        "objective": {"metric": "score", "mode": "max"},
        "execution": {"failure_retries": 0, "runs_per_gpu": 2, "max_parallel": 6},
    }
    if design == "adaptive":
        raw["run"] = "tests.work_cases.RecoverableStudyWork"
        raw["search"] = {
            "trials": 2,
            "startup_trials": 2,
            "runs_per_gpu": 2,
            "max_parallel": 6,
            "failure_retries": 0,
            "confirmation_seeds": [],
            "early_stopping": False,
            "choice": {"values": [0, 1]},
        }
    elif design == "sweep":
        raw["sweep"] = {"space": {"choice": [0, 1]}}
    config = WorkConfig.from_mapping(raw, source=tmp_path / "study.yaml")
    first = WorkRunner().run(config)
    assert first.status == "failed"
    completed = {run.run_id: run for run in first.runs if run.ok}
    failed = {run.run_id: run for run in first.runs if not run.ok}
    assert completed and failed
    retained = {
        run.run_dir: (run.run_dir / "result.json").read_bytes() for run in completed.values()
    }
    manifest_path = first.execution_dir / "execution.json"
    manifest = manifest_path.read_bytes()
    assert json.loads(manifest)["resources"]["gpu_count"] == 3
    decisions = first.execution_dir / "hpo-control" / "decisions.jsonl"
    prior_decisions = decisions.read_bytes() if decisions.exists() else None
    controller = first.execution_dir / "hpo-control" / "state.json"
    if design == "adaptive":
        before_control = json.loads(controller.read_text())
        spent_runs = len(before_control["runs"])
    else:
        from lambdaforge.work.recovery import fixed_inventory

        spent_runs = fixed_inventory(first.execution_dir)["spent_runs"]

    recovered = True
    # The remaining grant is NOT the old first token. Recovery must discover it afresh.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c")
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert second.execution_id == first.execution_id
    assert config.resources.gpu_count == 3 and manifest_path.read_bytes() == manifest
    assert all((path / "result.json").read_bytes() == data for path, data in retained.items())
    for run in second.runs:
        if run.run_id in completed:
            assert run.run_dir == completed[run.run_id].run_dir
        else:
            assert run.run_id in failed
            assert run.attempt_number == failed[run.run_id].attempt_number + 1
    resumed_launches = [item for item in launches if item[0]]
    assert resumed_launches and {item[1:4] for item in resumed_launches} == {("GPU-c", 0, 1)}
    assert len(resumed_launches) == len(failed)
    assert {count for is_resume, count in queries if is_resume} == {1}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c"
    assert {tokens for is_resume, tokens, _width in dispatches if is_resume} == {("GPU-c",)}
    assert all(width <= 2 for is_resume, _tokens, width in dispatches if is_resume)
    if design == "adaptive":
        after_control = json.loads(controller.read_text())
        assert after_control["public_trial_map"] == before_control["public_trial_map"]
        assert len(after_control["runs"]) == spent_runs + len(failed)
    else:
        assert fixed_inventory(first.execution_dir)["spent_runs"] == spent_runs + len(failed)
    if prior_decisions is not None:
        assert decisions.read_bytes().startswith(prior_decisions)


@pytest.mark.parametrize("mode", ["command", "scheduler", "exclusive", "shared", ""])
@pytest.mark.parametrize("disabled", ["", "-1", "none", "void", "nodevfiles"])
def test_explicitly_disabled_visibility_never_becomes_physical_indices(monkeypatch, mode, disabled):
    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setenv("LAMBDAFORGE_GPU_ACCESS_MODE", mode)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", disabled)
    with pytest.raises(RuntimeError, match="empty/disabled.*refusing to invent"):
        runner._visible_gpu_tokens(3)
    assert runner._visible_gpu_tokens(0) == ()
