"""Fixed evidence recovery reuses owned Attempts; no adaptive state is invented."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import atomic_json
from lambdaforge.work.recovery import fixed_inventory, fixed_recovery_preview

SEEDS = [4, 7, 32, 54, 65, 94, 109, 124, 142, 167]


def config_at(root: Path, **overrides: Any) -> WorkConfig:
    return WorkConfig.from_mapping(
        {
            "name": "fixed",
            "run": "tests.work_cases.FixedRecoveryWork",
            "seeds": SEEDS,
            "resources": {"cpu": 1},
            "execution": {"failure_retries": 0},
            "objective": {"metric": "score", "mode": "max"},
            **overrides,
        },
        source=root / "fixed.yaml",
    )


@pytest.fixture
def inline_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Run lifecycle/files with deterministic in-process toy scientific execution."""
    runner = importlib.import_module("lambdaforge.work.runner")

    def dispatch(specifications: Any, **kwargs: Any) -> tuple[Any, ...]:
        queue = list(specifications)
        results = []
        while queue:
            value = queue.pop(0)
            result = runner._execute_run(value)
            results.append(result)
            queue = list(kwargs["on_result"](result, queue, ()))
        return tuple(results)

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", dispatch)


def test_repeated_recovery_reuses_nine_and_retries_only_seed_54(
    tmp_path: Path,
    inline_dispatch: None,
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    assert sum(run.ok for run in first.runs) == 9
    assert not (first.execution_dir / "hpo-control" / "state.json").exists()
    preview = fixed_recovery_preview(first.execution_dir)
    assert (preview["reuse_runs"], preview["retry_runs"], preview["pending_runs"]) == (9, 1, 0)
    assert [row["seed"] for row in preview["runs"] if row["action"] == "retry"] == [54]
    before = {
        run.run_dir: (run.run_dir / "result.json").read_bytes() for run in first.runs if run.ok
    }
    manifest = (first.execution_dir / "execution.json").read_bytes()
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert second.execution_id == first.execution_id
    assert len(second.runs) == 10
    bad = next(run for run in first.runs if not run.ok)
    retried = next(run for run in second.runs if run.seed == 54)
    assert retried.run_id == bad.run_id
    assert retried.attempt_number == bad.attempt_number + 1
    assert retried.resumed_from_checkpoint
    assert all((path / "result.json").read_bytes() == value for path, value in before.items())
    assert (first.execution_dir / "execution.json").read_bytes() == manifest
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 11
    third = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert [run.run_dir for run in third.runs] == [run.run_dir for run in second.runs]
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 11


def test_sweep_recovery_preserves_every_completed_cell(
    tmp_path: Path, inline_dispatch: None
) -> None:
    config = config_at(tmp_path, seeds=[54, 7], sweep={"space": {"choice": [0, 1]}})
    first = WorkRunner().run(config)
    assert sum(run.ok for run in first.runs) == 3
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert len(second.runs) == 4
    assert {run.run_dir for run in first.runs if run.ok} <= {run.run_dir for run in second.runs}


def test_multiple_failures_keep_latest_outcome_and_physical_cost(
    tmp_path: Path, inline_dispatch: None
) -> None:
    config = config_at(tmp_path, **{"with": {"failures": 2}})
    first = WorkRunner().run(config)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "failed"
    preview = fixed_recovery_preview(first.execution_dir)
    assert preview["reuse_runs"] == 9
    assert preview["spent_attempts"] == 11
    third = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert third.status == "succeeded"
    assert next(run for run in third.runs if run.seed == 54).attempt_number == 3
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 12


def test_dry_run_no_mutation_and_controller_lock(tmp_path: Path, inline_dispatch: None) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    before = {path: path.read_bytes() for path in first.execution_dir.rglob("*") if path.is_file()}
    plan = WorkRunner().run(config, resume_execution=first.execution_dir, dry_run=True)
    assert plan.execution_id == first.execution_id
    fixed_recovery_preview(first.execution_dir)
    assert before == {
        path: path.read_bytes() for path in first.execution_dir.rglob("*") if path.is_file()
    }
    with CrossProcessFileLock(
        first.execution_dir / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            WorkRunner().run(config, resume_execution=first.execution_dir)


@pytest.mark.parametrize("status", ["cancelled", "interrupted", "timeout"])
def test_cancelled_interrupted_terminal_run_is_retried(
    tmp_path: Path, inline_dispatch: None, status: str
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    bad = next(run for run in first.runs if not run.ok)
    result = json.loads((bad.run_dir / "result.json").read_text())
    atomic_json(bad.run_dir / "result.json", {**result, "status": status})
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert next(run for run in second.runs if run.seed == 54).run_id == bad.run_id


def test_changed_scientific_identity_and_missing_results_fail_closed(
    tmp_path: Path, inline_dispatch: None
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    changed = config_at(tmp_path, seeds=SEEDS[:-1])
    with pytest.raises(ValueError, match="unchanged configuration|design"):
        WorkRunner().run(changed, resume_execution=first.execution_dir, accept_code_change=True)
    path = first.runs[0].run_dir / "result.json"
    path.unlink()
    with pytest.raises(ValueError, match="regular persisted file"):
        fixed_recovery_preview(first.execution_dir)


def test_code_acknowledgement_preserves_logical_run_and_checkpoint(
    tmp_path: Path, inline_dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setattr(runner, "_code_identity", lambda root: {"source": "fixed-model"})
    with pytest.raises(ValueError, match="Consumer code changed"):
        WorkRunner().run(config, resume_execution=first.execution_dir)
    second = WorkRunner().run(config, resume_execution=first.execution_dir, accept_code_change=True)
    assert second.status == "succeeded"
    assert (
        next(run for run in second.runs if run.seed == 54).run_id
        == next(run for run in first.runs if run.seed == 54).run_id
    )


def test_exhausted_run_budget_is_not_reset(tmp_path: Path, inline_dispatch: None) -> None:
    config = config_at(tmp_path, execution={"max_runs": 10, "failure_retries": 0})
    first = WorkRunner().run(config)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "failed"
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 10
    assert {run.run_dir for run in second.runs} == {run.run_dir for run in first.runs}


def test_fixed_recovery_with_real_spawned_cpu_workers(tmp_path: Path) -> None:
    config = config_at(tmp_path, seeds=[7, 54])
    first = WorkRunner().run(config)
    assert sum(run.ok for run in first.runs) == 1
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert next(run for run in second.runs if run.seed == 7).run_dir == first.runs[0].run_dir
    assert next(run for run in second.runs if run.seed == 54).attempt_number == 2


def test_controller_interruption_after_request_before_result(
    tmp_path: Path, inline_dispatch: None
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    bad = next(run for run in first.runs if not run.ok)
    # Simulate a killed worker/controller before publishing its terminal outcome.
    (bad.run_dir / "result.json").unlink()
    (first.execution_dir / "result.json").unlink()
    preview = fixed_recovery_preview(first.execution_dir)
    assert (preview["reuse_runs"], preview["retry_runs"]) == (9, 1)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert next(run for run in second.runs if run.seed == 54).attempt_number == 2


def test_pending_cells_are_not_confused_with_completed_evidence(
    tmp_path: Path, inline_dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_at(tmp_path, **{"with": {"failures": 0}})
    runner = importlib.import_module("lambdaforge.work.runner")
    dispatch = runner._execute_adaptive_dispatch
    monkeypatch.setattr(
        runner,
        "_execute_adaptive_dispatch",
        lambda values, **kwargs: dispatch(values[:3], **kwargs),
    )
    first = WorkRunner().run(config)
    assert first.status == "failed"
    preview = fixed_recovery_preview(first.execution_dir)
    assert (preview["reuse_runs"], preview["retry_runs"], preview["pending_runs"]) == (3, 0, 7)
    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", dispatch)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert len(second.runs) == 10
    assert {run.run_dir for run in first.runs} <= {run.run_dir for run in second.runs}


@pytest.mark.parametrize("damage", ["design", "parameters", "budget", "checkpoint"])
def test_corrupt_or_unsafe_fixed_evidence_fails_before_recovery_writes(
    tmp_path: Path, inline_dispatch: None, damage: str
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    bad = next(run for run in first.runs if not run.ok)
    if damage == "design":
        path = first.execution_dir / "resolved-configuration.json"
        value = json.loads(path.read_text())
        value["levels"][0][0]["design"]["evidence"]["requirements"] = []
        atomic_json(path, value)
    elif damage == "parameters":
        path = bad.run_dir / "result.json"
        value = json.loads(path.read_text())
        value["parameters"]["failures"] = 999
        atomic_json(path, value)
    elif damage == "budget":
        path = first.execution_dir / "hpo-control" / "fixed-recovery-state.json"
        value = json.loads(path.read_text())
        atomic_json(path, {**value, "elapsed_seconds": -1})
    else:
        root = bad.run_dir.parent.parent / "checkpoints"
        shutil.rmtree(root)
        root.symlink_to(tmp_path)
    before = (first.execution_dir / "current-code.json").read_bytes()
    with pytest.raises(ValueError):
        WorkRunner().run(config, resume_execution=first.execution_dir, dry_run=True)
    assert (first.execution_dir / "current-code.json").read_bytes() == before
    assert not (first.execution_dir / "recovery-history.jsonl").exists()


def test_spent_time_budget_is_not_reset(tmp_path: Path, inline_dispatch: None) -> None:
    config = config_at(tmp_path, execution={"max_time": "1h", "failure_retries": 0})
    first = WorkRunner().run(config)
    path = first.execution_dir / "hpo-control" / "fixed-recovery-state.json"
    value = json.loads(path.read_text())
    atomic_json(path, {**value, "elapsed_seconds": 3600})
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "failed"
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 10


def test_missing_checkpoints_restart_only_the_failed_run(
    tmp_path: Path, inline_dispatch: None
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    bad = next(run for run in first.runs if not run.ok)
    shutil.rmtree(bad.run_dir.parent.parent / "checkpoints")
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    retried = next(run for run in second.runs if run.seed == 54)
    assert retried.run_id == bad.run_id and retried.attempt_number == 2
    assert not retried.resumed_from_checkpoint
    assert not retried.ok  # Toy Work fails again when application state is absent.
    assert sum(run.ok for run in second.runs) == 9


def test_paired_sweep_recovery_restores_committed_shared_blocks(
    tmp_path: Path, inline_dispatch: None
) -> None:
    from lambdaforge.reproducibility.SeedProvider import ProjectSeedStream

    raw = {
        "name": "fixed",
        "run": "tests.work_cases.FixedRecoveryWork",
        "sweep": {"space": {"choice": [0, 1]}},
        "resources": {"cpu": 1},
        "execution": {"max_runs": 8, "failure_retries": 0},
        "objective": {"metric": "score", "mode": "max", "range": [0, 1]},
    }
    initial = WorkConfig.from_mapping(raw, source=tmp_path / "fixed.yaml")
    design = initial.levels[0].runs[0].study_design.to_dict()
    stream = ProjectSeedStream.from_mapping(design["seed_source"])
    config = WorkConfig.from_mapping(
        {**raw, "with": {"fail_seed": stream.at(1).value}}, source=initial.source
    )
    first = WorkRunner().run(config)
    assert first.status == "failed"
    preview = fixed_recovery_preview(first.execution_dir)
    assert preview["retry_runs"] == 1
    assert preview["reuse_runs"] + preview["pending_runs"] == 3
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    bad = next(run for run in first.runs if not run.ok)
    retried = next(run for run in second.runs if run.run_id == bad.run_id)
    assert retried.attempt_number == bad.attempt_number + 1
    assert {run.run_dir for run in first.runs if run.ok} <= {run.run_dir for run in second.runs}
    assert fixed_inventory(first.execution_dir)["spent_runs"] <= 8


def test_lost_worker_keeps_its_real_run_and_attempt_identity(
    tmp_path: Path, inline_dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = importlib.import_module("lambdaforge.work.runner")
    execute = runner._execute_run

    def killed(specification: Any) -> Any:
        result = execute(specification)
        if result.seed != 54:
            return result
        (result.run_dir / "result.json").unlink()
        location = result.run_dir / "worker-checkpoint.json"
        atomic_json(location, {"run_dir": str(result.run_dir), "run_id": result.run_id})
        failure = runner._controller_failure_result(
            {**specification, "hpo_checkpoint_manifest_path": location},
            RuntimeError("worker killed"),
        )
        assert failure.run_id == result.run_id
        assert failure.attempt_number == result.attempt_number
        return failure

    monkeypatch.setattr(runner, "_execute_run", killed)
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    assert fixed_recovery_preview(first.execution_dir)["retry_runs"] == 1
    monkeypatch.setattr(runner, "_execute_run", execute)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert fixed_inventory(first.execution_dir)["spent_runs"] == 11


@pytest.mark.parametrize("change", ["input", "objective", "parameters", "sweep"])
def test_code_acknowledgement_cannot_bypass_scientific_identity_changes(
    tmp_path: Path, inline_dispatch: None, change: str
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("original content")
    config = config_at(tmp_path, **{"with": {"source": {"file": str(source)}}})
    first = WorkRunner().run(config)
    raw = dict(config.raw)
    if change == "input":
        source.write_text("different content")
    elif change == "objective":
        raw["objective"] = {"metric": "score", "mode": "min"}
    elif change == "parameters":
        raw["with"] = {**raw["with"], "choice": 1}
    else:
        raw["sweep"] = {"space": {"choice": [0, 1]}}
    changed = WorkConfig.from_mapping(raw, source=config.source)
    with pytest.raises(ValueError, match="identity|parameters/inputs|unchanged configuration"):
        WorkRunner().run(changed, resume_execution=first.execution_dir, accept_code_change=True)
