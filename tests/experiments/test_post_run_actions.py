"""Focused tests for generic, restart-safe post-run actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.experiments import (
    ExperimentRunner,
    ExperimentValidator,
    PostRunContext,
    PostRunResult,
    PostRunService,
    RunResult,
    RunStatus,
)
from lambdaforge.experiments.postrun import TrainingCompletionStore
from lambdaforge.experiments.results import RunFingerprint


class RecordingAction:
    """Write one neutral report and expose checkpoint selection in its output."""

    calls = 0

    def __init__(self, filename: str = "analysis/report.json", value: str = "first") -> None:
        self.filename = filename
        self.value = value

    def run(self, context: PostRunContext) -> PostRunResult:
        """Persist one declared report below the owning run."""
        type(self).calls += 1
        path = context.artifact_path(self.filename, create_parent=True)
        path.write_text(
            json.dumps(
                {
                    "value": self.value,
                    "checkpoint": (
                        context.selected_checkpoint.name
                        if context.selected_checkpoint is not None
                        else None
                    ),
                    "role": context.selected_checkpoint_role,
                }
            ),
            encoding="utf-8",
        )
        return PostRunResult(
            outputs={"value": self.value},
            artifacts=[
                {
                    "name": "report",
                    "path": self.filename,
                    "kind": "report",
                    "media_type": "application/json",
                }
            ],
        )


class FailingAction:
    """Raise a deterministic project error for failure-policy tests."""

    def run(self, context: PostRunContext) -> PostRunResult:
        """Fail after receiving a valid stable context."""
        del context
        raise RuntimeError("project analysis failed")


class InterruptOnceAction:
    """Model a process interruption that must not become a failed receipt."""

    interrupted = False

    def run(self, context: PostRunContext) -> PostRunResult:
        """Interrupt once and finish when the whole run is relaunched."""
        if not type(self).interrupted:
            type(self).interrupted = True
            context.state_dir.joinpath("partial.txt").write_text("partial", encoding="utf-8")
            raise KeyboardInterrupt
        path = context.artifact_path("analysis/resumed.json", create_parent=True)
        path.write_text("{}", encoding="utf-8")
        return PostRunResult(artifacts=[{"name": "resumed", "path": "analysis/resumed.json"}])


class CancellationAwareAction:
    """Verify that cancellation state remains live while an action executes."""

    event: Event
    observations: tuple[bool, bool] | None = None

    def run(self, context: PostRunContext) -> PostRunResult:
        """Observe the context before and after the shared event changes."""
        before = context.stop_requested
        type(self).event.set()
        type(self).observations = (before, context.stop_requested)
        return PostRunResult()


def _run_evidence(tmp_path: Path, config: dict[str, Any]) -> RunResult:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = checkpoint_dir / "best-epoch=1.ckpt"
    last = checkpoint_dir / "last.ckpt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    return RunResult(
        name="neutral",
        run_dir=run_dir,
        variant="base",
        seed=7,
        status=RunStatus.OK,
        best_model_path=best,
        last_model_path=last,
        config_fingerprint=RunFingerprint.digest(config),
    )


def _action_config(
    *, required: bool = True, checkpoint: str = "best", value: str = "first"
) -> dict[str, Any]:
    return {
        "model": {"target": "torch.nn.Identity", "params": {}},
        "post_run": [
            {
                "name": "analysis",
                "target": "tests.experiments.test_post_run_actions.RecordingAction",
                "params": {"value": value},
                "checkpoint": checkpoint,
                "required": required,
            }
        ],
    }


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    (("best", "best-epoch=1.ckpt"), ("last", "last.ckpt"), ("current", "last.ckpt")),
)
def test_explicit_checkpoint_selection_and_artifact_provenance(
    tmp_path: Path, checkpoint: str, expected: str
) -> None:
    config = _action_config(checkpoint=checkpoint)
    result = PostRunService().run(config, _run_evidence(tmp_path, config))

    assert result.status is RunStatus.OK, result.post_run_actions[0].get("error")
    assert result.post_run_actions[0]["outputs"]["value"] == "first"
    assert result.artifacts[0]["name"] == "report"
    assert result.artifacts[0]["metadata"]["producer"].endswith("RecordingAction")
    expected_digest = hashlib.sha256(
        (tmp_path / "run" / "checkpoints" / expected).read_bytes()
    ).hexdigest()
    assert result.post_run_actions[0]["checkpoint_sha256"] == expected_digest
    payload = json.loads((tmp_path / "run" / "analysis" / "report.json").read_text())
    assert payload["checkpoint"] == expected
    assert PostRunService().is_complete(config, result)


def test_required_and_optional_failures_have_distinct_completion_semantics(
    tmp_path: Path,
) -> None:
    base = {
        "model": {"target": "torch.nn.Identity", "params": {}},
        "post_run": [
            {
                "name": "failure",
                "target": "tests.experiments.test_post_run_actions.FailingAction",
                "checkpoint": "none",
                "required": True,
            }
        ],
    }
    required = PostRunService().run(base, _run_evidence(tmp_path, base))
    assert required.status is RunStatus.FAILED
    assert not PostRunService().is_complete(base, required)
    assert "Required post-run" in str(required.error)

    optional_config = json.loads(json.dumps(base))
    optional_config["post_run"][0]["required"] = False
    optional = PostRunService().run(optional_config, _run_evidence(tmp_path, optional_config))
    assert optional.status is RunStatus.OK
    assert optional.post_run_actions[0]["status"] == "failed"
    assert optional.post_run_warnings
    assert PostRunService().is_complete(optional_config, optional)


def test_missing_global_required_artifact_prevents_success(tmp_path: Path) -> None:
    config = {
        "experiment": {"required_artifacts": ["analysis/missing.json"]},
        "model": {"target": "torch.nn.Identity", "params": {}},
    }
    result = PostRunService().run(config, _run_evidence(tmp_path, config))

    assert result.status is RunStatus.FAILED
    assert "analysis/missing.json" in str(result.error)


def test_schema_and_import_validation_accepts_public_post_run_yaml() -> None:
    config = {
        "schema_version": "1.1",
        "experiment": {"name": "post-run-schema"},
        "data": {},
        "model": {"target": "torch.nn.Identity", "params": {}},
        "losses": [{"target": "lambdaforge.nn.losses.MeanSquaredErrorLoss"}],
        "post_run": [
            {
                "name": "analysis",
                "target": "tests.experiments.test_post_run_actions.RecordingAction",
                "checkpoint": "best",
                "required": True,
                "artifacts": [{"name": "report", "path": "analysis/report.json", "kind": "report"}],
            }
        ],
    }

    report = ExperimentValidator().validate(config)
    assert report.is_valid, report.errors


def test_action_identity_reruns_changed_postprocessing_without_changing_training(
    tmp_path: Path,
) -> None:
    first_config = _action_config(value="first")
    second_config = _action_config(value="second")
    assert RunFingerprint.digest(first_config) == RunFingerprint.digest(second_config)
    RecordingAction.calls = 0
    evidence = _run_evidence(tmp_path, first_config)

    first = PostRunService().run(first_config, evidence)
    repeated = PostRunService().run(first_config, evidence)
    changed = PostRunService().run(second_config, evidence)

    assert first.post_run_actions[0]["skipped_existing"] is False
    assert repeated.post_run_actions[0]["skipped_existing"] is True
    assert changed.post_run_actions[0]["outputs"]["value"] == "second"
    payload = json.loads((tmp_path / "run" / "analysis" / "report.json").read_text())
    assert payload["value"] == "second"


def test_interrupted_action_keeps_partial_state_and_retries(tmp_path: Path) -> None:
    config = {
        "model": {"target": "torch.nn.Identity", "params": {}},
        "post_run": [
            {
                "name": "interruptible",
                "target": "tests.experiments.test_post_run_actions.InterruptOnceAction",
                "checkpoint": "none",
            }
        ],
    }
    evidence = _run_evidence(tmp_path, config)
    InterruptOnceAction.interrupted = False

    with pytest.raises(KeyboardInterrupt):
        PostRunService().run(config, evidence)
    receipt = tmp_path / "run" / ".lambdaforge/post-run/actions/interruptible/result.json"
    assert not receipt.exists()

    resumed = PostRunService().run(config, evidence)
    assert resumed.status is RunStatus.OK
    assert (tmp_path / "run" / "analysis" / "resumed.json").is_file()


def test_non_global_rank_does_not_execute_or_write_artifacts(tmp_path: Path) -> None:
    config = _action_config()
    evidence = _run_evidence(tmp_path, config)
    RecordingAction.calls = 0

    unchanged = PostRunService().run(config, evidence, is_global_zero=False)

    assert unchanged is evidence
    assert RecordingAction.calls == 0
    assert not (tmp_path / "run" / "analysis").exists()


def test_post_run_context_exposes_live_cooperative_cancellation(tmp_path: Path) -> None:
    event = Event()
    CancellationAwareAction.event = event
    CancellationAwareAction.observations = None
    config = {
        "model": {"target": "torch.nn.Identity", "params": {}},
        "post_run": [
            {
                "name": "cancellation-aware",
                "target": f"{__name__}.CancellationAwareAction",
                "checkpoint": "none",
            }
        ],
    }

    result = PostRunService().run(
        config,
        _run_evidence(tmp_path, config),
        stop_event=event,
    )

    assert result.status is RunStatus.OK, result.post_run_actions[0].get("error")
    assert CancellationAwareAction.observations == (False, True)


def test_training_completion_identity_excludes_post_run_but_tracks_fidelity(
    tmp_path: Path,
) -> None:
    first = _action_config(value="first")
    first["trainer"] = {"max_epochs": 3, "checkpoint_policy": "last"}
    second = _action_config(value="second")
    second["trainer"] = {"max_epochs": 3, "checkpoint_policy": "last"}
    evidence = _run_evidence(tmp_path, first)
    store = TrainingCompletionStore()
    store.write(first, evidence)

    assert store.load(second, evidence.run_dir) is not None
    changed_training = json.loads(json.dumps(second))
    changed_training["trainer"]["max_epochs"] = 4
    assert store.load(changed_training, evidence.run_dir) is None


def test_adaptive_scope_defaults_to_confirmation_and_never_runs_pauses() -> None:
    service = PostRunService()
    config = _action_config()
    assert not service.should_run_for_adaptive(config, phase="search", trial_status="completed")
    assert service.should_run_for_adaptive(config, phase="confirmation", trial_status="completed")
    config = {"post_run": {"scope": "all_runs", "actions": config["post_run"]}}
    assert service.should_run_for_adaptive(config, phase="search", trial_status="early_stopped")
    assert not service.should_run_for_adaptive(config, phase="search", trial_status="paused")

    confirmation = _action_config()
    confirmation["metadata"] = {
        "adaptive": {
            "phase": "confirmation",
            "target_budget": 10,
            "max_budget": 10,
        }
    }
    evidence = RunResult(name="confirmation", run_dir="run", status=RunStatus.OK)
    assert service.should_run_for_materialized_adaptive(confirmation, evidence)

    pause = json.loads(json.dumps(confirmation))
    pause["metadata"]["adaptive"].update(phase="search", target_budget=5)
    assert not service.should_run_for_materialized_adaptive(pause, evidence)


def test_runner_reuses_training_when_only_action_configuration_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "reuse" / "base" / "seed=7"
    fit_calls = 0

    def fit(**_kwargs: Any) -> Any:
        nonlocal fit_calls
        fit_calls += 1
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best = checkpoint_dir / "best-epoch=1.ckpt"
        last = checkpoint_dir / "last.ckpt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        callback = SimpleNamespace(
            best_model_path=str(best),
            last_model_path=str(last),
            best_model_score=None,
            monitor=None,
        )
        return SimpleNamespace(
            checkpoint_callback=callback,
            callback_metrics={},
            is_global_zero=True,
        )

    runner = ExperimentRunner()
    monkeypatch.setattr(runner, "_build_datamodule", lambda *_args, **_kwargs: object())
    task = SimpleNamespace(train_metrics=[], val_metrics=[], test_metrics=[])
    monkeypatch.setattr(runner, "_build_task", lambda *_args, **_kwargs: (task, []))
    monkeypatch.setattr(
        runner,
        "_build_runner",
        lambda *_args, **_kwargs: SimpleNamespace(fit=fit),
    )

    def config(value: str) -> dict[str, Any]:
        return {
            "experiment": {
                "name": "reuse",
                "output_root": str(tmp_path),
                "seed": 7,
                "variant": "base",
                "required_artifacts": ["analysis/report.json"],
            },
            "model": {"target": "torch.nn.Identity", "params": {}},
            "trainer": {"checkpoint_policy": "last_and_best"},
            **_action_config(value=value),
        }

    first = runner.run_single_experiment(config("first"))
    second = runner.run_single_experiment(config("second"))

    assert first.status is RunStatus.OK
    assert second.status is RunStatus.OK
    assert fit_calls == 1
    assert second.post_run_actions[0]["outputs"]["value"] == "second"
