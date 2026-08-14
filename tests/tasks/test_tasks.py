"""Generic task Schema, planning, execution, artifacts and CLI integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from lambdaforge import LambdaForge, TaskExecutionPlan, TaskResult, TaskRun
from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.tasks import TaskArtifact, TaskConfig, TaskStatus, TaskValidator


class TestGenericTasks:
    """Exercise the complete non-training task lifecycle."""

    @staticmethod
    def config(
        tmp_path: Path, *, target: str = "tests.fixtures.UserTask.UserTask"
    ) -> dict[str, Any]:
        """Return one minimal valid task mapping."""
        return {
            "schema_version": "1.0",
            "kind": "task",
            "name": "prepare-data",
            "output_root": str(tmp_path / "runs"),
            "task": {"target": target, "params": {"message": "ready"}},
        }

    @staticmethod
    def write(path: Path, value: dict[str, Any]) -> Path:
        """Persist one test YAML document."""
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return path

    def test_validation_checks_schema_imports_and_constructor_without_outputs(
        self, tmp_path: Path
    ) -> None:
        """Validation must catch constructor errors without constructing the task."""
        value = self.config(tmp_path)
        path = self.write(tmp_path / "task.yaml", value)
        report = TaskValidator().validate_file(path)
        assert report.is_valid
        assert report.schema_version == "1.0"
        assert not (tmp_path / "runs").exists()

        value["task"]["params"] = {}
        invalid = TaskValidator().validate(value)
        assert not invalid.is_valid
        assert any("missing a required argument: 'message'" in error for error in invalid.errors)

    def test_dry_run_is_an_immutable_side_effect_free_plan(self, tmp_path: Path) -> None:
        """Inspect and dry-run must agree without importing or creating run paths."""
        path = self.write(tmp_path / "task.yaml", self.config(tmp_path))
        task = TaskRun.from_yaml(path)
        inspected = task.inspect()
        dry_run = task.run(dry_run=True)
        assert isinstance(dry_run, TaskExecutionPlan)
        assert dry_run.to_dict() == inspected.to_dict()
        assert dry_run.will_run
        assert not Path(dry_run.run_dir).exists()
        try:
            dry_run["action"] = "skip"
        except TypeError:
            pass
        else:
            raise AssertionError("Task execution plans must be immutable.")
        try:
            dry_run.execution["mode"] = "parallel"
        except TypeError:
            pass
        else:
            raise AssertionError("Nested task plan values must be immutable.")

    def test_execution_persists_typed_result_artifacts_and_catalog(self, tmp_path: Path) -> None:
        """A duck-typed task should receive context and publish complete provenance."""
        path = self.write(tmp_path / "task.yaml", self.config(tmp_path))
        result = LambdaForge.run(path)
        assert isinstance(result, TaskResult)
        assert result.status is TaskStatus.OK
        assert result.outputs == {"message": "ready"}
        assert result.metrics == {"message_length": 5}
        assert len(result.artifacts) == 1
        artifact = result.artifacts[0]
        assert isinstance(artifact, TaskArtifact)
        assert artifact.path == "output.txt"
        assert len(artifact.sha256) == 64
        run_dir = Path(result.run_dir)
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "environment.json").is_file()
        assert (run_dir / "task.log").is_file()
        assert TaskResult.read_json(run_dir / "result.json").to_dict() == result.to_dict()
        records = TaskRun.from_yaml(path).result_catalog().records()
        assert len(records) == 1
        assert records[0].metrics == {"message_length": 5}

    def test_completion_skip_and_explicit_rerun_keep_attempt_history(self, tmp_path: Path) -> None:
        """Successful content is reused unless rerun_completed explicitly requests an attempt."""
        value = self.config(tmp_path)
        path = self.write(tmp_path / "task.yaml", value)
        first = TaskRun.from_yaml(path).run()
        second = TaskRun.from_yaml(path).run()
        assert isinstance(first, TaskResult) and isinstance(second, TaskResult)
        assert second.skipped_existing
        assert second.attempt_id == first.attempt_id

        value["rerun_completed"] = True
        self.write(path, value)
        third = TaskRun.from_yaml(path).run()
        assert isinstance(third, TaskResult)
        assert third.status is TaskStatus.OK
        assert third.attempt_id != first.attempt_id
        attempts = list(Path(third.run_dir).glob(".lambdaforge/attempts/result-*.json"))
        assert len(attempts) == 1

    def test_zero_argument_duck_task_and_structured_failure(self, tmp_path: Path) -> None:
        """Both concise duck typing and persisted exceptions follow explicit contracts."""
        value = self.config(tmp_path, target="tests.fixtures.NoContextTask.NoContextTask")
        value["task"]["params"] = {}
        zero_path = self.write(tmp_path / "zero.yaml", value)
        zero = TaskRun.from_yaml(zero_path).run()
        assert isinstance(zero, TaskResult)
        assert zero.outputs == {"zero_argument": True}

        value["name"] = "failure"
        value["task"] = {
            "target": "tests.fixtures.FailingTask.FailingTask",
            "params": {"message": "broken input"},
        }
        failed_path = self.write(tmp_path / "failed.yaml", value)
        failed = TaskRun.from_yaml(failed_path).run()
        assert isinstance(failed, TaskResult)
        assert failed.status is TaskStatus.FAILED
        assert failed.error is not None
        assert failed.error["type"] == "RuntimeError"
        assert failed.error["message"] == "broken input"

    def test_local_input_content_participates_in_identity(self, tmp_path: Path) -> None:
        """Changing declared input bytes must select a new content-addressed run."""
        source = tmp_path / "raw.txt"
        source.write_text("first", encoding="utf-8")
        value = self.config(tmp_path)
        value["inputs"] = [{"name": "raw", "path": "raw.txt"}]
        path = self.write(tmp_path / "task.yaml", value)
        first = TaskConfig.from_yaml(path)
        first_fingerprint = first.fingerprint
        first_run_dir = first.run_dir
        source.write_text("second", encoding="utf-8")
        second = TaskConfig.from_yaml(path)
        assert second.fingerprint != first_fingerprint
        assert second.run_dir != first_run_dir

    def test_symbolic_link_inputs_and_artifacts_are_rejected(self, tmp_path: Path) -> None:
        """Content identity must never silently follow symbolic-link indirection."""
        source = tmp_path / "raw.txt"
        source.write_text("source", encoding="utf-8")
        input_link = tmp_path / "raw-link.txt"
        try:
            os.symlink(source, input_link)
        except (OSError, NotImplementedError):
            return
        value = self.config(tmp_path)
        value["inputs"] = [{"path": "raw-link.txt"}]
        path = self.write(tmp_path / "task.yaml", value)
        report = TaskValidator().validate_file(path)
        assert not report.is_valid
        assert any("symbolic links" in error for error in report.errors)

        run_dir = tmp_path / "artifact-run"
        run_dir.mkdir()
        artifact_link = run_dir / "output.txt"
        os.symlink(source, artifact_link)
        try:
            TaskArtifact.materialize("output.txt", run_dir)
        except ValueError as error:
            assert "symbolic links" in str(error)
        else:
            raise AssertionError("Symbolic-link artifacts must be rejected.")

    def test_task_owned_log_link_is_rejected_before_execution(self, tmp_path: Path) -> None:
        """Runner metadata writes must not follow a pre-existing link outside the run."""
        path = self.write(tmp_path / "task.yaml", self.config(tmp_path))
        task = TaskRun.from_yaml(path)
        run_dir = task.config.run_dir
        run_dir.mkdir(parents=True)
        outside = tmp_path / "outside.log"
        outside.write_text("unchanged", encoding="utf-8")
        try:
            os.symlink(outside, run_dir / "task.log")
        except (OSError, NotImplementedError):
            return
        try:
            task.run()
        except ValueError as error:
            assert "task-owned path" in str(error)
        else:
            raise AssertionError("Task-owned metadata links must be rejected.")
        assert outside.read_text(encoding="utf-8") == "unchanged"

    def test_cli_dispatches_validate_inspect_dry_run_and_run(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Existing command names should work unchanged for task YAML."""
        path = self.write(tmp_path / "task.yaml", self.config(tmp_path))
        assert CommandLineInterface.main(["validate", str(path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["kind"] == "task"
        assert CommandLineInterface.main(["inspect", str(path)]) == 0
        assert json.loads(capsys.readouterr().out)["action"] == "run"
        assert CommandLineInterface.main(["run", str(path), "--dry-run"]) == 0
        assert json.loads(capsys.readouterr().out)["kind"] == "task"
        assert CommandLineInterface.main(["run", str(path)]) == 0
        assert "status=ok" in capsys.readouterr().out
