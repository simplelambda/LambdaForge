"""Concise authoring compilation and compatibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from lambdaforge.configuration import AuthoringConfig, ConfigurationKind
from lambdaforge.tasks import TaskContext, TaskResult, TaskRun, TaskStatus, TaskValidator


class TestAuthoringConfig:
    """Verify friendly YAML compiles to the established strict runners."""

    def test_concise_preprocessing_materializes_and_runs(self, tmp_path: Path) -> None:
        """A beginner-facing document should need no kind, Schema or object boilerplate."""
        source = tmp_path / "records.jsonl"
        source.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
        config = tmp_path / "preprocess.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "name": "friendly",
                    "output_root": str(tmp_path / "runs"),
                    "inputs": {"raw": "records.jsonl"},
                    "outputs": {"processed": "processed"},
                    "preprocess": {
                        "function": "tests.fixtures.normalize_record.normalize_record",
                        "input": "raw",
                        "output": "processed",
                        "key_field": "id",
                        "workers": 2,
                        "workload": "io",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        materialized = AuthoringConfig.from_yaml(config).materialize()
        assert materialized.kind is ConfigurationKind.TASK
        assert materialized.values["kind"] == "task"
        assert materialized.values["schema_version"] == "1.0"
        report = TaskValidator().validate_file(config)
        assert report.is_valid, report.summary()
        result = TaskRun.from_yaml(config).run()
        assert isinstance(result, TaskResult)
        assert result.status is TaskStatus.OK
        record_path = next((Path(result.run_dir) / "processed").glob("*.json"))
        assert json.loads(record_path.read_text(encoding="utf-8"))["value"]["text"] == "HELLO"
        assert result.metadata["workers"] == 2

    def test_model_string_is_target_shorthand(self) -> None:
        """Only object-bearing fields should interpret strings as import targets."""
        materialized = AuthoringConfig(
            {
                "experiment": {"name": "x"},
                "model": "tests.fixtures.UserModel.UserModel",
            }
        ).materialize()
        assert materialized.kind is ConfigurationKind.EXPERIMENT
        assert materialized.values["model"] == {"target": "tests.fixtures.UserModel.UserModel"}

    def test_common_optimizer_shorthand_compiles_to_the_strict_object_spec(self) -> None:
        materialized = AuthoringConfig(
            {
                "experiment": {"name": "optimizer-short"},
                "model": "tests.fixtures.UserModel.UserModel",
                "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.01},
            }
        ).materialize()

        assert materialized.values["optimizer"] == {
            "ref": "torch.optim.AdamW",
            "params": {"lr": 0.001, "weight_decay": 0.01},
        }

    def test_existing_strict_task_is_unchanged(self) -> None:
        """The authoring compiler must preserve the prior strict contract."""
        strict = {
            "schema_version": "1.0",
            "kind": "task",
            "name": "legacy",
            "task": {"target": "tests.fixtures.UserTask.UserTask", "params": {"value": 1}},
        }
        assert AuthoringConfig(strict).materialize().to_dict() == {**strict, "inputs": []}

    def test_named_output_creates_only_its_parent(self, tmp_path: Path) -> None:
        """Named outputs must work for either a file or a directory."""
        context = TaskContext(
            name="output-contract",
            run_dir=tmp_path / "run",
            source_dir=tmp_path,
            attempt_id="attempt",
            config_fingerprint="sha256:test",
            resume=False,
            outputs={"report": "reports/result.json"},
        )
        output = context.output("report", create=True)
        assert output == tmp_path / "run/reports/result.json"
        assert output.parent.is_dir()
        assert not output.exists()
