"""Composable, resumable and sharded preprocessing with dataset identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.preprocessing import DatasetArtifact, PreprocessingManifest
from lambdaforge.tasks import TaskConfig, TaskResult, TaskRun, TaskStatus, TaskValidator


class TestPreprocessing:
    """Exercise preprocessing through the public generic-task YAML surface."""

    @staticmethod
    def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
        """Write deterministic compact JSON Lines test input."""
        path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    @staticmethod
    def config(tmp_path: Path, *, transforms: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return a complete built-in preprocessing task mapping."""
        return {
            "schema_version": "1.0",
            "kind": "task",
            "name": "records",
            "output_root": str(tmp_path / "runs"),
            "inputs": [{"name": "raw", "path": "records.jsonl"}],
            "task": {
                "target": "lambdaforge.preprocessing.PreprocessingTask",
                "params": {
                    "source": {
                        "target": "lambdaforge.preprocessing.JsonLinesSource",
                        "params": {"path": "records.jsonl", "key_field": "id"},
                    },
                    "transforms": transforms
                    if transforms is not None
                    else [
                        {
                            "target": "tests.fixtures.UppercaseRecordValue.UppercaseRecordValue",
                            "params": {"field": "text"},
                        }
                    ],
                    "sink": {
                        "target": "lambdaforge.preprocessing.JsonDirectorySink",
                        "params": {"output_dir": "processed"},
                    },
                },
            },
        }

    @staticmethod
    def write_config(path: Path, value: dict[str, Any]) -> Path:
        """Persist one preprocessing YAML document."""
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return path

    def test_pipeline_writes_records_progress_and_dataset_artifact(self, tmp_path: Path) -> None:
        """The built-in pipeline should produce inspectable content-addressed outputs."""
        self.write_jsonl(
            tmp_path / "records.jsonl",
            [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}],
        )
        path = self.write_config(tmp_path / "preprocess.yaml", self.config(tmp_path))
        report = TaskValidator().validate_file(path)
        assert report.is_valid, report.summary()
        result = TaskRun.from_yaml(path).run()
        assert isinstance(result, TaskResult)
        assert result.status is TaskStatus.OK
        assert result.metrics == {
            "records_selected": 2,
            "records_processed": 2,
            "records_resumed": 0,
            "records_failed": 0,
        }
        run_dir = Path(result.run_dir)
        outputs = sorted((run_dir / "processed").glob("*.json"))
        assert len(outputs) == 2
        values = [json.loads(output.read_text(encoding="utf-8"))["value"] for output in outputs]
        assert {value["text"] for value in values} == {"ALPHA", "BETA"}
        manifest = PreprocessingManifest.read_json(run_dir / "preprocessing-manifest.json")
        assert manifest.complete
        assert manifest.successful_keys == {"a", "b"}
        dataset = DatasetArtifact.read_json(run_dir / "dataset-artifact.json")
        assert dataset.sample_count == 2
        assert dataset.splits == {"all": 2}
        assert dataset.preprocessing_fingerprint == result.config_fingerprint
        assert dataset.dataset_id == result.outputs["dataset_id"]
        assert "resolved_path" not in dataset.source["inputs"][0]

    def test_failed_attempt_resumes_only_verified_record_outputs(self, tmp_path: Path) -> None:
        """A new attempt should skip successful records and retry the failed key."""
        self.write_jsonl(
            tmp_path / "records.jsonl",
            [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}],
        )
        transform = {
            "target": "tests.fixtures.FailOnceRecordTransform.FailOnceRecordTransform",
            "params": {"key": "b"},
        }
        path = self.write_config(
            tmp_path / "preprocess.yaml",
            self.config(tmp_path, transforms=[transform]),
        )
        first = TaskRun.from_yaml(path).run()
        assert isinstance(first, TaskResult)
        assert first.status is TaskStatus.FAILED
        partial_outputs = list(Path(first.run_dir).glob("processed/*.json"))
        assert len(partial_outputs) == 1
        partial_outputs[0].write_text("corrupt", encoding="utf-8")
        second = TaskRun.from_yaml(path).run()
        assert isinstance(second, TaskResult)
        assert second.status is TaskStatus.OK
        assert second.metrics["records_resumed"] == 0
        assert second.metrics["records_processed"] == 2
        assert len(list(Path(second.run_dir).glob(".lambdaforge/attempts/result-*.json"))) == 1

    def test_shards_are_deterministic_disjoint_and_cover_the_source(self, tmp_path: Path) -> None:
        """Stable keys should map to exactly one independently runnable shard."""
        values = [{"id": f"sample-{index}", "text": str(index)} for index in range(20)]
        self.write_jsonl(tmp_path / "records.jsonl", values)
        successful: list[set[str]] = []
        for shard_index in (0, 1):
            value = self.config(tmp_path, transforms=[])
            value["name"] = f"records-shard-{shard_index}"
            value["task"]["params"].update({"shard_count": 2, "shard_index": shard_index})
            path = self.write_config(tmp_path / f"shard-{shard_index}.yaml", value)
            result = TaskRun.from_yaml(path).run()
            assert isinstance(result, TaskResult)
            manifest = PreprocessingManifest.read_json(
                Path(result.run_dir) / "preprocessing-manifest.json"
            )
            successful.append(set(manifest.successful_keys))
        assert successful[0].isdisjoint(successful[1])
        assert successful[0] | successful[1] == {value["id"] for value in values}

    def test_preprocessing_requires_declared_content_inputs(self, tmp_path: Path) -> None:
        """Built-in preprocessing must not silently fingerprint only a mutable path string."""
        value = self.config(tmp_path)
        value.pop("inputs")
        report = TaskValidator().validate(value)
        assert not report.is_valid
        assert any("requires at least one top-level 'inputs'" in error for error in report.errors)

    def test_built_in_source_must_match_the_declared_content_input(self, tmp_path: Path) -> None:
        """A dummy input must not make an undeclared mutable source look reproducible."""
        self.write_jsonl(tmp_path / "records.jsonl", [{"id": "a", "text": "alpha"}])
        (tmp_path / "dummy.txt").write_text("dummy", encoding="utf-8")
        value = self.config(tmp_path)
        value["inputs"] = [{"name": "dummy", "path": "dummy.txt"}]
        path = self.write_config(tmp_path / "preprocess.yaml", value)
        report = TaskValidator().validate_file(path)
        assert not report.is_valid
        assert any("is not covered" in error for error in report.errors)

    def test_input_change_selects_a_new_preprocessing_identity(self, tmp_path: Path) -> None:
        """Changing raw dataset bytes must invalidate task completion automatically."""
        source = tmp_path / "records.jsonl"
        self.write_jsonl(source, [{"id": "a", "text": "alpha"}])
        path = self.write_config(tmp_path / "preprocess.yaml", self.config(tmp_path))
        first = TaskConfig.from_yaml(path)
        first_fingerprint = first.fingerprint
        self.write_jsonl(source, [{"id": "a", "text": "changed"}])
        second = TaskConfig.from_yaml(path)
        assert second.fingerprint != first_fingerprint
        assert second.run_dir != first.run_dir

    def test_packaged_example_is_current_and_constructible(self) -> None:
        """The public preprocessing example must remain valid against packaged Schema 1.0."""
        path = Path("examples/preprocessing.yaml")
        report = TaskValidator().validate_file(path)
        assert report.is_valid, report.summary()
