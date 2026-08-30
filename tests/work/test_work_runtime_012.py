"""Essential behavioral contract for the Work-centric 0.12 runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lambdaforge import Work
from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner


def _yaml(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "work.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_only_work_subclasses_are_executable(tmp_path: Path) -> None:
    for target, message in (
        ("tests.work_cases.function_target", "Python function"),
        ("tests.work_cases.ForeignClass", "not a LambdaForge Work"),
    ):
        report = WorkConfig.validate_file(_yaml(tmp_path, {"name": "bad", "run": target}))
        assert not report.valid
        assert message in " ".join(report.errors)


def test_parameter_study_intent_is_available_before_execution(tmp_path: Path) -> None:
    source = _yaml(
        tmp_path,
        {
            "name": "seed-study",
            "run": "tests.work_cases.SeedWork",
            "seeds": [4, 7],
            "objective": {"metric": "score", "mode": "max"},
        },
    )

    config = WorkConfig.from_yaml(source)
    descriptor = ConfigurationDescriptor.from_path(source)

    assert config.has_parameter_study
    assert descriptor.metadata()["study_expected"] is True

    workflow = WorkConfig.from_mapping(
        {
            "name": "preprocessing",
            "steps": [
                {"name": "first", "run": "tests.work_cases.Producer"},
                {"name": "second", "run": "tests.work_cases.Producer"},
            ],
        },
        source=tmp_path / "workflow.yaml",
    )
    assert workflow.planned_runs == 2
    assert not workflow.has_parameter_study


def test_normal_work_does_not_publish_study_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_root = tmp_path / "job-study"
    monkeypatch.setenv("LAMBDAFORGE_STUDY_PATH", str(study_root))
    config = WorkConfig.from_yaml(
        _yaml(tmp_path, {"name": "preprocessing", "run": "tests.work_cases.Producer"})
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert not study_root.exists()


def test_required_constructor_is_rejected_at_class_definition() -> None:
    with pytest.raises(TypeError, match="constructors are framework-owned"):
        type(
            "Invalid",
            (Work,),
            {"__init__": lambda self, required: None, "run": lambda self: None},
        )


def test_signature_types_and_old_fields_fail_locally(tmp_path: Path) -> None:
    bad_parameter = _yaml(
        tmp_path,
        {
            "name": "bad",
            "run": "tests.work_cases.CompleteWork",
            "with": {"source": {"file": "missing"}, "counter": 2},
        },
    )
    errors = " ".join(WorkConfig.validate_file(bad_parameter).errors)
    assert "Unknown parameter 'counter'" in errors
    assert "missing" in errors
    missing = WorkConfig.validate_file(
        _yaml(tmp_path, {"name": "missing", "run": "tests.work_cases.Consumer"})
    )
    assert "Missing required parameter" in " ".join(missing.errors)
    wrong_type = WorkConfig.validate_file(
        _yaml(
            tmp_path,
            {
                "name": "wrong-type",
                "run": "tests.work_cases.CompleteWork",
                "with": {"source": {"file": __file__}, "count": "three"},
            },
        )
    )
    assert "expects int" in " ".join(wrong_type.errors)
    with pytest.raises(ValueError, match="Unknown top-level"):
        WorkConfig.from_mapping({"name": "legacy", "kind": "task", "run": "x.Y"})


def test_runtime_outputs_metrics_inputs_and_immutable_views(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "work-test-project"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    source = tmp_path / "input.txt"
    source.write_text("evidence", encoding="utf-8")
    config = WorkConfig.from_yaml(
        _yaml(
            tmp_path,
            {
                "name": "complete",
                "run": "tests.work_cases.CompleteWork",
                "with": {"source": {"file": "input.txt"}, "count": 3},
            },
        )
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    run = result.runs[0]
    assert run.primary_result == {"count": 3}
    assert run.outputs == {"numbers": [0, 2, 4]}
    assert run.metrics == {"score": 6.0}
    assert run.inputs[0].sha256
    assert (run.run_dir / run.artifacts[0].path).read_text(encoding="utf-8") == "evidence"
    execution = json.loads((result.execution_dir / "execution.json").read_text(encoding="utf-8"))
    assert execution["consumer_package"] == {
        "name": "work-test-project",
        "version": "1.2.3",
    }
    assert execution["lambdaforge_version"] == "0.13.0"
    with pytest.raises(TypeError):
        config.raw["name"] = "changed"  # type: ignore[index]

    immutable = WorkRunner().run(
        WorkConfig.from_yaml(
            _yaml(
                tmp_path,
                {
                    "name": "immutable",
                    "run": "tests.work_cases.ImmutableWork",
                    "with": {"source": {"file": "input.txt"}, "count": 3},
                },
            )
        ),
        rerun=True,
    )
    assert immutable.runs[0].primary_result == {
        "inputs": True,
        "parameters": True,
        "resources": True,
    }


def test_work_publishes_exact_job_level_result_for_control_plane_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_result = tmp_path / "job" / "result.json"
    monkeypatch.setenv("LAMBDAFORGE_JOB_RESULT_PATH", str(job_result))
    config = WorkConfig.from_mapping(
        {"name": "job-result", "run": "tests.work_cases.Producer"},
        source=tmp_path / "work.yaml",
    )

    result = WorkRunner().run(config)

    persisted = json.loads(job_result.read_text(encoding="utf-8"))
    assert result.status == "succeeded"
    assert persisted["execution_id"] == result.execution_id
    assert persisted["status"] == "succeeded"


def test_print_and_managed_work_log_are_captured_and_flushed(tmp_path: Path) -> None:
    result = WorkRunner().run(
        WorkConfig.from_yaml(
            _yaml(tmp_path, {"name": "logging", "run": "tests.work_cases.LoggingWork"})
        )
    )

    assert result.status == "succeeded"
    text = (result.runs[0].run_dir / "work.log").read_text(encoding="utf-8")
    assert "ordinary print is captured" in text
    assert "[INFO] managed message" in text
    assert "[WARNING] a warning" in text


def test_map_rejects_duplicate_stable_keys(tmp_path: Path) -> None:
    result = WorkRunner().run(
        WorkConfig.from_yaml(
            _yaml(tmp_path, {"name": "duplicate-map", "run": "tests.work_cases.DuplicateMapWork"})
        )
    )
    assert result.status == "failed"
    assert "keys must be unique" in result.runs[0].failure["message"]

    unsafe = WorkRunner().run(
        WorkConfig.from_yaml(
            _yaml(tmp_path, {"name": "unsafe-json", "run": "tests.work_cases.UnsafeJsonWork"})
        )
    )
    assert unsafe.status == "failed"
    assert "Out of range float" in unsafe.runs[0].failure["message"]


def test_sequence_parallel_references_seeds_and_hpo(tmp_path: Path) -> None:
    config = WorkConfig.from_yaml(
        _yaml(
            tmp_path,
            {
                "name": "study",
                "steps": [
                    {"name": "prepare", "run": "tests.work_cases.Producer"},
                    {
                        "parallel": [
                            {
                                "name": "consume",
                                "run": "tests.work_cases.Consumer",
                                "with": {"number": {"from": "prepare.number"}},
                            },
                            {
                                "name": "seeds",
                                "run": "tests.work_cases.SeedWork",
                                "seeds": [2, 3],
                                "search": {
                                    "strategy": "exhaustive",
                                    "scale": {"values": [1.0, 2.0]},
                                },
                                "objective": {"metric": "score", "mode": "max"},
                            },
                        ]
                    },
                ],
            },
        )
    )
    assert config.planned_runs == 6
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    assert len(result.runs) == 6
    assert result.runs[1].primary_result == {"consumed": 4}
    assert result.summary["best"]["value"] == 5.0
    assert result.summary["best"]["seeds"] == [2, 3]
    assert len(result.summary["candidates"]) == 2


def test_failed_run_retries_same_run_and_resumes_checkpoint(tmp_path: Path) -> None:
    source = _yaml(tmp_path, {"name": "resume", "run": "tests.work_cases.ResumeWork"})
    config = WorkConfig.from_yaml(source)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    assert first.runs[0].failure["type"] == "RuntimeError"
    assert first.runs[0].failure["diagnostic"]["category"] == "execution"
    source.write_text("name: changed\nrun: not.a.RealWork\n", encoding="utf-8")
    stored = ResultStore(tmp_path / ".lambdaforge" / "runs").configuration(first.execution_id)
    second = WorkRunner().run(stored)
    assert second.status == "succeeded"
    assert second.runs[0].attempt_number == 2
    assert second.runs[0].resumed_from_checkpoint
    assert json.loads((second.execution_dir / "result.json").read_text())["status"] == "succeeded"


def test_ambiguous_output_reference_is_rejected(tmp_path: Path) -> None:
    report = WorkConfig.validate_file(
        _yaml(
            tmp_path,
            {
                "name": "ambiguous",
                "steps": [
                    {"name": "many", "run": "tests.work_cases.SeedWork", "seeds": [1, 2]},
                    {
                        "name": "consumer",
                        "run": "tests.work_cases.Consumer",
                        "with": {"number": {"from": "many.score"}},
                    },
                ],
            },
        )
    )
    assert not report.valid
    assert "ambiguous" in " ".join(report.errors)


def test_work_publishes_streamed_dataset_and_registry_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "datasets.json"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    config = WorkConfig.from_yaml(
        _yaml(tmp_path, {"name": "publish", "run": "tests.work_cases.PublishDataset"})
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    dataset = result.runs[0].datasets["dataset"]
    assert dataset["name"] == "test-work-dataset"
    assert dataset["sample_count"] == 1
    from lambdaforge.data.DatasetRegistry import DatasetRegistry

    restored = DatasetRegistry(tmp_path / "datasets.json").get("test-work-dataset@1")
    assert restored.dataset_id == dataset["dataset_id"]


def test_local_execution_delete_is_preview_first_idempotent_and_bounded(tmp_path: Path) -> None:
    config = WorkConfig.from_yaml(
        _yaml(tmp_path, {"name": "deletable", "run": "tests.work_cases.Producer"})
    )
    result = WorkRunner().run(config)
    store = ResultStore(tmp_path / ".lambdaforge" / "runs")
    preview = store.delete(result.execution_id)
    assert preview["applied"] is False
    assert result.execution_dir.is_dir()
    applied = store.delete(result.execution_id, apply=True)
    assert applied["applied"] is True
    assert not result.execution_dir.exists()
    assert store.delete(result.execution_id, apply=True)["already_deleted"] is True

    outside = tmp_path / "valuable"
    outside.mkdir()
    malicious = tmp_path / ".lambdaforge/runs/bad/execution-fake"
    malicious.mkdir(parents=True)
    (malicious / "result.json").symlink_to(outside / "result.json")
    with pytest.raises(KeyError):
        store.delete("execution-fake", apply=True)


def test_local_result_store_exposes_logs_source_and_corruption(tmp_path: Path) -> None:
    config_path = _yaml(tmp_path, {"name": "local-operations", "run": "tests.work_cases.Producer"})
    result = WorkRunner().run(WorkConfig.from_yaml(config_path))
    store = ResultStore(tmp_path / ".lambdaforge" / "runs")

    assert store.source(result.execution_id) == config_path.resolve()
    assert store.logs(result.execution_id) == ""

    manifest = result.execution_dir / "result.json"
    manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Corrupt Work result manifest"):
        store.list()


def test_result_comparison_uses_persisted_run_metrics(tmp_path: Path) -> None:
    config_path = _yaml(
        tmp_path,
        {
            "name": "comparison",
            "run": "tests.work_cases.SeedWork",
            "with": {"scale": 2.0},
            "seeds": [2, 4],
        },
    )
    first = WorkRunner().run(WorkConfig.from_yaml(config_path))
    config_path.write_text(config_path.read_text().replace("2.0", "3.0"), encoding="utf-8")
    second = WorkRunner().run(WorkConfig.from_yaml(config_path))

    comparison = ResultStore(tmp_path / ".lambdaforge" / "runs").compare(
        (first.execution_id, second.execution_id), metric="score", mode="max"
    )
    assert comparison["ranking"][0]["execution_id"] == second.execution_id
    assert comparison["ranking"][0]["mean"] == 9.0
