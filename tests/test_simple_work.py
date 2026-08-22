"""Function-first authoring and runtime integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import lambdaforge as lf
from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.diagnostics import LambdaForgeError
from lambdaforge.tasks.TaskResult import TaskStatus
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.workflows.Workflow import Workflow


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_callable_file_metrics_artifact_and_final_return(tmp_path: Path) -> None:
    source = tmp_path / "number.txt"
    source.write_text("4", encoding="utf-8")
    config = _write(
        tmp_path / "work.yaml",
        {
            "name": "ordinary-python",
            "run": "tests.fixtures.SimpleWork.process",
            "with": {"source": {"file": "number.txt"}, "factor": 3},
            "resources": {"cpu": 1, "memory": "32MB"},
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = TaskRun.from_yaml(config).run()
    assert not isinstance(result, list)
    assert result.status is TaskStatus.OK
    assert result.outputs == {"value": 12, "seed": None}
    assert result.metrics == {"value": 12}
    assert {artifact.name for artifact in result.artifacts} == {"metrics", "value"}
    records = [
        json.loads(line)
        for line in (Path(result.run_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert records == [{"name": "value", "split": None, "step": 0, "value": 12}]
    materialized = AuthoringConfig.from_yaml(config).materialize()
    assert materialized.kind is ConfigurationKind.TASK
    assert materialized.values["inputs"][0]["path"] == "number.txt"


def test_runtime_api_fails_outside_a_run() -> None:
    with pytest.raises(RuntimeError, match="active LambdaForge run"):
        lf.current()
    with pytest.raises(RuntimeError, match="active LambdaForge run"):
        lf.metric("loss", 1.0)


def test_streaming_dataset_publication_is_registered_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "number.txt"
    source.write_text("content", encoding="utf-8")
    registry = tmp_path / "state" / "datasets.json"
    publication = tmp_path / "datasets"
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(registry))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(publication))
    config = _write(
        tmp_path / "publish.yaml",
        {
            "name": "publish",
            "run": "tests.fixtures.SimpleWork.publish",
            "with": {"source": {"file": "number.txt"}},
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = TaskRun.from_yaml(config).run()
    assert result.status is TaskStatus.OK
    record = DatasetRegistry(registry).get("simple-members@1")
    assert record.sample_count == 1
    root = Path(record.placements[0].root)
    assert (root / "index.jsonl").is_file()
    assert (root / "dataset-artifact.json").is_file()


def test_managed_dataset_argument_resolves_to_a_path_and_keeps_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "member.txt").write_text("content", encoding="utf-8")
    registry_path = tmp_path / ".lambdaforge" / "datasets.json"
    registry = DatasetRegistry(registry_path)
    record = DatasetPublisher(registry).publish_members(
        "managed",
        "1",
        ({"id": "one", "assets": {"data": "member.txt"}},),
        source_root=source_root,
        publication_root=tmp_path / "datasets",
        build_provenance={"identity": {"fixture": 1}},
    )
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(registry_path))
    config = _write(
        tmp_path / "consume.yaml",
        {
            "name": "consume",
            "run": "tests.fixtures.SimpleWork.inspect_dataset",
            "with": {"dataset": {"dataset": "managed@1"}},
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = TaskRun.from_yaml(config).run()
    assert result.status is TaskStatus.OK
    assert result.outputs == {"manifest": True}
    inputs = result.metadata["inputs"]
    assert inputs[0]["dataset_reference"] == "dataset:managed@1"
    assert inputs[0]["identity"]["value"] == record.dataset_id


def test_dataset_publication_keeps_portable_member_paths_collision_free(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "member.txt").write_text("content", encoding="utf-8")
    record = DatasetPublisher(DatasetRegistry(tmp_path / "datasets.json")).publish_members(
        "portable",
        "1",
        (
            {"id": "a/b", "path": "member.txt"},
            {"id": "a-b", "path": "member.txt"},
        ),
        source_root=source,
        publication_root=tmp_path / "published",
        build_provenance={"identity": {"fixture": 1}},
    )
    root = Path(record.placements[0].root)
    paths = {
        json.loads(line)["assets"]["data"]["path"]
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert len(paths) == 2


def test_runtime_and_dataset_assets_reject_symlink_traversal(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "unsafe.yaml",
        {
            "name": "unsafe-artifact",
            "run": "tests.fixtures.SimpleWork.symlink_artifact",
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = TaskRun.from_yaml(config).run()
    assert result.status is TaskStatus.FAILED
    assert "symbolic links" in result.error["message"]

    source = tmp_path / "source"
    source.mkdir()
    (source / "actual.txt").write_text("content", encoding="utf-8")
    (source / "linked.txt").symlink_to(source / "actual.txt")
    with pytest.raises(ValueError, match="symbolic link"):
        DatasetPublisher(DatasetRegistry(tmp_path / "datasets.json")).publish_members(
            "unsafe",
            "1",
            ({"id": "one", "path": "linked.txt"},),
            source_root=source,
            publication_root=tmp_path / "published",
            build_provenance={"identity": {"fixture": 1}},
        )
    with pytest.raises(ValueError, match="path-free"):
        DatasetPublisher(DatasetRegistry(tmp_path / "datasets.json")).publish_members(
            "../escape",
            "1",
            (),
            source_root=source,
            publication_root=tmp_path / "published",
            build_provenance={"identity": {"fixture": 1}},
        )


def test_steps_parallel_seeds_and_finite_search_compile_to_one_existing_dag(
    tmp_path: Path,
) -> None:
    steps = _write(
        tmp_path / "steps.yaml",
        {
            "name": "pipeline",
            "steps": [
                {"name": "first", "run": "tests.fixtures.SimpleWork.identity"},
                {
                    "parallel": [
                        {
                            "name": "left",
                            "run": "tests.fixtures.SimpleWork.identity",
                            "with": {"value": 2},
                        },
                        {
                            "name": "right",
                            "run": "tests.fixtures.SimpleWork.identity",
                            "with": {"value": 3},
                        },
                    ]
                },
                {"name": "last", "run": "tests.fixtures.SimpleWork.identity"},
            ],
            "output_root": str(tmp_path / "workflows"),
        },
    )
    plan = Workflow.from_yaml(steps).inspect()
    assert plan.levels == (("first",), ("left", "right"), ("last",))

    study = _write(
        tmp_path / "study.yaml",
        {
            "name": "study",
            "run": "tests.fixtures.SimpleWork.identity",
            "seeds": [7, 17],
            "search": {"value": {"values": [1, 2]}},
        },
    )
    materialized = AuthoringConfig.from_yaml(study).materialize()
    assert materialized.kind is ConfigurationKind.WORKFLOW
    assert len(materialized.values["nodes"]) == 4


def test_seeded_parameter_study_executes_and_selects_the_objective(tmp_path: Path) -> None:
    study = _write(
        tmp_path / "study.yaml",
        {
            "name": "study",
            "run": "tests.fixtures.SimpleWork.score",
            "with": {"value": 1},
            "seeds": [7, 17],
            "search": {"value": {"values": [1, 3]}},
            "objective": {"metric": "score", "mode": "max"},
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = Workflow.from_yaml(study).run()
    assert result.status == "ok"
    assert result.summary["best"]["value"] == 3
    assert result.summary["evaluated"] == 4


def test_explicit_class_form_constructs_only_during_execution(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "class.yaml",
        {
            "name": "class-work",
            "run": {
                "class": "tests.fixtures.SimpleWork.Multiplier",
                "init": {"factor": 4},
                "method": "calculate",
                "with": {"value": 3},
            },
            "output_root": str(tmp_path / "runs"),
        },
    )
    result = TaskRun.from_yaml(config).run()
    assert result.status is TaskStatus.OK
    assert result.outputs == {"result": 12}


def test_legacy_task_remains_materializable(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "legacy.yaml",
        {
            "schema_version": "1.0",
            "kind": "task",
            "name": "legacy",
            "task": {"target": "tests.fixtures.UserTask.UserTask", "params": {"message": "ok"}},
        },
    )
    assert AuthoringConfig.from_yaml(path).materialize().kind is ConfigurationKind.TASK


def test_same_simple_work_is_refused_only_on_the_same_active_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(
        tmp_path / "work.yaml",
        {
            "name": "duplicate",
            "run": "tests.fixtures.SimpleWork.identity",
            "code_version": "test",
        },
    )
    catalog = ClusterCatalog(
        {
            name: ClusterProfile(
                name,
                transport="ssh",
                scheduler="local",
                host=f"{name}.invalid",
                workspace=f"/work/{name}",
            )
            for name in ("a", "b")
        }
    )
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"))

    class Process:
        pid = 1

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: Process())
    service = SubmissionService(catalog, jobs)
    assert service.enqueue(config, cluster="a").cluster == "a"
    with pytest.raises(LambdaForgeError):
        service.enqueue(config, cluster="a")
    assert service.enqueue(config, cluster="b").cluster == "b"
