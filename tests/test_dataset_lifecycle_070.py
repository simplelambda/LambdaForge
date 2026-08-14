"""Acceptance coverage for the LambdaForge 0.7 dataset lifecycle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from lambdaforge.controlplane import JobHandle, JobState
from lambdaforge.data import (
    DatasetBuildService,
    DatasetIndex,
    DatasetRecipeConfig,
    DatasetRecord,
    DatasetRegistry,
    DatasetResolver,
    DatasetService,
    InvalidDatasetBuildError,
)
from lambdaforge.preprocessing import DatasetArtifact
from lambdaforge.tasks import TaskConfig, TaskRun


def _task(
    path: Path,
    stage: str,
    *,
    upstream: str | None = None,
    marker: str = "v1",
    fail: bool = False,
) -> None:
    params: dict[str, object] = {"stage": stage, "marker": marker, "fail": fail}
    if upstream is not None:
        params["upstream"] = upstream
    path.write_text(
        yaml.safe_dump(
            {
                "kind": "task",
                "schema_version": "1.0",
                "name": stage,
                "required_artifacts": ["dataset"],
                "task": {
                    "target": "tests.fixtures.DatasetPipelineTask.DatasetPipelineTask",
                    "params": params,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _recipe(root: Path, *, version: str = "1", fail_features: bool = False) -> Path:
    _task(root / "roster.yaml", "roster")
    _task(
        root / "features.yaml",
        "features",
        upstream="BOUND",
        fail=fail_features,
    )
    _task(root / "annotation.yaml", "annotation", upstream="BOUND")
    path = root / f"dataset-{version}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "kind": "dataset",
                "dataset": {
                    "name": "generic-records",
                    "version": version,
                    "target_schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                        "required": ["score"],
                    },
                    "global_assets": {"vocabulary": "vocabulary.json"},
                },
                "output_root": str(root / "dataset-runs"),
                "stages": {
                    "roster": {"task": "roster.yaml"},
                    "features": {
                        "task": "features.yaml",
                        "needs": ["roster"],
                        "bindings": {
                            "task.params.upstream": "${nodes.roster.artifacts.dataset}"
                        },
                    },
                    "annotation": {
                        "task": "annotation.yaml",
                        "needs": ["features"],
                        "bindings": {
                            "task.params.upstream": "${nodes.features.artifacts.dataset}"
                        },
                    },
                },
                "publish": {"from": "annotation", "root": "dataset", "index": "members.jsonl"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_recipe_build_reuse_changed_downstream_force_failure_and_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry.json"))
    recipe_path = _recipe(tmp_path)
    recipe = DatasetRecipeConfig.from_yaml(recipe_path)
    service = DatasetBuildService(DatasetRegistry(tmp_path / "registry.json"))
    first = service.build(recipe)
    assert first.status == "ok"
    record = service.registry.get("generic-records@1")
    assert record.sample_count == 2
    assert record.partitions["split"] == {"test": 1, "train": 1}

    repeated = service.plan(recipe)
    assert [stage.action for stage in repeated.stages] == ["REUSE", "REUSE", "REUSE"]
    assert repeated.publish_action == "NOOP"

    annotation = yaml.safe_load((tmp_path / "annotation.yaml").read_text(encoding="utf-8"))
    annotation["task"]["params"]["marker"] = "v2"
    (tmp_path / "annotation.yaml").write_text(
        yaml.safe_dump(annotation, sort_keys=False), encoding="utf-8"
    )
    changed = service.plan(DatasetRecipeConfig.from_yaml(recipe_path))
    assert [stage.action for stage in changed.stages] == ["REUSE", "REUSE", "EXECUTE"]
    forced = service.plan(recipe, force_stages=("roster",))
    assert [stage.action for stage in forced.stages] == ["EXECUTE", "EXECUTE", "EXECUTE"]

    failed_recipe = DatasetRecipeConfig.from_yaml(
        _recipe(tmp_path, version="2", fail_features=True)
    )
    failed = service.build(failed_recipe)
    assert failed.status == "failed"
    assert all(record.version != "2" for record in service.registry.records())

    resolution = DatasetResolver(
        service.registry, environment="local", source_dir=tmp_path
    ).resolve("dataset:generic-records@1")
    assert resolution.identity["dataset_id"] == record.dataset_id


def test_v2_identity_is_path_independent_and_build_provenance_is_separate(tmp_path: Path) -> None:
    recipe = DatasetRecipeConfig.from_yaml(_recipe(tmp_path))
    result = DatasetBuildService(DatasetRegistry(tmp_path / "registry.json")).build(recipe)
    assert result.record is not None
    root = Path(result.record.placements[0].root)
    artifact = DatasetArtifact.read_json(root / "dataset-artifact.json")
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    copied_artifact = DatasetArtifact.read_json(copied / "dataset-artifact.json")
    assert copied_artifact.content_id == artifact.content_id

    index = DatasetIndex(copied / str(artifact.index["path"]))
    alternate = DatasetArtifact.create_v2(
        name=artifact.name,
        version=artifact.version,
        index=index,
        index_path=str(artifact.index["path"]),
        global_assets=artifact.global_assets,
        build_provenance={"recipe_fingerprint": "sha256:" + "f" * 64},
    )
    assert alternate.content_id == artifact.content_id
    assert alternate.build_id != artifact.build_id


def test_preprocessing_no_longer_publishes_implicitly(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"id": "one"}) + "\n", encoding="utf-8")
    config = TaskConfig(
        {
            "kind": "task",
            "schema_version": "1.0",
            "name": "plain-preprocessing",
            "output_root": str(tmp_path / "runs"),
            "inputs": [{"name": "raw", "path": str(source)}],
            "task": {
                "target": "lambdaforge.preprocessing.PreprocessingTask",
                "params": {
                    "source": {
                        "target": "lambdaforge.preprocessing.JsonLinesSource",
                        "params": {"path": str(source), "key_field": "id"},
                    },
                    "transforms": [],
                    "sink": {
                        "target": "lambdaforge.preprocessing.JsonDirectorySink",
                        "params": {"output_dir": "processed"},
                    },
                },
            },
        }
    )
    result = TaskRun(config).run()
    assert not (Path(result.run_dir) / "dataset-artifact.json").exists()  # type: ignore[union-attr]


def test_materialize_build_submits_recipe_and_present_version_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry.json"))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    recipe_path = _recipe(tmp_path)
    registry = DatasetRegistry(tmp_path / "registry.json")
    service = DatasetService(registry)

    submitted: list[tuple[str, str]] = []

    def submit(
        self: DatasetBuildService,
        recipe: DatasetRecipeConfig,
        *,
        cluster: str,
        force: bool = False,
        force_stages=(),
        dry_run: bool = False,
    ) -> JobHandle:
        del self, force, force_stages, dry_run
        submitted.append((recipe.selector, cluster))
        return JobHandle("job-20260814000000-12345678", cluster, JobState.QUEUED, "1")

    monkeypatch.setattr(DatasetBuildService, "submit", submit)
    plan = service.materialize("generic-records@1", cluster="local", apply=True)
    assert plan.action == "BUILD"
    assert plan.job_id == "job-20260814000000-12345678"
    assert submitted == [("generic-records@1", "local")]

    result = DatasetBuildService(registry).build(DatasetRecipeConfig.from_yaml(recipe_path))
    assert result.status == "ok"
    noop = service.materialize("generic-records@1", cluster="local", apply=True)
    assert noop.action == "NOOP"
    assert noop.job_id is None


def test_registry_rejects_one_alias_for_different_content(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "registry.json")
    common = ("immutable", "1")
    registry.register(
        DatasetRecord(
            *common,
            "sha256:" + "1" * 64,
            0,
            {},
            "2026-08-14T00:00:00+00:00",
        )
    )
    with pytest.raises(InvalidDatasetBuildError, match="different immutable identity"):
        registry.register(
            DatasetRecord(
                *common,
                "sha256:" + "2" * 64,
                0,
                {},
                "2026-08-14T00:00:01+00:00",
            )
        )
