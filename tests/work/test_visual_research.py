"""Synthetic ETL→native Dataset→two-input model→native sweep, outside any GUI."""

import csv
import json

import pytest

from lambdaforge.data import DatasetRegistry
from lambdaforge.work import WorkConfig, WorkRunner


@pytest.mark.parametrize("design", ["sweep", "search"])
def test_ordinary_work_lifecycle_for_visual_operations(tmp_path, monkeypatch, design):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "datasets.json"))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="visual-operations-test"\nversion="1"\n'
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "samples.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "x0", "x1", "y"])
        writer.writeheader()
        for i in range(30):
            writer.writerow({"id": f"sample-{i}", "x0": i / 30, "x1": i % 2, "y": i / 20})
    (raw / "context.json").write_text(
        json.dumps([{"id": f"sample-{i}", "context": [i % 3, 1]} for i in range(30)])
    )
    prepare = WorkConfig.from_mapping(
        {
            "name": "prepare-visual",
            "run": "examples.visual_research.PrepareSamples",
            "with": {"raw": {"file": "raw"}},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "prepare.yaml",
    )
    result = WorkRunner().run(prepare)
    assert result.status == "succeeded"
    assert DatasetRegistry(tmp_path / "datasets.json").get("visual-demo@1").sample_count == 30
    authored = {
        "name": "train-visual",
        "run": "examples.visual_research.TrainSamples",
        "with": {"dataset": {"dataset": "visual-demo@1"}, "epochs": 1},
        "seeds": [7],
        "objective": {"metric": "val_loss", "mode": "min"},
        "resources": {"cpu": 1},
        "execution": {"max_parallel": 1},
    }
    space = {"activation": ["relu", "tanh"], "use_context": [True, False]}
    if design == "search":
        authored["search"] = {
            "budget": {"candidates": 4, "runs": 4},
            "space": space,
            "replication": {"minimum": 1, "confirmation": []},
        }
    else:
        authored["sweep"] = {"space": space}
    train = WorkConfig.from_mapping(
        authored,
        source=tmp_path / "train.yaml",
    )
    execution = WorkRunner().run(train)
    assert execution.status == "succeeded" and len(execution.runs) == 4
    assert all(run.ok for run in execution.runs)


def test_preparation_checkpoint_recovery_without_partial_publication(tmp_path, monkeypatch):
    from examples.visual_research import create_raw
    from lambdaforge.etl import WriteRecords

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "datasets.json"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "pyproject.toml").write_text('[project]\nname="visual-recovery"\nversion="1"\n')
    create_raw(tmp_path / "raw")
    config = WorkConfig.from_mapping(
        {
            "name": "prepare-visual",
            "run": "examples.visual_research.PrepareSamples",
            "with": {"raw": {"file": "raw"}},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "prepare.yaml",
    )
    original = WriteRecords.__call__
    counts = {"npz": 0, "json": 0}
    failed = False

    def fail_once(self, file, value):
        nonlocal failed
        if self.format == "json" and counts["json"] == 1 and not failed:
            failed = True
            raise RuntimeError("Synthetic interruption before publication")
        counts[self.format] += 1
        return original(self, file, value)

    monkeypatch.setattr(WriteRecords, "__call__", fail_once)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    assert not (tmp_path / "datasets.json").exists()
    second = WorkRunner().run(config)
    assert second.status == "succeeded"
    assert counts == {"npz": 30, "json": 30}
    assert DatasetRegistry(tmp_path / "datasets.json").get("visual-demo@1").sample_count == 30
