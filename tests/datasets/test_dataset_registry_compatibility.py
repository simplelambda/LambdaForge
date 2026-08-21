"""Read-only compatibility regression for a real-world-shaped legacy registry."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lambdaforge.cli import CommandLineInterface
from lambdaforge.data import (
    AmbiguousDatasetVersionError,
    DatasetRegistry,
    DatasetResolver,
    DatasetService,
)
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "dataset-registry-0.9.2.json"


def test_092_dataset_list_show_resolve_and_consume_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_path = tmp_path / ".lambdaforge" / "datasets.json"
    registry_path.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, registry_path)
    original = registry_path.read_bytes()
    registry = DatasetRegistry(registry_path)
    service = DatasetService(registry)

    listed = service.list()
    assert [record.key for record in listed] == ["wisdom-dna@1"]
    shown = service.show(listed[0].key)
    assert shown.dataset_id == "sha256:" + "a" * 64
    assert shown.placements[0].cluster == "citius-ctgpgpu12"
    resolution = DatasetResolver(
        registry, environment="citius-ctgpgpu12"
    ).resolve("dataset:wisdom-dna@1")
    assert resolution.location.uri.endswith("/wisdom-dna/1/aaaaaaaaaaaaaaaa")

    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(registry_path))
    experiment = ExperimentConfig(
        {
            "schema_version": "1.1",
            "experiment": {"name": "consumer"},
            "data": {
                "datamodule": {
                    "target": "builtins.dict",
                    "params": {"root": {"dataset": "wisdom-dna", "version": "1"}},
                }
            },
            "model": {"target": "builtins.dict"},
            "losses": [{"target": "builtins.dict"}],
            "extensions": {"authoring": {"environment": "citius-ctgpgpu12"}},
        },
        source=tmp_path / "experiment.yaml",
    )
    assert experiment["data"]["datamodule"]["params"]["root"].endswith(
        "/wisdom-dna/1/aaaaaaaaaaaaaaaa"
    )
    monkeypatch.chdir(tmp_path)
    assert CommandLineInterface.main(["datasets", "list"]) == 0
    listed_output = capsys.readouterr().out
    assert "wisdom-dna@1" in listed_output
    assert CommandLineInterface.main(["datasets", "show", "wisdom-dna@1"]) == 0
    shown_output = capsys.readouterr().out
    assert "Dataset: wisdom-dna@1" in shown_output
    assert "citius-ctgpgpu12  AVAILABLE" in shown_output
    assert registry_path.read_bytes() == original


def test_registry_never_guesses_an_unversioned_dataset(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "datasets.json")
    first = DatasetService(DatasetRegistry(FIXTURE)).show("wisdom-dna@1")
    registry.register(first)
    registry.register(
        type(first)(
            first.name,
            "2",
            "sha256:" + "c" * 64,
            first.sample_count,
            first.splits,
            "2026-08-21T00:00:00+00:00",
            first.placements,
        )
    )

    with pytest.raises(AmbiguousDatasetVersionError):
        registry.get("wisdom-dna")
