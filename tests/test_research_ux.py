"""Research-facing execution, catalog and explanation contracts for 0.10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.cli import CommandLineInterface
from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
from lambdaforge.configuration.ProjectConfigService import ProjectConfigService
from lambdaforge.controlplane import ClusterCatalog, JobService, JobStore
from lambdaforge.execution import ResourceRequest

ROOT = Path(__file__).resolve().parents[1]


def test_all_document_families_use_the_canonical_local_dry_run(capsys: Any) -> None:
    examples = (
        ROOT / "examples/preprocessing.yaml",
        ROOT / "examples/experiment.yaml",
        ROOT / "examples/workflow.yaml",
        ROOT / "examples/dataset-recipe.yaml",
    )
    for config in examples:
        assert CommandLineInterface.main(["run", str(config), "--dry-run"]) == 0
        assert capsys.readouterr().out.strip()

    dataset = ROOT / "examples/dataset-recipe.yaml"
    assert CommandLineInterface.main(["run", str(dataset), "--dry-run"]) == 0
    canonical = json.loads(capsys.readouterr().out)
    assert CommandLineInterface.main(["datasets", "plan", str(dataset), "--json"]) == 0
    alias = json.loads(capsys.readouterr().out)
    assert canonical["dataset"] == alias["dataset"]
    assert canonical["resources"] == alias["resources"]
    assert canonical["stages"] == alias["stages"]


def test_experiment_catalog_is_derived_from_config_and_jobs(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    config = tmp_path / "baseline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.1",
                "experiment": {"name": "baseline", "seeds": [7, 17, 27]},
                "data": {},
                "model": {"target": "tests.fixtures.UserModel.UserModel"},
                "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
                "extensions": {"code_version": "fixture-revision"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    descriptor = ConfigurationDescriptor.from_path(config)
    jobs = JobService(ClusterCatalog.load(), JobStore(tmp_path / "jobs"))
    jobs.reserve(
        cluster="local",
        resources=ResourceRequest(),
        config_path=config,
        metadata=descriptor.metadata(),
        job_type="experiment",
    )

    record = ProjectConfigService(tmp_path, jobs=jobs).list(kind="experiment")[0]
    assert record.name == "baseline"
    assert record.scientific_revision == descriptor.revision
    assert record.state == "preparing"
    assert record.planned_runs == 3
    assert record.attempt_count == 1

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "empty-state"))
    assert CommandLineInterface.main(["experiments", "--root", str(tmp_path), "list"]) == 0
    rendered = capsys.readouterr().out
    assert "EXPERIMENT" in rendered and "baseline" in rendered


def test_explain_config_reports_scientific_intent_without_importing_targets(
    tmp_path: Path, capsys: Any
) -> None:
    config = tmp_path / "study.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "study",
                "code_version": "fixture-revision",
                "data": {},
                "model": "consumer.models.NotInstalledYet",
                "loss": "consumer.losses.CustomLoss",
                "optimizer": {"type": "adamw", "lr": 0.0003},
                "trainer": {"epochs": 12},
                "experiment": {"seeds": [7, 17]},
                "resources": {"cpus": 4, "memory": "8GiB", "gpus": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert CommandLineInterface.main(["explain", str(config), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "study"
    assert payload["model"]["target"] == "consumer.models.NotInstalledYet"
    assert payload["training"]["max_epochs"] == 12
    assert payload["seeds"] == [7, 17]
    assert payload["planned_work"] == {"total": 2, "unit": "runs"}
    assert payload["resources"]["gpu_count"] == 1
