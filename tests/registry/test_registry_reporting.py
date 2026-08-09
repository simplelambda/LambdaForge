"""Focused registry query/export/comparison/report/dashboard tests."""

from pathlib import Path

import pytest

from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.registry import (
    ExperimentComparator,
    ExperimentRegistry,
    LocalDashboard,
    RegistryQuery,
    ReportBuilder,
)


def write_result(root: Path, name: str, value: float, tag: str) -> None:
    run = root / name
    run.mkdir(parents=True)
    RunResult(
        name=name,
        run_dir=run,
        status=RunStatus.OK,
        final_metrics={"score": value},
        attempt_id=f"attempt-{name}",
        config_fingerprint=f"sha256:{name}",
    ).write_json(run / "result.json")
    (run / "config.yaml").write_text(
        f"metadata:\n  tags: [{tag}]\nmodel:\n  width: {value}\n", encoding="utf-8"
    )


def test_registry_is_catalog_backed_and_reports_objective_data(tmp_path: Path) -> None:
    write_result(tmp_path, "a", 0.7, "baseline")
    write_result(tmp_path, "b", 0.9, "candidate")
    registry = ExperimentRegistry(tmp_path)
    baseline = registry.query(RegistryQuery.create(tags=["baseline"]))
    candidate = registry.query(RegistryQuery.create(tags=["candidate"]))
    assert len(baseline) == len(candidate) == 1
    assert registry.export(tmp_path / "registry.csv").is_file()
    comparison = ExperimentComparator().compare(
        {"baseline": baseline, "candidate": candidate}, metric="score"
    )
    assert comparison["mean_effects"]["baseline - candidate"] == pytest.approx(-0.2)
    assert ReportBuilder().write(comparison, tmp_path / "report.md").is_file()
    assert (tmp_path / "report-means.png").is_file()
    dashboard = LocalDashboard().build(tmp_path, tmp_path / "dashboard.html")
    assert "Read-only snapshot" in dashboard.read_text(encoding="utf-8")
