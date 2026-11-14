"""Owned local telemetry and descriptive charts, without any workload or remote I/O."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.analysis.ObservedStudyChart import ObservedStudyChart
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.ProjectContext import ProjectContext


def sample() -> dict[str, Any]:
    return {"objective": {"metric": "score"}, "candidates": [
        {"trial": 1, "mean": 0.2, "n": 1, "parameters": {"lr": 0.03, "width": 4}},
        {"trial": 2, "mean": 0.6, "n": 20, "parameters": {"lr": 0.03, "width": 8}},
        {"trial": 3, "mean": 0.8, "n": 1, "parameters": {"lr": 0.1, "width": 4}},
        {"trial": 4, "mean": None, "n": 0, "best_observed_objective": 1.0,
         "parameters": {"lr": 0.03, "width": 16}},
    ]}


def test_observed_groups_are_equal_weight_and_do_not_include_censored_best() -> None:
    source = sample()
    before = json.dumps(source, sort_keys=True)
    result = ObservedStudyChart.summarize(source, metrics=["score"], x="lr")
    assert len(result["observations"]) == 3
    first = result["records"][0]
    assert first["y"] == pytest.approx(0.4)
    assert first["n"] == 2 and first["trials"] == [1, 2]
    assert first["error"] == pytest.approx(0.282842712474619)
    assert result["records"][1]["error"] is None
    assert json.dumps(source, sort_keys=True) == before


def test_observed_median_ranges_and_pair_groups_preserve_missing_cells() -> None:
    value = sample()
    value["candidates"].append(
        {"trial": 5, "mean": 8.0, "n": 1, "parameters": {"lr": 0.03, "width": 8}}
    )
    result = ObservedStudyChart.summarize(
        value, metrics=["score"], x="lr", center="median", spread="iqr"
    )
    assert result["records"][0]["y"] == 0.6
    assert result["records"][0]["lower"] == pytest.approx(0.4)
    assert result["records"][0]["upper"] == pytest.approx(4.3)
    with pytest.raises(ValueError, match="median"):
        ObservedStudyChart.summarize(value, metrics=["score"], center="median", spread="se")
    pair = ObservedStudyChart.summarize(
        value, metrics=["score"], x="lr", second="width", spread="range"
    )
    assert len(pair["records"]) == 3
    assert not any(p["x"] == 0.1 and p["parameters"]["width"] == 8 for p in pair["records"])


def test_observed_metrics_require_native_completed_support_independently() -> None:
    value = {"objective": {"metric": "score"}, "candidates": [
        {"trial": 1, "mean": 99, "n": 0, "parameters": {"lr": 0.03},
         "diagnostic_metrics": {"loss": {"mean": 0.2, "n": 2}}},
        {"trial": 2, "mean": None, "n": 3, "parameters": {"lr": 0.03},
         "diagnostic_metrics": {"loss": {"mean": 99, "n": 0}}},
    ]}
    result = ObservedStudyChart.summarize(value, metrics=["score", "loss"], x="lr")
    assert len(result["observations"]) == 1
    assert result["observations"][0]["series"] == "loss"
    assert result["records"][0]["y"] == 0.2


@pytest.fixture
def owned_local(tmp_path: Path) -> tuple[JobService, JobRecord, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="fixture"\n[tool.lambdaforge]\nproject_id="fixture-local"\n'
    )
    source = project / "experiments" / "study.yaml"
    source.parent.mkdir()
    source.write_text("name: study\n")
    fingerprint = "sha256:" + "a" * 64
    execution_id = "execution-" + "a" * 16
    execution = project / ".lambdaforge" / "runs" / "study" / execution_id
    execution.mkdir(parents=True)
    (execution / "execution.json").write_text(json.dumps({
        "execution_manifest_version": 1, "execution_id": execution_id, "job_id": "job-owned",
        "scientific_fingerprint": fingerprint, "name": "study", "source": str(source),
    }))
    run = execution / "runs" / "run-owned" / "attempts" / "attempt-0001"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"step":1,"score":0.4}\n')
    record = JobRecord(
        job_id="job-owned", cluster="local", scheduler="local", scheduler_id=None,
        state=JobState.SUCCEEDED, command=(),
        work_dir=str(tmp_path / "provider/jobs/job-owned/work"),
        resources={}, created_at_utc="2026-10-25T00:00:00Z", updated_at_utc="2026-10-25T00:00:00Z",
        bundle_id="bundle", config_path=str(source), metadata={
            "source_config_path": str(source), "scientific_identity": fingerprint,
            "name": "study", **ProjectContext.discover(project).to_dict(),
            "project_root": str(project),
        },
    )
    jobs = JobService.__new__(JobService)
    jobs.catalog = ClusterCatalog({"local": ClusterProfile("local")})
    return jobs, record, run / "metrics.jsonl"


def test_owned_local_scientific_evidence_is_separate_from_provider_directory(owned_local) -> None:
    jobs, record, path = owned_local
    assert str(jobs._owned_study_path(record, str(path))) == str(path)
    for forged in (
        replace(record, metadata={**record.metadata, "scientific_identity": "sha256:" + "b" * 64}),
        replace(record, metadata={**record.metadata, "project_id": "other"}),
    ):
        with pytest.raises(RuntimeError, match="escaped"):
            jobs._owned_study_path(forged, str(path))
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(record, str(path.parents[4] / "another-execution/secret"))
    origin = path.parents[4] / "execution.json"
    value = json.loads(origin.read_text())
    value["job_id"] = None
    origin.write_text(json.dumps(value))
    assert str(jobs._owned_study_path(record, str(path))) == str(path)
    value["job_id"] = "job-other"
    origin.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(record, str(path))


def test_unstamped_deliberate_rerun_is_not_owned_by_matching_fingerprint_alone(
    owned_local,
) -> None:
    jobs, record, path = owned_local
    execution = path.parents[4]
    rerun = execution.with_name(execution.name + "-1234abcd")
    selected = rerun / path.relative_to(execution)
    selected.parent.mkdir(parents=True)
    selected.write_text("measured")
    origin = json.loads((execution / "execution.json").read_text())
    origin.update(job_id=None, execution_id=rerun.name)
    (rerun / "execution.json").write_text(json.dumps(origin))
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(record, str(selected))
    origin["job_id"] = record.job_id
    (rerun / "execution.json").write_text(json.dumps(origin))
    assert str(jobs._owned_study_path(record, str(selected))) == str(selected)

def test_owned_local_evidence_rejects_symlink_and_remote_whitelist(owned_local, tmp_path) -> None:
    jobs, record, path = owned_local
    outside = tmp_path / "outside"
    outside.write_text("secret")
    linked = path.parent / "linked"
    linked.symlink_to(outside)
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(record, str(linked))
    jobs.catalog = SimpleNamespace(get=lambda name: SimpleNamespace(transport="ssh"))
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(record, str(path))
