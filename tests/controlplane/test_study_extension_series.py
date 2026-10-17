"""Accumulated fixed designs retain exact native evidence through successive deltas."""

import pytest
import yaml

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.work import WorkConfig, WorkRunner
from tests.controlplane.test_study_extension_apply import capture_preparation
from tests.controlplane.test_study_extension_preview import execute, extend_yaml, service


@pytest.fixture(name="project")
def fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    return project.__wrapped__(tmp_path, monkeypatch)


def complete(svc, accepted, monkeypatch):
    """Inline only the provider handoff; use the real native scientific runner."""
    record = svc.jobs.get(accepted["job_id"], refresh=False)
    work = svc.project.root / ".lambdaforge/jobs" / record.job_id / "work"
    work.mkdir(parents=True)
    svc.jobs.store.write(record.with_updates(work_dir=str(work)))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", record.job_id)
    monkeypatch.setenv("LAMBDAFORGE_JOB_RESULT_PATH", str(work.parent / "result.json"))
    result = WorkRunner().run(WorkConfig.from_yaml(accepted["snapshot"]))
    assert result.status == "succeeded"
    svc.jobs.store.write(
        svc.jobs.get(record.job_id, refresh=False).with_updates(state=JobState.SUCCEEDED)
    )
    svc.refresh(accepted["study_series_id"])
    return result


def test_retry_preparation_failure_keeps_reviewed_delta_and_series_ownership(project, monkeypatch):
    origin = execute(project)
    saved = {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}
    extend_yaml(project)
    svc = service(project)
    launches = capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    original_job = svc.jobs.get(accepted["job_id"], refresh=False)
    assert original_job.scheduler_id is None
    # This disposable local provider has prepared an owned workspace, but no Execution.
    work = project["root"] / ".lambdaforge/jobs" / original_job.job_id / "work"
    work.mkdir(parents=True)
    original_job = original_job.with_updates(work_dir=str(work))
    svc.jobs.store.write(original_job)
    svc.jobs.fail_preparation(original_job.job_id, ConnectionError("fixture: SSH exec too large"))
    inspection = svc.jobs.retry_preview(original_job.job_id)
    assert inspection["resumable"] is False
    assert "reviewed extension delta only" in inspection["will_retry"]
    handle = svc.jobs.retry(original_job.job_id)
    retry = svc.jobs.get(handle.job_id, refresh=False)
    assert retry.retry_of == original_job.job_id
    assert retry.metadata["study_extension"] == original_job.metadata["study_extension"]
    assert retry.metadata["pinned_scientific_identity"] == original_job.metadata[
        "pinned_scientific_identity"
    ]
    assert "recovery_execution_dir" not in retry.metadata
    assert "prepared_extension_attestation" not in retry.metadata
    assert len(launches) == 2
    with pytest.raises(Exception, match="active|already|duplicate"):
        svc.jobs.retry(original_job.job_id)
    assert len(launches) == 2
    delta = complete(svc, {**accepted, "job_id": handle.job_id}, monkeypatch)
    assert len(delta.runs) == 15
    view = svc.overview(accepted["study_series_id"])
    assert view["counts"]["completed"] == 35 and view["physical_cost"]["spent_attempts"] == 35
    assert view["operations"][0]["recovery_jobs"] == [handle.job_id]
    assert svc.jobs.get(original_job.job_id, refresh=False).state is JobState.FAILED
    assert saved == {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}


def test_twenty_to_thirty_five_to_fifty_six_preserves_every_identity(project, monkeypatch):
    origin = execute(project)
    original_bytes = {
        path: path.read_bytes() for path in origin.execution_dir.rglob("*") if path.is_file()
    }
    extend_yaml(project)
    svc = service(project)
    launches = capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    first = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    pending = svc.overview(first["study_series_id"])
    assert pending["counts"]["completed"] == 20 and pending["counts"]["pending"] == 15
    delta = complete(svc, first, monkeypatch)
    view = svc.overview(first["study_series_id"])
    assert view["counts"] == {"candidates": 7, "required": 35, "completed": 35, "pending": 0}
    original_revision = svc.show(first["study_series_id"])["revisions"][0]
    full = yaml.safe_load(project["path"].read_text())
    full["seeds"].extend([73, 79, 83])
    project["path"].write_text(yaml.safe_dump(full))
    second_plan = svc.preview(first["study_series_id"], project["path"])
    assert second_plan["expected_revision"] == 1
    assert second_plan["comparison"]["counts"] == {"reused": 35, "new": 21, "total": 56}
    assert second_plan["budget"]["historical_attempts"] == 35
    second = svc.apply(
        first["study_series_id"], project["path"], plan_hash=second_plan["plan_hash"]
    )
    last = complete(svc, second, monkeypatch)
    assert len(last.runs) == 21
    observed = {
        (r.trial["parameters"]["strength"], r.seed) for r in (*origin.runs, *delta.runs, *last.runs)
    }
    assert len(observed) == 56
    view = svc.overview(first["study_series_id"])
    assert view["counts"]["completed"] == 56 and view["counts"]["pending"] == 0
    assert view["physical_cost"]["spent_attempts"] == 56
    assert len(launches) == 2
    assert svc.show(first["study_series_id"])["revisions"][0] == original_revision
    analysis = svc.analysis(first["study_series_id"])
    provenance = analysis["study_series"]["provenance"]
    assert len(provenance) == 56
    assert {row["execution_id"] for row in provenance} == {
        origin.execution_id,
        delta.execution_id,
        last.execution_id,
    }
    assert all(row["run_id"] and row["attempt_id"] for row in provenance)
    assert "exploratory" in analysis["study_series"]["interpretation"]
    assert original_bytes == {
        path: path.read_bytes() for path in origin.execution_dir.rglob("*") if path.is_file()
    }
    unchanged = svc.preview(first["study_series_id"], project["path"])
    assert unchanged["comparison"]["counts"]["new"] == 0
    before = svc.show(first["study_series_id"])
    assert (
        svc.apply(first["study_series_id"], project["path"], plan_hash=unchanged["plan_hash"])[
            "state"
        ]
        == "no-op"
    )
    assert svc.show(first["study_series_id"]) == before


def test_conditional_selective_delta_and_successive_revision_use_native_evidence(
    project, monkeypatch
):
    raw = project["raw"]
    raw["seeds"] = [11, 23]
    raw["sweep"]["space"] = {
        "strength": [0.0, 0.1],
        "fail": {"values": [False], "when": {"strength": 0.0}},
    }
    project["path"].write_text(yaml.safe_dump(raw))
    origin = execute(project)
    before = {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}
    svc = service(project)
    capture_preparation(monkeypatch)
    selector = origin.execution_id
    for index, strength in enumerate([0.2, 0.3], 1):
        raw["sweep"].setdefault("include", []).append({"strength": strength})
        project["path"].write_text(yaml.safe_dump(raw))
        plan = svc.preview(selector, project["path"])
        assert plan["comparison"]["counts"] == {
            "reused": 2 + index * 2, "new": 2, "total": 4 + index * 2
        }
        applied = svc.apply(selector, project["path"], plan_hash=plan["plan_hash"])
        selector = applied["study_series_id"]
        delta = complete(svc, applied, monkeypatch)
        assert len(delta.runs) == 2
        assert all(dict(r.trial["parameters"]) == {"strength": strength} for r in delta.runs)
    view = svc.overview(applied["study_series_id"])
    assert view["counts"] == {"candidates": 4, "required": 8, "completed": 8, "pending": 0}
    assert view["physical_cost"]["spent_attempts"] == 8
    analysis = svc.analysis(applied["study_series_id"])
    assert len(analysis["study_series"]["provenance"]) == 8
    assert analysis["search_space"]["fail"]["when"] == {"strength": 0.0}
    assert before == {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}


def test_seed_extension_runs_only_twelve_new_cells(project, monkeypatch):
    origin = execute(project)
    raw = dict(project["raw"], seeds=[11, 23, 47, 59, 71, 73, 79, 83])
    project["path"].write_text(yaml.safe_dump(raw))
    svc = service(project)
    capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    assert plan["comparison"]["counts"] == {"reused": 20, "new": 12, "total": 32}
    applied = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    delta = complete(svc, applied, monkeypatch)
    assert {run.seed for run in delta.runs} == {73, 79, 83}
    assert svc.overview(applied["study_series_id"])["counts"]["completed"] == 32
    from lambdaforge.tui.services import ConsoleServices

    console = object.__new__(ConsoleServices)
    console.catalog, console.jobs, console.results = svc.catalog, svc.jobs, svc.results
    snapshot = {
        "work": {
            "items": [
                {
                    "execution_id": origin.execution_id,
                    "name": "regularization",
                    "study_expected": True,
                },
                {
                    "primary_job_id": applied["job_id"],
                    "name": "regularization",
                    "study_expected": True,
                },
                {
                    "primary_job_id": "job-unrelated",
                    "name": "regularization",
                    "study_expected": True,
                },
            ]
        }
    }
    console._include_study_series(snapshot)
    rows = snapshot["work"]["items"]
    assert len(rows) == 2 and any(row.get("primary_job_id") == "job-unrelated" for row in rows)
    assert any(row.get("study_series_id") == applied["study_series_id"] for row in rows)
    exported = svc.export(applied["study_series_id"], project["root"] / "exports")
    import json
    from pathlib import Path

    output = Path(exported["path"])
    collection = json.loads((output / "collection.json").read_text())
    assert collection["read_only"] and len(collection["executions"]) == 2
    for entry in collection["executions"]:
        assert (output / entry["package"]["manifest"]).is_file()
    assert (output / "analysis.html").is_file() and (output / "analysis.json").is_file()


def test_missing_delta_and_insufficient_cumulative_budget_fail_closed(project, monkeypatch):
    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    applied = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    delta = complete(svc, applied, monkeypatch)
    raw = yaml.safe_load(project["path"].read_text())
    raw["seeds"].append(73)
    raw["execution"] = {"max_runs": 40}
    project["path"].write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="budget|Budget"):
        svc.preview(origin.execution_id, project["path"])
    raw["execution"]["max_runs"] = 60
    project["path"].write_text(yaml.safe_dump(raw))
    (delta.execution_dir / "execution.json").rename(delta.execution_dir / "missing.json")
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        svc.preview(origin.execution_id, project["path"])


def test_crash_before_native_job_is_reconciled_with_same_preallocated_id(project, monkeypatch):
    from lambdaforge.controlplane.SubmissionService import SubmissionService

    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    enqueue = SubmissionService.enqueue

    def killed(*args, **kwargs):
        raise SystemExit("fixture: controller died before native reservation")

    monkeypatch.setattr(SubmissionService, "enqueue", killed)
    with pytest.raises(SystemExit):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    operation = svc.show(plan["study_series_id"])["operations"][0]
    assert operation["state"] == "dispatching" and svc.jobs.list(refresh=False) == ()
    monkeypatch.setattr(SubmissionService, "enqueue", enqueue)
    launches = capture_preparation(monkeypatch)
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert accepted["job_id"] == operation["job_id"]
    assert accepted["reconciled_pre_job_handoff"] and len(launches) == 1
    repeated = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert repeated["job_id"] == accepted["job_id"] and not repeated["created"]
    assert len(launches) == 1


def test_missing_native_job_with_submission_evidence_never_replays(project, monkeypatch):
    from lambdaforge.controlplane.SubmissionService import SubmissionService

    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])

    def killed(*args, **kwargs):
        raise SystemExit("fixture: controller died")

    monkeypatch.setattr(SubmissionService, "enqueue", killed)
    with pytest.raises(SystemExit):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    operation = svc.show(plan["study_series_id"])["operations"][0]
    (svc.jobs.store.root / "submissions" / operation["job_id"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="uncertain replay"):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert svc.jobs.list(refresh=False) == ()


def test_failed_delta_native_retry_keeps_successful_cells_and_attempt_cost(project, monkeypatch):
    import functools

    from lambdaforge.work.config import import_work_class

    project["raw"]["seeds"] = [11, 23]
    project["raw"]["sweep"]["space"]["strength"] = [0.0, 0.1]
    project["raw"]["execution"] = {"failure_retries": 0}
    project["path"].write_text(yaml.safe_dump(project["raw"]))
    origin = execute(project)
    raw = yaml.safe_load(project["path"].read_text())
    raw["sweep"]["space"]["strength"].append(0.2)
    project["path"].write_text(yaml.safe_dump(raw))
    svc = service(project)
    capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    record = svc.jobs.get(accepted["job_id"], refresh=False)
    owned = project["root"] / ".lambdaforge/jobs" / record.job_id / "work"
    owned.mkdir(parents=True)
    svc.jobs.store.write(record.with_updates(work_dir=str(owned)))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", record.job_id)
    monkeypatch.setenv("LAMBDAFORGE_JOB_RESULT_PATH", str(owned.parent / "result.json"))
    config = WorkConfig.from_yaml(accepted["snapshot"])
    work = import_work_class(config.levels[0].runs[0].work_class)
    normal = work.run

    @functools.wraps(normal)
    def transient(self, *args, **kwargs):
        if self.seed == 23:
            raise RuntimeError("disposable fixture: injected resource failure")
        return normal(self, *args, **kwargs)

    monkeypatch.setattr(work, "run", transient)
    first = WorkRunner().run(config)
    assert first.status in {"failed", "completed_with_failures"}
    successful = next(run for run in first.runs if run.status == "succeeded")
    saved = (successful.run_dir / "result.json").read_bytes()
    monkeypatch.setattr(work, "run", normal)
    svc.jobs.store.write(
        svc.jobs.get(record.job_id, refresh=False).with_updates(state=JobState.FAILED)
    )
    # A remote-prepared extension keeps its original Execution pin through native retry.
    failed = svc.jobs.get(record.job_id, refresh=False)
    attestation = {"scientific_identity": first.scientific_fingerprint,
                   "configuration_sha256": "a" * 64, "code_sha256": "b" * 64}
    svc.jobs.store.write(failed.with_updates(metadata={
        **failed.metadata, "prepared_extension_attestation": attestation,
    }))
    svc.refresh(accepted["study_series_id"])
    assert svc.show(accepted["study_series_id"])["operations"][0]["state"] == "failed"
    live = svc.overview(accepted["study_series_id"], include_live=True)
    assert any(row.get("live_unsealed") and row["status"] == "succeeded" for row in live["runs"])
    assert any(row.get("live_unsealed") and row["status"] == "failed" for row in live["runs"])
    assert live["counts"]["completed"] == 4  # Live display is not sealed scientific evidence.
    assert svc.analysis(accepted["study_series_id"])["summary"]["complete_run_count"] == 4
    recovered = svc.jobs.retry(record.job_id)
    retry = svc.jobs.get(recovered.job_id, refresh=False)
    assert retry.metadata["prepared_extension_attestation"] == attestation
    assert retry.metadata["study_extension"]["series_id"] == accepted["study_series_id"]
    assert retry.metadata["recovery_execution_dir"] == str(first.execution_dir)
    owned_retry = project["root"] / ".lambdaforge/jobs" / retry.job_id / "work"
    owned_retry.mkdir(parents=True)
    svc.jobs.store.write(retry.with_updates(work_dir=str(owned_retry)))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", retry.job_id)
    monkeypatch.setenv("LAMBDAFORGE_JOB_RESULT_PATH", str(owned_retry.parent / "result.json"))
    final = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert final.status == "succeeded" and final.execution_id == first.execution_id
    assert (successful.run_dir / "result.json").read_bytes() == saved
    svc.jobs.store.write(
        svc.jobs.get(retry.job_id, refresh=False).with_updates(state=JobState.SUCCEEDED)
    )
    svc.refresh(accepted["study_series_id"])
    view = svc.overview(accepted["study_series_id"])
    assert view["counts"]["completed"] == 6
    assert view["physical_cost"]["spent_attempts"] == 7
    assert view["operations"][0]["recovery_jobs"] == [retry.job_id]
    assert view["operations"][0]["execution_id"] == first.execution_id
