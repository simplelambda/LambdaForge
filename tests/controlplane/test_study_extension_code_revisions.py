"""Revisioned extension is not an equivalence waiver or a replay of completed Runs."""

import json

import pytest
import yaml

from lambdaforge.work.extension_compatibility import code_review_pin
from tests.controlplane.test_study_extension_apply import capture_preparation
from tests.controlplane.test_study_extension_preview import commit, execute, extend_yaml, service
from tests.controlplane.test_study_extension_series import complete


@pytest.fixture(name="project")
def fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    return project.__wrapped__(tmp_path, monkeypatch)


def test_placement_independent_contract_never_changes_native_seed_identity(project):
    from lambdaforge.controlplane.StudyExtensionDispatch import native_delta_contract
    from lambdaforge.work import WorkConfig, WorkRunner

    config = WorkConfig.from_yaml(project["path"])
    original = WorkRunner._study_identity(config)
    design = json.dumps(config.levels[0].runs[0].study_design.to_dict(), sort_keys=True)
    contract = native_delta_contract(config)
    assert "namespace" not in json.dumps(contract["study_design"]["seed_source"])
    assert contract["seeds"] == project["raw"]["seeds"]
    assert WorkRunner._study_identity(config) == original
    assert json.dumps(config.levels[0].runs[0].study_design.to_dict(), sort_keys=True) == design


def test_dirty_origin_accepts_reviewed_current_code_and_keeps_inference_separate(
    project, monkeypatch
):
    commit(project)
    project["code"].write_text(project["code"].read_text() + "\n# original uncommitted code\n")
    origin = execute(project)
    preserved = {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}
    extend_yaml(project)
    project["code"].write_text(project["code"].read_text().replace(
        "score = 1.0 - strength", "score = 100.0 - strength"
    ))
    # Disposable inline runner: mirror the fresh scientific import in a real native worker.
    import importlib
    import sys

    importlib.reload(sys.modules[project["code"].parent.name])
    svc = service(project)
    launches = capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    assert plan["apply_supported"] and plan["code_compatibility"]["kind"] == "revisioned-current"
    assert not plan["code_compatibility"]["equivalent"]
    assert plan["comparison"]["counts"] == {"reused": 20, "new": 15, "total": 35}
    pin = plan["code_compatibility"]["review_pin"]
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    provisional = svc.analysis(accepted["study_series_id"])
    assert provisional["pooled_inference"] is False
    assert code_review_pin(svc.project) == pin  # Native generated snapshot is attested/ignored.
    assert svc.jobs.get(accepted["job_id"], refresh=False).metadata["study_extension"][
        "code_review_pin"
    ] == pin
    delta = complete(svc, accepted, monkeypatch)
    assert len(delta.runs) == 15 and len(launches) == 1
    assert all(run.metrics["score"] > 90 for run in delta.runs)
    view = svc.overview(accepted["study_series_id"])
    assert view["mixed_code"] and view["counts"]["completed"] == 35
    assert {r["code_cohort"] for r in view["runs"]} == {"revision-0", "revision-1"}
    result = svc.analysis(accepted["study_series_id"])
    assert result["pooled_inference"] is False
    cohorts = result["code_cohort_analyses"]
    assert len(cohorts["revision-0"]["study_series"]["provenance"]) == 20
    assert len(cohorts["revision-1"]["study_series"]["provenance"]) == 15
    assert len(result["study_series"]["provenance"]) == 15
    assert preserved == {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}
    assert not svc.apply(
        origin.execution_id, project["path"], plan_hash=plan["plan_hash"]
    )["created"]
    from lambdaforge.analysis.Report import write_html

    report = write_html(result, project["root"] / "cohorts.html").read_text()
    assert "cross-revision inference disabled" in report
    assert "Code · revision-0" in report and "Code · revision-1" in report
    assert "Native LambdaForge analysis" in report


def test_seed_append_never_pools_two_code_revisions_into_a_single_mean(project, monkeypatch):
    origin = execute(project)
    project["code"].write_text(project["code"].read_text() + "\n# new science version\n")
    raw = dict(project["raw"], seeds=[11, 23, 47, 59, 71, 73])
    project["path"].write_text(yaml.safe_dump(raw))
    svc = service(project)
    capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    complete(svc, accepted, monkeypatch)
    view = svc.overview(accepted["study_series_id"])
    assert view["counts"]["completed"] == 24
    assert all(candidate["selection_objective"] is None for candidate in view["candidates"])
    result = svc.analysis(accepted["study_series_id"])
    assert len(result["study_series"]["provenance"]) == 4
    assert all(row["origin_revision"] == 1 for row in result["study_series"]["provenance"])


def test_successive_code_revisions_compare_the_whole_accumulated_design(project, monkeypatch):
    project["raw"]["seeds"] = [11, 23]
    project["raw"]["sweep"]["space"]["strength"] = [0.0, 0.1]
    project["path"].write_text(yaml.safe_dump(project["raw"]))
    origin = execute(project)
    svc = service(project)
    capture_preparation(monkeypatch)
    selector = origin.execution_id
    for revision in (1, 2):
        project["code"].write_text(project["code"].read_text() + f"\n# revision {revision}\n")
        if revision == 1:
            project["raw"]["sweep"]["space"]["strength"].append(0.2)
        else:
            project["raw"]["seeds"].append(47)
        project["path"].write_text(yaml.safe_dump(project["raw"]))
        plan = svc.preview(selector, project["path"])
        assert plan["comparison"]["counts"] == (
            {"reused": 4, "new": 2, "total": 6} if revision == 1
            else {"reused": 6, "new": 3, "total": 9}
        )
        accepted = svc.apply(selector, project["path"], plan_hash=plan["plan_hash"])
        complete(svc, accepted, monkeypatch)
        selector = accepted["study_series_id"]
    view = svc.overview(selector)
    assert view["counts"]["completed"] == 9 and view["physical_cost"]["spent_attempts"] == 9
    analysis = svc.analysis(selector)
    assert {name: len(row["study_series"]["provenance"])
            for name, row in analysis["code_cohort_analyses"].items()} == {
        "revision-0": 4, "revision-1": 2, "revision-2": 3,
    }


def test_changed_code_after_review_cannot_submit_or_recover_an_old_approval(project, monkeypatch):
    origin = execute(project)
    extend_yaml(project)
    project["code"].write_text(project["code"].read_text() + "\n# first change\n")
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    project["code"].write_text(project["code"].read_text() + "\n# subsequent change\n")
    launches = capture_preparation(monkeypatch)
    with pytest.raises(ValueError, match="plan changed"):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert not launches


def test_remote_changed_code_preview_uses_normal_managed_deployment_not_old_parser(
    project, monkeypatch
):
    from tests.controlplane.test_study_extension_remote import remote_fixture

    project["raw"]["seeds"] = [11, 23]
    project["raw"]["sweep"]["space"]["strength"] = [0.0, 0.1]
    project["path"].write_text(yaml.safe_dump(project["raw"]))
    svc, job_id, _, _ = remote_fixture(project, monkeypatch)
    project["code"].write_text(project["code"].read_text() + "\n# compatible/new reviewed code\n")
    raw = dict(project["raw"], sweep={"space": {"strength": [0.0, 0.1, 0.2]}})
    project["path"].write_text(yaml.safe_dump(raw))
    plan = svc.preview(job_id, project["path"])
    assert plan["comparison"]["counts"] == {"reused": 4, "new": 2, "total": 6}
    assert plan["code_compatibility"]["kind"] == "revisioned-current"
    assert "retained_runtime" not in plan["runtime_compatibility"]
    launches = capture_preparation(monkeypatch)
    accepted = svc.apply(job_id, project["path"], plan_hash=plan["plan_hash"])
    request = json.loads((svc.jobs.store.root / "submissions" / accepted["job_id"]
                          / "request.json").read_text())
    assert request["profile"]["environment"] == "managed"
    assert not svc.jobs.get(accepted["job_id"], refresh=False).metadata["study_extension"].get(
        "retained_runtime"
    )
    assert len(launches) == 1


def test_current_code_is_checked_again_by_the_detached_native_worker(project, monkeypatch):
    import importlib

    from lambdaforge.controlplane.jobs import JobState

    origin = execute(project)
    extend_yaml(project)
    project["code"].write_text(project["code"].read_text() + "\n# approved source\n")
    svc = service(project)
    capture_preparation(monkeypatch)
    plan = svc.preview(origin.execution_id, project["path"])
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    project["code"].write_text(project["code"].read_text() + "\n# unapproved edit\n")
    worker = importlib.import_module("lambdaforge.controlplane.SubmissionWorker")
    dispatched = []
    monkeypatch.setattr(worker.ControlPlane, "submit", lambda *a, **kw: dispatched.append(kw))
    request = svc.jobs.store.root / "submissions" / accepted["job_id"] / "request.json"
    assert worker.serve(request) == 1
    assert not dispatched
    assert svc.jobs.get(accepted["job_id"], refresh=False).state == JobState.FAILED


@pytest.mark.parametrize("tamper", [False, True])
def test_remote_current_code_attests_actual_host_design_and_collects_only_new_cells(
    project, monkeypatch, tamper
):
    """Intercept only deployment; actual host preflight and CPU Runs use native code."""
    import importlib
    import venv
    from pathlib import Path
    from types import SimpleNamespace

    from lambdaforge.controlplane.jobs import JobState
    from lambdaforge.controlplane.LocalTransport import LocalTransport
    from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
    from tests.controlplane.test_study_extension_remote import remote_fixture

    project["raw"]["seeds"] = [11, 23]
    project["raw"]["sweep"]["space"]["strength"] = [0.0, 0.1]
    project["path"].write_text(yaml.safe_dump(project["raw"]))
    svc, job_id, _, origin_work = remote_fixture(project, monkeypatch)
    before = {p: p.read_bytes() for p in origin_work.rglob("*") if p.is_file()}
    project["code"].write_text(project["code"].read_text().replace(
        "score = 1.0 - strength", "score = 100.0 - strength"
    ))
    raw = dict(project["raw"], sweep={"space": {"strength": [0.0, 0.1, 0.2]}})
    project["path"].write_text(yaml.safe_dump(raw))
    plan = svc.preview(job_id, project["path"])
    capture_preparation(monkeypatch)
    receipt = svc.apply(job_id, project["path"], plan_hash=plan["plan_hash"])
    profile = svc.catalog.get("loopback")
    prefix = Path(profile.storage.environment_root) / ("env-" + "b" * 24)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    (prefix / ".lambdaforge-environment.json").write_text(json.dumps({
        "environment_id": prefix.name, "packages": ["disposable-fixture.whl"],
    }))
    work = Path(profile.storage.run_root) / receipt["job_id"] / "work"
    work.mkdir(parents=True)
    (work / "pyproject.toml").write_bytes((project["root"] / "pyproject.toml").read_bytes())
    (work / "code-identity.json").write_text(json.dumps(
        CodeIdentity.capture(project["root"]).to_dict()
    ))
    config = work / "config.yaml"
    config.write_bytes(Path(receipt["snapshot"]).read_bytes())
    if tamper:
        changed = yaml.safe_load(config.read_text())
        changed["seeds"] = [999]
        config.write_text(yaml.safe_dump(changed))
    prepared = SimpleNamespace(
        profile=profile, transport=LocalTransport(), python=str(prefix / "bin/python"),
        work_dir=str(work), config=str(config), dry_run=False,
    )
    worker = importlib.import_module("lambdaforge.controlplane.SubmissionWorker")
    entries = []

    class Provider:
        def __init__(self, catalog, **kwargs):
            assert catalog.get("loopback").environment == "managed"

        def submit(self, source, **kwargs):
            assert kwargs["run_arguments"] == ()
            entries.append(kwargs["entrypoint_builder"](prepared))

    monkeypatch.setattr(worker, "ControlPlane", Provider)
    request = svc.jobs.store.root / "submissions" / receipt["job_id"] / "request.json"
    if tamper:
        assert worker.serve(request) == 1 and not entries
        assert svc.jobs.get(receipt["job_id"], refresh=False).state == JobState.FAILED
        assert not (work.parent / "result.json").exists()
        return
    assert worker.serve(request) == 0 and len(entries) == 1
    result = LocalTransport().run((
        "env", "LAMBDAFORGE_EXECUTION_MODE=worker", "LAMBDAFORGE_BUNDLE=1",
        f"LAMBDAFORGE_JOB_ID={receipt['job_id']}", "LAMBDAFORGE_CLUSTER=loopback",
        f"LAMBDAFORGE_RUN_ROOT={work / '.lambdaforge/runs'}",
        f"LAMBDAFORGE_JOB_RESULT_PATH={work.parent / 'result.json'}",
        str(prefix / "bin/python"), "-m", *entries[0],
    ), cwd=work, timeout=90)
    assert result.returncode == 0, result.stderr
    job = svc.jobs.get(receipt["job_id"], refresh=False)
    pointer = json.loads((work.parent / "result.json").read_text())
    assert job.metadata["prepared_extension_attestation"][
        "scientific_identity"
    ] == pointer["scientific_fingerprint"]
    assert pointer["scientific_fingerprint"] != job.metadata["pinned_scientific_identity"]
    svc.jobs.store.write(job.with_updates(
        state=JobState.SUCCEEDED, work_dir=str(work), metadata={
            **job.metadata, "environment_id": prefix.name,
            "provider_storage": profile.storage.to_dict(),
        },
        command=(str(prefix / "bin/python"), "-m", *entries[0]),
    ))
    history = svc.refresh(receipt["study_series_id"])
    assert history["operations"][0]["state"] == "succeeded"
    view = svc.overview(receipt["study_series_id"])
    assert view["counts"]["completed"] == 6 and view["mixed_code"]
    assert all(r["metrics"]["score"] > 90 for r in view["runs"] if r["origin_revision"] == 1)
    assert before == {p: p.read_bytes() for p in origin_work.rglob("*") if p.is_file()}
