"""Native delta Work execution and durable handoff, never a replay of origin cells."""

import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.StudySeriesStore import StudySeriesStore
from tests.controlplane.test_study_extension_preview import execute, extend_yaml, service


@pytest.fixture(name="project")
def extension_fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    return project.__wrapped__(tmp_path, monkeypatch)


def capture_preparation(monkeypatch):
    module = importlib.import_module("lambdaforge.controlplane.SubmissionService")
    launched = []

    def launch(argv, **kwargs):
        launched.append(argv)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(
        module, "subprocess", SimpleNamespace(Popen=launch, DEVNULL=subprocess.DEVNULL)
    )
    return launched


def test_apply_reuses_twenty_executes_fifteen_via_native_runner_and_keeps_history(
    project, monkeypatch
):
    origin = execute(project)
    original_bytes = {p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()}
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    launched = capture_preparation(monkeypatch)
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert accepted["created"] and accepted["state"] == "submitted"
    assert len(launched) == 1
    request_path = svc.jobs.store.root / "submissions" / accepted["job_id"] / "request.json"
    request = json.loads(request_path.read_text())
    assert request["job_id"] == accepted["job_id"]
    assert request["config_sha256"] == accepted["snapshot_sha256"]
    assert request["config"] != str(project["path"])
    assert project["path"].read_text().count("0.5") == 1
    assert WorkConfig.from_yaml(request["config"]).planned_runs == 15
    repeated = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert not repeated["created"] and repeated["job_id"] == accepted["job_id"]
    assert len(launched) == 1

    # Only the provider/process handoff is inlined in this disposable CPU fixture.
    # The same native WorkRunner performs every scientific Run and writes all evidence.
    record = svc.jobs.get(accepted["job_id"], refresh=False)
    owned_work = project["root"] / ".lambdaforge/jobs" / accepted["job_id"] / "work"
    owned_work.mkdir(parents=True)
    svc.jobs.store.write(record.with_updates(work_dir=str(owned_work)))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", accepted["job_id"])
    monkeypatch.setenv("LAMBDAFORGE_JOB_RESULT_PATH", str(owned_work.parent / "result.json"))
    delta = WorkRunner().run(WorkConfig.from_yaml(request["config"]))
    assert delta.status == "succeeded" and len(delta.runs) == 15
    original_cells = {(r.trial["parameters"]["strength"], r.seed) for r in origin.runs}
    added_cells = {(r.trial["parameters"]["strength"], r.seed) for r in delta.runs}
    assert not original_cells & added_cells and len(original_cells | added_cells) == 35
    assert original_bytes == {
        p: p.read_bytes() for p in origin.execution_dir.rglob("*") if p.is_file()
    }
    for apply in (False, True):
        with pytest.raises(ValueError, match="retained by a Study series"):
            svc.results.delete(delta.execution_id, apply=apply)
    record = svc.jobs.get(accepted["job_id"], refresh=False)
    svc.jobs.store.write(record.with_updates(state=JobState.SUCCEEDED))
    reconciled = svc.refresh(accepted["study_series_id"])
    assert reconciled["operations"][0]["execution_id"] == delta.execution_id
    assert reconciled["operations"][0]["state"] == "succeeded"
    assert len(reconciled["history"]) == 4
    assert svc.jobs.series_dependents(accepted["job_id"]) == (accepted["study_series_id"],)
    assert svc.jobs.recovery_dependents(accepted["job_id"]) == ()
    with pytest.raises(ValueError, match="retained by a Study series"):
        svc.jobs.delete(accepted["job_id"])


def test_changed_plan_creates_no_native_job(project, monkeypatch):
    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    launched = capture_preparation(monkeypatch)
    project["path"].write_text(project["path"].read_text().replace("- 1.0", "- 1.5"))
    with pytest.raises(ValueError, match="plan changed"):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert not launched and svc.jobs.list(refresh=False) == ()


def test_handoff_failure_is_durable_and_not_blindly_duplicated(project, monkeypatch):
    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    module = importlib.import_module("lambdaforge.controlplane.SubmissionService")

    def fail(*args, **kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(
        module, "subprocess", SimpleNamespace(Popen=fail, DEVNULL=subprocess.DEVNULL)
    )
    with pytest.raises(OSError, match="synthetic launch"):
        svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    state = StudySeriesStore(svc.project).read(plan["study_series_id"])
    assert state["operations"][0]["state"] == "dispatching"
    assert len(svc.jobs.list(refresh=False)) == 1
    previous = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    assert not previous["created"] and previous["job"]["state"] == "failed"
    assert len(svc.jobs.list(refresh=False)) == 1


def test_concurrent_apply_returns_one_native_job(project, monkeypatch):
    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    launched = capture_preparation(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(
            pool.map(
                lambda _: svc.apply(
                    origin.execution_id, project["path"], plan_hash=plan["plan_hash"]
                ),
                range(2),
            )
        )
    assert len(launched) == 1
    assert len({reply["job_id"] for reply in replies}) == 1
    assert sum(reply["created"] for reply in replies) == 1


def test_snapshot_preserves_relative_inputs_and_implicit_name(project, monkeypatch):
    import yaml

    project["raw"].pop("name")
    data = project["root"] / "source.txt"
    data.write_text("exact immutable bytes")
    project["raw"]["with"] = {"source": {"file": "../source.txt"}}
    project["path"].write_text(yaml.safe_dump(project["raw"]))
    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    capture_preparation(monkeypatch)
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    config = WorkConfig.from_yaml(accepted["snapshot"])
    assert config.name == origin.name
    assert config.source.parent == project["path"].parent
    delta = WorkRunner().run(config)
    assert delta.status == "succeeded" and len(delta.runs) == 15
    assert {run.inputs[0].sha256 for run in delta.runs} == {
        run.inputs[0].sha256 for run in origin.runs
    }


def test_preparation_rejects_executable_changes_after_accepted_plan(project, monkeypatch):
    from lambdaforge.controlplane.ControlPlane import ControlPlane

    origin = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(origin.execution_id, project["path"])
    capture_preparation(monkeypatch)
    accepted = svc.apply(origin.execution_id, project["path"], plan_hash=plan["plan_hash"])
    project["code"].write_text(project["code"].read_text() + "\n# changed executable sources\n")
    with pytest.raises(ValueError, match="Pinned submission configuration/code/inputs changed"):
        ControlPlane(svc.catalog, jobs=svc.jobs).submit(
            accepted["snapshot"], cluster="local", reserved_job_id=accepted["job_id"]
        )
    assert len(svc.results.catalog()) == 1
