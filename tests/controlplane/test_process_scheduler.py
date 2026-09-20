"""Focused POSIX integration coverage for the durable process scheduler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from lambdaforge.controlplane import ClusterProfile, JobState, LocalTransport, ProcessScheduler
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from lambdaforge.execution import ResourceRequest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Process groups require POSIX.")


def scheduler(tmp_path: Path) -> ProcessScheduler:
    profile = ClusterProfile(
        "local",
        python=sys.executable,
        workspace=str(tmp_path),
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "jobs"),
        },
    )
    return ProcessScheduler(LocalTransport(), profile)


def wait_for(
    value: ProcessScheduler, job_id: str, states: set[JobState], timeout: float = 15.0
) -> JobState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = value.state(job_id)
        if state in states:
            return state
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not reach {states}.")


def test_submit_returns_and_reconnects_to_durable_state(tmp_path: Path) -> None:
    value = scheduler(tmp_path)
    job_id = "job-060-reconnect"
    started = time.monotonic()
    submission = value.submit(
        (
            sys.executable,
            "-c",
            "import os,time; print(os.environ['LAMBDAFORGE_CACHE_ROOT'], flush=True); "
            "time.sleep(1)",
        ),
        ResourceRequest(),
        work_dir=tmp_path,
        job_id=job_id,
    )
    assert time.monotonic() - started < 5
    assert submission.state is JobState.STAGING
    reconnected = scheduler(tmp_path)
    assert wait_for(reconnected, job_id, {JobState.RUNNING, JobState.SUCCEEDED}) in {
        JobState.RUNNING,
        JobState.SUCCEEDED,
    }
    assert wait_for(reconnected, job_id, {JobState.SUCCEEDED}) is JobState.SUCCEEDED
    assert str(tmp_path / "cache") in reconnected.logs(job_id)
    request = reconnected.inventory()[0]["request"]
    assert request["job_id"] == job_id
    assert request["cache_root"] == str(tmp_path / "cache")


def test_inventory_projects_study_without_transferring_candidates_or_runs(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "job-study-inventory"
    study_dir = job_dir / "study"
    study_dir.mkdir(parents=True)
    (job_dir / "state.json").write_text(
        json.dumps({"job_id": job_dir.name, "state": "running"}), encoding="utf-8"
    )
    (job_dir / "request.json").write_text(
        json.dumps({"job_id": job_dir.name, "command": ["python"], "resources": {}}),
        encoding="utf-8",
    )
    (study_dir / "summary.json").write_text(
        json.dumps(
            {
                "study_telemetry_version": 1,
                "objective": {"metric": "score", "mode": "max"},
                "counts": {"candidates": 2, "active_runs": 1},
                "candidates": [
                    {"trial": 1, "selection_objective": 0.6, "runs": [{"seed": 1}]},
                    {"trial": 2, "selection_objective": 0.8, "runs": [{"seed": 2}]},
                ],
            }
        ),
        encoding="utf-8",
    )

    state = ProcessSupervisor.inventory(tmp_path / "jobs")[0]["state"]

    assert state["study"]["detail_level"] == "overview"
    assert state["study"]["counts"]["candidates"] == 2
    assert state["study"]["leader"]["trial"] == 2
    assert "candidates" not in state["study"]


def test_supervisor_does_not_copy_an_already_staged_workspace_onto_itself(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "jobs" / "job-already-staged"
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "config.yaml").write_text("name: retained\n", encoding="utf-8")
    request = job_dir / "request.json"
    request.write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "command": [sys.executable, "-c", "print('launched')"],
                "cluster": "remote",
                "source_work_dir": str(work_dir),
                "stage_source": True,
                "resources": {},
                "lease_root": str(tmp_path / "gpu-leases"),
                "resource_lease_root": str(tmp_path / "process-leases"),
            }
        ),
        encoding="utf-8",
    )

    assert ProcessSupervisor.serve(request) == 0
    assert (work_dir / "config.yaml").is_file()
    assert "launched" in (job_dir / "stdout.log").read_text(encoding="utf-8")


def test_prelaunch_failure_has_empty_stable_log_streams(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "job-prelaunch-failure"
    job_dir.mkdir(parents=True)
    request = job_dir / "request.json"
    request.write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "command": [sys.executable, "-c", "print('never')"],
                "cluster": "remote",
                "source_work_dir": str(tmp_path / "missing"),
                "stage_source": True,
                "resources": {},
                "lease_root": str(tmp_path / "gpu-leases"),
                "resource_lease_root": str(tmp_path / "process-leases"),
            }
        ),
        encoding="utf-8",
    )

    assert ProcessSupervisor.serve(request) == 1
    assert (job_dir / "stdout.log").read_text(encoding="utf-8") == ""
    assert (job_dir / "stderr.log").read_text(encoding="utf-8") == ""
    state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    assert "Unsafe or missing staged source" in state["message"]
    observed = scheduler(tmp_path)
    assert observed.state(job_dir.name) is JobState.FAILED
    assert "Unsafe or missing staged source" in str(observed.details(job_dir.name)["message"])


def test_external_gpu_wrapper_environment_is_preserved_without_local_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "jobs" / "job-external-gpu"
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-granted-7,GPU-granted-9")
    request = job_dir / "request.json"
    request.write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "command": [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); "
                    "print(os.environ['LAMBDAFORGE_GPU_ACCESS_MODE']); "
                    "print(os.environ['LAMBDAFORGE_GPU_VISIBILITY_COMMAND'])",
                ],
                "cluster": "remote",
                "source_work_dir": str(work_dir),
                "stage_source": False,
                "resources": {"gpu_count": 2},
                "gpu_access": {
                    "mode": "command",
                    "command_prefix": ["gpu"],
                    "visibility_command": ["gpu", "env"],
                },
                "lease_root": str(tmp_path / "gpu-leases"),
                "resource_lease_root": str(tmp_path / "process-leases"),
            }
        ),
        encoding="utf-8",
    )

    assert ProcessSupervisor.serve(request) == 0
    output = (job_dir / "stdout.log").read_text(encoding="utf-8")
    assert output.splitlines() == [
        "GPU-granted-7,GPU-granted-9",
        "command",
        '["gpu", "env"]',
    ]
    state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    assert state["allocated_gpus"] == []


def test_log_reader_treats_legacy_missing_streams_as_no_output(tmp_path: Path) -> None:
    value = scheduler(tmp_path)
    (tmp_path / "jobs" / "job-legacy-no-streams").mkdir(parents=True)

    assert value.logs("job-legacy-no-streams") == ""


def test_dataset_environment_wrapper_preserves_supervisor_python(tmp_path: Path) -> None:
    """Dataset env assignments must not be mistaken for the control interpreter."""
    profile = ClusterProfile(
        "local",
        python=sys.executable,
        workspace=str(tmp_path),
        command_prefix=("env", "LAMBDAFORGE_SITE=testing"),
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "jobs"),
        },
    )
    value = ProcessScheduler(LocalTransport(), profile)
    scientific = (
        *profile.command_prefix,
        "env",
        "LAMBDAFORGE_DATASET_ROOT=/datasets",
        sys.executable,
        "-m",
        "lambdaforge.data.DatasetBuildWorker",
        "config.yaml",
    )

    submission = value.submit(
        scientific,
        ResourceRequest(),
        work_dir=tmp_path,
        job_id="job-dataset-env-wrapper",
        dry_run=True,
    )

    assert submission.command == (
        *profile.command_prefix,
        sys.executable,
        "-m",
        "lambdaforge.controlplane.ProcessSupervisor",
        "launch",
        str(tmp_path / "jobs/job-dataset-env-wrapper/request.json"),
    )


def test_pause_resume_cancel_and_runtime_timeout(tmp_path: Path) -> None:
    value = scheduler(tmp_path)
    paused_id = "job-060-pause"
    value.submit(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        ResourceRequest(),
        work_dir=tmp_path,
        job_id=paused_id,
    )
    wait_for(value, paused_id, {JobState.RUNNING})
    value.pause(paused_id)
    assert value.state(paused_id) is JobState.PAUSED
    value.resume(paused_id)
    assert value.state(paused_id) is JobState.RUNNING
    value.cancel(paused_id)
    assert wait_for(value, paused_id, {JobState.CANCELLED}) is JobState.CANCELLED

    timeout_id = "job-060-timeout"
    value.submit(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        ResourceRequest(runtime_seconds=0.5),
        work_dir=tmp_path,
        job_id=timeout_id,
    )
    assert wait_for(value, timeout_id, {JobState.TIMEOUT}) is JobState.TIMEOUT


def test_process_identity_rejects_reused_or_mismatched_pid() -> None:
    process = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(10)"))
    try:
        command = tuple(__import__("psutil").Process(process.pid).cmdline())
        identity = ProcessIdentity.create(process.pid, os.getpgid(process.pid), command, "job-pid")
        assert identity.matches()
        mismatched = ProcessIdentity(
            identity.pid,
            identity.process_group,
            identity.create_time + 1,
            identity.command_sha256,
            identity.job_id,
        )
        assert not mismatched.matches()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_detached_supervisor_cancels_the_scientific_child_tree(tmp_path: Path) -> None:
    value = scheduler(tmp_path)
    job_id = "job-060-child-tree"
    child_pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    value.submit(
        (sys.executable, "-c", code),
        ResourceRequest(),
        work_dir=tmp_path,
        job_id=job_id,
    )
    wait_for(value, job_id, {JobState.RUNNING})
    deadline = time.monotonic() + 5
    while not child_pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    assert psutil.pid_exists(child_pid)
    value.cancel(job_id)
    wait_for(value, job_id, {JobState.CANCELLED})
    deadline = time.monotonic() + 5
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(child_pid)


def test_cancel_stops_multiple_owned_workers_even_when_they_open_new_sessions(
    tmp_path: Path,
) -> None:
    value = scheduler(tmp_path)
    job_id = "job-012-cancel-owned-sessions"
    child_pid_file = tmp_path / "children.json"
    child_code = "import time; time.sleep(60)"
    code = (
        "import json,pathlib,subprocess,sys,time; "
        f"children=[subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True) for _ in range(2)]; "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(json.dumps([p.pid for p in children])); "
        "time.sleep(60)"
    )
    value.submit(
        (sys.executable, "-c", code),
        ResourceRequest(),
        work_dir=tmp_path,
        job_id=job_id,
    )
    wait_for(value, job_id, {JobState.RUNNING})
    deadline = time.monotonic() + 5
    while not child_pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pids = json.loads(child_pid_file.read_text(encoding="utf-8"))
    assert all(psutil.pid_exists(pid) for pid in child_pids)

    value.cancel(job_id)
    wait_for(value, job_id, {JobState.CANCELLED})

    deadline = time.monotonic() + 5
    while any(psutil.pid_exists(pid) for pid in child_pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(psutil.pid_exists(pid) for pid in child_pids)
    cancellation = value.details(job_id)["cancellation"]
    assert cancellation["observed_processes"] >= 3
    assert cancellation["remaining_processes"] == 0
    value.cancel(job_id)
    assert value.details(job_id)["cancellation"]["remaining_processes"] == 0


def test_supervisor_reaps_owned_workers_left_after_main_process_exits(tmp_path: Path) -> None:
    value = scheduler(tmp_path)
    job_id = "job-012-finish-owned-session"
    child_pid_file = tmp_path / "orphan.pid"
    child_code = "import time; time.sleep(60)"
    code = (
        "import pathlib,subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))"
    )
    value.submit(
        (sys.executable, "-c", code),
        ResourceRequest(),
        work_dir=tmp_path,
        job_id=job_id,
    )
    assert wait_for(value, job_id, {JobState.SUCCEEDED}) is JobState.SUCCEEDED
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(child_pid)
