"""Focused POSIX integration coverage for the durable 0.6 process scheduler."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from lambdaforge.controlplane import ClusterProfile, JobState, LocalTransport, ProcessScheduler
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
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
        (sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(1)"),
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
    assert "ready" in reconnected.logs(job_id)
    assert reconnected.inventory()[0]["request"]["job_id"] == job_id


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
