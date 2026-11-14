"""Transient Job inputs cross anonymous pipes only, never durable scheduler metadata."""

import hashlib
import os
import sys
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.execution.ResourceRequest import ResourceRequest
from tests.controlplane.test_execution_queue import profile, wait_for

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Native detached POSIX fixture")


def test_secret_survives_handoff_without_becoming_a_file_argument_or_log(tmp_path):
    selected = profile(tmp_path)
    jobs = JobService(ClusterCatalog({"local": selected}), JobStore(tmp_path / "history"))
    secret = b"fixture-only secret : with spaces and quotes ' \""
    ready, release = tmp_path / "ready", tmp_path / "release"
    command = [
        sys.executable,
        "-c",
        "import sys,pathlib,hashlib,time;secret=sys.stdin.buffer.read();"
        "assert hashlib.sha256(secret).hexdigest()==sys.argv[1];"
        "pathlib.Path(sys.argv[2]).touch();"
        "\nwhile not pathlib.Path(sys.argv[3]).exists():time.sleep(.05)",
        hashlib.sha256(secret).hexdigest(),
        str(ready),
        str(release),
    ]
    handle = jobs.submit(
        command,
        cluster="local",
        resources=ResourceRequest(cpu_cores=1),
        work_dir=tmp_path,
        stdin_data=secret,
    )
    try:
        wait_for(ready.exists)
        assert jobs.get(handle.job_id, include_study=False).state.value == "running"
        root = Path(selected.storage.job_root) / handle.job_id
        for file in (*root.rglob("*"), *jobs.store.root.rglob("*")):
            if file.is_file():
                assert secret not in file.read_bytes(), file
        release.touch()
        wait_for(lambda: jobs.get(handle.job_id, include_study=False).state.value == "succeeded")
        with pytest.raises(ValueError, match="non-persisted credential"):
            jobs.retry(handle.job_id, dry_run=True)
    finally:
        release.touch()
        if not jobs.get(handle.job_id, refresh=False).state.terminal:
            jobs.cancel(handle.job_id)


@pytest.mark.parametrize("secret", [b"", b"x" * 4097])
def test_invalid_ephemeral_input_does_not_launch_a_child(tmp_path, secret):
    from lambdaforge.controlplane.LocalTransport import LocalTransport
    from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler

    scheduler = ProcessScheduler(LocalTransport(), profile(tmp_path))
    with pytest.raises(ValueError, match="1–4096"):
        scheduler.submit(
            [sys.executable, "-c", "raise AssertionError('launched')"],
            ResourceRequest(),
            work_dir=tmp_path,
            job_id="job-no-launch",
            stdin_data=secret,
        )
    assert not list(tmp_path.rglob("request.json"))
