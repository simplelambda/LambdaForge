"""A lost controller must not permanently wedge preparation, or steal live ownership."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import PurePosixPath

import psutil
import pytest

from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease
from lambdaforge.controlplane.LocalTransport import LocalTransport


def stale_marker(tmp_path, **changes):
    marker = tmp_path / "cache/.environment-build-env-test.lock"
    marker.mkdir(parents=True)
    owner = {
        "host": socket.gethostname(),
        "pid": 2147483647,
        "created_at": time.time() - 3600,
        "heartbeat": time.time() - 3600,
        **changes,
    }
    (marker / "owner.json").write_text(json.dumps(owner))
    return marker


def test_dead_same_controller_is_recovered_by_real_acquisition_and_clean_release(tmp_path):
    marker = stale_marker(tmp_path)
    CacheBuildLease.acquire(
        LocalTransport(), PurePosixPath(marker), python=sys.executable, timeout=0
    )
    owner = json.loads((marker / "owner.json").read_text())
    assert owner["pid"] == os.getpid() and owner["process_birth"] == psutil.Process().create_time()
    with CacheBuildLease.maintain(LocalTransport(), PurePosixPath(marker), python=sys.executable):
        assert marker.is_dir()
    assert not marker.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"host": "foreign-controller"},
        {"pid": os.getpid(), "created_at": lambda: time.time() + 1},
        {"heartbeat": time.time},
        {"heartbeat": float("nan")},
    ],
)
def test_live_foreign_fresh_or_invalid_ownership_is_not_stolen(tmp_path, changes):
    # A fresh heartbeat must be fresh at execution, not when a long suite is collected.
    changes = {key: value() if callable(value) else value for key, value in changes.items()}
    marker = stale_marker(tmp_path, **changes)
    before = (marker / "owner.json").read_bytes()
    with pytest.raises(TimeoutError):
        CacheBuildLease.acquire(
            LocalTransport(), PurePosixPath(marker), python=sys.executable, timeout=0
        )
    assert (marker / "owner.json").read_bytes() == before


@pytest.mark.parametrize("final_prefix", [False, True])
def test_remote_installer_survives_dead_controller_and_blocks_reclamation(tmp_path, final_prefix):
    marker = stale_marker(tmp_path)
    name = "env-test" if final_prefix else ".env-test.tmp-owned"
    prefix = marker.parent / "environments" / name / "bin/python"
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(prefix)]
    ) as process:
        try:
            assert not CacheBuildLease._reclaim_dead_controller(
                LocalTransport(), PurePosixPath(marker), python=sys.executable
            )
            assert marker.exists()
        finally:
            process.terminate()
            process.wait(timeout=5)
    assert CacheBuildLease._reclaim_dead_controller(
        LocalTransport(), PurePosixPath(marker), python=sys.executable
    )


def test_changed_snapshot_and_unexpected_entries_are_protected(tmp_path):
    marker = stale_marker(tmp_path)

    class RacedTransport(LocalTransport):
        def run(self, command, **kwargs):
            if "lambdaforge-cache-build-reclaim" in " ".join(command):
                owner = json.loads((marker / "owner.json").read_text())
                owner["heartbeat"] = time.time()
                (marker / "owner.json").write_text(json.dumps(owner))
            return super().run(command, **kwargs)

    assert not CacheBuildLease._reclaim_dead_controller(
        RacedTransport(), PurePosixPath(marker), python=sys.executable
    )
    (marker / "owner.json").write_text(
        json.dumps(
            {"host": socket.gethostname(), "pid": 2147483647, "heartbeat": time.time() - 3600}
        )
    )
    (marker / "unexpected").write_text("protected")
    assert not CacheBuildLease._reclaim_dead_controller(
        LocalTransport(), PurePosixPath(marker), python=sys.executable
    )
    assert (marker / "unexpected").read_text() == "protected"


def test_pid_reuse_and_unreadable_owner_are_distinguished(monkeypatch):
    from types import SimpleNamespace

    owner = {
        "host": socket.gethostname(),
        "pid": 123,
        "heartbeat": time.time() - 3600,
        "process_birth": 1,
    }
    monkeypatch.setattr(psutil, "Process", lambda _: SimpleNamespace(create_time=lambda: 100))
    assert CacheBuildLease._controller_dead(owner)

    def inaccessible(_):
        raise psutil.AccessDenied(123)

    monkeypatch.setattr(psutil, "Process", inaccessible)
    assert not CacheBuildLease._controller_dead(owner)


def test_symlinked_marker_remains_untouched(tmp_path):
    target = stale_marker(tmp_path)
    linked = target.parent / ".environment-build-env-linked.lock"
    linked.symlink_to(target, target_is_directory=True)
    assert not CacheBuildLease._reclaim_dead_controller(
        LocalTransport(), PurePosixPath(linked), python=sys.executable
    )
    assert target.is_dir() and linked.is_symlink()


def test_two_recovery_requests_never_acquire_the_same_marker(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    marker = stale_marker(tmp_path)

    def acquire(_):
        try:
            CacheBuildLease.acquire(
                LocalTransport(), PurePosixPath(marker), python=sys.executable, timeout=0
            )
            return True
        except TimeoutError:
            return False

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(acquire, range(2))) == [False, True]
    assert json.loads((marker / "owner.json").read_text())["pid"] == os.getpid()
