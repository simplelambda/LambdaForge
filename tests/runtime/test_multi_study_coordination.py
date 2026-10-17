"""Real process/file-lock races with synthetic memory, no CUDA or scientific workload."""

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from lambdaforge.runtime.GPUObservation import GPUObservationCache
from lambdaforge.runtime.SharedGPUAdmission import shared_gpu_launch


def _observer(root, barrier, counter, output):
    def probe():
        with counter.get_lock():
            counter.value += 1
        return {"quality": "measured", "observed_at": time.time(), "devices": []}

    barrier.wait(timeout=10)
    output.put(GPUObservationCache(Path(root), probe=probe).read()["quality"])


def _hold_startup(root, ready, stop, output):
    import psutil

    identity = Path(root) / "fixture-worker.json"

    def launch():
        identity.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "controller_pid": os.getpid(),
                    "create_time": psutil.Process().create_time(),
                    "process_group": os.getpgrp(),
                }
            )
        )
        return "study-a-started"

    result = shared_gpu_launch(
        Path(root),
        "GPU-shared",
        launch,
        stagger_seconds=0,
        startup_bytes=80,
        metadata={"study": "a", "worker_identity_path": str(identity)},
    )
    output.put(result)
    ready.set()
    stop.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX process/file-lock fixture")
def test_cross_process_observation_is_single_flight(tmp_path):
    ctx = multiprocessing.get_context("fork")
    counter, barrier, output = ctx.Value("i", 0), ctx.Barrier(2), ctx.Queue()
    workers = [
        ctx.Process(target=_observer, args=(str(tmp_path / "cache"), barrier, counter, output))
        for _ in range(2)
    ]
    try:
        for process in workers:
            process.start()
        assert [output.get(timeout=10) for _ in workers] == ["measured", "measured"]
        assert counter.value == 1
    finally:
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
    assert all(process.exitcode == 0 for process in workers)


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX process/file-lock fixture")
def test_two_studies_cannot_both_spend_unobserved_startup_memory(tmp_path):
    ctx = multiprocessing.get_context("fork")
    ready, stop, output = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=_hold_startup, args=(str(tmp_path), ready, stop, output))
    process.start()
    try:
        assert ready.wait(timeout=10)
        assert output.get(timeout=5) == "study-a-started"
        launched = []
        result = shared_gpu_launch(
            tmp_path,
            "GPU-shared",
            lambda: launched.append("b"),
            stagger_seconds=0,
            startup_bytes=80,
            metadata={"study": "b"},
            capacity_check=lambda extra: 80 <= 100 - extra,
        )
        assert result is None and launched == []
        # A different granted device is independent, not blocked by a host-wide launch lock.
        assert (
            shared_gpu_launch(
                tmp_path,
                "MIG-independent",
                lambda: "b-other-device",
                stagger_seconds=0,
                metadata={"study": "b"},
                startup_bytes=80,
                capacity_check=lambda extra: 80 <= 100 - extra,
            )
            == "b-other-device"
        )
    finally:
        stop.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
    assert process.exitcode == 0


def test_coordinator_persists_before_spawn_not_after(tmp_path, monkeypatch):
    calls = []

    def fail_write(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("lambdaforge.runtime.SharedGPUAdmission.atomic_write_text", fail_write)
    with pytest.raises(OSError, match="synthetic"):
        shared_gpu_launch(
            tmp_path, "GPU-one", lambda: calls.append("spawn"), stagger_seconds=0, startup_bytes=40
        )
    assert calls == []
