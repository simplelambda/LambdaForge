"""Competing scientific owners share startup capacity, not scientific decisions or GPU grants."""

import json
import os

import pytest

from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.runtime.SharedGPUAdmission import _fairness_peer, shared_gpu_launch


def launch(root, study, free, *, worker=None, observed=None, device="GPU-one"):
    return shared_gpu_launch(
        root,
        device,
        lambda: study,
        stagger_seconds=0,
        startup_bytes=60,
        metadata={"study": study, "worker_identity_path": str(worker) if worker else None},
        observed_pid_bytes=observed or {},
        capacity_check=lambda extra: free - extra >= 60,
    )


def test_delayed_startup_consumption_blocks_peer_then_reconciles_without_double_count(tmp_path):
    worker = tmp_path / "worker.json"
    worker.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "create_time": __import__("psutil").Process().create_time(),
                "controller_pid": os.getpid(),
            }
        )
    )
    root = tmp_path / "coord"
    assert launch(root, "study-a", 100, worker=worker) == "study-a"
    # Physical bytes still look free while the first Run starts.
    assert launch(root, "study-b", 100) is None
    # Only the unobserved deficit remains; observed 40 already occupies physical VRAM.
    assert launch(root, "study-b", 80, observed={os.getpid(): 40}) == "study-b"


def test_own_native_envelope_not_reserved_again(tmp_path):
    root = tmp_path / "coord"
    assert launch(root, "study-a", 100) == "study-a"
    assert launch(root, "study-a", 100) == "study-a"


def test_fairness_yields_to_older_feasible_peer_without_forced_alternation():
    waiters = {
        "own": {"waiting_since": 95, "launches": 10, "priority": 1, "required_bytes": 20},
        "peer": {"waiting_since": 60, "launches": 0, "priority": 1, "required_bytes": 30},
    }
    assert _fairness_peer(waiters, "own", 40, now=100, minimum_wait=30)
    # Heavy peer cannot use the opportunity; light work must not stall behind it.
    assert not _fairness_peer(waiters, "own", 25, now=100, minimum_wait=30)
    assert not _fairness_peer(waiters, "own", 40, now=70, minimum_wait=30)
    waiters["own"]["waiting_since"] = 0
    waiters["own"]["priority"] = 10
    assert not _fairness_peer(waiters, "own", 40, now=100, minimum_wait=30)


def test_independent_gpu_and_mig_namespaces_are_not_serialized(tmp_path):
    root = tmp_path / "coord"
    assert launch(root, "a", 100) == "a"
    assert launch(root, "b", 100, device="GPU-two") == "b"
    assert launch(root, "c", 100, device="MIG-instance") == "c"


def test_missing_worker_identity_is_pending_not_failure(tmp_path):
    root = tmp_path / "coord"
    assert launch(root, "a", 100, worker=tmp_path / "not-yet-created") == "a"
    assert launch(root, "b", 100) is None


def test_confirmed_worker_death_releases_capacity(tmp_path, monkeypatch):
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps({"pid": 123456, "create_time": 1, "controller_pid": os.getpid()}))
    root = tmp_path / "coord"
    assert launch(root, "a", 100, worker=worker) == "a"
    monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "dead")
    assert launch(root, "b", 100) == "b"


def test_unresolved_worker_owner_never_reclaims_commitment(tmp_path, monkeypatch):
    root = tmp_path / "coord"
    assert launch(root, "a", 100) == "a"
    monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "unresolved")
    assert launch(root, "b", 100) is None


def test_raw_ordinals_and_unsafe_roots_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="UUID"):
        launch(tmp_path / "coord", "a", 100, device="0")
