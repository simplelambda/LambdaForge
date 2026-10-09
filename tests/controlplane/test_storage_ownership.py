"""Lease lifetime regressions never touch real project leases."""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import psutil
import pytest

from lambdaforge.controlplane.StorageAdmission import StorageAdmission, StorageOwnershipError
from lambdaforge.diagnostics.service import DiagnosticClassifier, DiagnosticContext
from tests.controlplane.test_storage_management import capacity, descriptor, owner


@pytest.mark.parametrize("state", ["dead", "zombie", "reused", "inaccessible", "alive"])
def test_owner_lifetime_is_resolved_without_trusting_pid_alone(tmp_path, monkeypatch, state):
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    original, incoming = owner("job-original"), owner("job-incoming")
    assert StorageAdmission.reserve(config, original, 70)["admitted"]
    lease = tmp_path / "host-leases/storage-leases/scratch/job-original.json"

    def process(pid):
        assert pid == original.pid
        if state == "dead":
            raise psutil.NoSuchProcess(pid)
        if state == "inaccessible":
            raise psutil.AccessDenied(pid)
        return SimpleNamespace(
            create_time=lambda: original.create_time + (1 if state == "reused" else 0),
            status=lambda: psutil.STATUS_ZOMBIE if state == "zombie" else psutil.STATUS_RUNNING,
        )

    monkeypatch.setattr(psutil, "Process", process)
    before = lease.read_bytes()
    observed = StorageAdmission.observe(config)["scratch"]
    assert lease.read_bytes() == before  # Observation never collects even proven dead leases.
    result = StorageAdmission.reserve(config, incoming, 60)
    if state in {"dead", "zombie", "reused"}:
        assert result["admitted"] and not lease.exists()
        assert observed["reserved_bytes"] == 0
    elif state == "alive":
        assert not result["admitted"] and result["reserved_bytes"] == 70
        assert lease.read_bytes() == before
    else:
        assert result["reason"] == "unresolved-storage-owner"
        assert result["blocking_lease"] == str(lease)
        assert observed["ownership_unresolved"]
        assert lease.read_bytes() == before


def test_same_process_exec_keeps_commitment_and_exact_release_identity(tmp_path, monkeypatch):
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    original = owner("job-original")
    StorageAdmission.reserve(config, original, 70)
    # Argv may change on exec. It is unsafe to erase a living owner's commitment, but
    # also unnecessary to globally block unrelated admission while birth identity matches.
    changed = replace(original, command_sha256="0" * 64)
    StorageAdmission.release(config, changed)
    assert StorageAdmission.observe(config)["scratch"]["reserved_bytes"] == 70
    assert not StorageAdmission.reserve(config, changed, 0)["admitted"]
    StorageAdmission.release(config, original)
    assert StorageAdmission.reserve(config, changed, 60)["admitted"]


def test_zero_byte_clone_still_reports_ownership_not_enospc(tmp_path, monkeypatch):
    capacity(monkeypatch, [1000])
    config = descriptor(tmp_path)
    original = owner("job-original")
    StorageAdmission.reserve(config, original, 0)
    lease = tmp_path / "host-leases/storage-leases/scratch/job-original.json"
    record = json.loads(lease.read_text())
    record["host"] = "inaccessible-other-host"
    lease.write_text(json.dumps(record))
    monkeypatch.setenv("LAMBDAFORGE_STORAGE_POLICY", json.dumps(config))
    with pytest.raises(StorageOwnershipError) as caught:
        with StorageAdmission.transaction(tmp_path / "clone", 0, purpose="publication-reflink"):
            pytest.fail("Unknown ownership cannot authorize a clone")
    assert not isinstance(caught.value, OSError)
    assert str(lease) in str(caught.value) and "not exhausted disk" in str(caught.value)
    diagnostic = DiagnosticClassifier().classify(
        caught.value, DiagnosticContext.from_argv(["datasets", "publish-candidate", "candidate"])
    )
    assert diagnostic.category.value == "operation_refused"
    from lambdaforge.diagnostics.failure import classify_failure

    restored = pickle.loads(pickle.dumps(caught.value))
    assert str(restored) == str(caught.value)
    disposition = classify_failure(restored)
    assert disposition.category.value == "storage"
    assert disposition.reason == "storage_ownership"
    assert disposition.automatic_recovery_eligible is False
    assert lease.exists()


def test_nested_zero_byte_transactions_release_only_their_own_leases(tmp_path, monkeypatch):
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_STORAGE_POLICY", json.dumps(config))
    with StorageAdmission.transaction(tmp_path / "outer", 70, purpose="outer"):
        with StorageAdmission.transaction(tmp_path / "clone", 0, purpose="publication-reflink"):
            assert StorageAdmission.observe(config)["scratch"]["reserved_bytes"] == 70
            assert StorageAdmission.observe(config)["scratch"]["reservations"] == 2
        assert StorageAdmission.observe(config)["scratch"]["reservations"] == 1
    assert StorageAdmission.observe(config)["scratch"]["reservations"] == 0


def test_concurrent_admission_cannot_overcommit_or_release_another_owner(tmp_path, monkeypatch):
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    owners = [owner(f"job-thread-{index}") for index in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: StorageAdmission.reserve(config, item, 60), owners))
    assert sum(result["admitted"] for result in results) == 1
    winner = owners[next(index for index, result in enumerate(results) if result["admitted"])]
    StorageAdmission.release(config, replace(winner, create_time=winner.create_time + 1))
    assert StorageAdmission.observe(config)["scratch"]["reserved_bytes"] == 60
    StorageAdmission.release(config, winner)
    assert StorageAdmission.observe(config)["scratch"]["reserved_bytes"] == 0
