"""Storage admission/collection regressions use only synthetic owned roots."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import psutil
import pytest

from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.StorageAdmission import StorageAdmission
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.cache import WorkCache


def descriptor(root: Path) -> dict[str, object]:
    return {
        "state_root": str(root / "state"),
        "cache_root": str(root / "cache"),
        "run_root": str(root / "jobs"),
        "dataset_root": str(root / "datasets"),
        "lease_root": str(root / "host-leases"),
        "safety": {"min_free": 50, "min_free_percent": 0},
    }


@pytest.mark.parametrize("operation", ["status", "reconcile", "gc", "prune_environments"])
@pytest.mark.parametrize("populated", [False, True])
def test_storage_read_only_operations_do_not_create_roots_or_lock_files(
    tmp_path: Path, operation: str, populated: bool
) -> None:
    config = descriptor(tmp_path)
    if populated:
        content = tmp_path / "cache/work/entry/content"
        content.parent.mkdir(parents=True)
        content.write_bytes(b"cached")
        (content.parent.parent / ".entry.lease").write_bytes(b"")
    before = {path: path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}
    if operation == "gc":
        StorageOperations.gc(config, {}, apply=False)
    elif operation == "prune_environments":
        StorageOperations.prune_environments(config, (), apply=False)
    else:
        getattr(StorageOperations, operation)(config)
    assert {
        path: path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")
    } == before


def test_job_compaction_preview_never_creates_execution_writer_metadata(tmp_path: Path) -> None:
    job_id = "job-test"
    attempt = (
        tmp_path
        / "jobs"
        / job_id
        / "work/.lambdaforge/runs/work/execution-test"
        / "runs/run-test/attempts/attempt-0001"
    )
    artifacts = attempt / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "partial.bin").write_bytes(b"interrupted output")
    before = {path: path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}
    preview = StorageOperations.compact_job(descriptor(tmp_path), job_id, apply=False)
    assert preview["reclaimable_bytes"] == len(b"interrupted output")
    assert {
        path: path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")
    } == before


def owner(name: str) -> ProcessIdentity:
    return ProcessIdentity.create(os.getpid(), os.getpgrp(), psutil.Process().cmdline(), name)


def capacity(monkeypatch: pytest.MonkeyPatch, free: list[int], *, device: str = "scratch") -> None:
    monkeypatch.setattr(
        StorageAdmission,
        "filesystem",
        staticmethod(
            lambda path, config: {
                "device": device,
                "capacity_bytes": 1000,
                "free_bytes": free[0],
                "used_bytes": 1000 - free[0],
                "safety_bytes": 50,
                "pressure": "NORMAL",
                "free_inodes": 100,
                "inode_accounting": True,
            }
        ),
    )


def test_publication_reserves_destination_volume_and_releases_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = descriptor(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_STORAGE_POLICY", json.dumps(config))
    destination = tmp_path / "external/report.json"
    capacity(monkeypatch, [200], device="durable")
    with pytest.raises(RuntimeError, match="copy failed"):
        with StorageAdmission.transaction(destination, 100, purpose="test-publication"):
            leases = list((tmp_path / "host-leases/storage-leases/durable").glob("job-*.json"))
            assert len(leases) == 1
            record = json.loads(leases[0].read_text())
            assert record["requested_bytes"] == 100
            assert record["run_root"] == str(destination.parent)
            assert record["purpose"] == "test-publication"
            assert record["destination"] == str(destination)
            raise RuntimeError("copy failed")
    assert not list((tmp_path / "host-leases/storage-leases/durable").glob("job-*.json"))
    assert not destination.exists()


def test_reflink_does_not_require_duplicate_payload_but_fallback_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.work import snapshot

    capacity(monkeypatch, [60])  # Only ten bytes above the safety floor.
    monkeypatch.setenv("LAMBDAFORGE_STORAGE_POLICY", json.dumps(descriptor(tmp_path)))
    monkeypatch.setattr(StorageOperations, "gc", lambda *_args, **_kw: {})
    source = tmp_path / "source"
    source.write_bytes(b"large" * 100)

    def clone(source, target):
        # Synthetic kernel clone, independent bytes and no reported physical allocation.
        target.write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setattr(snapshot, "_try_reflink", clone)
    destination = tmp_path / "clone"
    assert snapshot.copy_file(source, destination) == str(destination)
    assert destination.read_bytes() == source.read_bytes()
    monkeypatch.setattr(snapshot, "_try_reflink", lambda *_args: None)
    with pytest.raises(OSError, match="publication-copy blocked"):
        snapshot.copy_file(source, tmp_path / "full-copy")
    assert not (tmp_path / "full-copy").exists()
    assert not list((tmp_path / "host-leases/storage-leases/scratch").glob("job-*.json"))


def test_publication_pressure_never_inventories_researcher_destination_for_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = descriptor(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_STORAGE_POLICY", json.dumps(config))
    capacity(monkeypatch, [60])
    collected: list[dict] = []
    monkeypatch.setattr(
        StorageOperations, "gc", lambda descriptor, *_args, **_kw: collected.append(descriptor)
    )
    with pytest.raises(OSError, match="No partial publication"):
        with StorageAdmission.transaction(tmp_path / "research/job-evidence", 40, purpose="copy"):
            pytest.fail("Insufficient space cannot begin a copy")
    assert len(collected) == 1
    assert collected[0]["run_root"] == config["run_root"]
    assert not list((tmp_path / "host-leases/storage-leases/scratch").glob("job-*.json"))


def test_reconcile_preview_is_read_only_and_apply_records_drift(tmp_path: Path) -> None:
    config = descriptor(tmp_path)
    cache = tmp_path / "cache/work/data"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"first")
    preview = StorageOperations.reconcile(config)
    assert not preview["baseline_available"] and not preview["applied"]
    assert not (tmp_path / "state").exists()
    StorageOperations.reconcile(config, apply=True)
    unchanged = StorageOperations.reconcile(config)
    assert unchanged["baseline_available"]
    assert unchanged["drift"]["state"] == {"bytes_delta": 0, "files_delta": 0}
    cache.write_bytes(b"first plus five")
    changed = StorageOperations.reconcile(config)
    assert changed["drift"]["work_cache"]["bytes_delta"] == 10
    assert cache.read_bytes() == b"first plus five"


def test_gc_audit_counts_actual_collection_not_a_plan_that_became_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cache/work/entry"
    path.mkdir(parents=True)
    (path / "data").write_bytes(b"still in use")
    monkeypatch.setattr(StorageOperations, "_work_cache_busy", lambda _path: False)

    def busy():
        raise TimeoutError("A worker acquired the cache after preview")

    monkeypatch.setattr(
        StorageOperations,
        "_work_cache_lock",
        lambda _path: SimpleNamespace(acquire=busy, release=lambda: None),
    )
    result = StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
    assert len(result["candidates"]) == 1 and result["reclaimed_bytes"] == 0
    record = json.loads((tmp_path / "state/storage-gc.jsonl").read_text())
    assert record["planned_items"] == 1 and record["items"] == 0
    assert record["collected"] == []
    assert (path / "data").read_bytes() == b"still in use"


def test_storage_cli_routes_preserve_preview_and_explain_drift(tmp_path: Path) -> None:
    from lambdaforge.cli.CommandLineInterface import CommandLineInterface
    from lambdaforge.cli.parser import build_parser

    args = build_parser().parse_args(["storage", "reconcile", "--on", "scratch", "--json"])
    assert args.on == "scratch" and args.json and not args.apply
    args = build_parser().parse_args(["storage", "reconcile", "--apply"])
    assert args.apply
    snapshot = {"cluster": "scratch", **StorageOperations.reconcile(descriptor(tmp_path))}
    human = CommandLineInterface._storage_reconcile_summary(snapshot)
    assert "read-only preview" in human and "No previous baseline" in human
    assert "delta bytes" in human and "datasets" in human


@pytest.mark.parametrize("content", ["not json", '{"storage_ledger_version":9,"categories":{}}'])
def test_reconcile_rejects_corrupt_ledger_without_overwriting(tmp_path: Path, content: str) -> None:
    state = tmp_path / "state"
    state.mkdir()
    ledger = state / "storage-ledger.json"
    ledger.write_text(content)
    with pytest.raises(ValueError, match="ledger"):
        StorageOperations.reconcile(descriptor(tmp_path), apply=True)
    assert ledger.read_text() == content


def test_runtime_build_uses_gc_handshake_even_with_completion(tmp_path: Path) -> None:
    from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease
    from lambdaforge.controlplane.PythonRuntimeResolver import PythonRuntimeResolver

    cache = tmp_path / "cache"
    cache.mkdir()
    marker = PurePosixPath(str(cache / ".python-runtime-python-test.lock"))
    complete = PurePosixPath(str(cache / "runtimes/python-test/.lambdaforge-python-runtime.json"))
    Path(complete).parent.mkdir(parents=True)
    Path(complete).write_text("{}")
    transport = LocalTransport()
    resolver = PythonRuntimeResolver(lock_timeout=0)
    with CrossProcessFileLock(
        cache / ".gc.lock", shared=False, timeout_seconds=1, poll_interval_seconds=0.01
    ):
        assert not resolver._acquire(transport, marker, complete)
        assert not Path(marker).exists()
    assert resolver._acquire(transport, marker, complete)
    with CacheBuildLease.maintain(transport, marker, python="python3"):
        owner = json.loads((Path(marker) / "owner.json").read_text())
        assert owner["operation"] == "runtime-build"
        assert owner["environment_id"] == "python-test"
        assert not resolver._acquire(transport, marker, complete)
    assert not Path(marker).exists()


def test_stale_runtime_build_reclaims_only_its_owned_temporary(tmp_path: Path) -> None:
    config = descriptor(tmp_path)
    root = tmp_path / "cache/runtimes"
    abandoned = root / ".python-test.tmp-owned"
    abandoned.mkdir(parents=True)
    (abandoned / "large.bin").write_bytes(b"old bytes")
    unknown = root / ".python-unknown.tmp-other"
    unknown.mkdir()
    (unknown / "large.bin").write_bytes(b"keep uncertain ownership")
    marker = tmp_path / "cache/.python-runtime-python-test.lock"
    marker.mkdir()
    (marker / "owner.json").write_text(
        json.dumps(
            {
                "pid": 2147483647,
                "host": socket.gethostname(),
                "heartbeat": time.time() - 3600,
            }
        )
    )
    preview = StorageOperations.gc(config, {})
    assert [row["path"] for row in preview["candidates"]] == [str(abandoned)]
    StorageOperations.gc(config, {}, apply=True)
    assert not abandoned.exists() and not marker.exists()
    assert (unknown / "large.bin").read_bytes() == b"keep uncertain ownership"


def test_dead_build_controller_does_not_reclaim_running_or_inaccessible_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cache/.environment-build-env-test.lock"
    marker.mkdir(parents=True)
    (marker / "owner.json").write_text(
        json.dumps(
            {
                "pid": 2147483647,
                "host": socket.gethostname(),
                "heartbeat": time.time() - 3600,
            }
        )
    )
    child = SimpleNamespace(
        cmdline=lambda: [
            str(tmp_path / "cache/environments/.env-test.tmp-build/bin/python"),
            "-m",
            "pip",
            "install",
        ]
    )
    monkeypatch.setattr(psutil, "process_iter", lambda: iter((child,)))
    assert not StorageOperations._orphan_lease(marker)

    def inaccessible():
        raise psutil.AccessDenied(1)

    monkeypatch.setattr(
        psutil, "process_iter", lambda: iter((SimpleNamespace(cmdline=inaccessible),))
    )
    assert not StorageOperations._orphan_lease(marker)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter(()))
    assert StorageOperations._orphan_lease(marker)


def test_invalid_complete_environment_is_never_replaced_behind_live_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.controlplane.CommandResult import CommandResult
    from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle

    environment(tmp_path, "env-existing")
    provider = ManagedEnvironmentProvider()
    monkeypatch.setattr(
        provider,
        "_verify_reusable",
        lambda *_args: CommandResult(1, stderr="corrupt Python inventory"),
    )
    bundle = ExecutionBundle(
        "bundle",
        tmp_path,
        tmp_path / "config.yaml",
        tmp_path / "manifest.json",
        0,
        environment_id="env-existing",
    )
    profile = ClusterProfile(
        "test",
        transport="local",
        workspace=str(tmp_path),
        python="python3",
        environment="managed",
        storage=ClusterStoragePolicy.from_mapping(descriptor(tmp_path), workspace=str(tmp_path)),
    )
    with pytest.raises(RuntimeError, match="prefix was preserved"):
        provider.prepare(profile, LocalTransport(), bundle, remote_bundle_dir=tmp_path)
    assert (tmp_path / "cache/environments/env-existing/.lambdaforge-environment.json").exists()
    assert not list((tmp_path / "cache/environments").glob(".env-existing.*"))
    assert not list((tmp_path / "cache").glob(".environment-build-*.lock"))


def test_commitments_survive_low_observed_usage_and_release_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    assert StorageAdmission.reserve(config, owner("job-first"), 40)["admitted"]
    assert StorageAdmission.reserve(config, owner("job-second"), 40)["admitted"]
    denied = StorageAdmission.reserve(config, owner("job-third"), 100)
    assert not denied["admitted"] and denied["reserved_bytes"] == 80
    StorageAdmission.release(config, owner("job-first"))
    StorageAdmission.release(config, owner("job-second"))
    assert StorageAdmission.reserve(config, owner("job-third"), 100)["admitted"]


def test_observation_is_lightweight_and_does_not_double_reserve_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    StorageAdmission.reserve(config, owner("job-first"), 40)
    monkeypatch.setattr(StorageOperations, "_usage", lambda _path: pytest.fail("Deep scan"))
    assert StorageAdmission.observe(config)["scratch"]["admissible_bytes"] == 60
    assert (
        StorageAdmission.observe(config, owner_job_id="job-first")["scratch"]["admissible_bytes"]
        == 100
    )


def test_corrupt_commitment_cannot_increase_admissible_space(tmp_path: Path, monkeypatch) -> None:
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    StorageAdmission.reserve(config, owner("job-first"), 40)
    lease = tmp_path / "host-leases/storage-leases/scratch/job-first.json"
    record = json.loads(lease.read_text())
    record["requested_bytes"] = -400
    lease.write_text(json.dumps(record))
    observed = StorageAdmission.observe(config)["scratch"]
    assert observed["ownership_unresolved"] and observed["admissible_bytes"] == 0
    assert not StorageAdmission.reserve(config, owner("job-second"), 1)["admitted"]


def test_foreign_host_commitment_is_not_reclaimed_from_local_pid_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity(monkeypatch, [150])
    config = descriptor(tmp_path)
    StorageAdmission.reserve(config, owner("job-first"), 40)
    lease = tmp_path / "host-leases/storage-leases/scratch/job-first.json"
    record = json.loads(lease.read_text())
    record.update(host="another-host", owner={**record["owner"], "pid": 2147483647})
    lease.write_text(json.dumps(record))
    assert StorageAdmission.observe(config)["scratch"]["ownership_unresolved"]
    denied = StorageAdmission.reserve(config, owner("job-second"), 1)
    assert denied["reason"] == "unresolved-storage-host"
    assert lease.exists()


def test_environment_build_acquisition_is_atomic_with_collection(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = PurePosixPath(str(cache / ".environment-build-env-one.lock"))
    complete = PurePosixPath(str(cache / "environments/env-one/.lambdaforge-environment.json"))
    transport = LocalTransport()
    with CrossProcessFileLock(
        cache / ".gc.lock", shared=False, timeout_seconds=1, poll_interval_seconds=0.01
    ):
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            ManagedEnvironmentProvider._acquire(transport, marker, complete, timeout=0)
        assert not Path(marker).exists()
    assert ManagedEnvironmentProvider._acquire(transport, marker, complete, timeout=1)
    record = json.loads((Path(marker) / "owner.json").read_text())
    assert record["environment_id"] == "env-one"
    assert record["host"] == socket.gethostname() and record["pid"] == os.getpid()
    environment(tmp_path, "env-one")
    plan = StorageOperations.gc({**descriptor(tmp_path), "cache_max_age": 0}, {}, apply=True)
    assert not plan["candidates"]
    assert plan["protected_items"][0]["reason"] == "environment-build"
    # A ready prefix still cannot bypass another builder's ownership during reuse.
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        ManagedEnvironmentProvider._acquire(transport, marker, complete, timeout=0)


def test_unknown_job_or_incomplete_environment_blocks_bootstrap_eviction(tmp_path: Path) -> None:
    environment(tmp_path, "env-ready")
    incomplete = environment(tmp_path, ".env-incomplete.tmp-owned")
    (incomplete / ".lambdaforge-environment.json").unlink()
    unknown = tmp_path / "jobs/job-unknown"
    unknown.mkdir(parents=True)
    plan = StorageOperations.prune_environments(descriptor(tmp_path), (), apply=True)
    assert not plan["pruned"]
    assert incomplete.exists() and (tmp_path / "cache/environments/env-ready").exists()


def test_host_inventory_does_not_override_recovery_protected_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "jobs/job-original"
    job.mkdir(parents=True)
    (job / "state.json").write_text('{"state":"succeeded"}')
    monkeypatch.setattr(
        StorageOperations,
        "compact_job",
        lambda *args, **kwargs: pytest.fail("Recovery dependency must not be compacted"),
    )
    plan = StorageOperations.gc(
        descriptor(tmp_path),
        {"protected_jobs": ["job-original"], "terminal_jobs": ["job-original"]},
        apply=True,
    )
    assert not plan["candidates"]


def test_cleanup_summary_reports_actual_reclamation_not_the_preview_estimate() -> None:
    from lambdaforge.cli.CommandLineInterface import CommandLineInterface

    message = CommandLineInterface._clean_summary(
        {
            "applied": True,
            "candidates": [{}],
            "reclaimable_bytes": 1024,
            "reclaimed_bytes": 0,
        }
    )
    assert "0 bytes reclaimed" in message and "1 planned" in message
    assert "Removed" not in message


def test_insufficient_disk_waits_and_gc_can_enable_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free = [150]
    capacity(monkeypatch, free)
    config = descriptor(tmp_path)
    assert not StorageAdmission.reserve(config, owner("job-first"), 120)["admitted"]
    free[0] += 20  # Safe collector frees actual physical bytes, not an apparent-size prediction.
    assert StorageAdmission.reserve(config, owner("job-first"), 120)["admitted"]


def test_volume_capacity_and_inode_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_blocks=1000,
            f_frsize=1,
            f_bavail=150 if Path(path).name == "scratch" else 900,
            f_files=200,
            f_favail=100,
        ),
    )
    scratch, data = tmp_path / "scratch", tmp_path / "data"
    scratch.mkdir()
    data.mkdir()
    policy = {**descriptor(tmp_path), "safety": {"min_free": 20, "min_free_percent": 10}}
    observed = StorageAdmission.filesystem(scratch, policy)
    assert observed["safety_bytes"] == 100 and observed["free_bytes"] == 150
    assert StorageAdmission.filesystem(data, policy)["free_bytes"] == 900
    # The publication volume's free bytes never enter scratch admission.
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(
            f_blocks=1000,
            f_frsize=1,
            f_bavail=900,
            f_files=200,
            f_favail=0,
        ),
    )
    assert StorageAdmission.filesystem(scratch, policy)["pressure"] == "CRITICAL"


def environment(root: Path, name: str) -> Path:
    path = root / "cache" / "environments" / name
    path.mkdir(parents=True)
    (path / ".lambdaforge-environment.json").write_text("{}")
    (path / "heavy.bin").write_bytes(b"x" * 30)
    return path


def test_exact_active_and_job_environment_references(tmp_path: Path) -> None:
    for name in ("env-active", "env-job", "env-old-a", "env-old-b"):
        environment(tmp_path, name)
    state = tmp_path / "state"
    state.mkdir()
    (state / "active-environment").write_text(
        str(tmp_path / "cache/environments/env-active/bin/python")
    )
    job = tmp_path / "jobs/job-one"
    job.mkdir(parents=True)
    (job / "state.json").write_text(json.dumps({"state": "running"}))
    (job / "request.json").write_text(
        json.dumps(
            {
                "command": [str(tmp_path / "cache/environments/env-job/bin/python")],
            }
        )
    )
    config = {**descriptor(tmp_path), "cache_max_age": 0}
    result = StorageOperations.gc(config, {}, apply=True)
    assert {row["name"] for row in result["candidates"]} == {"env-old-a", "env-old-b"}
    assert {row["name"] for row in result["protected_items"]} == {"env-active", "env-job"}


def test_active_venv_interpreter_symlink_retains_invoked_environment(tmp_path: Path) -> None:
    active = environment(tmp_path, "env-active")
    environment(tmp_path, "env-old")
    (active / "bin").mkdir()
    (active / "bin/python").symlink_to(os.sys.executable)
    state = tmp_path / "state"
    state.mkdir()
    (state / "active-environment").write_text(str(active / "bin/python"))
    result = StorageOperations.gc({**descriptor(tmp_path), "cache_max_age": 0}, {}, apply=True)
    assert active.exists()
    assert [row["name"] for row in result["candidates"]] == ["env-old"]


@pytest.mark.parametrize("key", ["cache_max_age", "terminal_jobs"])
def test_policy_rejects_nonfinite_retention(key: str) -> None:
    values = {key: float("nan") if key == "cache_max_age" else {"grace_period": float("nan")}}
    with pytest.raises(ValueError):
        ClusterStoragePolicy.from_mapping(values, workspace="/owned")


@pytest.mark.parametrize("alive", [True, False])
def test_environment_lease_and_orphan_temporary(tmp_path: Path, alive: bool) -> None:
    cache = tmp_path / "cache"
    temporary = environment(tmp_path, ".env-one.tmp-abc")
    (temporary / ".lambdaforge-environment.json").unlink()
    marker = cache / ".environment-build-env-one.lock"
    marker.mkdir()
    (marker / "owner.json").write_text(
        json.dumps(
            {
                "pid": os.getpid() if alive else 2147483647,
                "host": socket.gethostname(),
                "heartbeat": time.time() - 600,
            }
        )
    )
    work = WorkCache(cache / "work/old")
    work.put("key", b"reconstructible")
    result = StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
    assert any(row["category"] == "work_cache" for row in result["candidates"])
    assert temporary.exists() is alive
    assert marker.exists() is alive


def test_live_work_cache_does_not_block_unrelated_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    active = WorkCache(cache / "work/active", hold_gc_lease=True)
    inactive = WorkCache(cache / "work/inactive")
    active.put("keep", b"keep")
    inactive.put("remove", b"remove")
    try:
        result = StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
        assert {row["name"] for row in result["candidates"]} == {"inactive"}
        assert active.get("keep") == b"keep"
    finally:
        active.close()


def test_package_caches_participate_in_quota_and_preview_is_read_only(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for name in ("pip", "conda-pkgs", "runtime-packages", "runtime-managers"):
        path = cache / name / "entry"
        path.mkdir(parents=True)
        (path / "bytes").write_bytes(b"x" * 50)
    config = {**descriptor(tmp_path), "cache_max_size": 50}
    preview = StorageOperations.gc(config, {})
    assert preview["reclaimable_bytes"] == 150
    assert len(list(cache.glob("*/entry/bytes"))) == 4
    result = StorageOperations.gc(config, {}, apply=True)
    assert result["reclaimed_bytes"] == 150
    assert len(list(cache.glob("*/entry/bytes"))) == 1


def test_crash_after_cache_rename_is_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = WorkCache(tmp_path / "cache/work/entry")
    work.put("data", b"cached")
    remove = StorageOperations._remove_owned
    monkeypatch.setattr(
        StorageOperations,
        "_remove_owned",
        staticmethod(lambda *args: (_ for _ in ()).throw(OSError("injected crash"))),
    )
    with pytest.raises(OSError, match="injected crash"):
        StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
    monkeypatch.setattr(StorageOperations, "_remove_owned", remove)
    assert not (tmp_path / "cache/work/entry").exists()
    StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
    assert not list((tmp_path / "cache/work").glob(".gc-*-*"))


def test_symlinks_unknown_owners_and_datasets_are_never_collected(tmp_path: Path) -> None:
    outside = tmp_path / "user-data"
    outside.mkdir()
    (outside / "evidence").write_bytes(b"science")
    cache = tmp_path / "cache"
    (cache / "work").mkdir(parents=True)
    (cache / "work/attack").symlink_to(outside, target_is_directory=True)
    dataset = tmp_path / "datasets/release"
    dataset.mkdir(parents=True)
    (dataset / "science").write_bytes(b"science")
    unknown = environment(tmp_path, ".env-unknown.tmp-a")
    (unknown / ".lambdaforge-environment.json").unlink()
    StorageOperations.gc({**descriptor(tmp_path), "cache_max_size": 1}, {}, apply=True)
    assert outside.joinpath("evidence").is_file() and dataset.joinpath("science").is_file()
    assert unknown.exists()
    with pytest.raises(ValueError, match="symbolic"):
        StorageOperations.gc({**descriptor(tmp_path), "cache_root": str(cache / "work/attack")}, {})


def test_storage_policy_validates_and_roundtrips() -> None:
    value = ClusterStoragePolicy.from_mapping(
        {
            "safety": {"min_free": "20GiB", "min_free_percent": 10},
            "terminal_jobs": {"grace_period": "2d"},
        },
        workspace="/work",
    )
    assert value.safety_min_free_bytes == 20 * 1024**3
    assert ClusterStoragePolicy.from_mapping(value.to_dict(), workspace="/work") == value
    with pytest.raises(ValueError, match="safety"):
        ClusterStoragePolicy.from_mapping({"safety": {"min_free_percent": 100}}, workspace="/work")
