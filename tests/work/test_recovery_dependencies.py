"""Failure, revision and quota fixtures never contact a live service or cluster."""

import io
import json
import shutil
import urllib.request

import pytest
import yaml

from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.work import WorkConfig, WorkRunner, recovery_dependency_preview
from lambdaforge.work.recovery_dependencies import repair_recovery_dependencies
from tests.controlplane.test_storage_management import descriptor
from tests.work.test_single_work_recovery import snapshot


def setup(root, monkeypatch):
    work = root / "jobs/job-test/work"
    work.mkdir(parents=True)
    source = work / "source.yaml"
    source.write_text(yaml.safe_dump({"run": "tests.work_cases.FrozenAcquisitionRecoveryWork"}))
    monkeypatch.setenv("LAMBDAFORGE_CACHE_ROOT", str(root / "cache"))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-test")
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b"original observation")

    monkeypatch.setattr(urllib.request, "urlopen", request)
    config = WorkConfig.from_yaml(source)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    return config, first, calls


def test_terminal_quota_gc_and_accepted_revision_do_not_reacquire(tmp_path, monkeypatch):
    from lambdaforge.work import runner

    config, first, calls = setup(tmp_path, monkeypatch)
    plan = StorageOperations.gc(
        {**descriptor(tmp_path), "cache_max_size": 1, "automatic_gc": True}, {}, apply=True
    )
    assert any(p["reason"] == "retained-recovery-dependency" for p in plan["protected_items"])
    assert plan["protected_bytes"] > 0 and plan["quota_unresolved_bytes"] > 0
    assert all(
        p.get("reference_chains") for p in plan["protected_items"] if p["category"] == "work_cache"
    )
    original = runner._code_identity
    monkeypatch.setattr(
        runner, "_code_identity", lambda root: {**original(root), "revision": "fix"}
    )
    second = WorkRunner().run(config, resume_execution=first.execution_dir, accept_code_change=True)
    assert second.status == "succeeded"
    assert len(calls) == 2
    assert second.runs[-1].run_id == first.runs[0].run_id
    preview = recovery_dependency_preview(first.execution_dir, tmp_path / "cache")
    assert preview["present"] == 2 and preview["unavailable"] == 0
    shutil.rmtree(tmp_path / "jobs/job-test")  # fixture simulates explicit history deletion
    released = StorageOperations.gc({**descriptor(tmp_path), "cache_max_size": 1}, {}, apply=False)
    assert not any(
        p["reason"] == "retained-recovery-dependency" for p in released["protected_items"]
    )


def test_missing_observation_does_not_replay_changed_url(tmp_path, monkeypatch):
    config, first, calls = setup(tmp_path, monkeypatch)
    for cache in (tmp_path / "cache/work").iterdir():
        if cache.is_dir() and not cache.name.startswith("."):
            shutil.rmtree(cache)
    before = snapshot(tmp_path)
    preview = repair_recovery_dependencies(first.execution_dir, tmp_path / "cache")
    assert preview["unavailable"] == 2
    assert snapshot(tmp_path) == before
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network replay"))
    with pytest.raises(ValueError, match="Missing original recovery input"):
        WorkRunner().run(config, resume_execution=first.execution_dir)
    assert len(calls) == 2


def test_source_receipts_are_logical_not_absolute_paths(tmp_path, monkeypatch):
    _, first, _ = setup(tmp_path, monkeypatch)
    path = next((first.runs[0].run_dir.parent.parent / "checkpoints/maps/frozen").glob("*.json"))
    reference = json.loads(path.read_text())["result"]["source"]["$lambdaforge_managed_file"]
    assert reference["observed"] and len(reference["namespace"]) == 64
    assert "path" not in reference


def test_legacy_artifact_copy_requires_original_digest_and_no_live_controller(
    tmp_path, monkeypatch
):
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    config, first, calls = setup(tmp_path, monkeypatch)
    output = first.runs[0].run_dir / "artifacts/original-copy"
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b"original observation")
    for cache in (tmp_path / "cache/work").iterdir():
        if cache.is_dir() and not cache.name.startswith("."):
            shutil.rmtree(cache)
    before = snapshot(tmp_path)
    plan = repair_recovery_dependencies(first.execution_dir, tmp_path / "cache")
    assert plan["copy_candidates"] == 2
    assert snapshot(tmp_path) == before
    with CrossProcessFileLock(
        first.execution_dir / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            repair_recovery_dependencies(first.execution_dir, tmp_path / "cache", apply=True)
    applied = repair_recovery_dependencies(first.execution_dir, tmp_path / "cache", apply=True)
    assert applied["repaired"] == 2
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded" and len(calls) == 2


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_bound_map_records_input_and_policy_for_spawn(tmp_path, executor):
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {"run": "tests.work_cases.BoundMapRecoveryWork", "with": {"executor": executor}}
        )
    )
    config = WorkConfig.from_yaml(source)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    record = next((first.runs[0].run_dir.parent.parent / "checkpoints/maps/bound").glob("*.json"))
    value = json.loads(record.read_text())
    assert value["dependencies"][0]["$lambdaforge_managed_file"]["scope"] == "checkpoint"
    assert len(value["scientific_policy"]) == 64
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert json.loads(record.read_text()) == value


def test_local_job_result_reference_keeps_external_execution_namespace(tmp_path, monkeypatch):
    config, _, calls = setup(tmp_path, monkeypatch)
    other = WorkRunner(execution_root=tmp_path / "local-executions").run(config)
    (tmp_path / "jobs/job-test/result.json").write_text(json.dumps(other.to_dict()))
    plan = StorageOperations.gc({**descriptor(tmp_path), "cache_max_size": 1}, {}, apply=False)
    assert any(
        str(other.execution_dir) in chain
        for row in plan["protected_items"]
        for chain in row.get("reference_chains", [])
    )
    assert len(calls) == 2  # same exact identity, not a new observation
