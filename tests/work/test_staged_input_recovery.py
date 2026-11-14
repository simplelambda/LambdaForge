"""Role-bound historical recovery across native bounded-input staging names."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.recovery import restore_seed_context, validate_execution
from tests.work.test_single_work_recovery import snapshot


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-fixture")
    work = tmp_path / "job-fixture/work"
    inputs = work / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "0000-data.txt").write_text("science")
    (inputs / "0001-data.txt").write_text("reference")
    raw = {
        "name": "original",
        "run": "tests.work_cases.BoundedInputRecoveryWork",
        "with": {
            "source": {"file": "inputs/0000-data.txt"},
            "reference": {"file": "inputs/0001-data.txt"},
        },
    }
    source = work / "config.yaml"
    source.write_text(yaml.safe_dump(raw))
    manifest = {
        "bundle_version": 4,
        "config": raw,
        "staged": [
            [f"inputs/{p.name}", "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()]
            for p in inputs.iterdir()
        ],
    }
    (work / "manifest.json").write_text(json.dumps(manifest))
    config = WorkConfig.from_yaml(source)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    local = tmp_path / "relocated"
    (local / "a").mkdir(parents=True)
    (local / "b").mkdir()
    (local / "a/data.txt").write_text("science")
    (local / "b/data.txt").write_text("reference")
    current = {
        **raw,
        "with": {"source": {"file": "a/data.txt"}, "reference": {"file": "b/data.txt"}},
    }
    selected = local / "renamed.yaml"
    selected.write_text(yaml.safe_dump(current))
    return first, config, WorkConfig.from_yaml(selected)


def test_worker_reconstructs_original_identity_and_retains_checkpoint_bytes(staged, monkeypatch):
    from lambdaforge.work import runner

    first, _, selected = staged
    original = runner._code_identity
    monkeypatch.setattr(
        runner, "_code_identity", lambda root: {**original(root), "revision": "fix"}
    )
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError, match="code changed"):
        WorkRunner().run(selected, resume_execution=first.execution_dir)
    plan = WorkRunner().run(
        selected, resume_execution=first.execution_dir, accept_code_change=True, dry_run=True
    )
    assert plan.execution_id == first.execution_id
    assert snapshot(first.execution_dir) == before
    recovered = WorkRunner().run(
        selected, resume_execution=first.execution_dir, accept_code_change=True
    )
    assert recovered.status == "succeeded"
    assert recovered.runs[-1].run_id == first.runs[0].run_id
    assert recovered.runs[-1].attempt_number == 2
    assert recovered.runs[-1].parameters == first.runs[0].parameters
    assert (first.execution_dir / "execution.json").read_bytes() == before[
        str(first.execution_dir / "execution.json")
    ]
    checkpoint = first.runs[0].run_dir.parent.parent / "checkpoints/producers/0.bin"
    assert checkpoint.read_bytes() == before[str(checkpoint)]


@pytest.mark.parametrize(
    "change", ["same-size", "roles", "receipt", "source", "parameter", "missing-receipt"]
)
def test_changed_content_or_binding_is_not_authorized_by_code_ack(staged, change):
    first, config, selected = staged
    if change == "same-size":
        (selected.source.parent / "a/data.txt").write_text("SCIENCE")
    elif change == "roles":
        raw = json.loads(json.dumps(selected.raw))
        raw["with"]["source"], raw["with"]["reference"] = (
            raw["with"]["reference"],
            raw["with"]["source"],
        )
        selected = WorkConfig.from_mapping(raw, source=selected.source)
    elif change == "receipt":
        path = config.source.parent / "manifest.json"
        raw = json.loads(path.read_text())
        raw["staged"][0][1] = "sha256:" + "f" * 64
        path.write_text(json.dumps(raw))
    elif change == "source":
        (config.source.parent / "inputs/0000-data.txt").write_text("SCIENCE")
    elif change == "missing-receipt":
        (config.source.parent / "manifest.json").unlink()
    else:
        raw = json.loads(json.dumps(selected.raw))
        raw["objective"] = {"metric": "score", "mode": "max"}
        selected = WorkConfig.from_mapping(raw, source=selected.source)
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError) as error:
        WorkRunner().run(selected, resume_execution=first.execution_dir, accept_code_change=True)
    if change in {"same-size", "roles", "source", "receipt"}:
        assert "role" in str(error.value)
    assert snapshot(first.execution_dir) == before


def test_public_retry_and_dependencies_share_preview_contract(staged, tmp_path):
    first, original, selected = staged
    profile = ClusterProfile(
        "fixture",
        transport="ssh",
        host="never-connect.invalid",
        workspace=str(tmp_path / "storage"),
    )
    factory = SimpleNamespace(
        transport=lambda _: LocalTransport(), scheduler=lambda *_: SimpleNamespace()
    )
    jobs = JobService(
        ClusterCatalog({profile.name: profile}), JobStore(tmp_path / "index"), factory
    )
    now = datetime.now(timezone.utc).isoformat()
    jobs.store.write(
        JobRecord(
            "job-fixture",
            profile.name,
            "local",
            None,
            JobState.FAILED,
            (),
            str(original.source.parent),
            original.resources.to_dict(),
            now,
            now,
            config_path=str(original.source),
            job_type="work",
            metadata={
                "source_config_path": str(selected.source),
                "submission_mode": "asynchronous",
                "study_expected": False,
                "run_arguments": [],
            },
        )
    )
    before = snapshot(tmp_path)
    handle = jobs.retry("job-fixture", accept_code_change=True, dry_run=True)
    assert handle.recovery_plan["configuration"]["scientific_compatibility"] == "verified"
    assert handle.recovery_plan["runs"][0]["run_id"] == first.runs[0].run_id
    assert before == snapshot(tmp_path)
    import sys

    jobs.factory.reader_command = lambda *_: (sys.executable,)
    dependencies = jobs.recovery_dependencies("job-fixture", limit=1)
    assert dependencies["unavailable"] == 0
    assert before == snapshot(tmp_path)
    manifest = validate_execution(first.execution_dir)
    restored = restore_seed_context(selected, manifest)
    assert (
        WorkRunner._study_identity(restored, code_identity=manifest["code_identity"])
        == manifest["scientific_fingerprint"]
    )


def test_failed_checkpoint_storage_is_not_disposable(staged, tmp_path):
    from lambdaforge.controlplane.StorageOperations import StorageOperations

    first, _, _ = staged
    before = snapshot(first.execution_dir)
    descriptor = {
        "state_root": str(tmp_path / "state"),
        "cache_root": str(tmp_path / "cache"),
        "run_root": str(tmp_path),
        "lease_root": str(tmp_path / "leases"),
        "dataset_root": str(tmp_path / "datasets"),
    }
    plan = StorageOperations.gc(
        {**descriptor, "automatic_gc": True, "cache_max_size": 1}, {}, apply=False
    )
    assert not any(item["category"] == "checkpoints" for item in plan["candidates"])
    assert snapshot(first.execution_dir) == before


def test_new_staging_prefix_and_configuration_name_do_not_change_recovery(staged, tmp_path):
    first, _, selected = staged
    work = tmp_path / "new-worker"
    (work / "inputs").mkdir(parents=True)
    (work / "inputs/0042-current-source.txt").write_bytes(
        (selected.source.parent / "a/data.txt").read_bytes()
    )
    (work / "inputs/0099-current-reference.txt").write_bytes(
        (selected.source.parent / "b/data.txt").read_bytes()
    )
    raw = json.loads(json.dumps(selected.raw))
    raw["with"] = {
        "source": {"file": "inputs/0042-current-source.txt"},
        "reference": {"file": "inputs/0099-current-reference.txt"},
    }
    source = work / "config.yaml"
    source.write_text(yaml.safe_dump(raw))
    recovered = WorkRunner().run(
        WorkConfig.from_yaml(source), resume_execution=first.execution_dir, accept_code_change=True
    )
    assert recovered.status == "succeeded"
    assert recovered.runs[-1].run_id == first.runs[0].run_id
