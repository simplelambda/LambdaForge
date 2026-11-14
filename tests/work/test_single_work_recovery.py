"""Native single Work recovery uses owned files, never a fabricated Study controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.recovery import fixed_recovery_preview


def config(root: Path, *, ram: str = "16GiB", filename: str = "config.yaml", **changes):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input.txt"
    if not source.exists():
        source.write_text("exact scientific input")
    raw = {
        "run": "tests.work_cases.SingleFileRecoveryWork",
        "with": {"source": {"file": str(source)}, "count": 3},
        "resources": {"cpu": 1, "memory": ram},
        **changes,
    }
    path = root / filename
    import yaml

    path.write_text(yaml.safe_dump(raw))
    return WorkConfig.from_yaml(path)


def snapshot(root: Path):
    return {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_single_recovery_reuses_files_with_new_attempt_resources_and_staging_label(tmp_path):
    original = config(tmp_path)
    first = WorkRunner().run(original)
    assert first.status == "failed" and len(first.runs) == 1
    run = first.runs[0]
    checkpoints = run.run_dir.parent.parent / "checkpoints"
    files = {p: p.read_bytes() for p in checkpoints.glob("producers/*.bin")}
    assert len(files) == 2
    assert not (checkpoints / "producers/2.bin").exists()
    manifest = (first.execution_dir / "execution.json").read_bytes()
    before = snapshot(first.execution_dir)
    preview = fixed_recovery_preview(first.execution_dir)
    assert preview["strategy"] == "single"
    assert preview["retry_runs"] == 1 and preview["preserved_completed_runs"] == 0
    row = preview["runs"][0]
    assert row["next_attempt"] == 2 and row["run_id"] == run.run_id
    assert row["checkpoint_collection"]["registered_files"] == 3  # files plus failure marker
    assert row["checkpoint_collection"]["payload_hashes"] == "not_checked"
    current = config(tmp_path, filename="enrich_structures.yaml", ram="96GiB")
    WorkRunner().run(current, dry_run=True, resume_execution=first.execution_dir)
    assert before == snapshot(first.execution_dir)
    second = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert second.status == "succeeded" and second.execution_id == first.execution_id
    latest = second.runs[-1]
    assert latest.run_id == run.run_id and latest.attempt_number == 2
    assert latest.resumed_from_checkpoint and latest.primary_result == {
        "files": 3,
        "resuming": True,
    }
    assert latest.requested_resources.memory == 96 * 1024**3
    assert all(path.read_bytes() == content for path, content in files.items())
    assert (first.execution_dir / "execution.json").read_bytes() == manifest
    assert not (first.execution_dir / "hpo-control/state.json").exists()
    assert fixed_recovery_preview(first.execution_dir)["reuse_runs"] == 1
    again = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert again.runs[-1].attempt_number == 2


@pytest.mark.parametrize("change", ["parameters", "input", "seeds", "objective"])
def test_single_recovery_rejects_scientific_changes_even_with_code_ack(tmp_path, change):
    first = WorkRunner().run(config(tmp_path))
    kwargs = {}
    if change == "parameters":
        kwargs["with"] = {"source": {"file": str(tmp_path / "input.txt")}, "count": 4}
    elif change == "input":
        (tmp_path / "input.txt").write_text("changed input bytes")
    elif change == "seeds":
        kwargs["seeds"] = [1]
    else:
        kwargs["objective"] = {"metric": "score", "mode": "max"}
    current = config(tmp_path, **kwargs)
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError):
        WorkRunner().run(current, resume_execution=first.execution_dir, accept_code_change=True)
    assert before == snapshot(first.execution_dir)


def test_single_code_ack_and_live_controller_lock(tmp_path, monkeypatch):
    from lambdaforge.work import runner

    current = config(tmp_path)
    first = WorkRunner().run(current)
    origin_code = runner._code_identity
    monkeypatch.setattr(
        runner, "_code_identity", lambda root: {**origin_code(root), "revision": "compatible-fix"}
    )
    with pytest.raises(ValueError, match="code changed"):
        WorkRunner().run(current, resume_execution=first.execution_dir)
    with CrossProcessFileLock(
        first.execution_dir / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            WorkRunner().run(current, resume_execution=first.execution_dir, accept_code_change=True)
    assert not (first.execution_dir / "recovery-history.jsonl").exists()
    result = WorkRunner().run(
        current, resume_execution=first.execution_dir, accept_code_change=True
    )
    assert result.runs[-1].run_id == first.runs[0].run_id
    assert result.status == "succeeded"


def test_single_interrupted_attempt_without_result_recovers_public_files(tmp_path):
    first = WorkRunner().run(config(tmp_path))
    first.runs[0].run_dir.joinpath("result.json").unlink()
    (first.execution_dir / "result.json").unlink()
    assert fixed_recovery_preview(first.execution_dir)["runs"][0]["action"] == "retry"
    second = WorkRunner().run(config(tmp_path), resume_execution=first.execution_dir)
    assert second.status == "succeeded" and second.runs[-1].attempt_number == 2


@pytest.mark.parametrize("corruption", ["receipt", "foreign", "symlink", "payload"])
def test_corrupt_checkpoint_is_rejected_or_rebuilt_not_used_as_evidence(tmp_path, corruption):
    first = WorkRunner().run(config(tmp_path))
    root = first.runs[0].run_dir.parent.parent / "checkpoints"
    records = list((root / ".managed/records").glob("*.json"))
    record = next(p for p in records if json.loads(p.read_text())["key"] == "producers/0.bin")
    if corruption == "receipt":
        record.write_text("broken-json")
    elif corruption == "foreign":
        value = json.loads(record.read_text())
        value["owner"]["run_id"] = "run-" + "f" * 20
        record.write_text(json.dumps(value))
    elif corruption == "symlink":
        root.joinpath("escape").symlink_to(tmp_path / "input.txt")
    else:
        # Same-size corruption is detected by the public managed reader, not metadata preview.
        payload = root / "producers/0.bin"
        payload.write_bytes(b"!" * payload.stat().st_size)
        assert (
            fixed_recovery_preview(first.execution_dir)["runs"][0]["checkpoint_collection"][
                "payload_hashes"
            ]
            == "not_checked"
        )
        second = WorkRunner().run(config(tmp_path), resume_execution=first.execution_dir)
        # The fixture forbids rebuilding earlier files, so corruption must fail loudly.
        assert second.status == "failed"
        return
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError):
        WorkRunner().run(config(tmp_path), resume_execution=first.execution_dir)
    assert before == snapshot(first.execution_dir)


def test_dataset_is_published_only_after_complete_validated_file_recovery(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(registry))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "datasets"))
    current = config(tmp_path, run="tests.work_cases.SingleFileRecoveryDatasetWork")
    first = WorkRunner().run(current)
    assert first.status == "failed" and not first.runs[-1].datasets
    assert not registry.exists()
    second = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert registry.exists() and len(second.runs[-1].datasets) == 1
    assert second.runs[-1].primary_result["files"] == 3


@pytest.mark.parametrize(
    "resources", [{"cpu": 2, "memory": "96GiB", "time": "2h"}, {"cpu": 1, "memory": "8GiB"}]
)
def test_reviewed_operational_cpu_memory_wall_time_can_change(tmp_path, resources):
    first = WorkRunner().run(config(tmp_path))
    current = config(tmp_path, resources=resources)
    outcome = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert outcome.status == "succeeded"
    audit = json.loads((first.execution_dir / "recovery-history.jsonl").read_text())
    assert audit["effective_resources"] == current.resources.to_dict()


def test_single_resource_review_does_not_waive_gpu_topology(tmp_path):
    first = WorkRunner().run(config(tmp_path))
    current = config(tmp_path, resources={"cpu": 1, "gpu": 1})
    with pytest.raises(ValueError, match="CPU/RAM/wall-time"):
        WorkRunner().run(current, dry_run=True, resume_execution=first.execution_dir)


@pytest.mark.parametrize("policy", [{"max_runs": 1}, {"max_time": 0.000001}])
def test_single_recovery_does_not_refund_spent_execution_budgets(tmp_path, policy):
    current = config(tmp_path, execution=policy)
    first = WorkRunner().run(current)
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError, match="budget is exhausted"):
        fixed_recovery_preview(first.execution_dir)
    with pytest.raises(ValueError, match="budget is exhausted"):
        WorkRunner().run(current, resume_execution=first.execution_dir)
    assert snapshot(first.execution_dir) == before
