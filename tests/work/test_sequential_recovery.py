"""Native sequential recovery preserves acquisition sessions, typed outputs and Attempts."""

import json

import pytest
import yaml

from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.recovery import sequential_recovery_preview
from tests.work.test_single_work_recovery import snapshot


def config(root, *, fail_first=True, failures=1):
    path = root / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "pipeline",
                "steps": [
                    {
                        "name": "discover",
                        "run": "tests.work_cases.SequentialRecoveryWork",
                        "with": {"label": "discovery", "failures": failures if fail_first else 0},
                    },
                    {
                        "name": "select",
                        "run": "tests.work_cases.SequentialRecoveryWork",
                        "with": {
                            "label": "selection",
                            "source": {"from": "discover.answer"},
                            "failures": 0 if fail_first else failures,
                        },
                    },
                    {"name": "finish", "run": "tests.work_cases.Consumer", "with": {"number": 42}},
                ],
            }
        )
    )
    return WorkConfig.from_yaml(path)


@pytest.mark.parametrize("fail_first", [True, False])
def test_native_sequence_retry_keeps_completed_steps_and_map_items(tmp_path, fail_first):
    current = config(tmp_path, fail_first=fail_first)
    first = WorkRunner().run(current)
    assert first.status == "failed"
    before = snapshot(first.execution_dir)
    plan = sequential_recovery_preview(first.execution_dir)
    assert plan["reuse_runs"] == (0 if fail_first else 1)
    assert plan["retry_runs"] == 1
    assert plan["pending_runs"] == (2 if fail_first else 1)
    WorkRunner().run(current, resume_execution=first.execution_dir, dry_run=True)
    assert before == snapshot(first.execution_dir)
    second = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert second.execution_id == first.execution_id
    failed = first.runs[-1]
    root = failed.run_dir.parent.parent / "checkpoints"
    assert json.loads((root / "calls.json").read_text()) == [0, 1, 2, 2, 3]
    assert json.loads((root / "session.json").read_text()) == {"created": 1}
    assert (root / "session.json").read_bytes() == before[str(root / "session.json")]
    retried = [r for r in second.runs if r.run_id == failed.run_id]
    assert [r.attempt_number for r in retried] == [1, 2]
    if not fail_first:
        assert second.runs[0].run_dir == first.runs[0].run_dir
    assert sequential_recovery_preview(first.execution_dir)["reuse_runs"] == 3


def test_repeated_interruption_and_code_ack(tmp_path, monkeypatch):
    from lambdaforge.work import runner

    current = config(tmp_path, failures=2)
    first = WorkRunner().run(current)
    original = runner._code_identity
    monkeypatch.setattr(
        runner, "_code_identity", lambda root: {**original(root), "revision": "fix"}
    )
    with pytest.raises(ValueError, match="code changed"):
        WorkRunner().run(current, resume_execution=first.execution_dir)
    second = WorkRunner().run(
        current, resume_execution=first.execution_dir, accept_code_change=True
    )
    assert second.status == "failed"
    third = WorkRunner().run(current, resume_execution=first.execution_dir)
    assert third.status == "succeeded"
    assert third.runs[2].attempt_number == 3


def test_corrupt_predecessor_and_live_lock_refuse_before_mutation(tmp_path):
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    current = config(tmp_path, fail_first=False)
    first = WorkRunner().run(current)
    with CrossProcessFileLock(
        first.execution_dir / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            WorkRunner().run(current, resume_execution=first.execution_dir)
    artifact = first.runs[0].run_dir / first.runs[0].artifacts[0].path
    artifact.write_text("corrupt")
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError, match="corrupt output"):
        WorkRunner().run(current, resume_execution=first.execution_dir)
    assert snapshot(first.execution_dir) == before


@pytest.mark.parametrize("kind", ["receipt", "missing-request", "parameter", "interrupted"])
def test_sequence_corruption_identity_and_interruption(tmp_path, kind):
    current = config(tmp_path)
    first = WorkRunner().run(current)
    run = first.runs[0]
    if kind == "parameter":
        raw = json.loads(json.dumps(current.raw))
        raw["steps"][0]["with"]["label"] = "changed science"
        current.source.write_text(yaml.safe_dump(raw))
        current = WorkConfig.from_yaml(current.source)
    elif kind == "receipt":
        receipt = next((run.run_dir.parent.parent / "checkpoints/.managed/records").glob("*.json"))
        receipt.write_text("corrupt")
    else:
        (run.run_dir / "result.json").unlink()
        if kind == "missing-request":
            (run.run_dir / "request.json").unlink()
    before = snapshot(first.execution_dir)
    if kind == "interrupted":
        second = WorkRunner().run(current, resume_execution=first.execution_dir)
        assert second.status == "succeeded"
        assert second.runs[0].run_id == run.run_id
        assert second.runs[0].attempt_number == 2
    else:
        with pytest.raises(ValueError):
            WorkRunner().run(current, resume_execution=first.execution_dir, accept_code_change=True)
        assert snapshot(first.execution_dir) == before


def test_sequential_spent_budget_is_not_refunded(tmp_path):
    current = config(tmp_path)
    raw = json.loads(json.dumps(current.raw))
    raw["steps"][0]["execution"] = {"max_runs": 1}
    current.source.write_text(yaml.safe_dump(raw))
    current = WorkConfig.from_yaml(current.source)
    first = WorkRunner().run(current)
    before = snapshot(first.execution_dir)
    with pytest.raises(ValueError, match="budget exhausted"):
        WorkRunner().run(current, resume_execution=first.execution_dir)
    assert before == snapshot(first.execution_dir)
