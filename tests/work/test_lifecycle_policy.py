"""Operational failures must not overwrite logical evidence or invent retry permission."""

from __future__ import annotations

import copy
import errno
import json
import pickle
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.diagnostics.failure import classify_failure
from lambdaforge.diagnostics.models import (
    ErrorCategory,
    FailureDisposition,
    LambdaForgeError,
    RetryDisposition,
    diagnostic,
)
from lambdaforge.diagnostics.service import work_failure_diagnostic
from lambdaforge.work.attempt_history import retain_attempt
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.models import WorkResources, WorkResult
from lambdaforge.work.recovery import latest_outcomes
from lambdaforge.work.ResultStore import ResultStore
from lambdaforge.work.runner import WorkRunner, _is_gpu_memory_failure, _run_termination
from lambdaforge.work.state import StudyState, execution_evidence, required_evidence
from lambdaforge.work.study import StudyTelemetry


def result_at(root: Path, **changes: Any) -> WorkResult:
    result = WorkResult(
        name="study",
        work_class="tests.work_cases.AdaptiveScoreWork",
        execution_id="execution-1",
        run_id="run-1",
        attempt_id="attempt-0001",
        attempt_number=1,
        scientific_fingerprint="sha256:test",
        status="failed",
        run_dir=root,
        created_at_utc="2026-10-07T00:00:00+00:00",
        started_at_utc="2026-10-07T00:00:00+00:00",
        finished_at_utc="2026-10-07T00:00:01+00:00",
        duration_seconds=1.0,
        seed=4,
        trial={"index": 1, "parameters": {}},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 1, 0, None, 0, 1),
        failure={"type": "OutOfMemoryError", "message": "allocation failed"},
        termination_type="resource_failed",
    )
    return replace(result, **changes)


@pytest.mark.parametrize(
    "kind,message,is_oom",
    [
        ("OutOfMemoryError", "allocator error", True),
        ("RuntimeError", "CUDA out of memory", True),
        ("RuntimeError", "CUDA error: an illegal memory access was encountered", False),
        ("RuntimeError", "expected BFloat16 but found Float", False),
        ("ValueError", "CUDA out of memory is not a valid parameter", False),
        ("EOFError", "invalid consumer input", False),
    ],
)
def test_one_classification_for_retry_result_and_termination(
    tmp_path: Path,
    kind: str,
    message: str,
    is_oom: bool,
) -> None:
    failure = {"type": kind, "message": message}
    policy = classify_failure(failure)
    result = result_at(tmp_path, failure=failure)
    termination, evidence = _run_termination(status="failed", failure=failure, stop_path=None)
    assert _is_gpu_memory_failure(result) is is_oom
    assert policy.automatic_recovery_eligible is is_oom
    assert termination == ("resource_failed" if is_oom else "scientific_failed")
    assert evidence["failure_disposition"] == result.to_dict()["failure_disposition"]
    diagnostic = work_failure_diagnostic(name="study", source="study.yaml", error=failure)
    assert diagnostic.to_dict()["failure_disposition"] == policy.to_dict()
    assert pickle.loads(pickle.dumps(policy)) == policy


def test_worker_boundary_is_required_for_lost_process_retry() -> None:
    for error in (EOFError("pipe closed"), BrokenProcessPool("terminated abruptly")):
        assert not classify_failure(error, phase="run work").automatic_recovery_eligible
        policy = classify_failure(error, phase="worker-process")
        assert policy.reason == "lost_worker"
        assert policy.automatic_recovery_eligible
        assert policy.termination_type == "infrastructure_failed"
    assert not classify_failure(
        ValueError("worker process killed by signal"), phase="worker-process"
    ).automatic_recovery_eligible


def test_disk_full_needs_a_fix_not_blind_retries() -> None:
    policy = classify_failure(OSError(errno.ENOSPC, "No space left on device"))
    assert policy.category is ErrorCategory.STORAGE
    assert policy.retryable is RetryDisposition.AFTER_FIX
    assert not policy.automatic_recovery_eligible


def test_typed_control_plane_diagnostics_survive_serialization() -> None:
    for category in ErrorCategory:
        retryable = (
            RetryDisposition.NO
            if category is ErrorCategory.CANCELLED
            else RetryDisposition.AFTER_FIX
        )
        source = diagnostic(
            category, "Failure", "Specific typed cause", reason="test", retryable=retryable
        )
        typed = classify_failure(LambdaForgeError(source))
        restored = classify_failure({"type": "LambdaForgeError", "diagnostic": source.to_dict()})
        assert restored == typed
        assert not restored.automatic_recovery_eligible

    supplied = FailureDisposition(
        ErrorCategory.RESOURCE,
        "specific_framework_policy",
        "after_fix",
        RetryDisposition.AFTER_FIX,
        termination_type="resource_failed",
    )
    structured = diagnostic(
        ErrorCategory.RESOURCE,
        "Failure",
        "Typed resource cause",
        reason="test",
        failure_disposition=supplied,
    )
    assert (
        classify_failure({"type": "LambdaForgeError", "diagnostic": structured.to_dict()})
        == supplied
    )
    invalid = {**supplied.to_dict(), "automatic_recovery_eligible": "true"}
    with pytest.raises(ValueError, match="retry eligibility"):
        FailureDisposition.from_dict(invalid)
    with pytest.raises(ValueError, match="version"):
        FailureDisposition.from_dict({**supplied.to_dict(), "failure_disposition_version": True})
    restored = classify_failure(
        {
            "type": "LambdaForgeError",
            "diagnostic": {
                **structured.to_dict(),
                "failure_disposition": invalid,
            },
        }
    )
    assert restored.category is ErrorCategory.RESOURCE
    assert not restored.automatic_recovery_eligible

    class LostWorker(BrokenProcessPool):
        pass

    error = LostWorker("worker disappeared")
    restored = classify_failure(
        {
            "type": type(error).__name__,
            "phase": "worker-process",
            "exception_types": [base.__name__ for base in type(error).__mro__],
        }
    )
    assert restored == classify_failure(error, phase="worker-process")


def test_latest_attempt_is_independent_of_delivery_order(tmp_path: Path) -> None:
    failed = result_at(tmp_path)
    success = replace(
        failed,
        status="succeeded",
        attempt_number=2,
        attempt_id="attempt-0002",
        failure=None,
        termination_type="completed",
    )
    assert latest_outcomes((success, failed, success)) == (success,)
    next_rung = replace(success, fidelity={"target": 100})
    assert len(latest_outcomes((success, next_rung))) == 2
    state = StudyState.from_execution(
        {"evidence": {"required_missing": 0}},
        [success.to_dict(), failed.to_dict()],
        status="succeeded",
    )
    assert state.evidence == "complete"
    assert state.health == "degraded"
    assert state.prior_attempt_failures == 1
    assert state.physical_history_complete
    missing_history = StudyState.from_execution({}, [success.to_dict()], status="succeeded")
    assert not missing_history.physical_history_complete


def test_final_execution_uses_logical_success_and_preserves_physical_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    runner = importlib.import_module("lambdaforge.work.runner")
    config = WorkConfig.from_mapping(
        {
            "name": "study",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [4, 7],
            "objective": {"metric": "score", "mode": "max"},
        },
        source=tmp_path / "study.yaml",
    )
    failed = result_at(tmp_path / "run-1/attempt-0001")
    success = replace(
        failed,
        status="succeeded",
        failure=None,
        termination_type="completed",
        attempt_id="attempt-0002",
        attempt_number=2,
        metrics={"score": 0.8},
        run_dir=tmp_path / "run-1/attempt-0002",
    )
    other = replace(
        success,
        seed=7,
        run_id="run-2",
        attempt_number=1,
        attempt_id="attempt-0001",
        run_dir=tmp_path / "run-2/attempt-0001",
        metrics={"score": 0.6},
    )
    monkeypatch.setattr(runner, "_execute_group", lambda *_args, **_kw: (success, failed, other))
    monkeypatch.setattr(WorkRunner, "_compact_outcomes", lambda *_args: None)
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    assert len(result.runs) == 3  # No failed physical Attempt is erased or refunded.
    assert result.summary["failed_runs"] == 0
    assert result.summary["completed_runs"] == 2
    assert result.summary["best"]["value"] == pytest.approx(0.7)
    assert result.to_dict()["lifecycle"]["health"] == "degraded"
    assert result.to_dict()["lifecycle"]["prior_attempt_failures"] == 1


def test_execution_obligations_do_not_mix_phases_fidelities_or_delivery_order() -> None:
    requirements = [
        {
            "key": "search-5",
            "required": True,
            "candidate": 1,
            "seed": 4,
            "phase": "search",
            "fidelity": 5,
        },
        {
            "key": "search-10",
            "required": True,
            "candidate": 1,
            "seed": 4,
            "phase": "search",
            "fidelity": 10,
        },
        {
            "key": "confirmation",
            "required": True,
            "candidate": 1,
            "seed": 4,
            "phase": "confirmation",
        },
    ]
    design = {"type": "adaptive", "evidence": {"requirements": requirements}}
    base = {
        "trial": {"index": 1},
        "seed": 4,
        "attempt_number": 2,
        "status": "succeeded",
        "termination_type": "completed",
    }
    runs = [
        {**base, "fidelity": {"target": 5}, "study_phase": "search"},
        {
            **base,
            "fidelity": {"target": 5},
            "study_phase": "search",
            "attempt_number": 1,
            "status": "failed",
            "termination_type": "resource_failed",
        },
        {**base, "study_phase": "confirmation"},
    ]
    evidence = execution_evidence(runs, design)
    assert evidence["required_completed"] == 2
    assert evidence["missing_requirement_keys"] == ["search-10"]
    assert execution_evidence(list(reversed(runs)), design) == evidence


def test_composed_study_debt_is_not_limited_to_the_first_definition(tmp_path: Path) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "composed",
            "steps": [
                {"name": name, "run": "tests.work_cases.AdaptiveScoreWork", "seeds": [4, 7]}
                for name in ("first", "second")
            ],
        },
        source=tmp_path / "composed.yaml",
    )
    success = result_at(
        tmp_path, name="first", status="succeeded", failure=None, termination_type="completed"
    )
    summary = WorkRunner._summary(config, [success, replace(success, seed=7, run_id="run-2")])
    assert summary["evidence"]["required_runs"] == 4
    assert summary["evidence"]["required_missing"] == 2
    state = StudyState.from_execution(summary, [success.to_dict()], status="completed")
    assert state.evidence == "incomplete"
    assert state.final_status == "failed"


def test_physical_history_is_bounded_but_cost_and_failures_are_not() -> None:
    record: dict[str, Any] = {}
    for index in range(1, 21):
        record.update(
            attempt_id=f"attempt-{index:04d}",
            state="failed",
            duration_seconds=5.0,
            failure={"type": "OutOfMemoryError", "message": str(index)},
        )
        record = retain_attempt(record)
        assert retain_attempt(record) == record
    assert len(record["attempt_history"]) == 8
    assert record["attempt_statistics"]["failed"] == 20
    assert record["attempt_statistics"]["duration_seconds"] == 100
    assert record["attempt_statistics"]["history_complete"] is True
    delayed = {**record, "attempt_id": "attempt-0001"}
    assert retain_attempt(delayed)["attempt_statistics"] == record["attempt_statistics"]
    legacy = retain_attempt({"attempt_id": "attempt-0021", "state": "failed"})
    assert legacy["attempt_statistics"]["history_complete"] is False
    gap = retain_attempt({**record, "attempt_id": "attempt-0022"})
    assert gap["attempt_statistics"]["history_complete"] is False


def test_telemetry_archives_failure_before_retry_and_keeps_success_logical(tmp_path: Path) -> None:
    telemetry = StudyTelemetry(tmp_path / "study")
    requirement = {"key": "required-1", "required": True, "candidate": 1, "seed": 4}
    specification = {
        "trial_index": 1,
        "seed": 4,
        "trial_parameters": {},
        "evidence_requirement": requirement,
    }
    telemetry.initialize(
        name="study",
        execution_id="execution-1",
        strategy="repeated",
        objective=None,
        specifications=(specification,),
        design={"type": "repeated", "evidence": {"requirements": [requirement]}},
    )
    telemetry.schedule((specification,))
    first = result_at(tmp_path / "run" / "attempt-0001")
    telemetry.run_finished(specification, first)
    telemetry.run_finished(specification, first)  # Duplicate observation is not another Attempt.
    telemetry.run_retrying(specification, reason="safer headroom", retry=1)
    snapshot = telemetry.refresh()
    assert snapshot["counts"]["active_runs"] == 0
    assert snapshot["counts"]["queued_runs"] == 1
    assert snapshot["lifecycle"]["evidence"] == "pending_recovery"
    run = snapshot["candidates"][0]["runs"][0]
    assert run["attempt_history"][0]["failure"]["type"] == "OutOfMemoryError"
    assert run["attempt_statistics"]["failed"] == 1
    second = tmp_path / "run" / "attempt-0002"
    telemetry.run_started(
        specification,
        run_dir=second,
        metrics_path=second / "metrics.jsonl",
        training_metrics_path=second / "training-metrics.jsonl",
    )
    telemetry.run_finished(
        specification,
        replace(
            first,
            run_dir=second,
            attempt_id=second.name,
            attempt_number=2,
            status="succeeded",
            failure=None,
            termination_type="completed",
        ),
    )
    telemetry.candidate_states(active=(), ranked=(1,), finished=True)
    snapshot = telemetry.refresh()
    assert snapshot["lifecycle"]["evidence"] == "complete"
    assert snapshot["lifecycle"]["health"] == "degraded"
    assert snapshot["required_failed"] == 0
    assert snapshot["lifecycle"]["prior_attempt_failures"] == 1
    assert snapshot["candidates"][0]["runs"][0]["attempt_statistics"]["completed"] == 2
    compact = json.loads((telemetry.root / "overview.json").read_text())
    assert compact["lifecycle"] == snapshot["lifecycle"]
    assert "attempt_history" not in json.dumps(compact)
    telemetry.run_finished(specification, first)  # A delayed old observation cannot undo recovery.
    assert telemetry.refresh()["lifecycle"] == snapshot["lifecycle"]


def test_required_debt_and_censoring_are_not_operational_failure() -> None:
    design = {
        "type": "adaptive",
        "evidence": {
            "requirements": [
                {"key": "a", "candidate": 1, "required": True},
                {"key": "b", "candidate": 1, "required": True},
                {"key": "future", "candidate": 2, "required": True},
            ]
        },
    }
    snapshot = {
        "finished": True,
        "candidates": [
            {
                "trial": 1,
                "runs": [
                    {"state": "pruned", "evidence_requirement": {"key": "a", "required": True}}
                ],
            }
        ],
    }
    before = copy.deepcopy(snapshot)
    obligations = required_evidence(snapshot, design)
    assert obligations["required_runs"] == 2
    assert obligations["required_missing"] == 1
    state = StudyState.from_snapshot({**snapshot, **obligations})
    assert state.health == "healthy"
    assert state.evidence == "incomplete"
    assert state.censored_runs == 1
    assert snapshot == before
    assert StudyState.from_snapshot({"status": "unknown"}).health != "terminal_failure"


def test_legacy_result_lifecycle_projection_does_not_modify_files(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    execution = root / "study" / "execution-1"
    execution.mkdir(parents=True)
    manifest = execution / "result.json"
    manifest.write_text(
        json.dumps(
            {
                "execution_result_version": 1,
                "name": "study",
                "execution_id": "execution-1",
                "status": "failed",
                "summary": {"evidence": {"required_missing": 1}},
                "runs": [result_at(execution).to_dict()],
            }
        )
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    store = ResultStore(root)
    assert store.list()[0]["lifecycle"]["evidence"] == "incomplete"
    assert store.select("execution-1")["lifecycle"]["health"] == "terminal_failure"
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
