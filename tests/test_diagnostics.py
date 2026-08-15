"""Focused acceptance coverage for actionable operational diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, ControlPlane, JobStore
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.diagnostics import (
    DiagnosticClassifier,
    DiagnosticContext,
    DiagnosticRenderer,
    ErrorCategory,
    LambdaForgeError,
    diagnostic,
    job_failure_diagnostic,
)


def _break_dispatch(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    def broken(cls: type[CommandLineInterface], argv: object) -> int:
        del cls, argv
        raise RuntimeError(message)

    monkeypatch.setattr(CommandLineInterface, "_dispatch", classmethod(broken))


def test_unexpected_cli_error_is_persisted_and_debug_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    _break_dispatch(monkeypatch, "impossible invariant")

    assert CommandLineInterface.main(["project", "status"]) == 10
    normal = capsys.readouterr()
    assert "INTERNAL ERROR" in normal.err
    assert "probably a LambdaForge bug" in normal.err
    assert "Traceback (most recent call last)" not in normal.err
    records = tuple((tmp_path / "lambdaforge/logs/errors").glob("error-*.json"))
    assert len(records) == 1
    persisted = json.loads(records[0].read_text(encoding="utf-8"))
    assert "Traceback (most recent call last)" in persisted["traceback"]
    assert persisted["diagnostic"]["category"] == "internal"

    assert CommandLineInterface.main(["project", "status", "--debug"]) == 10
    debug = capsys.readouterr()
    assert "Debug" in debug.err
    assert "RuntimeError" in debug.err
    assert "Traceback (most recent call last)" in debug.err


def test_json_error_is_equivalent_and_contains_no_human_preamble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    _break_dispatch(monkeypatch, "unclassified state")

    assert CommandLineInterface.main(["project", "status", "--json"]) == 10
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["status"] == "error"
    assert payload["category"] == "internal"
    assert payload["exit_code"] == 10
    assert payload["retryable"] == "unknown"
    assert payload["commands"]
    assert "debug" not in payload


def test_secret_values_are_redacted_from_terminal_and_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    secret_message = (
        "password=swordfish token=topsecret "
        "https://researcher:credential-value@example.invalid/data"
    )
    _break_dispatch(monkeypatch, secret_message)

    assert CommandLineInterface.main(["project", "status", "--debug"]) == 10
    terminal = capsys.readouterr().err
    record = next((tmp_path / "lambdaforge/logs/errors").glob("error-*.json"))
    persisted = record.read_text(encoding="utf-8")
    for secret in ("swordfish", "topsecret", "credential-value"):
        assert secret not in terminal
        assert secret not in persisted
    assert "***" in terminal and "***" in persisted


def test_missing_remote_dataset_root_is_a_preflight_configuration_error() -> None:
    profile = ClusterProfile(
        "atlas",
        transport="ssh",
        scheduler="slurm",
        host="atlas.example.invalid",
        workspace="/work/researcher",
        environment="existing",
    )

    with pytest.raises(LambdaForgeError) as raised:
        ControlPlane(ClusterCatalog({"atlas": profile})).submit(
            Path("examples/dataset-recipe.yaml"), cluster="atlas"
        )

    value = raised.value.diagnostic
    assert value.category is ErrorCategory.CONFIGURATION
    assert value.context["cluster"] == "atlas"
    assert value.context["dataset"] == "example-records@1"
    assert "No job was submitted" in value.impact[0]
    commands = dict(value.commands)
    assert commands["Configure dataset storage"] == (
        "lf clusters set atlas storage.dataset_root /persistent/path/to/datasets"
    )
    assert "lf datasets build" in commands["Retry after configuring"]


def test_failed_dataset_job_reports_one_root_and_derived_blocking() -> None:
    record = JobRecord(
        job_id="job-20260815120000-12345678",
        cluster="atlas",
        scheduler="slurm",
        scheduler_id="42",
        state=JobState.FAILED,
        command=("python", "worker.py"),
        work_dir="/work/jobs/42",
        resources={},
        created_at_utc="2026-08-15T12:00:00+00:00",
        updated_at_utc="2026-08-15T12:01:00+00:00",
        metadata={"name": "wisdom-dna@1"},
        job_type="dataset-build",
    )
    logs = json.dumps(
        {
            "kind": "dataset-build",
            "dataset": "wisdom-dna@1",
            "stages": {
                "curate": {
                    "status": "failed",
                    "error": {"type": "ValueError", "message": "duplicate record key"},
                },
                "geometry": {"status": "blocked", "blocked_by": ["curate"]},
                "annotate": {
                    "status": "blocked",
                    "blocked_by": ["curate", "geometry"],
                },
            },
        }
    )

    value = job_failure_diagnostic(record, logs)
    rendered = DiagnosticRenderer().human(value)
    assert value.category is ErrorCategory.EXECUTION
    assert rendered.count("curate FAILED") == 1
    assert "geometry BLOCKED by curate" in rendered
    assert "annotate BLOCKED by curate, geometry" in rendered
    assert f"lf jobs logs {record.job_id} --tail 300" in rendered


def test_jobs_show_failed_uses_the_same_json_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    record = JobRecord(
        job_id="job-20260815130000-12345678",
        cluster="local",
        scheduler="local",
        scheduler_id=None,
        state=JobState.FAILED,
        command=("python", "project_task.py"),
        work_dir=str(tmp_path / "job"),
        resources={},
        created_at_utc="2026-08-15T13:00:00+00:00",
        updated_at_utc="2026-08-15T13:01:00+00:00",
        stderr="ValueError: project task rejected one record\n",
        metadata={"name": "project-task"},
        job_type="task",
    )
    JobStore().write(record)

    assert CommandLineInterface.main(["jobs", "show", record.job_id, "--json"]) == 4
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["category"] == "execution"
    assert payload["job_id"] == record.job_id
    assert payload["context"]["state"] == "failed"
    assert any("jobs logs" in value["command"] for value in payload["commands"])


def test_warning_renderer_describes_continuation_without_error_exit() -> None:
    value = diagnostic(
        ErrorCategory.WARNING,
        "Legacy Python policy is surprising.",
        "The scalar form still works but selects strategy=existing.",
        reason="Managed fallback is disabled by the legacy scalar form.",
        impact=("Execution may continue with the existing compatible interpreter.",),
        fixes=("Use the explicit mapping form for new cluster profiles.",),
        commands=(("Inspect cluster", "lf clusters inspect atlas"),),
    )

    rendered = DiagnosticRenderer().human(value)
    assert value.exit_code == 0
    assert rendered.startswith("WARNING")
    assert "Execution may continue" in rendered


@pytest.mark.parametrize(
    ("message", "category"),
    (
        ("Permission denied (publickey).", ErrorCategory.AUTHENTICATION),
        ("Dataset publish path permission denied.", ErrorCategory.STORAGE),
        ("sbatch: error: invalid qos", ErrorCategory.RESOURCE),
    ),
)
def test_known_provider_messages_keep_distinct_categories(
    message: str, category: ErrorCategory
) -> None:
    value = DiagnosticClassifier().classify(
        RuntimeError(message),
        DiagnosticContext(("run", "experiment.yaml", "--on", "atlas"), "run", "atlas"),
    )

    assert value.category is category
    assert value.commands


def test_invalid_cli_arguments_use_the_configuration_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert CommandLineInterface.main(["does-not-exist"]) == 2
    captured = capsys.readouterr()
    assert "CONFIGURATION ERROR" in captured.err
    assert "lf --help" in captured.err
    assert "usage:" not in captured.err
