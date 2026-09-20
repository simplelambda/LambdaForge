"""Focused current-boundary diagnostic behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.cli import CommandLineInterface
from lambdaforge.controlplane import NativeEnvironmentError
from lambdaforge.diagnostics import (
    DiagnosticClassifier,
    DiagnosticContext,
    ErrorCategory,
    work_failure_diagnostic,
)


def test_invalid_work_is_local_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = tmp_path / "invalid.yaml"
    config.write_text("name: invalid\nrun: builtins.len\n", encoding="utf-8")
    assert CommandLineInterface.main(["validate", str(config), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "function" in " ".join(payload["errors"])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("--help",), "Run reproducible scientific Work"),
        (("help",), "Run reproducible scientific Work"),
        (("help", "clusters", "add"), "--gpu-claim-command"),
        (("clusters", "add", "help"), "--gpu-claim-command"),
        (("clusters", "help"), "bootstrap"),
    ),
)
def test_help_is_a_successful_cli_operation(
    arguments: tuple[str, ...], expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert CommandLineInterface.main(arguments) == 0
    captured = capsys.readouterr()
    assert expected in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    ("message", "category"),
    (
        ("Permission denied (publickey).", ErrorCategory.AUTHENTICATION),
        ("Dataset publish path permission denied.", ErrorCategory.STORAGE),
        ("sbatch: error: invalid qos", ErrorCategory.RESOURCE),
    ),
)
def test_provider_failures_keep_distinct_categories(message: str, category: ErrorCategory) -> None:
    value = DiagnosticClassifier().classify(
        RuntimeError(message),
        DiagnosticContext(("run", "work.yaml", "--on", "atlas"), "run", "atlas"),
    )
    assert value.category is category


def test_native_inventory_with_libssh2_is_environment_not_connection() -> None:
    context = DiagnosticContext(
        ("clusters", "bootstrap", "atlas", "--project", "."),
        "clusters bootstrap atlas",
        "atlas",
    )
    message = (
        "Native package inventory differs from the resolved environment identity: "
        "1 package difference(s): changed 'libssh2': subdir 'linux-64' -> 'noarch'"
    )

    typed = DiagnosticClassifier().classify(NativeEnvironmentError(message), context)
    legacy = DiagnosticClassifier().classify(RuntimeError(message), context)
    unrelated = DiagnosticClassifier().classify(
        RuntimeError("Package libssh2 was present in a report."), context
    )

    assert typed.category is ErrorCategory.ENVIRONMENT
    assert legacy.category is ErrorCategory.ENVIRONMENT
    assert unrelated.category is ErrorCategory.INTERNAL


@pytest.mark.parametrize(
    "message",
    (
        "ssh: connect to host atlas: Connection refused",
        "scp transport failed",
        "SFTP session closed",
        "Network is unreachable",
    ),
)
def test_connection_markers_require_transport_semantics(message: str) -> None:
    value = DiagnosticClassifier().classify(
        RuntimeError(message),
        DiagnosticContext(("doctor", "--on", "atlas"), "doctor", "atlas"),
    )

    assert value.category is ErrorCategory.CONNECTION


def test_gpu_admission_failures_are_never_reported_as_internal_framework_bugs() -> None:
    context = DiagnosticContext(("run", "study.yaml", "--on", "atlas"), "run", "atlas")

    impossible = DiagnosticClassifier().classify(
        ValueError(
            "resources.gpu_memory=90.0GiB exceeds total memory on every allocated GPU; "
            "no Run can ever be admitted."
        ),
        context,
    )
    probe = DiagnosticClassifier().classify(
        RuntimeError("CUDA GPU admission could not read current device memory safely."),
        context,
    )

    assert impossible.category is ErrorCategory.CONFIGURATION
    assert probe.category is ErrorCategory.ENVIRONMENT


def test_immutable_model_pickle_failure_is_classified_as_internal() -> None:
    value = DiagnosticClassifier().classify(
        TypeError("cannot pickle 'mappingproxy' object"),
        DiagnosticContext(("run", "study.yaml"), "run"),
    )

    assert value.category is ErrorCategory.INTERNAL
    assert "lambdaforge bug" in " ".join(value.details).lower()


@pytest.mark.parametrize(
    "message",
    (
        "'<' not supported between instances of 'ActiveResourceEvidence' and "
        "'ActiveResourceEvidence'",
        "Object of type PosixPath is not JSON serializable",
    ),
)
def test_internal_resource_and_analysis_type_errors_are_not_configuration_errors(
    message: str,
) -> None:
    value = DiagnosticClassifier().classify(
        TypeError(message),
        DiagnosticContext(("run", "study.yaml"), "run"),
    )

    assert value.category is ErrorCategory.INTERNAL


def test_work_failure_diagnostic_is_strict_json_with_a_path_context(tmp_path: Path) -> None:
    value = work_failure_diagnostic(
        name="study",
        source=tmp_path / "study.yaml",
        error={"type": "RuntimeError", "message": "failed"},
        run_dir=tmp_path / "runs" / "attempt-0001",
    ).to_dict()

    json.dumps(value, allow_nan=False)
    assert value["context"]["run_dir"] == str(tmp_path / "runs" / "attempt-0001")
