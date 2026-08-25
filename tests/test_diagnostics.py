"""Focused current-boundary diagnostic behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.cli import CommandLineInterface
from lambdaforge.controlplane import NativeEnvironmentError
from lambdaforge.diagnostics import DiagnosticClassifier, DiagnosticContext, ErrorCategory


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
