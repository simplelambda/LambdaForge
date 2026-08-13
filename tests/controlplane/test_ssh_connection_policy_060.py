"""SSH connection reuse and independent command timeout semantics."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane import (
    ClusterProfile,
    ControlPlaneFactory,
    SshConnectionPolicy,
    SshTransport,
)
from lambdaforge.controlplane.RemoteCommandTimeout import RemoteCommandTimeout


def test_openssh_enables_idle_multiplex_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    transport = SshTransport(
        "atlas",
        connection=SshConnectionPolicy(persist=75, command_timeout=None),
        control_root=tmp_path / "ssh",
    )
    transport.run(("true",))
    command, kwargs = calls[0]
    rendered = " ".join(command)
    assert "ControlMaster=auto" in rendered
    assert "ControlPersist=75s" in rendered
    assert "ControlPath=" in rendered
    assert kwargs["timeout"] is None


def test_explicit_command_timeout_is_not_a_connect_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    transport = SshTransport("atlas", control_root=tmp_path / "ssh")
    with pytest.raises(RemoteCommandTimeout, match="2"):
        transport.run(("sleep", "10"), timeout=2)


def test_factory_reuses_transport_for_same_effective_profile() -> None:
    factory = ControlPlaneFactory()
    profile = ClusterProfile("local")
    assert factory.transport(profile) is factory.transport(profile)
    assert factory.transport(
        ClusterProfile("local", python="different-python")
    ) is not factory.transport(profile)
