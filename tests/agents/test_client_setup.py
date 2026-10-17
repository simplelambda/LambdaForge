"""Official client handoff simulated in disposable config, without global client mutations."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from lambdaforge.agents.client import CodexClient
from lambdaforge.agents.doctor import agent_context, integration_doctor
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.agents.services import AgentServices


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="client-fixture"\nversion="1"\n')
    monkeypatch.setattr("lambdaforge.agents.client.shutil.which", lambda name: "/owned/fake-codex")
    value = CodexClient(ProjectScope(tmp_path))
    calls = []
    records = {"foreign": {"name": "foreign", "transport": {"command": "untouched"}}}

    def query(*args):
        calls.append(args)
        if args == ("--version",):
            output = "codex-cli fixture"
        elif args[-1] == "--help":
            output = "Usage: codex mcp add -- <COMMAND>... --json"
        elif args[:2] == ("mcp", "list"):
            output = json.dumps(list(records.values()))
        elif args[:2] == ("mcp", "get"):
            output = json.dumps(records[args[2]])
        else:
            assert args[:2] == ("mcp", "add")
            records[args[2]] = {
                "name": args[2],
                "enabled": True,
                "transport": {"command": args[4], "args": list(args[5:])},
            }
            output = "Registered"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(value, "_query", query)
    monkeypatch.setattr(
        "lambdaforge.agents.mcp_server.dependency_status", lambda: {"available": True}
    )
    return value, calls, records


def test_connect_is_preview_first_idempotent_and_preserves_foreign_entries(client):
    value, calls, records = client
    original = json.loads(json.dumps(records["foreign"]))
    plan = value.connect()
    assert not any(call[:2] == ("mcp", "add") and "--help" not in call for call in calls)
    with pytest.raises(ValueError, match="plan-hash"):
        value.connect(apply=True)
    applied = value.connect(apply=True, plan_hash=plan["plan_hash"])
    assert applied["configuration_modified"] and not applied["connection_verified"]
    assert records["foreign"] == original
    assert records[value.name]["transport"]["command"] == sys.executable
    plan = value.connect()
    assert plan["action"] == "unchanged"
    assert not value.connect(apply=True, plan_hash=plan["plan_hash"])["configuration_modified"]
    assert sum(call[:2] == ("mcp", "add") and "--help" not in call for call in calls) == 1


def test_existing_conflict_or_missing_sdk_cannot_be_overwritten_or_installed(client, monkeypatch):
    value, calls, records = client
    plan = value.connect()
    monkeypatch.setattr(
        "lambdaforge.agents.mcp_server.dependency_status", lambda: {"available": False}
    )
    with pytest.raises(ValueError, match="plan-hash"):
        value.connect(apply=True, plan_hash=plan["plan_hash"])
    current = value.connect()
    with pytest.raises(ValueError, match="SDK"):
        value.connect(apply=True, plan_hash=current["plan_hash"])
    records[value.name] = {"name": value.name, "transport": {"command": "foreign-program"}}
    with pytest.raises(ValueError, match="already has another"):
        value.connect()
    assert not any(call[:2] == ("mcp", "add") and "--help" not in call for call in calls)


def test_missing_codex_context_is_local_and_doctor_does_not_claim_handshake(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="context-fixture"\nversion="1"\n')
    monkeypatch.setattr("lambdaforge.agents.client.shutil.which", lambda name: None)
    service = AgentServices(tmp_path)
    from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile

    service._catalog = ClusterCatalog(
        {"local": ClusterProfile("local")}, project=service.scope.project
    )
    context = agent_context(service)
    assert context["permissions"] == ["READ", "PLAN"]
    assert context["clusters"] == ["local"]
    report = integration_doctor(service)
    assert not report["codex"]["installed"]
    assert not report["configuration_modified"] and not report["installation_performed"]
    assert "not performed" in report["client_handshake"]
    assert list(tmp_path.iterdir()) == [tmp_path / "pyproject.toml"]


def test_launch_needs_tty_and_does_not_upgrade_permissions(client, monkeypatch):
    value, calls, records = client
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="TTY"):
        value.launch()
    assert not any(call[:2] == ("mcp", "add") for call in calls)
