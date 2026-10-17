"""Bounded local integration context/doctor, delegating capability and client authorities."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from typing import Any

from lambdaforge.agents.client import CodexClient
from lambdaforge.agents.integration import IntegrationService
from lambdaforge.agents.services import AgentServices


def agent_context(service: AgentServices) -> dict[str, Any]:
    """No provider access, client launch, consumer imports, note contents or artifact reads."""
    from lambdaforge.agents.contracts import redact
    from lambdaforge.agents.mcp_server import dependency_status

    return redact(
        {
            "project": asdict(service.project_info()),
            "python": sys.executable,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "integration": IntegrationService(service.scope.root).status(),
            "agents_guide_present": (service.scope.root / "AGENTS.md").is_file(),
            "yaml": asdict(service.list_project_configs(limit=20)),
            "work_classes": asdict(service.find_work_classes(limit=20)),
            "clusters": service._clusters().names(),
            "mcp": dependency_status(),
            "permissions": list(service.permissions),
            "project_code_imports": service.allow_project_code,
            "limits": [
                "metadata snapshot, not provider observation",
                "no scientific action authorization",
                "missing source contracts are not inferred",
                "Codex registry is not a handshake",
            ],
        }
    )


def integration_doctor(service: AgentServices) -> dict[str, Any]:
    """Inspect installed configuration honestly, without registering or installing anything."""
    context = agent_context(service)
    try:
        client = CodexClient(service.scope).status()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        from lambdaforge.agents.contracts import redact

        client = {
            "status": "error",
            "diagnostic": redact(str(error)),
            "configuration_modified": False,
        }
    return {
        "project": context["project"],
        "mcp": context["mcp"],
        "context": context,
        "codex": client,
        "installation_performed": False,
        "configuration_modified": False,
        "client_handshake": "not performed; use the explicit native MCP test/client acceptance",
        "human_approval": "local signed knowledge edits and exact Job cancellation only",
        "http": "opt-in paired loopback READ/PLAN; no public-network serving",
    }
