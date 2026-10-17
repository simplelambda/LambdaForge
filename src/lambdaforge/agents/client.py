"""Official Codex CLI integration with explicit preview/apply, no scientific permission changes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from typing import Any

from lambdaforge.agents.contracts import redact
from lambdaforge.agents.scope import ProjectScope


class CodexClient:
    """Inspect/prepare the real installed client; never install it or rewrite its TOML ourselves."""

    def __init__(self, scope: ProjectScope) -> None:
        self.scope = scope
        self.executable = shutil.which("codex")
        self.name = (
            "lambdaforge-" + hashlib.sha256(scope.project.project_id.encode()).hexdigest()[:12]
        )

    def _query(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        if self.executable is None:
            raise FileNotFoundError(
                "Codex is not installed; install it explicitly or use portable MCP config."
            )
        self.scope.assert_current()
        value = subprocess.run(
            (self.executable, *arguments),
            cwd=self.scope.root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if len(value.stdout.encode()) + len(value.stderr.encode()) > 256 * 1024:
            raise ValueError("Installed client response exceeds the bounded metadata contract.")
        return value

    def status(self) -> dict[str, Any]:
        self.scope.assert_current()
        if self.executable is None:
            return {
                "installed": False,
                "registered": False,
                "server_name": self.name,
                "hint": "Install Codex explicitly or use the existing portable MCP configuration.",
            }
        version = self._query("--version")
        help_value = self._query("mcp", "add", "--help")
        query_help = self._query("mcp", "get", "--help")
        list_help = self._query("mcp", "list", "--help")
        supported = (
            version.returncode
            == help_value.returncode
            == query_help.returncode
            == list_help.returncode
            == 0
            and "-- <COMMAND>" in help_value.stdout
            and "--json" in query_help.stdout
            and "--json" in list_help.stdout
        )
        registered = None
        if supported:
            inventory = self._query("mcp", "list", "--json")
            if inventory.returncode:
                raise RuntimeError(
                    "Cannot inspect Codex registry: " + redact(inventory.stderr.strip())
                )
            rows = json.loads(inventory.stdout)
            if not isinstance(rows, list):
                raise ValueError("Installed Codex MCP inventory is not the verified list contract.")
            if any(row.get("name") == self.name for row in rows if isinstance(row, dict)):
                record = self._query("mcp", "get", self.name, "--json")
                if record.returncode:
                    raise RuntimeError(
                        "The named Codex entry cannot be verified; no configuration is changed."
                    )
                registered = json.loads(record.stdout)
        return {
            "installed": True,
            "executable": self.executable,
            "version": redact(version.stdout.strip()),
            "supported_cli": supported,
            "server_name": self.name,
            "registered": registered is not None,
            "registration_matches": self._matches(registered) if registered is not None else False,
            "client_handshake": "not performed; registry metadata is not an MCP connection",
            "configuration_modified": False,
            "permissions": ["READ", "PLAN"],
        }

    def _matches(self, registered: Any) -> bool:
        if not isinstance(registered, dict):
            return False
        transport = registered.get("transport", registered)
        if not isinstance(transport, dict):
            return False
        return (
            transport.get("command") == sys.executable
            and transport.get("args") == self.server_arguments()
            and registered.get("enabled", True) is True
        )

    def server_arguments(self) -> list[str]:
        return ["-m", "lambdaforge", "ai", "--project", str(self.scope.root), "mcp", "serve"]

    def preview(self) -> dict[str, Any]:
        state = self.status()
        if not state.get("installed") or not state.get("supported_cli"):
            raise ValueError(
                "Installed Codex does not expose the verified MCP add/get grammar. "
                "Use lf ai init --agents codex for a reviewable portable example."
            )
        if state["registered"] and not state["registration_matches"]:
            raise ValueError(
                "The project MCP name already has another configuration; reconcile it in Codex. "
                "LambdaForge will not overwrite a client server."
            )
        from lambdaforge.agents.mcp_server import dependency_status

        payload = {
            "project_id": self.scope.project.project_id,
            "project_root": str(self.scope.root),
            "client": state,
            "command": [
                self.executable,
                "mcp",
                "add",
                self.name,
                "--",
                sys.executable,
                *self.server_arguments(),
            ],
            "action": "unchanged" if state["registered"] else "register",
            "mcp": dependency_status(),
            "permissions": ["READ", "PLAN"],
            "configuration_scope": (
                "official client-managed configuration; apply is explicit consent"
            ),
            "configuration_modified": False,
        }
        payload["plan_hash"] = (
            "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        )
        return payload

    def connect(self, *, apply: bool = False, plan_hash: str | None = None) -> dict[str, Any]:
        if apply:
            from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

            lock = self.scope.path(".lambdaforge/agents/client-connect.lock", internal=True)
            with CrossProcessFileLock(
                lock, shared=False, timeout_seconds=30, poll_interval_seconds=0.05
            ):
                return self._connect_owned(plan_hash=plan_hash)
        return self.preview()

    def _connect_owned(self, *, plan_hash: str | None) -> dict[str, Any]:
        """Revalidate under the native project lock before handing off to the official client."""
        plan = self.preview()
        if plan_hash != plan["plan_hash"]:
            raise ValueError("Review the current connection plan and pass its exact --plan-hash.")
        if not plan["mcp"]["available"]:
            raise ValueError(
                "The optional MCP SDK is absent/incompatible; "
                "install the agents extra explicitly first."
            )
        if plan["action"] == "register":
            value = self._query(
                "mcp", "add", self.name, "--", sys.executable, *self.server_arguments()
            )
            if value.returncode:
                raise RuntimeError(
                    "Official Codex registration failed: " + redact(value.stderr.strip())
                )
        state = self.status()
        if not state.get("registration_matches"):
            raise RuntimeError(
                "Codex did not retain the expected exact project MCP entry; inspect client config."
            )
        return {
            **plan,
            "configuration_modified": plan["action"] == "register",
            "client": state,
            "connection_verified": False,
            "next": (
                "Open Codex /mcp or run lf ai mcp test. Registration is not a client handshake."
            ),
        }

    def launch(self) -> int:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError(
                "Interactive Codex launch requires a TTY; connect/status has a noninteractive path."
            )
        state = self.status()
        if not state.get("registration_matches"):
            raise ValueError("Configure this project's MCP entry with lf ai connect codex first.")
        print(
            f"Launching installed Codex in {self.scope.root}; client permissions are unchanged.",
            flush=True,
        )
        assert self.executable is not None
        return subprocess.call((self.executable,), cwd=self.scope.root)
