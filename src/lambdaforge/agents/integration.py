"""Installed assets and conservative, preview-first consumer integration.

This service never edits scientific configuration, client-global settings or knowledge notes.
CLI apply is a local operator action; it is deliberately not an MCP mutation tool.
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from lambdaforge._version import VERSION
from lambdaforge.agents.knowledge import KnowledgeService
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_bytes, atomic_write_text


def _assets() -> Any:
    return files("lambdaforge").joinpath("agents/assets")


def skill_names() -> tuple[str, ...]:
    return tuple(
        sorted(item.name for item in _assets().joinpath("skills").iterdir() if item.is_dir())
    )


def skill_metadata(name: str) -> dict[str, Any]:
    if name not in skill_names():
        raise ValueError("Unknown installed LambdaForge Skill.")
    document = _assets().joinpath("skills", name, "SKILL.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(document.split("---\n", 2)[1])
    return {**metadata, "version": VERSION, "document": document}


def example_assets() -> dict[str, str]:
    return {
        entry.name: entry.read_text(encoding="utf-8")
        for entry in _assets().joinpath("examples").iterdir()
        if entry.is_file()
    }


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class IntegrationService:
    """Protect personalized files with a managed-hash manifest and optimistic plan checks."""

    def __init__(self, root: str | Path) -> None:
        self.scope = ProjectScope(root)
        self.manifest_path = self.scope.path(".lambdaforge/agents/manifest.json", internal=True)

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": 1, "project_id": self.scope.project.project_id, "files": {}}
        value = json.loads(
            self.scope.file(".lambdaforge/agents/manifest.json", limit=512 * 1024, internal=True)
        )
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("project_id") != self.scope.project.project_id
            or not isinstance(value.get("files"), dict)
        ):
            raise ValueError("Invalid integration manifest; inspect it before syncing.")
        return value

    def _desired(self, agents: str) -> dict[str, str]:
        if agents not in {"portable", "codex", "none"}:
            raise ValueError("Agent mode must be portable, codex or none.")
        if agents == "none":
            return {}
        guide = _assets().joinpath("consumer/AGENTS.md").read_text(encoding="utf-8")
        guide = guide.replace("{{project_id}}", self.scope.project.project_id)
        guide = guide.replace("{{version}}", VERSION)
        desired = {"AGENTS.md": guide, ".agents/LAMBDAFORGE.md": guide}
        for name in skill_names():
            base = _assets().joinpath("skills", name)
            desired[f".agents/skills/{name}/SKILL.md"] = base.joinpath("SKILL.md").read_text(
                encoding="utf-8"
            )
            desired[f".agents/skills/{name}/references/procedure.md"] = base.joinpath(
                "references/procedure.md"
            ).read_text(encoding="utf-8")
        desired.update(
            {f".agents/examples/{name}": value for name, value in example_assets().items()}
        )
        if agents == "codex":
            desired[".lambdaforge/agents/codex-config.example.toml"] = (
                "# Review and merge into your client config; LambdaForge never edits it.\n"
                "[mcp_servers.lambdaforge]\ncommand = "
                + json.dumps(sys.executable, ensure_ascii=False)
                + "\nargs = "
                + json.dumps(
                    ["-m", "lambdaforge", "ai", "--project", str(self.scope.root), "mcp", "serve"],
                    ensure_ascii=False,
                )
                + "\n# Verify this interpreter has LambdaForge and the agents extra installed.\n"
            )
        return desired

    def preview(self, *, agents: str = "portable", knowledge: str = "none") -> dict[str, Any]:
        self.scope.assert_current()
        if knowledge not in {"none", "markdown", "obsidian"}:
            raise ValueError("Knowledge mode must be none, markdown or obsidian.")
        old = self._manifest()
        entries = []
        for relative, content in sorted(self._desired(agents).items()):
            path = self.scope.path(relative, internal=True)
            existing = (
                _sha(self.scope.file(relative, limit=512 * 1024, internal=True).decode("utf-8"))
                if path.exists()
                else None
            )
            expected = _sha(content)
            state = (
                "unchanged"
                if existing == expected
                else "create"
                if existing is None
                else "update"
                if old["files"].get(relative) == existing
                else "preserve_custom"
            )
            entries.append(
                {
                    "path": relative,
                    "action": state,
                    "current_sha256": existing,
                    "proposed_sha256": expected,
                }
            )
        payload = {
            "schema_version": 1,
            "project_id": self.scope.project.project_id,
            "version": VERSION,
            "agents": agents,
            "knowledge": knowledge,
            "files": entries,
            "notes_modified": False,
            "client_configuration_modified": False,
            "dependencies_installed": False,
        }
        payload["plan_hash"] = _sha(json.dumps(payload, sort_keys=True))
        return payload

    def apply(
        self, *, plan_hash: str, agents: str = "portable", knowledge: str = "none"
    ) -> dict[str, Any]:
        lock = self.scope.path(".lambdaforge/agents/integration.lock", internal=True)
        with CrossProcessFileLock(
            lock, shared=False, timeout_seconds=10, poll_interval_seconds=0.05
        ):
            preview = self.preview(agents=agents, knowledge=knowledge)
            if preview["plan_hash"] != plan_hash:
                raise ValueError("Integration plan changed; preview again before applying.")
            desired = self._desired(agents)
            manifest = self._manifest()
            for entry in preview["files"]:
                relative = entry["path"]
                if entry["action"] == "preserve_custom":
                    continue
                path = self.scope.path(relative, internal=True)
                if entry["action"] != "unchanged":
                    atomic_write_bytes(path, desired[relative].encode("utf-8"))
                manifest["files"][relative] = entry["proposed_sha256"]
            manifest.update({"version": VERSION, "agents": agents, "knowledge": knowledge})
            atomic_write_text(self.manifest_path, json.dumps(manifest, indent=2) + "\n")
            if knowledge != "none":
                KnowledgeService(self.scope).initialize(obsidian=knowledge == "obsidian")
        return {
            **preview,
            "applied": True,
            "custom_files_preserved": [
                e["path"] for e in preview["files"] if e["action"] == "preserve_custom"
            ],
        }

    def status(self) -> dict[str, Any]:
        self.scope.assert_current()
        manifest = self._manifest()
        return {
            "configured": self.manifest_path.exists(),
            "project_id": self.scope.project.project_id,
            "version": manifest.get("version"),
            "agents": manifest.get("agents", "none"),
            "knowledge": manifest.get("knowledge", "none"),
            "managed_files": len(manifest["files"]),
            "installed_skills": len(skill_names()),
        }
