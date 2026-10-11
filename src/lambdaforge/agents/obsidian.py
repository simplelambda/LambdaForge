"""Optional desktop helpers; Markdown access never depends on the Obsidian application."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import quote

from packaging.version import Version

from lambdaforge.agents.contracts import redact
from lambdaforge.agents.knowledge import KnowledgeService


class ObsidianService:
    """Inspect without launching; version probe and URI opening are explicit local actions."""

    def __init__(self, vault: KnowledgeService) -> None:
        self.vault = vault

    def status(self) -> dict[str, Any]:
        state = self.vault.status()
        return {
            "project_id": state["project_id"],
            "vault_configured": state["configured"],
            "obsidian_marker": state["obsidian_vault"],
            "directory": state["directory"],
            "cli_on_path": shutil.which("obsidian") is not None,
            "cli_usable": "not_probed",
            "app_installed": "not_verified",
            "app_running": "not_verified",
            "compatible_version": "not_probed",
            "minimum_cli_installer": "1.12.7",
            "desktop_environment_hint": sys.platform in {"darwin", "win32"}
            or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
            "markdown_usable_without_app": True,
            "network_used": False,
            "hint": "Open this project's knowledge folder as a vault in Obsidian. "
            "No plugins or desktop application are required for native Markdown operations. "
            "Official CLI must be enabled in Obsidian; even version probes may start the app.",
        }

    @staticmethod
    def _run(argv: list[str], *, timeout: int = 5) -> dict[str, Any]:
        # No shell, arbitrary commands, plugin execution or environment installation.
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            output.seek(0)
            text = redact(output.read(2048).decode("utf-8", errors="replace"))
        return {"return_code": process.returncode, "timed_out": timed_out, "diagnostic": text}

    def probe_cli(self, *, consent: bool = False) -> dict[str, Any]:
        """Explicit version probe may start desktop Obsidian; never required for reading notes."""
        if not consent:
            raise PermissionError("CLI probing can start Obsidian; use --probe-cli --apply.")
        state = self.status()
        executable = shutil.which("obsidian")
        if executable is None or not state["desktop_environment_hint"]:
            return {
                **state,
                "status": "unavailable",
                "cli_usable": False,
                "reason": "cli_missing" if executable is None else "headless_environment",
            }
        result = self._run([executable, "version"])
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", result["diagnostic"])
        version = match.group(1) if match else None
        usable = result["return_code"] == 0 and not result["timed_out"] and version is not None
        return {
            **state,
            **result,
            "status": "ok" if usable else "error",
            "cli_usable": usable,
            "version": version,
            "compatible_version": Version(version) >= Version("1.12.7") if version else None,
        }

    def open(self, *, apply: bool = False) -> dict[str, Any]:
        """Preview/request opening the exact owned README in a registered vault, not any URI."""
        state = self.status()
        scope = self.vault.scope
        readme = scope.path(self.vault.root.relative_to(scope.root) / "README.md")
        if not readme.is_file():
            raise FileNotFoundError("Initialize the knowledge folder before opening it.")
        uri = "obsidian://open?path=" + quote(str(readme), safe="")
        state = {
            **state,
            "uri": uri,
            "effect": "Ask desktop Obsidian to open the owned README. "
            "Register this folder as a vault manually first; no settings are changed by LF.",
        }
        if not apply:
            return {**state, "status": "preview", "application_started": False}
        if not state["desktop_environment_hint"]:
            return {
                **state,
                "status": "unavailable",
                "reason": "headless_environment",
                "application_started": False,
            }
        if sys.platform == "win32":
            os.startfile(uri)  # type: ignore[attr-defined]
            return {**state, "status": "open_requested", "opening_verified": False}
        executable = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
        if executable is None:
            return {
                **state,
                "status": "unavailable",
                "reason": "desktop_opener_missing",
                "application_started": False,
            }
        result = self._run([executable, uri])
        return {
            **state,
            **result,
            "status": "open_requested" if result["return_code"] == 0 else "error",
            "opening_verified": False,
        }
