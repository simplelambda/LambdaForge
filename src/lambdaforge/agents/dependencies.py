"""Explicit optional dependency management in the invoking project interpreter.

Only declared extras are installable here. No command or package supplied by an agent is
accepted, no system interpreter is mutated and no installation occurs on import/inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from lambdaforge.agents.contracts import redact
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

EXTRAS = {"agents": ("mcp>=2.3,<2.4",)}


class DependencyService:
    """Inspect, preview and deliberately install a fixed optional extra in this environment."""

    def __init__(self, scope: ProjectScope) -> None:
        self.scope = scope

    def check(self, extra: str = "agents") -> dict[str, Any]:
        self.scope.assert_current()
        if extra not in EXTRAS:
            raise ValueError(
                "Unknown optional dependency group; currently only agents is supported."
            )
        packages = []
        for declaration in EXTRAS[extra]:
            requirement = Requirement(declaration)
            try:
                installed = version(requirement.name)
            except PackageNotFoundError:
                installed = None
            packages.append(
                {
                    "name": requirement.name,
                    "requirement": declaration,
                    "installed": installed,
                    "compatible": installed is not None and installed in requirement.specifier,
                }
            )
        return {
            "schema_version": 1,
            "project_id": self.scope.project.project_id,
            "extra": extra,
            "interpreter": sys.executable,
            "environment": sys.prefix,
            "isolated_environment": sys.prefix != sys.base_prefix
            or (Path(sys.prefix) / "conda-meta").is_dir(),
            "available": all(row["compatible"] for row in packages),
            "packages": packages,
            "installation_performed": False,
        }

    def preview(self, extra: str = "agents", *, wheelhouse: str | None = None) -> dict[str, Any]:
        """Exact interpreter/group plan; no installation, network request or cache mutation."""
        status = self.check(extra)
        if not status["isolated_environment"]:
            raise PermissionError(
                "Dependency installation requires a virtualenv/Conda environment, "
                "not system Python."
            )
        argv = [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--prefix",
            sys.prefix,
        ]
        if wheelhouse is not None:
            folder = self.scope.path(wheelhouse)
            if not folder.is_dir():
                raise FileNotFoundError("Offline wheelhouse must be a project-owned directory.")
            argv.extend(["--no-index", "--find-links", str(folder)])
        argv.extend(EXTRAS[extra])
        value = {
            **status,
            "argv": argv,
            "network_access": wheelhouse is None,
            "installer_environment": {"PIP_CONFIG_FILE": os.devnull},
            "wheelhouse": wheelhouse,
            "consent_required": True,
            "effect": "Install declared optional Python dependencies, "
            "never external software or science.",
        }
        value["plan_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value

    def install(
        self,
        *,
        plan_hash: str,
        extra: str = "agents",
        wheelhouse: str | None = None,
        consent: bool = False,
        timeout_seconds: int = 600,
    ) -> dict[str, Any]:
        """Explicit local operator workflow only. This method is never an MCP operation."""
        if not consent:
            raise PermissionError(
                "Installing optional dependencies requires explicit --apply or --install-deps."
            )
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
            raise ValueError("Install timeout must be 1–600 seconds.")
        lock = self.scope.path(".lambdaforge/agents/dependencies.lock", internal=True)
        with CrossProcessFileLock(
            lock, shared=False, timeout_seconds=10, poll_interval_seconds=0.1
        ):
            plan = self.preview(extra, wheelhouse=wheelhouse)
            if plan["plan_hash"] != plan_hash:
                raise ValueError("Dependency plan changed; inspect again before installing.")
            if plan["available"]:
                return {**plan, "status": "already_available"}
            # Keep potentially large pip output out of JSON/stdout and bound diagnostics.
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    plan["argv"],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    # --isolated alone still permits system/site pip configuration. Disable
                    # all config files and bind destination to this environment explicitly.
                    env={**os.environ, **plan["installer_environment"]},
                    start_new_session=os.name != "nt",
                )
                timed_out = False
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                finally:
                    if process.poll() is None:
                        if os.name != "nt":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                output.seek(0, os.SEEK_END)
                output.seek(max(0, output.tell() - 12000))
                diagnostic = redact(output.read().decode("utf-8", errors="replace"))
            checked = self.check(extra)
            return {
                **checked,
                "installation_performed": True,
                "return_code": process.returncode,
                "timed_out": timed_out,
                "diagnostic": diagnostic,
                "status": "ok" if process.returncode == 0 and checked["available"] else "error",
                "hint": "Project files and notes are unchanged. "
                "Check network/offline wheels and recheck the environment.",
            }


__all__ = ["DependencyService", "EXTRAS"]
