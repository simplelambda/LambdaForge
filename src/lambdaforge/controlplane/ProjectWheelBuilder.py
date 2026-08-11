"""Build and cache exact pure-Python project wheels for remote deployment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from lambdaforge.reproducibility.CodeIdentity import CodeIdentity


class ProjectWheelBuilder:
    """Create a wheel from the actual local project state without cloning a branch."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def build(self, project_root: str | Path) -> Path:
        """Build or reuse a wheel keyed by code identity and packaging declarations."""
        project = Path(project_root).resolve()
        pyproject = project / "pyproject.toml"
        if not pyproject.is_file():
            raise FileNotFoundError(f"Managed deployment requires {pyproject}.")
        identity = CodeIdentity.capture(project).to_dict()
        payload = json.dumps(identity, sort_keys=True).encode("utf-8") + pyproject.read_bytes()
        key = hashlib.sha256(payload).hexdigest()[:24]
        destination = self.root / key
        wheels = tuple(destination.glob("*.whl")) if destination.is_dir() else ()
        if len(wheels) == 1:
            return wheels[0]
        destination.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(destination),
                str(project),
            ),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Could not build exact project wheel: {completed.stderr.strip()}")
        wheels = tuple(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one project wheel in {destination}, found {len(wheels)}.")
        return wheels[0]
