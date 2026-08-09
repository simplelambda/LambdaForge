"""Reproducible environment lock export."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from lambdaforge.EnvironmentManifest import EnvironmentManifest


class EnvironmentExporter:
    """Export pip, Conda or container-oriented environment snapshots explicitly."""

    def export(self, destination: str | Path, *, format: str = "pip") -> Path:
        """Write an environment file without changing the active environment."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "pip":
            completed = subprocess.run(
                (sys.executable, "-m", "pip", "freeze", "--all"),
                check=True,
                capture_output=True,
                text=True,
            )
            path.write_text(completed.stdout, encoding="utf-8")
        elif format == "conda":
            completed = subprocess.run(
                ("conda", "env", "export", "--no-builds"),
                check=True,
                capture_output=True,
                text=True,
            )
            path.write_text(completed.stdout, encoding="utf-8")
        elif format == "container":
            payload = {
                "base_image_hint": f"python:{platform.python_version()}",
                "manifest": EnvironmentManifest.capture().to_dict(),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            raise ValueError("Environment export format must be pip, conda or container.")
        return path
