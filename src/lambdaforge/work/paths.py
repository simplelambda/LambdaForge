"""Resolve researcher-owned paths consistently across local and remote Work runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkPathContext:
    """Map relative publication paths to the directory that owns the source YAML.

    Local execution uses the actual YAML directory. A remote execution bundle may
    carry a small trusted mapping to the equivalent directory below the cluster's
    configured project mirror. Relative destinations cannot escape that mirror;
    researchers can still opt into another persistent location with an absolute path.
    """

    source_directory: Path
    project_root: Path | None = None
    relative_publication: bool = True

    @classmethod
    def load(cls, source: Path) -> WorkPathContext:
        """Load an optional bundle path mapping, otherwise use the local YAML directory."""
        marker = source.parent / ".lambdaforge-paths.json"
        if not marker.exists():
            bundled_remote = (
                os.environ.get("LAMBDAFORGE_BUNDLE") == "1"
                and os.environ.get("LAMBDAFORGE_CLUSTER", "local") != "local"
            )
            return cls(source.parent.resolve(), relative_publication=not bundled_remote)
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 16 * 1024:
            raise ValueError(f"Invalid LambdaForge path-context file: {marker}")
        try:
            value: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid LambdaForge path-context file: {marker}") from error
        if not isinstance(value, dict) or value.get("path_context_version") != 1:
            raise ValueError(f"Unsupported LambdaForge path-context file: {marker}")
        root = Path(str(value.get("project_root", "")))
        relative = Path(str(value.get("source_relative", "")))
        if (
            not root.is_absolute()
            or root == Path("/")
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(f"Unsafe LambdaForge path-context file: {marker}")
        lexical_root = Path(os.path.abspath(root))
        source_directory = Path(os.path.abspath(lexical_root / relative))
        if not source_directory.is_relative_to(lexical_root):
            raise ValueError(f"Unsafe LambdaForge path-context file: {marker}")
        return cls(source_directory, lexical_root)

    def publication_path(self, value: str | Path) -> Path:
        """Resolve one explicit output destination without following hidden bundle paths."""
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return Path(os.path.abspath(raw))
        if not self.relative_publication:
            raise ValueError(
                "Relative publish_to on remote execution requires a cluster project_root. "
                "Configure the remote project mirror or use an explicit absolute remote path."
            )
        destination = Path(os.path.abspath(self.source_directory / raw))
        if self.project_root is not None and not destination.is_relative_to(self.project_root):
            raise ValueError(
                "Relative publish_to paths cannot escape the configured remote project_root. "
                "Use a path inside the mirror or an explicit absolute remote path."
            )
        return destination


__all__ = ["WorkPathContext"]
