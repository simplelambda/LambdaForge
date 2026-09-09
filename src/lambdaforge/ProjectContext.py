"""Project identity and path discovery, independent of Python environments and science."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli


@dataclass(frozen=True)
class ProjectContext:
    """One controller checkout; its ID scopes storage, never scientific fingerprints."""

    root: Path
    project_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.project_id):
            raise ValueError(
                "tool.lambdaforge.project_id must be 1–80 letters, digits, '_' or '-', "
                "starting with a letter or digit."
            )

    @classmethod
    def discover(cls, start: str | Path | None = None) -> ProjectContext:
        current = Path(start or Path.cwd()).expanduser().resolve()
        if current.is_file():
            current = current.parent
        root = next(
            (path for path in (current, *current.parents) if (path / "pyproject.toml").is_file()),
            current,
        )
        explicit = None
        manifest = root / "pyproject.toml"
        if manifest.is_file():
            with manifest.open("rb") as stream:
                configuration = tomli.load(stream)
            tools = configuration.get("tool", {})
            if not isinstance(tools, Mapping):
                raise TypeError("pyproject.toml [tool] must be a table.")
            lambdaforge = tools.get("lambdaforge", {})
            if not isinstance(lambdaforge, Mapping):
                raise TypeError("pyproject.toml [tool.lambdaforge] must be a table.")
            explicit = lambdaforge.get("project_id")
        if explicit is not None:
            if not isinstance(explicit, str):
                raise TypeError("tool.lambdaforge.project_id must be a string.")
            identifier = explicit
        else:
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", root.name).strip("-")[:40] or "project"
            identifier = f"{slug}-{hashlib.sha256(str(root).encode()).hexdigest()[:16]}"
        return cls(root, identifier)

    def owns_source(self, source: object) -> bool:
        """Require the nearest project root, not merely a shared ancestor directory."""
        if not isinstance(source, str) or not Path(source).is_absolute():
            return False
        return self.discover(Path(source).parent).root == self.root

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "root": str(self.root)}
