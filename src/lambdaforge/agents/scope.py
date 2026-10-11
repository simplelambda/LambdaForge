"""Pinned project scope and bounded, non-executing file discovery for agent adapters."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from lambdaforge.ProjectContext import ProjectContext


class ProjectScope:
    """A session never rediscovers its project from the changing current directory."""

    EXCLUDED = frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".lambdaforge",
            ".ssh",
            ".aws",
            ".codex",
            ".obsidian",
        }
    )
    SECRET_NAMES = frozenset({"credentials", "id_rsa", "id_ed25519", "secrets"})

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root).expanduser().absolute()
        if supplied.resolve() != supplied or not supplied.is_dir():
            raise ValueError("Agent project root must be an existing non-symbolic directory.")
        self.root = supplied
        self.file("pyproject.toml", limit=512 * 1024)
        self.project = ProjectContext.discover(supplied)
        if self.project.root != supplied:
            raise ValueError(
                "Specify the project root containing pyproject.toml, not a subdirectory."
            )

    def assert_current(self) -> None:
        """A changed project identity invalidates a session instead of silently rebinding it."""
        self.file("pyproject.toml", limit=512 * 1024)
        if ProjectContext.discover(self.root) != self.project:
            raise PermissionError("Project identity changed; reconnect with the intended project.")

    def path(self, relative: str | Path, *, internal: bool = False) -> Path:
        """Reject traversal, nested projects, secret names and symlink components."""
        raw = Path(relative)
        if raw.is_absolute() or not raw.parts or any(p in {"..", "."} for p in raw.parts):
            raise ValueError("Agent paths must be project-relative without traversal.")
        if not internal and any(
            part.casefold() in self.EXCLUDED
            or part.casefold().split(".", 1)[0] in self.SECRET_NAMES
            or part.casefold().startswith(".env")
            or part.casefold().endswith((".pem", ".key"))
            for part in raw.parts
        ):
            raise PermissionError("Secret or internal paths are not agent-readable.")
        path = self.root / raw
        if path.resolve() != path or not path.is_relative_to(self.root):
            raise PermissionError("Symbolic or external paths are not permitted.")
        if (path / "pyproject.toml").is_file():
            raise PermissionError("The selected directory belongs to another nested project.")
        for parent in path.parents:
            if parent == self.root:
                break
            if (parent / "pyproject.toml").exists():
                raise PermissionError("The selected path belongs to another nested project.")
        return path

    def file(self, relative: str | Path, *, limit: int, internal: bool = False) -> bytes:
        """Read a bounded regular file through a no-follow descriptor, not a universal API."""
        path = self.path(relative, internal=internal)
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise PermissionError("Agent reads require regular files.")
            if metadata.st_size > limit:
                raise ValueError(
                    f"Metadata exceeds its {limit}-byte limit; request a smaller view."
                )
            data = stream.read(limit + 1)
            if len(data) > limit or path.resolve() != path:
                raise ValueError(
                    "Metadata grew beyond its limit or changed ownership while reading."
                )
            return data

    def files(self, *, suffixes: tuple[str, ...], limit: int = 2000) -> tuple[str, ...]:
        """Inventory allowed source/config paths without walking environments or artifacts."""
        results: list[str] = []
        for directory, children, files in os.walk(self.root, followlinks=False):
            base = Path(directory)
            children[:] = sorted(
                child
                for child in children
                if child not in self.EXCLUDED
                and not child.startswith(".")
                and not (base / child).is_symlink()
                and not (base / child / "pyproject.toml").exists()
            )
            for name in sorted(files):
                if not name.endswith(suffixes):
                    continue
                relative = (base / name).relative_to(self.root)
                try:
                    path = self.path(relative)
                except (ValueError, PermissionError):
                    continue
                if path.is_file():
                    results.append(relative.as_posix())
                if len(results) >= limit:
                    raise ValueError("Project inventory is too large; use a narrower project.")
        return tuple(results)
