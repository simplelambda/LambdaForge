"""Scientific identity of consumer code."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """Identify clean/dirty Git code or an explicit release version."""

    provider: str
    revision: str
    dirty: bool = False
    changes_digest: str | None = None
    source_digest: str | None = None

    @classmethod
    def capture(
        cls, source_dir: str | Path, *, explicit_version: str | None = None
    ) -> CodeIdentity:
        """Capture code identity without modifying the repository."""
        if explicit_version is not None:
            if not explicit_version.strip():
                raise ValueError("Explicit code_version cannot be empty.")
            return cls("explicit", explicit_version)
        directory = Path(source_dir).resolve()
        try:
            root = Path(cls._git(directory, "rev-parse", "--show-toplevel")).resolve()
            revision = cls._git(root, "rev-parse", "HEAD")
            status = cls._git(root, "status", "--porcelain=v1", "--untracked-files=all")
            if not status:
                return cls("git", revision)
            hasher = hashlib.sha256()
            hasher.update(cls._git_bytes(root, "diff", "--binary", "HEAD"))
            untracked = cls._git_bytes(
                root, "ls-files", "--others", "--exclude-standard", "-z"
            ).split(b"\0")
            for relative_bytes in sorted(value for value in untracked if value):
                path = root / os.fsdecode(relative_bytes)
                if path.is_file() and not path.is_symlink():
                    hasher.update(relative_bytes)
                    hasher.update(path.read_bytes())
            return cls("git", revision, True, f"sha256:{hasher.hexdigest()}")
        except (OSError, subprocess.SubprocessError, ValueError):
            project = cls._project(directory)
            return project or cls("unversioned", "unknown")

    @classmethod
    def _project(cls, directory: Path) -> CodeIdentity | None:
        """Identify a non-Git installable project by version and available source bytes."""
        pyproject = next(
            (
                parent / "pyproject.toml"
                for parent in (directory, *directory.parents)
                if (parent / "pyproject.toml").is_file()
            ),
            None,
        )
        if pyproject is None:
            return None
        try:
            text = pyproject.read_text(encoding="utf-8")
            project_section = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", text)
            if project_section is None:
                return None
            name_match = re.search(r"""(?m)^name\s*=\s*["']([^"']+)["']""", project_section[1])
            version_match = re.search(
                r"""(?m)^version\s*=\s*["']([^"']+)["']""", project_section[1]
            )
            if name_match is None or version_match is None:
                return None
            name, version = name_match[1], version_match[1]
        except (OSError, UnicodeError):
            return None
        root = pyproject.parent
        candidates = tuple(sorted((root / "src").rglob("*.py"))) if (root / "src").is_dir() else ()
        if not candidates:
            return cls("distribution", f"{name}=={version}")
        hasher = hashlib.sha256()
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
                hasher.update(path.read_bytes())
        return cls(
            "distribution",
            f"{name}=={version}",
            source_digest=f"sha256:{hasher.hexdigest()}",
        )

    @staticmethod
    def _git(directory: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(directory), *arguments),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return completed.stdout.strip()

    @staticmethod
    def _git_bytes(directory: Path, *arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", "-C", str(directory), *arguments),
            check=True,
            capture_output=True,
            shell=False,
        )
        return completed.stdout

    def to_dict(self) -> dict[str, Any]:
        """Return a stable scientific identity component."""
        return {
            "provider": self.provider,
            "revision": self.revision,
            "dirty": self.dirty,
            "changes_digest": self.changes_digest,
            "source_digest": self.source_digest,
        }
