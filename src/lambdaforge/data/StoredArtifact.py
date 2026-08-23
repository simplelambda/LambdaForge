"""Content-verified artifact descriptor used by persisted dataset formats."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Describe one contained file/directory in an existing dataset manifest."""

    path: str
    kind: str
    sha256: str
    size_bytes: int
    name: str | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts:
            raise ValueError("Stored artifact paths must be contained and relative.")
        if len(self.sha256) != 64 or any(value not in "0123456789abcdef" for value in self.sha256):
            raise ValueError("Stored artifact sha256 must be lowercase hexadecimal SHA-256.")
        if self.size_bytes < 0:
            raise ValueError("Stored artifact size must be non-negative.")
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StoredArtifact:
        return cls(
            str(value["path"]),
            str(value["kind"]),
            str(value["sha256"]),
            int(value["size_bytes"]),
            str(value["name"]) if value.get("name") is not None else None,
            str(value["media_type"]) if value.get("media_type") is not None else None,
            value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": copy.deepcopy(self.metadata),
            **({"name": self.name} if self.name else {}),
            **({"media_type": self.media_type} if self.media_type else {}),
        }

    @staticmethod
    def fingerprint_path(path: str | Path) -> tuple[str, int]:
        root = Path(path)
        if root.is_symlink() or (not root.is_file() and not root.is_dir()):
            raise ValueError(f"Artifact path is unsafe: {root}")
        entries = (root,) if root.is_file() else tuple(sorted(root.rglob("*")))
        digest, size = hashlib.sha256(), 0
        for item in entries:
            if item.is_symlink():
                raise ValueError(f"Artifact tree contains a symbolic link: {item}")
            if item.is_file():
                relative = item.name if root.is_file() else item.relative_to(root).as_posix()
                digest.update(relative.encode())
                digest.update(b"\0")
                with item.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
        return digest.hexdigest(), size
