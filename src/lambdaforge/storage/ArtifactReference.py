"""Immutable logical artifact reference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Identify content independently of its local staging location."""

    store: str
    key: str
    sha256: str
    size_bytes: int
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not self.store or not self.key or not self.sha256.startswith("sha256:"):
            raise ValueError("Artifact references require store, key and sha256:<digest>.")
        if self.size_bytes < 0 or ".." in self.key.split("/"):
            raise ValueError("Artifact reference size/key is invalid.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible reference."""
        return {
            "store": self.store,
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }
