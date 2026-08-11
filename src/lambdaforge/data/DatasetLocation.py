"""Physical placement of one logical dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetLocation:
    """Locate dataset bytes in one named execution environment."""

    environment: str
    uri: str
    shared: bool = False

    def __post_init__(self) -> None:
        if not self.environment.strip() or not self.uri.strip():
            raise ValueError("Dataset location environment and URI cannot be empty.")

    def local_path(self, source_dir: str | Path) -> Path:
        """Resolve a local/file URI and reject unsupported remote protocols."""
        raw = self.uri.removeprefix("file://")
        if "://" in raw:
            raise ValueError(
                f"Dataset location {self.uri!r} needs an installed transfer/resolver provider."
            )
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (Path(source_dir) / candidate).resolve()

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable physical placement descriptor."""
        return {"environment": self.environment, "uri": self.uri, "shared": self.shared}
