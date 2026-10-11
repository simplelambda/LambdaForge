"""Opt-in checkpoint cadence and retention, never automatic scientific serialization."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def checkpoint_payloads(root: Path) -> Iterator[Path]:
    """List owned application payloads, not framework locks, records or temporary files."""
    if root.is_symlink() or root.resolve() != root.absolute():
        return
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.resolve() == path.absolute()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        ):
            yield path


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Advisory write cadence; retention applies only after verified scientific success.

    Storage durability is a project/operator declaration, not proof that a node-local path
    survives node loss. No frequency is imposed when both cadence fields are absent.
    """

    interval_steps: int | None = None
    interval_seconds: float | None = None
    retention_seconds: float | None = None
    storage_durability: str = "unspecified"

    def __post_init__(self) -> None:
        if self.interval_steps is not None and (
            type(self.interval_steps) is not int or self.interval_steps < 1
        ):
            raise ValueError("Checkpoint interval_steps must be a positive integer.")
        for name in ("interval_seconds", "retention_seconds"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in {int, float} or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Checkpoint {name} must be finite and positive.")
        if self.storage_durability not in {"unspecified", "ephemeral", "persistent", "shared"}:
            raise ValueError("Unknown checkpoint storage_durability declaration.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize an additive per-Run operational policy, not scientific design."""
        return asdict(self)
