"""Immutable context supplied to one configured post-run action."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.RunResult import RunResult


@dataclass(frozen=True, slots=True)
class PostRunContext:
    """Expose stable run evidence instead of live Lightning internals."""

    run_dir: Path
    config: FrozenJsonMapping
    result: RunResult
    seed: Any
    variant: str | None
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    selected_checkpoint: Path | None
    selected_checkpoint_role: str
    selected_checkpoint_sha256: str | None
    model_identity: FrozenJsonMapping
    action_name: str
    action_identity: str
    state_dir: Path
    _stop_requested: Callable[[], bool] = field(
        default=lambda: False,
        repr=False,
        compare=False,
    )

    @property
    def stop_requested(self) -> bool:
        """Return the current cooperative cancellation state."""
        return bool(self._stop_requested())

    def artifact_path(self, relative: str | Path, *, create_parent: bool = False) -> Path:
        """Resolve a safe run-relative artifact path for this action."""
        value = Path(relative)
        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError("Post-run artifact paths must be non-empty and run-relative.")
        root = self.run_dir.resolve()
        path = (root / value).resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError("Post-run artifact paths must remain inside the run directory.")
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
