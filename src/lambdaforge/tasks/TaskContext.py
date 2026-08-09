"""Runtime context passed to a generic task object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Expose safe paths, identity and cancellation state to one task attempt."""

    name: str
    run_dir: Path
    source_dir: Path
    attempt_id: str
    config_fingerprint: str
    resume: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    inputs: tuple[Mapping[str, Any], ...] = ()
    stop_event: Any = None

    def __post_init__(self) -> None:
        """Freeze nested public metadata and input descriptors defensively."""
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))
        object.__setattr__(
            self,
            "inputs",
            tuple(FrozenJsonMapping(value) for value in self.inputs),
        )

    @property
    def stop_requested(self) -> bool:
        """Return whether the owning execution requested cooperative cancellation."""
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def output_path(self, relative_path: str | Path, *, create_parent: bool = False) -> Path:
        """Resolve one safe run-relative output path and optionally create its parent."""
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Task output paths must be non-empty and run-relative: {relative!s}")
        root = self.run_dir.resolve()
        candidate = (root / relative).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError(f"Task output path escapes the run directory: {relative!s}")
        if create_parent:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def input_path(self, path: str | Path) -> Path:
        """Resolve a relative input against the YAML directory without requiring existence."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.source_dir / candidate).resolve()

    def declared_input_path(self, path: str | Path) -> Path:
        """Resolve a path and require it to belong to a content-addressed input."""
        candidate = self.input_path(path).resolve(strict=False)
        for value in self.inputs:
            declared = Path(str(value["resolved_path"])).resolve(strict=False)
            if candidate == declared or candidate.is_relative_to(declared):
                return candidate
        raise ValueError(f"Path is not covered by a declared task input: {path!s}")
