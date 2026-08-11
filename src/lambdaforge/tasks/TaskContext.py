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
    outputs: Mapping[str, str] = field(default_factory=dict)
    stop_event: Any = None

    def __post_init__(self) -> None:
        """Freeze nested public metadata and input descriptors defensively."""
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))
        object.__setattr__(
            self,
            "inputs",
            tuple(FrozenJsonMapping(value) for value in self.inputs),
        )
        object.__setattr__(self, "outputs", FrozenJsonMapping(self.outputs))

    @property
    def stop_requested(self) -> bool:
        """Return whether the owning execution requested cooperative cancellation."""
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def output_path(self, relative_path: str | Path, *, create_parent: bool = False) -> Path:
        """Resolve a legacy run-relative output path; prefer :meth:`output`."""
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

    def output(self, name: str, *, create: bool = False) -> Path:
        """Resolve one declared logical output name inside the run directory."""
        if name not in self.outputs:
            raise KeyError(f"Unknown task output {name!r}; declared: {sorted(self.outputs)}.")
        path = self.output_path(self.outputs[name])
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def input(self, name: str) -> Path:
        """Return the physical path for one declared logical input name."""
        matches = [value for value in self.inputs if value.get("name") == name]
        if not matches:
            raise KeyError(
                f"Unknown task input {name!r}; declared: {[v.get('name') for v in self.inputs]}."
            )
        return Path(str(matches[0]["resolved_path"]))

    def input_path(self, path: str | Path) -> Path:
        """Resolve a legacy YAML-relative input path; prefer :meth:`input`."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.source_dir / candidate).resolve()

    def declared_input_path(self, path: str | Path) -> Path:
        """Resolve a legacy declared path; prefer logical :meth:`input`."""
        candidate = self.input_path(path).resolve(strict=False)
        for value in self.inputs:
            declared = Path(str(value["resolved_path"])).resolve(strict=False)
            if candidate == declared or candidate.is_relative_to(declared):
                return candidate
        raise ValueError(f"Path is not covered by a declared task input: {path!s}")
