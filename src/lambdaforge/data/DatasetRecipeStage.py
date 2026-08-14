"""One stage declaration in a dataset recipe."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class DatasetRecipeStage:
    """Keep scientific necessity independent from execution reuse policy."""

    name: str
    task: Path | Mapping[str, Any]
    needs: tuple[str, ...] = ()
    bindings: Mapping[str, Any] = field(default_factory=dict)
    required: bool = True
    reuse: str = "auto"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset recipe stage names cannot be empty.")
        if self.reuse not in {"auto", "never"}:
            raise ValueError("Dataset stage reuse must be auto or never.")
        if self.name in self.needs or len(self.needs) != len(set(self.needs)):
            raise ValueError(f"Invalid dependencies for dataset stage {self.name!r}.")
        object.__setattr__(self, "bindings", FrozenJsonMapping(self.bindings))

    @classmethod
    def from_mapping(
        cls, name: str, value: Mapping[str, Any], source_dir: Path
    ) -> DatasetRecipeStage:
        """Parse a source-relative task stage."""
        raw_task = value.get("task")
        if raw_task is None:
            raise ValueError(f"Dataset stage {name!r} requires task.")
        task = (source_dir / raw_task).resolve() if isinstance(raw_task, str) else raw_task
        needs = value.get("needs", ())
        if isinstance(needs, str):
            needs = (needs,)
        if not isinstance(needs, Sequence) or isinstance(needs, (str, bytes, bytearray)):
            raise TypeError(f"Dataset stage {name!r} needs must be a list or string.")
        required = value.get("required", True)
        if not isinstance(required, bool):
            raise TypeError(f"Dataset stage {name!r} required must be a bool.")
        return cls(
            name,
            task,
            tuple(str(item) for item in needs),
            value.get("bindings", {}),
            required,
            str(value.get("reuse", "auto")),
        )

    def to_workflow_node(self) -> dict[str, Any]:
        """Compile this semantic stage to the existing Workflow node contract."""
        return {
            "config": str(self.task) if isinstance(self.task, Path) else copy.deepcopy(self.task),
            "needs": list(self.needs),
            "bindings": copy.deepcopy(self.bindings),
        }
