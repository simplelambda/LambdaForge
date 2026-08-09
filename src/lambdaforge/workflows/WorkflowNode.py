"""Immutable workflow node definition."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """Describe one task/experiment node and its explicit dependencies."""

    name: str
    config: Path | Mapping[str, Any]
    needs: tuple[str, ...] = ()
    bindings: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    continue_on_failure: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Workflow node names cannot be empty.")
        if len(self.needs) != len(set(self.needs)) or self.name in self.needs:
            raise ValueError(f"Invalid dependencies for workflow node {self.name!r}.")
        if not isinstance(self.config, (Path, Mapping)):
            raise TypeError("Workflow node config must be a path or mapping.")
        object.__setattr__(self, "bindings", FrozenJsonMapping(self.bindings))
        object.__setattr__(self, "resources", FrozenJsonMapping(self.resources))

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any], source_dir: Path) -> WorkflowNode:
        """Validate and materialize one YAML node mapping."""
        if "config" not in value:
            raise ValueError(f"Workflow node {name!r} requires 'config'.")
        raw_config = value["config"]
        config = (source_dir / raw_config).resolve() if isinstance(raw_config, str) else raw_config
        raw_needs = value.get("needs", ())
        if isinstance(raw_needs, str):
            raw_needs = (raw_needs,)
        if not isinstance(raw_needs, Sequence) or isinstance(raw_needs, (bytes, bytearray)):
            raise TypeError(f"Workflow node {name!r} needs must be a sequence.")
        return cls(
            name=name,
            config=config,
            needs=tuple(str(item) for item in raw_needs),
            bindings=value.get("bindings", {}),
            resources=value.get("resources", {}),
            continue_on_failure=bool(value.get("continue_on_failure", False)),
        )

    def materialize(
        self,
    ) -> tuple[dict[str, Any], Path | None, ResolvedConfiguration | None]:
        """Return an isolated composed node mapping and its source metadata."""
        if isinstance(self.config, Path):
            resolution = ConfigurationComposer().resolve(self.config)
            return copy.deepcopy(dict(resolution.values)), self.config, resolution
        return copy.deepcopy(dict(self.config)), None, None
