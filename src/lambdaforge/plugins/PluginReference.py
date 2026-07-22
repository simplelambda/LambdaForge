"""Immutable references to installed LambdaForge plugins."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.plugins.PluginKind import PluginKind


@dataclass(frozen=True, slots=True)
class PluginReference:
    """Identify exactly one entry point by supported kind and case-sensitive name."""

    kind: PluginKind
    name: str

    NAME_PATTERN = re.compile(r"^[\w.-]+$", re.UNICODE)

    def __post_init__(self) -> None:
        """Validate and normalize values supplied directly by Python callers."""
        object.__setattr__(self, "kind", PluginKind.from_value(self.kind))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Plugin name must be a non-empty string.")
        if self.name != self.name.strip():
            raise ValueError("Plugin name cannot start or end with whitespace.")
        if self.NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError(
                "Plugin name may contain only letters, numbers, underscores, dots and dashes."
            )

    @classmethod
    def from_value(cls, value: PluginReference | Mapping[str, Any]) -> PluginReference:
        """Build a reference from its object form or explicit YAML mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("'plugin' must be a mapping with 'kind' and 'name'.")
        unexpected = set(value) - {"kind", "name"}
        if unexpected:
            raise ValueError(f"Unexpected plugin reference keys: {sorted(unexpected)}.")
        missing = {"kind", "name"} - set(value)
        if missing:
            raise ValueError(f"Missing plugin reference keys: {sorted(missing)}.")
        return cls(kind=PluginKind.from_value(value["kind"]), name=value["name"])
