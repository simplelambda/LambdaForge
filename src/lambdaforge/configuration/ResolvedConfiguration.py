"""Immutable result of configuration composition."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.configuration.SecretValue import SecretValue
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True, init=False)
class ResolvedConfiguration:
    """Expose materialized values, value origins and loaded source files."""

    values: Mapping[str, Any]
    provenance: Mapping[str, str]
    sources: tuple[Path, ...]

    def __init__(
        self,
        values: Mapping[str, Any],
        provenance: Mapping[str, str],
        sources: tuple[Path, ...],
    ) -> None:
        object.__setattr__(self, "values", FrozenJsonMapping(values))
        object.__setattr__(self, "provenance", FrozenJsonMapping(provenance))
        object.__setattr__(self, "sources", sources)

    def materialized(self, *, reveal_secrets: bool = False) -> dict[str, Any]:
        """Return a mutable snapshot, redacting secrets unless explicitly requested."""

        def visit(value: Any) -> Any:
            if isinstance(value, SecretValue):
                return value.value if reveal_secrets else "***"
            if isinstance(value, Mapping):
                return {str(key): visit(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [visit(item) for item in value]
            return copy.deepcopy(value)

        return visit(self.values)

    @property
    def contains_secrets(self) -> bool:
        """Return whether any nested value came from a secret interpolation."""

        def visit(value: Any) -> bool:
            if isinstance(value, SecretValue):
                return True
            if isinstance(value, Mapping):
                return any(visit(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(visit(item) for item in value)
            return False

        return visit(self.values)
