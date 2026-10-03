"""Validated research-question declaration, independent of execution/objective policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.analysis.MetricCatalog import resolve_semantics
from lambdaforge.work.models import immutable_mapping


@dataclass(frozen=True, slots=True)
class AnalysisProfile:
    """Portable resolved metric catalog + questions + bounded discovery rules.

    Use ``resolve`` for class defaults and YAML overrides. The immutable document is
    persisted before execution; it is never a scheduling or objective-policy object.
    """

    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", immutable_mapping(self.document))

    @classmethod
    def resolve(
        cls,
        declared: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        *,
        objective: Mapping[str, Any] | None = None,
    ) -> AnalysisProfile:
        return cls(resolve_semantics(declared, overrides, objective=objective))

    @property
    def identity(self) -> str:
        return str(self.document["identity"])
