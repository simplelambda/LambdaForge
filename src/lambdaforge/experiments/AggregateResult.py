"""Typed collection of cross-seed variant aggregates."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.VariantAggregateResult import VariantAggregateResult


class AggregateResult(JsonResult):
    """Wrap variant aggregates while retaining the historical mapping surface."""

    def __init__(
        self,
        variants: Mapping[str, Mapping[str, Any] | VariantAggregateResult],
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        self._variants = {
            str(name): (
                value
                if isinstance(value, VariantAggregateResult)
                else VariantAggregateResult.from_mapping(value)
            )
            for name, value in variants.items()
        }
        self._summary = copy.deepcopy(dict(summary or {}))
        self._freeze_mapping({name: result.to_dict() for name, result in self._variants.items()})

    @property
    def variants(self) -> Mapping[str, VariantAggregateResult]:
        """Return a shallow copy of the immutable typed variant collection."""
        return dict(self._variants)

    def variant(self, name: str) -> VariantAggregateResult:
        """Return one typed variant or raise the ordinary mapping KeyError."""
        return self._variants[name]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AggregateResult:
        """Parse either the legacy variant map or a complete summary artifact."""
        raw_variants = value.get("variants")
        if isinstance(raw_variants, Mapping):
            summary = {key: item for key, item in value.items() if key != "variants"}
            return cls(raw_variants, summary=summary)
        return cls(value)

    @classmethod
    def read_json(cls, path: str | Path) -> AggregateResult:
        """Load a legacy aggregate or complete summary JSON object."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Aggregate result must contain a JSON object: {path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy JSON-compatible variant mapping."""
        return copy.deepcopy(dict(self))

    def to_summary_dict(self) -> dict[str, Any]:
        """Return summary metadata plus serialized variants when available."""
        payload = copy.deepcopy(self._summary)
        payload["variants"] = self.to_dict()
        return payload

    def write_summary_json(self, path: str | Path) -> Path:
        """Atomically write the complete aggregate summary rather than only variants."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.to_summary_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
