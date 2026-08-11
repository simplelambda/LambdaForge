"""Declarative, testable scientific plot specification."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """Describe plot data and policy independently from a rendering backend."""

    plot_type: str
    data_references: tuple[str, ...] = ()
    x: str | None = None
    y: str | None = None
    metrics: tuple[str, ...] = ()
    aggregation: str = "none"
    uncertainty: str = "none"
    labels: Mapping[str, str] = field(default_factory=dict)
    data: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Return a cache key for source references plus rendering policy."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reproducibility record."""
        return {
            "plot_spec_version": 1,
            "plot_type": self.plot_type,
            "data_references": list(self.data_references),
            "x": self.x,
            "y": self.y,
            "metrics": list(self.metrics),
            "aggregation": self.aggregation,
            "uncertainty": self.uncertainty,
            "labels": copy.deepcopy(dict(self.labels)),
            "data": [copy.deepcopy(dict(value)) for value in self.data],
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlotSpec:
        """Restore a persisted specification for regeneration."""
        references = value.get("data_references", ())
        metrics = value.get("metrics", ())
        data = value.get("data", ())
        if not isinstance(references, Sequence) or isinstance(references, str):
            raise TypeError("PlotSpec data_references must be a list.")
        if not isinstance(metrics, Sequence) or isinstance(metrics, str):
            raise TypeError("PlotSpec metrics must be a list.")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError("PlotSpec data must be a list.")
        return cls(
            plot_type=str(value["plot_type"]),
            data_references=tuple(str(item) for item in references),
            x=str(value["x"]) if value.get("x") is not None else None,
            y=str(value["y"]) if value.get("y") is not None else None,
            metrics=tuple(str(item) for item in metrics),
            aggregation=str(value.get("aggregation", "none")),
            uncertainty=str(value.get("uncertainty", "none")),
            labels=value.get("labels", {}),
            data=tuple(dict(item) for item in data if isinstance(item, Mapping)),
            metadata=value.get("metadata", {}),
        )
