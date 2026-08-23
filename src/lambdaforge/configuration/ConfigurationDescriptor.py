"""Side-effect-free description of the only executable configuration: Work YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.work import WorkConfig, WorkRunner


@dataclass(frozen=True, slots=True)
class ConfigurationDescriptor:
    """Expose Work identity and scheduling metadata to the control plane."""

    source: Path
    name: str
    scientific_identity: str
    datasets: tuple[str, ...]
    planned_units: int
    materialized: Mapping[str, Any]
    job_type: str = "work"
    unit: str = "runs"

    @classmethod
    def from_path(cls, path: str | Path) -> ConfigurationDescriptor:
        """Validate a Work document and derive metadata without executing it."""
        source = Path(path).expanduser().resolve()
        config = WorkConfig.from_yaml(source)
        plan = WorkRunner().plan(config)
        return cls(
            source,
            config.name,
            plan.scientific_fingerprint,
            cls.dataset_references(config.raw),
            config.planned_runs,
            config.to_dict(),
        )

    @staticmethod
    def dataset_references(values: Mapping[str, Any]) -> tuple[str, ...]:
        """Return all explicit dataset markers in the current document."""
        found: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                if set(item) == {"dataset"}:
                    reference = str(item["dataset"]).removeprefix("dataset:")
                    found.add(reference.split("/", 1)[0])
                else:
                    for nested in item.values():
                        visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(values)
        return tuple(sorted(found))

    @property
    def revision(self) -> str:
        """Return a concise human-facing scientific revision."""
        return self.scientific_identity.removeprefix("sha256:")[:12]

    def metadata(self) -> dict[str, Any]:
        """Return the stable Work fields persisted on the low-level Job."""
        return {
            "name": self.name,
            "kind": "work",
            "scientific_identity": self.scientific_identity,
            "scientific_revision": self.revision,
            "source_config_path": str(self.source),
            "datasets": list(self.datasets),
            "planned_units": self.planned_units,
            "unit": self.unit,
        }
