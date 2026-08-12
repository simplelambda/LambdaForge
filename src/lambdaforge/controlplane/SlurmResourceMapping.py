"""Portable-resource to SLURM-directive translation."""

from __future__ import annotations

import math
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class SlurmResourceMapping:
    """Translate every portable resource exactly once using validated templates."""

    rules: Mapping[str, Any] = field(default_factory=dict)

    DEFAULTS = {
        "processes": {"option": "ntasks", "value": "{processes}"},
        "cpu": {"option": "cpus-per-task", "value": "{cpu_per_process}"},
        "memory": {"option": "mem", "value": "{memory_mib}M"},
        "gpu": {"option": "gpus", "value": "{gpus}"},
        "time": {"option": "time", "value": "{minutes}"},
    }
    RULE_ALLOWED = {
        "processes": {"processes"},
        "cpu": {"cpu_cores", "cpu_per_process", "processes"},
        "memory": {"memory_bytes", "memory_mib", "memory_gib"},
        "gpu": {"gpus"},
        "time": {"seconds", "minutes", "hours"},
    }

    def __post_init__(self) -> None:
        unknown = set(self.rules) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SLURM resource mappings: {sorted(unknown)}.")
        normalized: dict[str, Any] = {}
        for name, default in self.DEFAULTS.items():
            raw = self.rules.get(name, default)
            if raw is None or raw is False or raw == "omit":
                normalized[name] = {"omit": True}
                continue
            if not isinstance(raw, Mapping):
                raise TypeError(f"SLURM resource mapping {name!r} must be a mapping or 'omit'.")
            rule = dict(raw)
            if rule.get("omit") is True:
                normalized[name] = {"omit": True}
                continue
            option = str(rule.get("option", default["option"])).removeprefix("--")
            template = str(rule.get("value", default["value"]))
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", option) is None:
                raise ValueError(f"Unsafe SLURM resource option {option!r}.")
            self._validate_template(template, allowed=self.RULE_ALLOWED[name])
            normalized[name] = {"option": option, "value": template}
        object.__setattr__(self, "rules", FrozenJsonMapping(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> SlurmResourceMapping:
        """Construct a mapping while filling omitted rules with portable defaults."""
        return cls(value or {})

    def render(self, resources: ResourceRequest) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return directive payloads and explicit omission warnings."""
        values = {
            "cpu_cores": resources.cpu_cores,
            "cpu_per_process": max(1, math.ceil(resources.cpu_cores / resources.processes)),
            "processes": resources.processes,
            "memory_bytes": resources.ram_bytes,
            "memory_mib": max(1, math.ceil(resources.ram_bytes / 1048576)),
            "memory_gib": max(1, math.ceil(resources.ram_bytes / (1024**3))),
            "gpus": resources.gpu_count,
            "seconds": math.ceil(resources.runtime_seconds or 0),
            "minutes": max(1, math.ceil((resources.runtime_seconds or 0) / 60)),
            "hours": max(1, math.ceil((resources.runtime_seconds or 0) / 3600)),
        }
        requested = {
            "processes": True,
            "cpu": True,
            "memory": resources.ram_bytes > 0,
            "gpu": resources.gpu_count > 0,
            "time": resources.runtime_seconds is not None,
        }
        directives: list[str] = []
        warnings: list[str] = []
        for name in ("processes", "cpu", "memory", "gpu", "time"):
            if not requested[name]:
                continue
            rule = self.rules[name]
            if rule.get("omit"):
                warnings.append(
                    f"Requested {name} is not emitted because this cluster mapping explicitly "
                    "omits it; the scheduler will not enforce that resource."
                )
                continue
            rendered = str(rule["value"]).format_map(values)
            if not rendered or "\n" in rendered or "\r" in rendered:
                raise ValueError(f"Unsafe rendered SLURM resource value for {name!r}.")
            directives.append(f"--{rule['option']}={rendered}")
        return tuple(directives), tuple(warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return all effective resource mappings for inspection/export."""
        return {name: dict(value) for name, value in self.rules.items()}

    @classmethod
    def _validate_template(cls, value: str, *, allowed: set[str]) -> None:
        if "\n" in value or "\r" in value:
            raise ValueError("SLURM resource templates cannot contain newlines.")
        for _, field_name, format_spec, conversion in string.Formatter().parse(value):
            if field_name is None:
                continue
            if field_name not in allowed or format_spec or conversion:
                raise ValueError(f"Unsafe SLURM resource placeholder {{{field_name}}}.")
