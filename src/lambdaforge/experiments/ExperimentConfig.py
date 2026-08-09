"""Object representation and expansion of a YAML experiment configuration."""

from __future__ import annotations

import copy
import itertools
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
    ExperimentConfigMigrationResult,
)
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)


class ExperimentConfig(Mapping[str, Any]):
    """Validated, task-agnostic experiment configuration.

    The object provides mapping access for interoperability while owning all
    path lookup, mutation, YAML loading, sweep expansion and output-layout
    rules. Expanded runs are independent deep copies, so one subprocess cannot
    mutate the configuration of another.
    """

    DEFAULT_OUTPUT_ROOT = Path("runs/experiments")
    DEFAULT_VARIANT = "base"

    def __init__(
        self,
        data: Mapping[str, Any],
        source: str | Path | None = None,
        *,
        _migration_result: ExperimentConfigMigrationResult | None = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("Experiment config must be a mapping.")
        self.source = Path(source) if source is not None else None
        if _migration_result is None:
            try:
                _migration_result = ExperimentConfigMigrator.default().preview_mapping(
                    data,
                    validate=False,
                )
            except (TypeError, ValueError) as error:
                label = str(self.source) if self.source is not None else "<mapping>"
                raise ValueError(
                    f"Cannot normalize experiment Schema for {label}: {error}. "
                    "Preview supported migrations with 'lambdaforge migrate <config>'."
                ) from error
        self.migration_result = _migration_result
        self._data = copy.deepcopy(dict(_migration_result.config))
        self._validate_identity(self._data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a safely composed experiment from a UTF-8 YAML file."""
        from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
        from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec

        path = Path(path)
        source_text = path.read_text(encoding="utf-8")
        loaded, _ = RoundTripYamlCodec().load_file(path)
        if "extends" not in loaded and "include" not in loaded and "${" not in source_text:
            try:
                result = ExperimentConfigMigrator.default().preview_file(path, validate=False)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot normalize experiment Schema for {path}: {error}. "
                    "Preview supported migrations with 'lambdaforge migrate <config>'."
                ) from error
            return cls(result.config, source=path, _migration_result=result)
        resolution = ConfigurationComposer().resolve(path)
        if resolution.contains_secrets:
            raise ValueError(
                "Training configuration cannot persist secret values safely. Keep provider "
                "credentials in their runtime environment instead of experiment YAML."
            )
        try:
            result = ExperimentConfigMigrator.default().preview_mapping(
                resolution.materialized(reveal_secrets=True),
                validate=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot normalize experiment Schema for {path}: {error}. "
                "Preview supported migrations with 'lambdaforge migrate <config>'."
            ) from error
        return cls(result.config, source=path, _migration_result=result)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive deep copy suitable for serialization."""
        return copy.deepcopy(self._data)

    def value(self, path: str, default: Any = None) -> Any:
        """Read a dot-separated path from this configuration."""
        return self.get_value(self._data, path, default)

    def set(self, path: str, value: Any) -> None:
        """Set a dot-separated path in this configuration."""
        self.set_value(self._data, path, value)

    @property
    def suite_dir(self) -> Path:
        """Return the directory shared by every expanded variant and seed."""
        return self.suite_dir_for(self._data)

    def expand(self) -> list[dict[str, Any]]:
        """Expand seeds, grid axes and named ablations into concrete runs."""
        return self._expand_current_mapping(self._data)

    @staticmethod
    def get_value(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
        """Read a dot-separated path from any mapping."""
        value: Any = config
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    @staticmethod
    def set_value(config: dict[str, Any], path: str, value: Any) -> None:
        """Set a dot-separated path on a mutable mapping."""
        parts = path.split(".")
        if not all(parts):
            raise ValueError("Config paths cannot contain empty segments.")
        node = config
        for part in parts[:-1]:
            current = node.get(part)
            if not isinstance(current, dict):
                current = {}
                node[part] = current
            node = current
        node[parts[-1]] = value

    @classmethod
    def suite_dir_for(cls, config: Mapping[str, Any]) -> Path:
        """Resolve ``<output_root>/<base_name>`` for any run mapping."""
        root = Path(str(cls.get_value(config, "experiment.output_root", cls.DEFAULT_OUTPUT_ROOT)))
        base = str(
            cls.get_value(
                config,
                "experiment.base_name",
                cls.get_value(config, "experiment.name", "experiment"),
            )
        )
        return root / base

    @classmethod
    def expand_mapping(cls, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Normalize and expand a raw mapping into materialized run mappings."""
        return cls(config).expand()

    @classmethod
    def _expand_current_mapping(cls, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Expand a mapping that already uses the current Schema version."""
        cls._validate_identity(config)
        base_name = str(cls.get_value(config, "experiment.name"))
        runs: list[dict[str, Any]] = []
        for grid_values in cls._grid_assignments(config):
            for ablation_name, ablation_values in cls._ablation_assignments(config):
                assignments = {**grid_values, **ablation_values}
                parts: list[str] = []
                if grid_values:
                    parts.append(cls._variant_from_assignments(grid_values))
                if ablation_name != cls.DEFAULT_VARIANT:
                    parts.append(ablation_name)
                variant = cls.slugify("__".join(parts) or cls.DEFAULT_VARIANT)

                for seed in cls._seed_values(config):
                    run = copy.deepcopy(dict(config))
                    run.pop("sweep", None)
                    for path, value in assignments.items():
                        cls.set_value(run, path, copy.deepcopy(value))
                    cls.set_value(run, "experiment.seed", seed)
                    cls.set_value(run, "experiment.base_name", base_name)
                    cls.set_value(run, "experiment.variant", variant)
                    cls.set_value(run, "experiment.name", cls.slugify(f"{base_name}__{variant}"))
                    runs.append(run)

        identities = [
            (
                str(cls.get_value(run, "experiment.variant")),
                cls.get_value(run, "experiment.seed"),
            )
            for run in runs
        ]
        duplicates = sorted(
            {identity for identity in identities if identities.count(identity) > 1},
            key=str,
        )
        if duplicates:
            raise ValueError(f"Sweep creates duplicate variant/seed runs: {duplicates}.")
        if not runs:
            raise ValueError("Experiment expansion produced no runs.")
        return runs

    @staticmethod
    def deep_update(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
        """Return a recursive, non-mutating mapping update."""
        output = copy.deepcopy(dict(base))
        for key, value in patch.items():
            if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
                output[key] = ExperimentConfig.deep_update(output[key], value)
            else:
                output[key] = copy.deepcopy(value)
        return output

    @staticmethod
    def slugify(value: str) -> str:
        """Convert arbitrary text into a portable run-directory segment."""
        normalized = value.strip().replace("\\", "/")
        normalized = re.sub(r"[^A-Za-z0-9_.=-]+", "-", normalized).strip("-")
        return normalized or "run"

    @classmethod
    def _seed_values(cls, config: Mapping[str, Any]) -> list[int | None]:
        seeds = cls.get_value(config, "experiment.seeds")
        if seeds is None:
            seed = cls.get_value(config, "experiment.seed")
            return [None if seed is None else int(seed)]
        if not isinstance(seeds, list):
            raise TypeError("experiment.seeds must be a list.")
        return [None if seed is None else int(seed) for seed in seeds]

    @classmethod
    def _grid_assignments(cls, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        grid = cls.get_value(config, "sweep.grid", {}) or {}
        if not isinstance(grid, Mapping):
            raise TypeError("sweep.grid must map config paths to value lists.")
        if not grid:
            return [{}]
        paths = list(grid)
        options: list[list[Any]] = []
        for path in paths:
            values = grid[path]
            if not isinstance(values, list) or not values:
                raise TypeError(f"sweep.grid.{path} must be a non-empty list.")
            options.append(values)
        return [
            dict(zip(paths, combination, strict=True))
            for combination in itertools.product(*options)
        ]

    @classmethod
    def _ablation_assignments(
        cls,
        config: Mapping[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        ablations = cls.get_value(config, "sweep.ablations", []) or []
        if not isinstance(ablations, list):
            raise TypeError("sweep.ablations must be a list.")
        output: list[tuple[str, dict[str, Any]]] = []
        if bool(cls.get_value(config, "sweep.include_base", True)):
            output.append((cls.DEFAULT_VARIANT, {}))
        for index, ablation in enumerate(ablations):
            if not isinstance(ablation, Mapping):
                raise TypeError("Each ablation must be a mapping.")
            values = ablation.get("set", {})
            if not isinstance(values, Mapping):
                raise TypeError("Ablation 'set' must be a mapping.")
            name = cls.slugify(str(ablation.get("name", f"ablation_{index}")))
            output.append((name, dict(values)))
        return output or [(cls.DEFAULT_VARIANT, {})]

    @classmethod
    def _variant_from_assignments(cls, assignments: Mapping[str, Any]) -> str:
        tokens = [f"{cls._short_key(path)}={value}" for path, value in assignments.items()]
        return cls.slugify("__".join(tokens))

    @staticmethod
    def _short_key(path: str) -> str:
        parts = path.split(".")
        ignored = {"params", "model", "training", "data", "task"}
        return next((part for part in reversed(parts) if part not in ignored), parts[-1])

    @classmethod
    def _validate_identity(cls, config: Mapping[str, Any]) -> None:
        from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
            ExperimentSchemaVersion,
        )

        version = ExperimentSchemaVersion.from_config(config)
        if version != ExperimentSchemaVersion.current():
            raise ValueError(
                f"ExperimentConfig requires current Schema "
                f"{ExperimentSchemaVersion.current()}, found {version}."
            )
        experiment = config.get("experiment")
        if not isinstance(experiment, Mapping):
            raise TypeError("'experiment' must be a mapping.")
        name = experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("experiment.name must be a non-empty string.")
