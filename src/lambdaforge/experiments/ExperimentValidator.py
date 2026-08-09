"""Non-mutating validation of experiment YAML and object references."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import (
    ExperimentSchemaCatalog,
)
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.ValidationReport import ValidationReport
from lambdaforge.plugins.PluginReference import PluginReference
from lambdaforge.plugins.PluginRegistry import PluginRegistry

if TYPE_CHECKING:
    from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig


class ExperimentValidator:
    """Validate structure, expansion, resources and import references safely.

    Validation imports referenced classes and callables when requested, but it
    never instantiates them, creates run directories or starts training.
    """

    def __init__(
        self,
        plugins: PluginRegistry | None = None,
        schema_catalog: ExperimentSchemaCatalog | None = None,
    ) -> None:
        """Use an injectable registry while defaulting to process-wide discovery."""
        self.plugins = plugins or PluginRegistry.default()
        self.schema_catalog = schema_catalog or ExperimentSchemaCatalog()
        self.migrator = ExperimentConfigMigrator(
            schema_catalog=self.schema_catalog,
        )

    def schema(self) -> dict[str, Any]:
        """Load the packaged Draft 2020-12 experiment schema."""
        return self.schema_catalog.schema()

    def validate_file(
        self,
        path: str | Path,
        *,
        check_imports: bool = True,
    ) -> ValidationReport:
        """Load and validate a UTF-8 YAML file without materializing a run."""
        path = Path(path)
        try:
            config = ExperimentConfig.from_yaml(path)
        except Exception as error:
            return ValidationReport(source=str(path), errors=(self._format_error(error),))
        return self.validate(config, check_imports=check_imports)

    def validate(
        self,
        config: ExperimentConfig | Mapping[str, Any],
        *,
        check_imports: bool = True,
    ) -> ValidationReport:
        """Return every discoverable error in one immutable report."""
        source = (
            str(config.source) if isinstance(config, ExperimentConfig) and config.source else None
        )
        try:
            migration = (
                config.migration_result
                if isinstance(config, ExperimentConfig)
                else self.migrator.preview_mapping(config, validate=False)
            )
        except Exception as error:
            return ValidationReport(
                source=source,
                errors=(self._format_error(error),),
                imports_checked=False,
            )
        data = copy.deepcopy(migration.config)
        errors = self._schema_errors(data)
        warnings: list[str] = []
        if migration.changed:
            warnings.append(
                f"Configuration normalized from "
                f"{migration.source_version.to_json_value() or 'unversioned'} to "
                f"{migration.target_version} using "
                f"{', '.join(step.identifier for step in migration.steps)}."
            )
        expanded_runs: int | None = None

        if not errors:
            try:
                normalized = ExperimentConfig(data, source=source)
                hpo = data.get("hpo", {})
                adaptive = isinstance(hpo, Mapping) and bool(hpo.get("enabled", False))
                if adaptive:
                    from lambdaforge.hpo.AdaptiveOptimizerConfig import (
                        AdaptiveOptimizerConfig,
                    )

                    optimizer = AdaptiveOptimizerConfig.from_experiment(data)
                    self._validate_adaptive(data, optimizer)
                    warnings.append(
                        "Adaptive HPO is dynamically materialized; no finite expanded-run count "
                        "exists before execution."
                    )
                else:
                    runs = normalized.expand()
                    expanded_runs = len(runs)
                    for run in runs:
                        errors.extend(self._schema_errors(run))
                ExecutionConfig.from_mapping(normalized)
                from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
                    ArtifactRetentionPolicy,
                )

                ArtifactRetentionPolicy.from_config(normalized)
            except Exception as error:
                errors.append(self._format_error(error))

        if check_imports:
            errors.extend(self._import_errors(data))
        else:
            warnings.append("Import references were not checked.")

        return ValidationReport(
            source=source,
            errors=tuple(dict.fromkeys(errors)),
            warnings=tuple(warnings),
            imports_checked=check_imports,
            expanded_runs=expanded_runs,
            source_schema_version=migration.source_version.to_json_value(),
            target_schema_version=migration.target_version.to_json_value(),
            migration_steps=tuple(step.to_dict() for step in migration.steps),
        )

    @staticmethod
    def _validate_adaptive(config: Mapping[str, Any], optimizer: AdaptiveOptimizerConfig) -> None:
        """Reject combinations that cannot preserve adaptive scientific identity."""
        sweep = config.get("sweep")
        if isinstance(sweep, Mapping) and any(sweep.values()):
            raise ValueError("An enabled hpo block cannot be combined with sweep.")
        checkpoint_policy = ExperimentConfig.get_value(
            config, "trainer.checkpoint_policy", "last_and_best"
        )
        if checkpoint_policy not in {"last", "last_and_best", "all"}:
            raise ValueError(
                "Adaptive fidelity requires trainer.checkpoint_policy to retain the last "
                "checkpoint."
            )
        if optimizer.min_budget_before_drop > optimizer.max_budget:
            raise ValueError("hpo.pruning.min_budget_before_drop cannot exceed fidelity.max.")
        if optimizer.max_search_seeds > len(optimizer.search_seeds):
            raise ValueError("hpo.seeds.max_search_seeds exceeds the declared search seeds.")
        if ExperimentConfig.get_value(config, "execution.mode", "sequential") == "ddp":
            raise ValueError(
                "Adaptive HPO currently schedules independent single-process trials; "
                "execution.mode ddp is unsupported."
            )
        if optimizer.memory_preflight and not ExperimentConfig.get_value(config, "execution.gpus"):
            raise ValueError("hpo.memory.preflight requires explicit execution.gpus.")
        forbidden = {"hpo", "sweep", "execution", "retention", "aggregation", "metadata"}
        for parameter in optimizer.space.parameters:
            if parameter.path.split(".", 1)[0] in forbidden:
                raise ValueError(
                    f"Adaptive search path {parameter.path!r} targets an operational section."
                )

    def _schema_errors(self, config: Mapping[str, Any]) -> list[str]:
        return list(self.schema_catalog.validation_errors(config))

    def _import_errors(self, value: Any, path: str = "<root>") -> list[str]:
        errors: list[str] = []
        if isinstance(value, Mapping):
            if (
                "plugin" in value
                and isinstance(value["plugin"], Mapping)
                and set(value) <= {"plugin", "params"}
            ):
                try:
                    self.plugins.resolve(
                        PluginReference.from_value(value["plugin"]),
                        record_usage=False,
                    )
                except Exception as error:
                    errors.append(f"import {path}.plugin: {self._format_error(error)}")
            for key in ("target", "ref"):
                if key in value and isinstance(value[key], str):
                    try:
                        ObjectFactory.import_object(value[key])
                    except Exception as error:
                        errors.append(f"import {path}.{key}: {self._format_error(error)}")
            for key, item in value.items():
                errors.extend(self._import_errors(item, f"{path}.{key}"))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                errors.extend(self._import_errors(item, f"{path}[{index}]"))
        return errors

    @staticmethod
    def _format_error(error: Exception) -> str:
        return f"{error.__class__.__name__}: {error}"
