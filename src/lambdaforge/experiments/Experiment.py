"""High-level object API for one YAML experiment suite."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.experiments.CheckpointChoice import CheckpointChoice
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig

if TYPE_CHECKING:
    from lambdaforge.experiments.AggregateResult import AggregateResult
    from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
    from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
    from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.RunLoader import RunLoader
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.experiments.ValidationReport import ValidationReport


class Experiment:
    """User-facing handle for configuration, execution, aggregation and loading."""

    def __init__(self, config: ExperimentConfig | Mapping[str, Any]) -> None:
        self.config = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        self._runner: ExperimentRunner | None = None
        self._aggregator: ExperimentAggregator | None = None

    @property
    def runner(self) -> ExperimentRunner:
        """Create the concrete run engine only when execution is requested."""
        if self._runner is None:
            from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

            self._runner = ExperimentRunner()
        return self._runner

    @property
    def aggregator(self) -> ExperimentAggregator:
        """Create the reporting engine only when aggregation is requested."""
        if self._aggregator is None:
            from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator

            self._aggregator = ExperimentAggregator()
        return self._aggregator

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        """Create an experiment from a YAML file."""
        return cls(ExperimentConfig.from_yaml(path))

    def expand(self) -> list[dict[str, Any]]:
        """Return all concrete variant/seed configurations without running them."""
        return self.config.expand()

    def validate(self, *, check_imports: bool = True) -> ValidationReport:
        """Validate schema, expansion, resources and optionally imports."""
        from lambdaforge.experiments.ExperimentValidator import ExperimentValidator

        return ExperimentValidator().validate(self.config, check_imports=check_imports)

    def run(
        self,
        *,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[RunResult]:
        """Execute the suite and return one typed result per materialized run."""
        return self.runner.run_experiment_config(
            self.config,
            dry_run=dry_run,
            execution_overrides=execution_overrides,
            aggregate_plots=aggregate_plots,
            on_run_finished=on_run_finished,
        )

    def aggregate(
        self,
        results: list[Mapping[str, Any]] | None = None,
        *,
        make_plots: bool = True,
    ) -> AggregateResult:
        """Regenerate artifacts and return typed cross-seed aggregates."""
        return self.aggregator.write(
            self.config,
            results or (),
            make_plots=make_plots,
            global_plots=True,
            variant_plot_policy="available",
        )

    def loader(self) -> RunLoader:
        """Create a loader rooted at this experiment's configured output root."""
        from lambdaforge.experiments.RunLoader import RunLoader

        output_root = self.config.value("experiment.output_root", "runs/experiments")
        return RunLoader(output_root)

    def result_catalog(self) -> ResultCatalog:
        """Create a catalog spanning current and archived attempts in this suite."""
        from lambdaforge.experiments.results.ResultCatalog import ResultCatalog

        return ResultCatalog(self.config.suite_dir)

    def results(
        self,
        *,
        status: str | None = None,
        fingerprint: str | None = None,
        include_archived: bool = True,
    ) -> tuple[ResultRecord, ...]:
        """Discover persisted attempts without starting or modifying a run."""
        return self.result_catalog().records(
            status=status,
            fingerprint=fingerprint,
            include_archived=include_archived,
        )

    def preview_retention(self) -> ArtifactRetentionPlan:
        """Build a read-only retention plan from the latest complete aggregation."""
        from lambdaforge.experiments.retention.ArtifactRetentionManager import (
            ArtifactRetentionManager,
        )

        return ArtifactRetentionManager().preview(self.config)

    def apply_retention(self) -> ArtifactRetentionResult:
        """Explicitly apply the current YAML retention policy after safety checks."""
        from lambdaforge.experiments.retention.ArtifactRetentionManager import (
            ArtifactRetentionManager,
        )

        return ArtifactRetentionManager().apply(self.config, explicit=True)

    def load_model(
        self,
        seed: Any = None,
        variant: str | None = None,
        which: CheckpointChoice | str = CheckpointChoice.BEST,
    ) -> Any:
        """Load a trained model from this suite by seed and optional variant."""
        name = str(self.config.value("experiment.name"))
        return self.loader().load_model(name, seed, variant, which)
