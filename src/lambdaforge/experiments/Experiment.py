"""High-level object API for one YAML experiment suite."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.experiments.CheckpointChoice import CheckpointChoice
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig

if TYPE_CHECKING:
    from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
    from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
    from lambdaforge.experiments.RunLoader import RunLoader


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

    def run(
        self,
        *,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the suite and return one result mapping per materialized run."""
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
    ) -> dict[str, Any]:
        """Regenerate cross-seed artifacts from the configured output folder."""
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

    def load_model(
        self,
        seed: Any = None,
        variant: str | None = None,
        which: CheckpointChoice | str = CheckpointChoice.BEST,
    ) -> Any:
        """Load a trained model from this suite by seed and optional variant."""
        name = str(self.config.value("experiment.name"))
        return self.loader().load_model(name, seed, variant, which)
