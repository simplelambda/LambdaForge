"""Single discoverable facade for LambdaForge's main workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ObjectFactory import ObjectFactory


class LambdaForge:
    """Framework facade for experiments and YAML object construction.

    Most users start with :meth:`experiment` or :meth:`run`. Lower-level
    components remain available from the documented ``lambdaforge.nn``,
    ``lambdaforge.metrics``, ``lambdaforge.training`` and
    ``lambdaforge.experiments`` namespaces.
    """

    VERSION = "0.1.0"

    @staticmethod
    def experiment(path: str | Path) -> Experiment:
        """Load a YAML experiment into the object API."""
        return Experiment.from_yaml(path)

    @staticmethod
    def run(
        path: str | Path,
        *,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
    ) -> list[dict[str, Any]]:
        """Load and execute a YAML experiment in one call."""
        return LambdaForge.experiment(path).run(
            dry_run=dry_run,
            execution_overrides=execution_overrides,
            aggregate_plots=aggregate_plots,
        )

    @staticmethod
    def build(spec: Any) -> Any:
        """Construct an object from a YAML-compatible ``target`` or ``ref`` spec."""
        return ObjectFactory.build(spec)
