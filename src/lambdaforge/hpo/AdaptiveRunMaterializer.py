"""Materialize one adaptive action as a normal LambdaForge training run."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig


class AdaptiveRunMaterializer:
    """Bridge adaptive decisions to the existing checkpoint-aware runner contract."""

    def materialize(
        self,
        base: Mapping[str, Any],
        action: AdaptiveAction,
        optimizer: AdaptiveOptimizerConfig,
    ) -> dict[str, Any]:
        """Return a standalone run mapping for the action's cumulative target budget."""
        output = copy.deepcopy(dict(base))
        output.pop("hpo", None)
        output.pop("sweep", None)
        for path, value in action.parameters.items():
            ExperimentConfig.set_value(output, path, copy.deepcopy(value))

        base_name = str(ExperimentConfig.get_value(base, "experiment.name", "experiment"))
        ExperimentConfig.set_value(output, "experiment.base_name", base_name)
        ExperimentConfig.set_value(output, "experiment.variant", action.config_id)
        ExperimentConfig.set_value(
            output,
            "experiment.name",
            ExperimentConfig.slugify(f"{base_name}__{action.config_id}"),
        )
        ExperimentConfig.set_value(output, "experiment.seed", action.seed)
        ExperimentConfig.set_value(output, "experiment.seeds", [action.seed])
        ExperimentConfig.set_value(output, "experiment.resume", True)
        ExperimentConfig.set_value(output, "experiment.rerun_completed", action.current_budget > 0)
        ExperimentConfig.set_value(output, "trainer.max_epochs", action.target_budget)
        ExperimentConfig.set_value(output, "trainer.write_epoch_metrics_csv", True)
        ExperimentConfig.set_value(
            output,
            "metadata.adaptive",
            {
                "action_id": action.action_id,
                "config_id": action.config_id,
                "phase": action.phase.value,
                "current_budget": action.current_budget,
                "target_budget": action.target_budget,
                "max_budget": optimizer.max_budget,
                "memory_budget_bytes": action.memory_reservation_bytes,
            },
        )
        return output
