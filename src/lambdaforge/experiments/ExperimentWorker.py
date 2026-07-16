"""Pickle-safe subprocess entry object for one experiment run."""

from __future__ import annotations

import json
import traceback
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.RunStatus import RunStatus


class ExperimentWorker:
    """Execute one materialized config and persist failures in its run folder."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def __call__(self, stop_event: Any) -> None:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
        from lambdaforge.experiments.StdIOCapture import StdIOCapture

        runner = ExperimentRunner()
        run_dir = runner.experiment_run_dir(self.config)
        run_dir.mkdir(parents=True, exist_ok=True)
        with StdIOCapture(run_dir / "train.log", echo=False):
            try:
                runner.run_single_experiment(self.config, stop_event=stop_event)
            except Exception:
                traceback.print_exc()
                payload = {
                    "name": ExperimentConfig.get_value(self.config, "experiment.name"),
                    "run_dir": str(run_dir),
                    "variant": ExperimentConfig.get_value(self.config, "experiment.variant"),
                    "seed": ExperimentConfig.get_value(self.config, "experiment.seed"),
                    "status": RunStatus.FAILED.value,
                    "error": traceback.format_exc().splitlines()[-1],
                }
                with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                raise
