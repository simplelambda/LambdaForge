"""Cross-seed artifact generation from persisted results."""

import json

from lambdaforge.experiments import ExperimentAggregator, ExperimentConfig


class TestExperimentAggregator:
    """Verify aggregation can rebuild a suite entirely from disk."""

    def test_aggregates_final_metrics_across_seeds(self, tmp_path) -> None:
        config = {
            "experiment": {"name": "demo", "output_root": str(tmp_path), "seeds": [1, 2]},
        }
        for index, run in enumerate(ExperimentConfig(config).expand(), start=1):
            run_dir = ExperimentConfig.suite_dir_for(run) / "base" / f"seed={index}"
            run_dir.mkdir(parents=True)
            result = {
                "variant": "base",
                "seed": index,
                "status": "ok",
                "seconds": float(index),
                "best_metric": {},
                "final_metrics": {"val_loss": float(index)},
            }
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

        aggregates = ExperimentAggregator().write(config, make_plots=False)
        assert aggregates["base"]["n_seeds"] == 2
        assert aggregates["base"]["metrics"]["final_val_loss"]["mean"] == 1.5
        assert (tmp_path / "demo" / "aggregate" / "summary.json").exists()
