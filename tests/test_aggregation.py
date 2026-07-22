"""Cross-seed artifact generation from persisted results."""

import json

import pytest

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

    @pytest.mark.parametrize(
        ("status", "expected_state", "expected_reason"),
        [
            ("failed", "ignored", "status=failed"),
            ("interrupted", "pending", "status=interrupted"),
            ("dry_run", "pending", "status=dry_run"),
            ("unknown", "pending", "status=unknown"),
            ("missing", "pending", "missing_result"),
        ],
    )
    def test_only_failed_results_are_terminally_ignored(
        self,
        tmp_path,
        status: str,
        expected_state: str,
        expected_reason: str,
    ) -> None:
        run_dir = tmp_path / status
        run_dir.mkdir()

        state, reason = ExperimentAggregator()._aggregate_state(
            {"status": status},
            run_dir,
        )

        assert state == expected_state
        assert reason == expected_reason

    def test_retryable_results_keep_variant_aggregation_non_terminal(self) -> None:
        expected_rows = [
            {
                "seed": 1,
                "status": "interrupted",
                "aggregate_status": "pending",
            }
        ]

        aggregate = ExperimentAggregator()._aggregate_variant(
            "base",
            expected_rows,
            seed_rows=[],
            metric_keys=[],
            metric_meta={},
        )

        assert not aggregate["terminal"]
        assert aggregate["pending_seeds"] == [1]
