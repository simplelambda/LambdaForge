"""Configurable confidence intervals and paired experiment comparisons."""

from __future__ import annotations

import copy
import csv
import json
import math
import statistics
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments import (
    ConfidenceIntervalMethod,
    ExperimentAggregator,
    ExperimentConfig,
    ExperimentValidator,
    PairedAlternative,
    PairedTestMethod,
    StatisticalComparisonConfig,
    WilcoxonCalculation,
    WilcoxonZeroMethod,
)
from lambdaforge.experiments.statistics import (
    BootstrapConfidenceInterval,
    WilcoxonSignedRankTest,
)


class TestComparisonStatistics:
    """Verify deterministic, bounded and YAML-selected comparison strategies."""

    @staticmethod
    def _schema_config(output_root: Path) -> dict:
        """Return a schema-valid experiment without constructing user objects."""
        return {
            "schema_version": "1.0",
            "experiment": {
                "name": "statistical-validation",
                "output_root": str(output_root),
                "seeds": [1, 2],
            },
            "data": {"train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"}},
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
        }

    @staticmethod
    def _comparison_protocol() -> dict:
        """Return a non-default protocol exercising every YAML option."""
        return {
            "aggregation": {
                "comparisons": {
                    "alpha": 0.01,
                    "target_power": 0.9,
                    "min_pairs_for_verdict": 4,
                    "confidence_interval": {
                        "method": "bootstrap_percentile",
                        "confidence_level": 0.9,
                        "resamples": 257,
                        "seed": 19,
                        "batch_size": 7,
                        "max_batch_elements": 20,
                    },
                    "paired_test": {
                        "method": "wilcoxon",
                        "alternative": "greater",
                        "calculation": "exact",
                        "zero_method": "pratt",
                        "continuity_correction": True,
                        "exact_max_pairs": 40,
                        "zero_tolerance": 1e-10,
                        "round_decimals": 10,
                    },
                }
            }
        }

    @staticmethod
    def _aggregation_config(output_root: Path) -> dict:
        """Return a two-variant experiment with explicit comparison methods."""
        return {
            "experiment": {
                "name": "comparison-integration",
                "output_root": str(output_root),
                "seeds": [1, 2, 3, 4],
            },
            "sweep": {
                "include_base": True,
                "ablations": [
                    {
                        "name": "candidate",
                        "set": {"metadata.selected": True},
                    }
                ],
            },
            "aggregation": {
                "comparisons": {
                    "confidence_interval": {
                        "method": "bootstrap_percentile",
                        "confidence_level": 0.9,
                        "resamples": 257,
                        "seed": 19,
                        "batch_size": 7,
                        "max_batch_elements": 20,
                    },
                    "paired_test": {
                        "method": "wilcoxon",
                        "alternative": "greater",
                        "calculation": "exact",
                        "zero_method": "wilcox",
                    },
                }
            },
        }

    @staticmethod
    def _write_aggregation_results(config: dict) -> None:
        """Persist paired final metrics using the framework's materialized layout."""
        values = {
            "base": [1.0, 1.2, 0.8, 1.1],
            "candidate": [0.8, 1.0, 0.7, 0.9],
        }
        for run in ExperimentConfig(config).expand():
            variant = str(ExperimentConfig.get_value(run, "experiment.variant"))
            seed = int(ExperimentConfig.get_value(run, "experiment.seed"))
            run_dir = ExperimentConfig.suite_dir_for(run) / variant / f"seed={seed}"
            run_dir.mkdir(parents=True)
            result = {
                "variant": variant,
                "seed": seed,
                "status": "ok",
                "best_metric": {},
                "final_metrics": {"val_loss": values[variant][seed - 1]},
            }
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    @staticmethod
    def _read_comparison(path: Path) -> tuple[list[str], dict[str, str]]:
        """Read the sole comparison and retain its exact CSV field order."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        assert len(rows) == 1
        return fields, rows[0]

    def test_runtime_and_schema_materialize_the_complete_yaml_protocol(self, tmp_path) -> None:
        mapping = self._comparison_protocol()
        protocol = StatisticalComparisonConfig.from_mapping(mapping)

        assert protocol.alpha == 0.01
        assert protocol.target_power == 0.9
        assert protocol.min_pairs_for_verdict == 4
        assert protocol.confidence_interval_method is ConfidenceIntervalMethod.BOOTSTRAP_PERCENTILE
        assert protocol.confidence_level == 0.9
        assert protocol.bootstrap_resamples == 257
        assert protocol.bootstrap_seed == 19
        assert protocol.bootstrap_batch_size == 7
        assert protocol.bootstrap_max_batch_elements == 20
        assert protocol.paired_test_method is PairedTestMethod.WILCOXON
        assert protocol.paired_alternative is PairedAlternative.GREATER
        assert protocol.wilcoxon_calculation is WilcoxonCalculation.EXACT
        assert protocol.wilcoxon_zero_method is WilcoxonZeroMethod.PRATT
        assert protocol.wilcoxon_continuity_correction is True
        assert protocol.wilcoxon_exact_max_pairs == 40
        assert protocol.zero_tolerance == 1e-10
        assert protocol.round_decimals == 10

        schema_config = self._schema_config(tmp_path)
        schema_config.update(copy.deepcopy(mapping))
        report = ExperimentValidator().validate(schema_config, check_imports=False)
        assert report.is_valid, report.errors

        invalid_schema = copy.deepcopy(schema_config)
        invalid_schema["aggregation"]["comparisons"]["paired_test"]["method"] = "paired_t"
        invalid_report = ExperimentValidator().validate(invalid_schema, check_imports=False)
        assert not invalid_report.is_valid
        assert any("paired_t" in error for error in invalid_report.errors)

        with pytest.raises(ValueError, match="unknown keys"):
            StatisticalComparisonConfig.from_mapping(
                {"aggregation": {"comparisons": {"confidence_interval": {"workers": 2}}}}
            )

    def test_omitted_protocol_preserves_normal_interval_and_sign_test_defaults(self) -> None:
        protocol = StatisticalComparisonConfig.from_mapping({})

        assert protocol.confidence_interval_method is ConfidenceIntervalMethod.NORMAL
        assert protocol.confidence_level == 0.95
        assert protocol.paired_test_method is PairedTestMethod.SIGN
        assert protocol.paired_alternative is PairedAlternative.OBSERVED_DIRECTION
        assert protocol.to_dict()["confidence_interval"]["method"] == "normal"
        assert protocol.to_dict()["paired_test"]["method"] == "sign"

    def test_direct_configuration_uses_the_same_validation_and_enum_normalization(self) -> None:
        protocol = StatisticalComparisonConfig(
            confidence_interval_method="normal",
            paired_test_method="wilcoxon",
            paired_alternative="two_sided",
        )
        assert protocol.confidence_interval_method is ConfidenceIntervalMethod.NORMAL
        assert protocol.paired_test_method is PairedTestMethod.WILCOXON
        assert protocol.paired_alternative is PairedAlternative.TWO_SIDED

        with pytest.raises(ValueError, match="between 0 and 1"):
            StatisticalComparisonConfig(alpha=2.0)
        with pytest.raises(ValueError, match="finite and non-negative"):
            WilcoxonSignedRankTest(zero_tolerance=float("nan"))

    def test_bootstrap_is_deterministic_per_identity_and_independent_of_batching(self) -> None:
        values = [-2.0, -0.5, 0.25, 1.5, 3.0]
        identity = ("base", "candidate", "final_score")
        large_batches = BootstrapConfidenceInterval(
            0.9,
            resamples=257,
            seed=17,
            batch_size=128,
            max_batch_elements=10_000,
        )
        small_batches = BootstrapConfidenceInterval(
            0.9,
            resamples=257,
            seed=17,
            batch_size=3,
            max_batch_elements=15,
        )

        first = large_batches.compute(values, identity=identity)
        repeated = large_batches.compute(values, identity=identity)
        rebatched = small_batches.compute(values, identity=identity)
        other_stream = large_batches.compute(values, identity=("base", "other", "final_score"))

        assert first.to_dict() == repeated.to_dict()
        assert first.effective_seed == rebatched.effective_seed
        assert first.effective_seed != other_stream.effective_seed
        assert first.lower == rebatched.lower
        assert first.upper == rebatched.upper
        assert first.standard_error == rebatched.standard_error
        assert first.status == "ok"
        assert first.method == "bootstrap_percentile"

        insufficient = large_batches.compute([4.0], identity=identity)
        assert insufficient.status == "unavailable"
        assert insufficient.reason == "insufficient_samples"
        assert insufficient.lower is None
        constant = large_batches.compute([4.0, 4.0, 4.0], identity=identity)
        assert constant.lower == constant.upper == 4.0
        assert constant.standard_error == 0.0
        assert constant.degenerate

    def test_bootstrap_caps_each_transient_index_batch(self, monkeypatch) -> None:
        generator = Mock()

        def zeros_for_indices(_low: int, _high: int, *, size: tuple[int, int]) -> np.ndarray:
            return np.zeros(size, dtype=np.int64)

        generator.integers.side_effect = zeros_for_indices
        constructor = Mock(return_value=generator)
        monkeypatch.setattr(np.random, "Generator", constructor)

        interval = BootstrapConfidenceInterval(
            resamples=11,
            seed=3,
            batch_size=9,
            max_batch_elements=10,
        ).compute([1.0, 2.0, 3.0, 4.0], identity=("memory-bound",))

        sizes = [call.kwargs["size"] for call in generator.integers.call_args_list]
        assert sizes == [(2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (1, 4)]
        assert all(rows * columns <= 10 for rows, columns in sizes)
        assert interval.status == "ok"
        assert constructor.call_count == 1

    def test_wilcoxon_exact_enumeration_handles_average_rank_ties(self) -> None:
        result = WilcoxonSignedRankTest(
            PairedAlternative.TWO_SIDED,
            calculation=WilcoxonCalculation.EXACT,
        ).compute([1.0, -1.0, 2.0])

        assert result.calculation_used == "exact"
        assert result.positive_statistic == 4.5
        assert result.negative_statistic == 1.5
        assert result.statistic == 1.5
        assert result.has_rank_ties
        assert result.p_value_better == pytest.approx(3 / 8)
        assert result.p_value_worse == pytest.approx(7 / 8)
        assert result.p_value_two_sided == pytest.approx(3 / 4)

    @pytest.mark.parametrize(
        ("zero_method", "positive", "negative"),
        [
            (WilcoxonZeroMethod.WILCOX, 1.0, 2.0),
            (WilcoxonZeroMethod.PRATT, 2.0, 3.0),
            (WilcoxonZeroMethod.ZSPLIT, 2.5, 3.5),
        ],
    )
    def test_wilcoxon_zero_conventions_are_explicit(
        self,
        zero_method: WilcoxonZeroMethod,
        positive: float,
        negative: float,
    ) -> None:
        result = WilcoxonSignedRankTest(
            PairedAlternative.TWO_SIDED,
            calculation=WilcoxonCalculation.EXACT,
            zero_method=zero_method,
        ).compute([0.0, 1.0, -2.0])

        assert result.n_pairs == 3
        assert result.n_effective == 2
        assert result.n_zero == 1
        assert result.positive_statistic == positive
        assert result.negative_statistic == negative
        assert result.status == "ok"

    def test_wilcoxon_reports_all_zero_and_forced_exact_limits(self) -> None:
        exact = WilcoxonSignedRankTest(
            PairedAlternative.TWO_SIDED,
            calculation=WilcoxonCalculation.EXACT,
            exact_max_pairs=3,
        )

        all_zero = exact.compute([0.0, 1e-13, -1e-13])
        assert all_zero.status == "unavailable"
        assert all_zero.reason == "no_nonzero_differences"
        assert all_zero.p_value is None
        assert all_zero.n_zero == 3

        over_limit = exact.compute([1.0, 2.0, 3.0, 4.0])
        assert over_limit.status == "unavailable"
        assert over_limit.reason == "exact_pair_limit_exceeded"
        assert over_limit.calculation_used is None

    def test_wilcoxon_auto_uses_the_documented_asymptotic_rank_variance(self) -> None:
        result = WilcoxonSignedRankTest(
            PairedAlternative.TWO_SIDED,
            calculation=WilcoxonCalculation.AUTO,
            exact_max_pairs=3,
        ).compute([1.0, 2.0, 3.0, 4.0])

        expected_z = 5.0 / math.sqrt(30.0 / 4.0)
        expected_better = 1.0 - statistics.NormalDist().cdf(expected_z)
        assert result.calculation_used == "asymptotic"
        assert result.z_statistic == pytest.approx(expected_z)
        assert result.p_value_better == pytest.approx(expected_better)
        assert result.p_value_two_sided == pytest.approx(2.0 * expected_better)

    def test_wilcoxon_zsplit_asymptotic_includes_split_zero_ranks(self) -> None:
        result = WilcoxonSignedRankTest(
            PairedAlternative.TWO_SIDED,
            calculation=WilcoxonCalculation.ASYMPTOTIC,
            zero_method=WilcoxonZeroMethod.ZSPLIT,
        ).compute([0.0, 1.0, 2.0, -3.0, -4.0])

        expected_z = (5.5 - 7.5) / math.sqrt(55.0 / 4.0)
        expected_worse = statistics.NormalDist().cdf(expected_z)
        assert result.positive_statistic == 5.5
        assert result.negative_statistic == 9.5
        assert result.z_statistic == pytest.approx(expected_z)
        assert result.p_value_two_sided == pytest.approx(2.0 * expected_worse)

    def test_one_sided_verdict_never_claims_the_opposite_hypothesis(self) -> None:
        protocol = StatisticalComparisonConfig(
            alpha=0.99,
            paired_alternative=PairedAlternative.GREATER,
        )
        verdict = ExperimentAggregator()._comparison_verdict(
            n_pairs=3,
            mean_improvement=-1.0,
            p_value=0.5,
            recommended_n=3,
            protocol=protocol,
        )
        assert verdict == "inconclusive"

    def test_aggregator_selects_yaml_methods_and_preserves_legacy_columns(self, tmp_path) -> None:
        config = self._aggregation_config(tmp_path)
        self._write_aggregation_results(config)

        result = ExperimentAggregator().write(
            config,
            make_plots=False,
            global_plots=False,
        )
        summary = result.to_summary_dict()
        aggregate_dir = tmp_path / "comparison-integration" / "aggregate"
        comparison_path = aggregate_dir / "baseline_comparisons.csv"
        fields, row = self._read_comparison(comparison_path)

        legacy_fields = {
            "ci95_improvement_low",
            "ci95_improvement_high",
            "wins",
            "losses",
            "ties",
            "p_value_sign_two_sided",
            "p_value_sign_better",
            "p_value_sign_worse",
            "p_value_directional",
            "q_value_bh_directional",
        }
        selected_fields = {
            "confidence_interval_method",
            "confidence_level",
            "confidence_interval_low",
            "confidence_interval_high",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_effective_seed",
            "paired_test_method",
            "paired_test_alternative",
            "paired_test_calculation_used",
            "paired_test_statistic",
            "paired_test_p_value_two_sided",
            "paired_test_p_value_better",
            "p_value_wilcoxon_two_sided",
            "p_value_wilcoxon_better",
        }
        assert legacy_fields | selected_fields <= set(fields)
        assert row["variant"] == "candidate"
        assert row["baseline_variant"] == "base"
        assert row["metric"] == "final_val_loss"
        assert row["mode"] == "min"
        assert row["paired_seeds"] == "1,2,3,4"
        assert row["confidence_interval_method"] == "bootstrap_percentile"
        assert float(row["confidence_level"]) == 0.9
        assert row["paired_test_method"] == "wilcoxon"
        assert row["paired_test_alternative"] == "greater"
        assert row["paired_test_calculation_used"] == "exact"
        assert float(row["p_value_directional"]) == pytest.approx(1 / 16)
        assert float(row["q_value_bh_directional"]) == pytest.approx(1 / 16)
        assert row["p_value_wilcoxon_better"] == row["paired_test_p_value_better"]
        assert row["p_value_sign_better"]
        assert row["ci95_improvement_low"]
        assert row["ci95_improvement_high"]
        assert row["bootstrap_effective_seed"]

        protocol = summary["reliability"]["statistical_protocol"]
        assert protocol["confidence_interval"]["method"] == "bootstrap_percentile"
        assert protocol["paired_test"]["method"] == "wilcoxon"
        reliability = json.loads((aggregate_dir / "reliability.json").read_text(encoding="utf-8"))
        assert reliability["statistical_protocol"] == protocol
        assert reliability["tests"]["confidence_interval"] == "bootstrap_percentile"
        assert reliability["tests"]["p_value"] == "wilcoxon"

        first_effective_seed = row["bootstrap_effective_seed"]
        ExperimentAggregator().write(config, make_plots=False, global_plots=False)
        _, repeated = self._read_comparison(comparison_path)
        assert repeated["bootstrap_effective_seed"] == first_effective_seed
        assert repeated["confidence_interval_low"] == row["confidence_interval_low"]
        assert repeated["confidence_interval_high"] == row["confidence_interval_high"]

        default_config = copy.deepcopy(config)
        default_config.pop("aggregation")
        ExperimentAggregator().write(default_config, make_plots=False, global_plots=False)
        _, compatible = self._read_comparison(comparison_path)
        assert compatible["confidence_interval_method"] == "normal"
        assert compatible["paired_test_method"] == "sign"
        assert compatible["p_value_directional"] == compatible["p_value_sign_better"]
        assert compatible["p_value_wilcoxon_two_sided"] == ""
        assert compatible["ci95_improvement_low"] == compatible["confidence_interval_low"]
        assert compatible["ci95_improvement_high"] == compatible["confidence_interval_high"]

    def test_cli_aggregate_applies_yaml_protocol_without_creating_plots(self, tmp_path) -> None:
        config = self._aggregation_config(tmp_path)
        self._write_aggregation_results(config)
        config_path = tmp_path / "comparison.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        assert CommandLineInterface.main(["aggregate", str(config_path), "--no-plots"]) == 0
        aggregate_dir = tmp_path / "comparison-integration" / "aggregate"
        _, row = self._read_comparison(aggregate_dir / "baseline_comparisons.csv")
        assert row["confidence_interval_method"] == "bootstrap_percentile"
        assert row["paired_test_method"] == "wilcoxon"
        assert not list((tmp_path / "comparison-integration").rglob("*.png"))
