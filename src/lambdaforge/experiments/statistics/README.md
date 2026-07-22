# Statistical comparisons

[Experiment guide](../README.md) · [Repository guide](../../../../README.md) ·
[Español](README.es.md)

This package provides object-based, YAML-selectable uncertainty estimates and paired tests for
cross-seed experiment comparisons. It operates on already materialised scalar metrics and does not
load models, datasets or checkpoints.

## Contents

- [Comparison contract](#comparison-contract)
- [Complete YAML reference](#complete-yaml-reference)
- [Pairing and improvement semantics](#pairing-and-improvement-semantics)
- [Confidence intervals](#confidence-intervals)
- [Paired tests](#paired-tests)
- [Artifacts and compatibility](#artifacts-and-compatibility)
- [Python API](#python-api)
- [Interpretation](#interpretation)

## Comparison contract

`StatisticalComparisonConfig.from_mapping` reads only `aggregation.comparisons`, rejects unknown
framework keys and materialises every default. Strategy-specific settings remain validated and are
written to `statistical_protocol` even when their strategy is not active, making a later method
change explicit and reproducible.

| YAML key | Default | Valid values |
|---|---:|---|
| `alpha` | `0.05` | Number strictly between 0 and 1. |
| `target_power` | `0.80` | Number strictly between 0 and 1. |
| `min_pairs_for_verdict` | `3` | Integer at least 1. |
| `confidence_interval.method` | `normal` | `normal`, `bootstrap_percentile`. |
| `confidence_interval.confidence_level` | `0.95` | Number strictly between 0 and 1. |
| `confidence_interval.resamples` | `10000` | Integer from 1 to 10,000,000. |
| `confidence_interval.seed` | `0` | Non-negative integer. |
| `confidence_interval.batch_size` | `1024` | Integer from 1 to 1,000,000. |
| `confidence_interval.max_batch_elements` | `1000000` | Integer from 1 to 100,000,000. |
| `paired_test.method` | `sign` | `sign`, `wilcoxon`. |
| `paired_test.alternative` | `observed_direction` | `two_sided`, `greater`, `less`, `observed_direction`. |
| `paired_test.calculation` | `auto` | `auto`, `exact`, `asymptotic`. |
| `paired_test.zero_method` | `wilcox` | `wilcox`, `pratt`, `zsplit`. |
| `paired_test.continuity_correction` | `false` | Boolean. |
| `paired_test.exact_max_pairs` | `50` | Integer from 0 to 200. |
| `paired_test.zero_tolerance` | `1.0e-12` | Finite non-negative number. |
| `paired_test.round_decimals` | `12` | `null` or integer from 0 to 15. |

Omitting `aggregation`, `comparisons` or either strategy mapping is valid. A completely omitted
block reproduces the prior aggregate protocol: 95% normal interval, exact paired sign test,
`observed_direction`, alpha 0.05, target power 0.80 and three pairs before a verdict.

## Complete YAML reference

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon
      alternative: two_sided
      calculation: auto
      zero_method: wilcox
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12
```

The canonical [experiment example](../../../../examples/experiment.yaml) carries the same block.

## Pairing and improvement semantics

Comparisons are paired, never treated as independent samples:

1. A variant named `parent__ablation` uses `parent` as baseline when it exists. Other non-base
   variants use literal `base` when present.
2. Only seeds with finite values on both sides enter a metric comparison. `n_pairs` and
   `paired_seeds` expose the resulting sample.
3. `delta = variant - baseline`.
4. `improvement = delta` for a `max` metric and `-delta` for a `min` metric. Positive improvement
   therefore always means better.

An explicit metric/monitor mode takes precedence. If no mode metadata exists, names containing
`loss`, `time`, `seconds`, `mem` or `rss` are treated as `min` and all others as `max`. Configure
monitor modes explicitly whenever this fallback could be ambiguous.

No common seeds, no finite pairs or no applicable baseline produce explicit unavailable metadata
instead of an unpaired fallback.

## Confidence intervals

Both estimators target the arithmetic mean of paired improvements.

### Normal

`NormalConfidenceInterval` uses the sample standard deviation, standard error and a two-sided
normal critical value at `confidence_level`. Fewer than two pairs return
`status: unavailable`/`reason: insufficient_samples`. Zero variance yields a degenerate interval.
This is the compatibility default.

### Deterministic percentile bootstrap

`BootstrapConfidenceInterval` samples paired improvements with replacement, stores each resampled
mean and takes linear lower/upper quantiles. It also reports the bootstrap standard deviation of
those means.

Reproducibility is comparison-local. LambdaForge hashes the base `seed` together with the canonical
`(baseline_variant, variant, metric)` identity using SHA-256 and initializes a PCG64 stream with the
resulting 64-bit `effective_seed`. Reordering metrics or adding an unrelated comparison therefore
does not alter an existing interval.

Memory use is deliberately bounded:

- the retained array is one `float64` mean per resample, or `O(resamples)`;
- the temporary index matrix has
  `min(batch_size, max(1, max_batch_elements // n_pairs)) * n_pairs` elements;
- when one resample itself exceeds `max_batch_elements`, one row is the unavoidable floor.

Fewer than two pairs are unavailable without allocating resample matrices. Constant samples return
their constant endpoints and `degenerate: true` without random sampling.

## Paired tests

`SignTest`, the compatibility default, applies an exact binomial test to the signs of improvements
outside `zero_tolerance`. Ties do not enter its effective sample size.

`WilcoxonSignedRankTest` ranks absolute paired improvements and evaluates their signed rank sum. It
uses average ranks for equal magnitudes, reports `has_rank_ties` and supports:

| Setting | Meaning |
|---|---|
| `alternative: two_sided` | Use the doubled smaller tail. |
| `alternative: greater` | Test for positive improvement (variant better). |
| `alternative: less` | Test for negative improvement (variant worse). |
| `alternative: observed_direction` | Select `greater` or `less` from the observed mean direction. |
| `calculation: exact` | Enumerate the conditional sign distribution; above `exact_max_pairs` return unavailable. |
| `calculation: asymptotic` | Use a normal approximation, optionally with continuity correction. |
| `calculation: auto` | Exact up to `exact_max_pairs` non-zero pairs, asymptotic above it. |

Exact calculation uses a deterministic dynamic program over the observed (half-rank-scaled) ranks,
so average-rank ties are supported without random tie breaking. `exact_max_pairs` bounds this work;
an explicitly requested exact calculation never silently changes to asymptotic.

`round_decimals` is applied before zero detection and ranking. Then values whose absolute magnitude
is at most `zero_tolerance` are zeros:

- `wilcox` discards zeros before assigning ranks;
- `pratt` includes zeros when assigning ranks but excludes their ranks from the random sign sum;
- `zsplit` follows Pratt ranking and splits the zero-rank contribution equally between the reported
  positive and negative statistics.

All-zero pairs return `status: unavailable`/`reason: no_nonzero_differences` rather than NaN.
`PairedTestResult` always exposes the selected p-value plus two-sided, better and worse diagnostics,
the requested/used calculation, rank statistics and effective/zero counts.

## Artifacts and compatibility

Aggregation schema version 4 writes:

| Artifact | Statistical content |
|---|---|
| `aggregate/baseline_comparisons.csv` | One row per baseline/variant/metric comparison with selected interval/test fields. |
| `aggregate/reliability.json` | Materialised protocol, baseline rule, all comparisons, primary comparisons and seed recommendations. |
| `aggregate/summary.json` | Protocol summary, comparison counts and artifact paths. |

Method-neutral columns include `confidence_interval_method`, `confidence_level`,
`confidence_interval_low/high`, `confidence_interval_standard_error`, interval status/reason,
bootstrap seed metadata, `paired_test_method`, alternative, requested/used calculation, rank
statistics, effective/zero counts, all paired p-values and test status/reason.

`p_value_directional` is the p-value selected by `alternative`. Benjamini-Hochberg correction across
all available comparison rows produces `q_value_bh_directional`, and the selected p-value drives the
verdict. The seed-count recommendation remains an observed-effect normal approximation and is
reported with its reason.

For compatibility, `ci95_improvement_low/high` always retain the historical 95% normal interval and
`wins`, `losses`, `ties` plus `p_value_sign_*` always retain the exact sign-test diagnostics.
`p_value_wilcoxon_*` is populated only when Wilcoxon is selected. Missing estimates and p-values use
JSON `null`/empty CSV cells, never NaN sentinels.

## Python API

The stable `lambdaforge.experiments` namespace exports:

- `StatisticalComparisonConfig`;
- `ConfidenceIntervalMethod` and `ConfidenceIntervalResult`;
- `PairedAlternative`, `PairedTestMethod` and `PairedTestResult`;
- `WilcoxonCalculation` and `WilcoxonZeroMethod`.

The focused `lambdaforge.experiments.statistics` namespace additionally exports
`StatisticalComparisonEngine`, `NormalConfidenceInterval`, `BootstrapConfidenceInterval`,
`SignTest` and `WilcoxonSignedRankTest`.

```python
from lambdaforge.experiments import StatisticalComparisonConfig
from lambdaforge.experiments.statistics import StatisticalComparisonEngine

protocol = StatisticalComparisonConfig.from_mapping(
    {
        "aggregation": {
            "comparisons": {
                "confidence_interval": {"method": "bootstrap_percentile", "seed": 17},
                "paired_test": {"method": "wilcoxon", "alternative": "two_sided"},
            }
        }
    }
)
engine = StatisticalComparisonEngine(protocol)
interval = engine.confidence_interval(
    [0.02, 0.01, 0.03],
    identity=("base", "candidate", "val_auroc"),
)
test = engine.paired_test([0.02, 0.01, 0.03])
assert interval.status == "ok"
assert test.method == "wilcoxon"
```

Both result classes are frozen, slotted dataclasses with `to_dict()` for artifact or integration
code. Config, estimator and test constructors validate numerical safety; YAML receives the same
checks through the packaged Schema and config object.

## Interpretation

These outputs are exploratory framework primitives, not a universal study design. Choose metric
direction, alternative, confidence level, seed count and multiplicity policy before interpreting a
confirmatory experiment. In particular, `observed_direction` intentionally follows the observed
mean and is useful for exploratory reports; prefer a predeclared `two_sided`, `greater` or `less`
alternative for confirmatory inference. Very small paired samples have low resolution even under an
exact test, and asymptotic Wilcoxon and seed recommendations remain approximations.
