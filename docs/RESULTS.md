[English](RESULTS.md) | [Español](RESULTS.es.md)

# Results and plotting guide

## 1. Select evidence safely

```bash
lambdaforge results list --root runs
lambdaforge results show baseline --root runs
lambdaforge results compare baseline ablation --metric val_loss --direction minimize
lambdaforge results export baseline --series --format csv --output analysis/curves.csv
```

Selectors accept an existing config/result path, exact attempt ID, fingerprint, run/experiment
name or variant. `show` returns every candidate and marks ambiguity; it never silently chooses the
latest. The older `lambdaforge results CONFIG --write-index --fail-on-ambiguous` audit remains
compatible (internally it is `results audit`). Publication selection still requires an explicit
attempt when successful results are ambiguous.

`MetricSeries` normalizes the existing dense `metrics.csv` into run, seed, variant, split, metric,
step, value and timestamp. It does not duplicate training logs. Metric names are exact; errors list
available names.
Comparison deltas use the first selector as baseline. Best/worst labels require explicit
`--direction minimize|maximize`, because LambdaForge does not guess metric semantics.

## 2. Learning curves and seed uncertainty

```bash
lambdaforge plot learning baseline --metric val_loss --aggregate mean \
  --uncertainty std --output plots/learning.svg
lambdaforge plot seeds baseline --metric val_accuracy --kind violin
```

`individual` draws each seed. `mean` groups the same variant/step. `std` shows sample deviation;
`ci` shows a normal mean interval. When `n=1`, `lower`/`upper` are null and no false uncertainty is
drawn. Multiple metrics use small multiples. For a running remote job,
`plot learning JOB --follow` repeatedly synchronizes only small metrics, writes the figure
atomically and exits at a terminal scheduler state or Ctrl-C.

## 3. Sweeps and HPO

```bash
lambdaforge plot sweep experiments/sweep.yaml --x optimizer.params.lr \
  --metric val_loss --uncertainty ci
lambdaforge plot sweep experiments/sweep.yaml --x model.params.width \
  --y optimizer.params.lr --metric val_accuracy --output sweep.html
lambdaforge plot sweep experiments/sweep.yaml --x model.params.width \
  --metric val_loss --metric val_accuracy --normalize
lambdaforge plot hpo runs/STUDY/.lambdaforge/adaptive/ID --parameter optimizer.params.lr
lambdaforge plot resources baseline
```

Sweep cells aggregate terminal metrics across seeds and expose `n`, sample deviation and the chosen
error bounds. Missing 2-D cells remain missing unless `--interpolate` explicitly records that
policy; the default static view is scatter/masked evidence, not invented science. `--normalize`
uses per-metric min-max scaling across observed cells and preserves raw values in the spec. HPO plotting reads
the existing durable state and shows objective, best-so-far, budget, status and optional parameter;
it does not change HPO algorithms. Resource plotting uses only resource-like metrics already
recorded and fails with an available-name list otherwise.

## 4. Reproducible output

`VisualizationService` first creates an immutable `PlotSpec`. `--json` returns that renderer-neutral
spec. PNG/SVG/PDF use Matplotlib; self-contained HTML needs `lambdaforge[viz]`. Rendering uses a
temporary file plus atomic replace. Every figure receives `FIGURE.plot.json`, containing the full
spec, fingerprint and generation timestamp; an identical request reuses the cached file. A plot
stored under a selected run's `plots/` exposes both figure and spec through `artifact list`.

Remote synchronization and explicit checkpoint download are described in [clusters](CLUSTERS.md).
