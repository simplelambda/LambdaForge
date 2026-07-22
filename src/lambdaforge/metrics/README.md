# LambdaForge metrics

[Repository guide](../../../README.md) · [Español](README.es.md)

Metrics are stateful objects with explicit lifecycle and optimisation direction. They are independent
of Lightning and can be used in any loop that supplies output and batch mappings.

## Contents

- [Metric contract](#metric-contract)
- [Distributed metrics](#distributed-metrics)
- [Binary classification](#binary-classification)
- [Exact and streaming curve metrics](#exact-and-streaming-curve-metrics)
- [Multiclass classification](#multiclass-classification)
- [Regression](#regression)
- [Writing a metric](#writing-a-metric)

## Metric contract

```python
metric.reset()
for outputs, batch in evaluation_stream:
    metric.update(outputs, batch)
value = metric.compute()
```

Every `Metric` has a stable `name`, `higher_is_better` and derived `direction` (`max` or `min`).
Implementations detach tensor state from autograd and normally store it on CPU. `LightningTask`
deep-copies configured metrics for each stage, resets at the start of an epoch, updates per batch and
logs the final scalar.

Metric names must be unique inside a stage because they become YAML monitor and logging keys. Wrap
an existing object in `MetricAlias` when the same metric implementation is needed more than once:

```yaml
val_metrics:
  - target: lambdaforge.metrics.MetricAlias
    params:
      name: strict_accuracy
      metric:
        target: lambdaforge.metrics.classification.BinaryAccuracy
        params: {threshold: 0.8}
```

Empty inputs and undefined cases return `NaN` where a meaningful number does not exist—for example,
AUROC with only one target class. Downstream aggregation should preserve that missing information
rather than replacing it with zero.

## Distributed metrics

Non-linear metrics cannot be computed independently on every rank and then averaged. Before
`compute`, `Metric.synchronize` gathers each rank's `distributed_state` and calls
`merge_distributed_state`. Built-ins merge predictions, targets, counts or sufficient statistics as
appropriate.

A custom stateful metric used under DDP must implement both methods. If the world size is greater
than one and the contract is absent, LambdaForge raises a descriptive error instead of returning a
plausible but incorrect value. The generic contract uses `all_gather_object`, so metrics that retain
sample tensors can be expensive. The streaming binary and multiclass curve metrics instead override
`synchronize()` with an additive tensor `all_reduce` whose size is independent of sample count.

## Binary classification

Binary and multiclass implementations live in separate physical subpackages so that each folder
has one clear contract. Their stable public imports remain
`lambdaforge.metrics.classification.<ClassName>`; consumers should not depend on the internal file
layout.

Confusion-derived metrics share configurable `pred_key`, `target_key` and threshold:

- `BinaryAccuracy`
- `BinaryBalancedAccuracy`
- `BinaryPrecision`
- `BinaryRecall`
- `BinarySpecificity`
- `BinaryF1`
- `BinaryMCC`
- `BinaryCohenKappa`

Use probabilities with a threshold such as `0.5`, or make the chosen threshold consistent with the
score representation. `BinaryConfusionCounts` stores TP/TN/FP/FN and is the shared sufficient-state
object.

## Exact and streaming curve metrics

`BinaryAUROC` and `BinaryAUPRC` are the exact variants. They accumulate every raw ranking score and
target on CPU, using memory proportional to the number of samples. Scores may be logits or
probabilities because monotonic sigmoid conversion does not alter exact ranking. TorchMetrics is the
installed implementation; scikit-learn remains an optional fallback.

`StreamingBinaryAUROC` and `StreamingBinaryAUPRC` are explicit bounded-memory alternatives. They
retain two CPU `int64` histograms—one positive and one negative—of `num_bins` elements each. Their
histogram payload therefore occupies exactly `16 * num_bins` bytes, excluding small Python and
allocator overheads. The default of 4096 bins uses 65,536 payload bytes per metric, independently of
dataset size; processing one update still needs temporary memory proportional to that batch.

```yaml
val_metrics:
  - target: lambdaforge.metrics.classification.StreamingBinaryAUROC
    params:
      pred_key: logits
      target_key: y
      from_logits: true
      num_bins: 4096
  - target: lambdaforge.metrics.classification.StreamingBinaryAUPRC
    params:
      pred_key: logits
      target_key: y
      from_logits: true
      num_bins: 4096
```

`from_logits` is deliberately explicit. Set it to `true` to apply sigmoid before binning; with
`false`, every score must already be finite and inside `[0, 1]`. Per-batch autodetection would be
unsafe because different batches could otherwise use incompatible score transformations. Targets
must contain only zero and one.

Bins are uniform probability intervals. Samples in the same bin are treated as tied:

- streaming AUROC counts concordant positive-negative pairs and gives half credit to within-bin
  pairs;
- streaming AUPRC uses average precision, weighting precision by recall increments while visiting
  bins from highest to lowest score. It is not trapezoidal integration.

More bins increase score resolution, but neither metric promises a universal error such as
`1 / num_bins`; error also depends on how positive and negative scores occupy bins. For scientific
use, compare several bin counts with the exact metric on a representative subset, select the
resolution before the final experiment and retain it in YAML. Exact and streaming metrics return
`NaN` when the state is empty or either target class is absent.

Streaming state is serializable and mergeable. `reset()` clears both histograms and begins a new
lifecycle. After DDP synchronization, updates are rejected until reset so globally reduced counts
cannot be mixed with new local counts accidentally. Synchronization sums a fixed `2 × num_bins`
tensor with `all_reduce`, giving `O(num_bins)` state and communication with respect to dataset size.
`MetricAlias` delegates to this specialized synchronization.

### Streaming multiclass curves

`StreamingMulticlassAUROC` and `StreamingMulticlassAUPRC` apply the same bounded histogram approach
one-vs-rest. Their persistent state is exactly two CPU `int64` tensors shaped
`(num_classes, num_bins)`, or `16 * num_classes * num_bins` payload bytes. The class count is
mandatory because allocating or changing this state implicitly would make the memory contract
unpredictable.

```yaml
val_metrics:
  - target: lambdaforge.metrics.StreamingMulticlassAUROC
    params:
      num_classes: 10
      num_bins: 4096
      average: macro
      undefined_class_policy: ignore
      pred_key: logits
      target_key: y
      from_logits: true
  - target: lambdaforge.metrics.StreamingMulticlassAUPRC
    params:
      num_classes: 10
      num_bins: 4096
      average: weighted
      undefined_class_policy: ignore
```

`average` accepts `macro`, `weighted` or `micro`. `compute_per_class()` always exposes every
one-vs-rest result. `undefined_class_policy` controls absent positive/negative classes with
`ignore`, `nan` or `zero`; it is therefore never silently inferred from a batch. With
`from_logits: true`, scores are transformed with softmax. With `false`, scores must be finite,
inside `[0, 1]` and, by default, each row must sum to one within `probability_tolerance`;
`validate_probability_sum` can disable only that final check.

The binning caveat is the same as for binary curves: resolution is configurable but no universal
error bound follows from the number of bins. Compare against the exact metric on a representative
subset before fixing the experiment configuration. DDP synchronization reduces one fixed
`(2, num_classes, num_bins)` tensor, so communication is also independent of sample count. The
averaging semantics follow the current one-vs-rest definitions documented by
[TorchMetrics AUROC](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html) and
[average precision](https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html).

## Multiclass classification

- `MulticlassAccuracy`
- `MulticlassBalancedAccuracy`
- `MulticlassF1`
- `MulticlassAUROC`
- `MulticlassAUPRC`
- `StreamingMulticlassAUROC`
- `StreamingMulticlassAUPRC`

Inputs are score/logit tensors shaped `(samples, classes)` and integer targets shaped `(samples,)`.
`num_classes` may be supplied for validation or inferred by exact metrics; streaming curves require
it explicitly. Curve metrics use one-vs-rest reductions and default to a macro average.

## Regression

- `MAE`, `MSE`, `RMSE`
- `R2Score`
- `PearsonCorrelation`, `SpearmanCorrelation`
- `MeanMetric` for averaging any named scalar output

Regression metrics read configurable prediction and target keys. Additive metrics merge sufficient
statistics across DDP ranks. Correlations merge sample values; Spearman assigns average ranks to
ties. R² and correlations return `NaN` when their mathematical denominator or sample count makes the
result undefined.

## Writing a metric

Create one class in one module and implement:

```python
class ProjectMetric(Metric):
    def update(self, outputs, batch, context=None): ...
    def compute(self) -> float: ...
    def reset(self) -> None: ...
    def distributed_state(self): ...
    def merge_distributed_state(self, state): ...
```

Keep state detached, validate shapes early and declare the correct `higher_is_better` direction. Use
generic mapping keys supplied to the constructor. Add tests for empty state, normal values, edge
cases, reset, repeated updates and merged states. Export the class from the appropriate
`classification`, `regression` and/or top-level package initializer. An external project does not
need to modify LambdaForge: point `train_metrics`, `val_metrics` or `test_metrics` at the fully
qualified class in its own package. Supply explicit stage lists when the metric owns a resource that
cannot be deep-copied.
