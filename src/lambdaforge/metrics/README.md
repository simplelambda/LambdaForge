# LambdaForge metrics

[Repository guide](../../../README.md) · [Español](README.es.md)

Metrics are stateful objects with explicit lifecycle and optimisation direction. They are independent
of Lightning and can be used in any loop that supplies output and batch mappings.

## Contents

- [Metric contract](#metric-contract)
- [Distributed metrics](#distributed-metrics)
- [Binary classification](#binary-classification)
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
plausible but incorrect value. Synchronisation uses `all_gather_object`; very large stored tensors can
be expensive, which motivates a future streaming metric implementation.

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

`BinaryAUROC` and `BinaryAUPRC` accumulate raw ranking scores plus targets. Scores may be logits or
probabilities because monotonic sigmoid conversion does not alter ranking. They return `NaN` when
both classes are not present. TorchMetrics is the installed implementation; scikit-learn remains an
optional fallback.

## Multiclass classification

- `MulticlassAccuracy`
- `MulticlassBalancedAccuracy`
- `MulticlassF1`
- `MulticlassAUROC`
- `MulticlassAUPRC`

Inputs are score/logit tensors shaped `(samples, classes)` and integer targets shaped `(samples,)`.
`num_classes` may be supplied for validation or inferred from scores. Curve metrics use one-vs-rest
TorchMetrics reductions; built-in defaults are macro averages.

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
