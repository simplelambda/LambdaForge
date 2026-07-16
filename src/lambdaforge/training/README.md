# LambdaForge training and process layer

[Repository guide](../../../README.md) · [Español](README.es.md)

This package connects generic PyTorch objects to Lightning and runs independent training jobs with
bounded resources and explicit shutdown behaviour.

## Contents

- [Object map](#object-map)
- [Default training contract](#default-training-contract)
- [Configuration](#configuration)
- [Metrics and logging](#metrics-and-logging)
- [Concurrent jobs](#concurrent-jobs)
- [Shutdown and cleanup](#shutdown-and-cleanup)
- [Customisation](#customisation)

## Object map

| Object | Responsibility |
|---|---|
| `LightningTask` | Map a batch to a model, losses, metrics, optimizer and scheduler. |
| `LightningDataModule` | Wrap already-created train/validation/test PyTorch datasets. |
| `LightningTrainConfig` | Validate common Trainer settings and forward advanced settings. |
| `LightningRunner` | Build Trainer callbacks/logger and execute fit/test. |
| `TaskLoggingConfig` | Control total/individual loss and metric publication. |
| `EpochMetricsCSV` | Write one dense scalar row per epoch. |
| `EpochLogPrinter` | Print compact epoch summaries into captured logs. |
| `EpochStats` | Record wall time, GPU peak memory and process RSS. |
| `TrainingJob` | Serializable name/callable/device request for one independent job. |
| `TrainingOrchestrator` | Schedule jobs, bind resources and control their process trees. |
| `ProcessGuard` | Parent-death, thread-pool, affinity and descendant cleanup utilities. |
| `WindowsJobObject` | Kill-on-close containment for Windows descendants. |
| `LogKeyFilter` | Apply include/exclude patterns to CSV and console keys. |

The physical package separates `callbacks/`, `data/` and `orchestration/`; the stable
`lambdaforge.training` namespace re-exports the main objects. `CheckpointPolicy`, `LoggerMode`,
`MatmulPrecision` and `MonitorMode` are enums for closed configuration values.

## Default training contract

`LightningTask` expects a mapping-shaped batch. If `model_input_key="x"`, it calls
`model(batch["x"])`; when the key is `None`, it passes the whole mapping. Tensor results are wrapped
under `model_output_key`, while mapping results are preserved.

Every loss receives `(outputs, batch)` and returns a differentiable scalar. Metrics receive the same
objects through `update`; each stage owns a deep copy of its metrics. The task logs total loss,
individual losses and computed metrics. Custom keys make the contract usable for classification,
regression or structured outputs without hard-coded domain names.

The optimizer is a class plus keyword arguments. A scheduler is optional and may also provide the
Lightning scheduler metadata mapping (`interval`, `frequency`, `monitor`, and similar fields).

## Configuration

`LightningTrainConfig` explicitly owns stable, common settings: epochs, accelerator, devices,
strategy, precision, matrix precision, gradient accumulation/clipping, validation cadence,
checkpoint policy, early stopping, logger and UI behaviour. Values are validated before Trainer
construction.

Uncommon or newly introduced Lightning parameters belong in `trainer_kwargs`; explicit fields cannot
be overridden there. A logger can be one of `none`, `csv`, `lightning_csv` or an already-built logger
object. `write_epoch_metrics_csv` independently preserves the canonical dense artifact required by
LambdaForge aggregation. Additional callback objects are passed to `LightningRunner` through
`runner.params.callbacks`. For convenience, experiment YAML may also declare the same list as
top-level `callbacks`.

Checkpoint and early-stopping monitors default to the first **validation** metric, not the first
training metric. `checkpoint_monitor`, `checkpoint_mode`, `early_stopping_monitor` and
`early_stopping_mode` make the choice explicit. A custom non-loss key requires an explicit `min` or
`max` mode so LambdaForge never guesses the scientific direction.

`LightningDataModule` controls dataset, batch size, shuffling, worker count, pinned memory,
persistent workers, prefetching, collator, worker initializer and training `drop_last`. Shared or
per-split `dataloader_kwargs` forward safe extra options. Framework-owned keys cannot be repeated in
those mappings.

LambdaForge does not cache the datasets: they remain normal `torch.utils.data.Dataset` objects and
decide how samples are stored or loaded.

## Metrics and logging

The built-in `csv` mode disables Lightning's sparse CSV logger. `EpochMetricsCSV` is controlled
separately by `write_epoch_metrics_csv` and produces one dense row per epoch, including when a custom
external logger is active. `EpochLogPrinter` mirrors scalars to stdout, which the experiment layer
captures in `train.log`. `EpochStats` adds `epoch_time_s`, peak allocated/reserved GPU memory and CPU
RSS when available.

`metrics` remains the shared backwards-compatible list; `train_metrics`, `val_metrics` and
`test_metrics` select each stage explicitly. Names must be unique inside a stage. `MetricAlias`
renames a delegated metric when, for example, two accuracies use different thresholds.

`TaskLoggingConfig` controls total/individual loss logging, step/epoch aggregation, progress-bar
visibility, logger publication and distributed synchronisation. `epoch_metrics_include` /
`epoch_metrics_exclude` filter the dense CSV, and `epoch_console_include` /
`epoch_console_exclude` filter the terminal table with shell-style patterns. With no validation
metric, monitor selection falls back to `val_loss` in `min` mode. Early stopping is active whenever
patience is configured.

## Concurrent jobs

`TrainingOrchestrator.run` accepts `TrainingJob` objects and a maximum concurrency. Jobs use
`torch.multiprocessing` with `spawn`, required for safe CUDA initialisation. GPU restrictions are set
inside each child before CUDA use. The parent `CUDA_VISIBLE_DEVICES` is only read, never modified.

Each child may receive independent values for PyTorch intra/inter-op threads, BLAS/OpenMP variables
and CPU affinity. `DataLoader` workers inherit affinity and install `GuardedWorkerInit`, which first
applies framework protection and then invokes the user's initializer.

Avoid closures, lambdas and interactive-notebook-only callables in `TrainingJob`: everything sent to
a spawned process must be importable and pickle-safe. Entry scripts must use the conventional
`if __name__ == "__main__":` guard on Windows.

## Shutdown and cleanup

SIGINT and SIGTERM request cooperative stop through one shared event. `StopEventCallback` checks it
at train/validation/test batch boundaries. The orchestrator joins for `grace_seconds`, recursively
terminates surviving descendants, and restores signal handlers and parent affinity.

On Windows, every spawned root is assigned to a Job Object configured with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. On Linux, child processes request a parent-death signal where
supported. psutil provides the portable recursive fallback. Cleanup lives in `finally` paths so a
Python exception does not bypass it.

The guarantee covers framework-owned process trees. A third-party executable that deliberately
becomes an unrelated daemon is outside that tree and needs its own stop integration.

## Customisation

The complete YAML extension surface is illustrated below:

```yaml
model:
  target: my_project.models.ProjectModel
  params: {width: 128}
losses:
  - target: my_project.losses.ProjectLoss
    params: {name: reconstruction}
train_metrics:
  - target: my_project.metrics.ProjectMetric
    params: {name: train_score}
val_metrics:
  - target: my_project.metrics.ProjectMetric
    params: {name: selection_score}
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_key: features
    logging: {loss_prog_bar: false, metric_prog_bar: true, logger: true}
trainer:
  logger:
    target: my_project.logging.ProjectLogger
    params: {project: demo}
  write_epoch_metrics_csv: true
  checkpoint_monitor: val_selection_score
  checkpoint_mode: max
callbacks:
  - target: my_project.callbacks.ProjectCallback
    params: {}
```

Models may be ordinary `torch.nn.Module` objects. Losses and metrics subclass the LambdaForge
contracts so mixed-precision and DDP behaviour stay explicit. Nested `target`/`ref` specifications
also work in parameters. The automated suite runs this same pattern end to end with external-style
model, loss, metric, logger and callback classes.

- Subclass or replace `LightningTask` for non-mapping batches, multiple optimizers or manual
  optimisation.
- Supply a custom Lightning data module when dataset preparation/splitting belongs in the run.
- Supply callback and logger objects through YAML `target` specifications.
- Replace `runner.target` when a different training backend is required; the built-in experiment
  runner expects compatible `fit` and `test` methods.
- Keep process-sensitive changes covered by spawned-process tests and validate interruption on the
  target operating system and GPU host.
