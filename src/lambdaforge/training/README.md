# LambdaForge training and process layer

[Repository guide](../../../README.md) · [Español](README.es.md)

This package connects generic PyTorch objects to Lightning and runs independent training jobs with
bounded resources and explicit shutdown behaviour.

## Contents

- [Object map](#object-map)
- [Default training contract](#default-training-contract)
- [Multiple inputs and optimizer groups](#multiple-inputs-and-optimizer-groups)
- [Configuration](#configuration)
- [Checkpoint lifecycle and suite locks](#checkpoint-lifecycle-and-suite-locks)
- [Metrics and logging](#metrics-and-logging)
- [Optional tracking adapters](#optional-tracking-adapters)
- [Concurrent jobs](#concurrent-jobs)
- [Shutdown and cleanup](#shutdown-and-cleanup)
- [Lifecycle verification](#lifecycle-verification)
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
| `MLflowTrackingLogger` | Optional local/remote MLflow logger from `lambdaforge.tracking`. |
| `TensorBoardTrackingLogger` | Optional TensorBoard event logger from `lambdaforge.tracking`. |
| `WeightsAndBiasesTrackingLogger` | Optional offline/online W&B logger from `lambdaforge.tracking`. |
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

Every loss receives `(outputs, batch, context)`, where `context` is the active `LightningTask`, and
returns a differentiable scalar. Custom losses should keep `context=None` in their signature so they
also remain usable independently. Metrics receive outputs and batch through `update`; each stage owns
a deep copy of its metrics. The task logs total loss, individual losses and computed metrics. Custom
keys make the contract usable for classification, regression or structured outputs without
hard-coded domain names.

The optimizer is a class plus keyword arguments. A scheduler is optional and may also provide the
Lightning scheduler metadata mapping (`interval`, `frequency`, `monitor`, and similar fields).

## Multiple inputs and optimizer groups

`model_input_keys` removes the need for a custom task when a model consumes several tensors. A
sequence routes positional arguments in order; a mapping routes named model arguments to batch
keys. It is mutually exclusive with a non-default `model_input_key`:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys:
      x: node_features
      edge_index: edge_index
```

Models may expose `parameter_groups()` as a mapping from stable group names to parameter iterables.
`optimizer_group_kwargs` then overrides optimizer options per group while the ordinary
`optimizer_kwargs` remain the defaults. Differentiable-tree models use this contract for routing,
leaf and ensemble parameters without coupling `LightningTask` to a specific architecture:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    optimizer_kwargs: {lr: 0.001, weight_decay: 0.0001}
    optimizer_group_kwargs:
      routing: {lr: 0.0005}
      leaves: {weight_decay: 0.0}
```

Unknown group names fail during optimizer construction. Parameters omitted from custom groups are
placed in a `task` group, so trainable loss/task parameters are not silently lost.

## Configuration

`LightningTrainConfig` explicitly owns stable, common settings: epochs, accelerator, devices,
strategy, precision, matrix precision, gradient accumulation/clipping, validation cadence,
checkpoint policy, early stopping, logger and UI behaviour. Values are validated before Trainer
construction.

Uncommon or newly introduced Lightning parameters belong in `trainer_kwargs`; explicit fields cannot
be overridden there. A logger can be `none`, `csv`, `lightning_csv`, a `target`/`ref`/installed
logger plugin, or a non-empty list of logger objects. `write_epoch_metrics_csv` independently
preserves the canonical dense artifact required by LambdaForge aggregation. Additional callbacks
use `target` or `kind: callback` plugin specifications through `runner.params.callbacks` or the
top-level `callbacks` convenience list. No external logger service is required. The optional
provider adapters and extras are documented in the [tracking guide](../tracking/README.md).

Checkpoint and early-stopping monitors default to the first **validation** metric, not the first
training metric. `checkpoint_monitor`, `checkpoint_mode`, `early_stopping_monitor` and
`early_stopping_mode` make the choice explicit. A custom non-loss key requires an explicit `min` or
`max` mode so LambdaForge never guesses the scientific direction.

`LightningDataModule` controls dataset, batch size, shuffling, worker count, pinned memory,
persistent workers, prefetching, collator, worker initializer and training `drop_last`. Shared or
per-split `dataloader_kwargs` forward safe extra options. Framework-owned keys cannot be repeated in
those mappings.

Datasets remain ordinary `torch.utils.data.Dataset` objects. Projects may opt into the public
`lambdaforge.data` wrappers for explicit file/mmap loading or bounded caching; the training layer
never inserts a cache implicitly. Data splits accept a local `target` or reusable `kind: dataset`
plugin. An `IterableDataset` requires a custom datamodule that does not force map-style shuffling.
Each worker owns a dataset replica, so read the
[data guide](../data/README.md) before combining `DatasetCache`, `num_workers` and
`persistent_workers`.

## Checkpoint lifecycle and suite locks

`trainer.checkpoint_policy` controls what Lightning creates during fitting: `none`, `last`, `best`,
`last_and_best` or `all`. The top-level Schema 1.1 `retention.checkpoints` policy independently
controls what may survive only after a complete successful suite has been aggregated. In
particular, `prune_unselected: false` preserves every created checkpoint, whatever `keep` says.

Resume and completion inspect safe files inside the run. `checkpoint_policy: all` therefore remains
reusable even if Lightning does not record a best/last callback path. Loading with
`CheckpointChoice.AUTO` resolves best, then last, then the latest safe local checkpoint. Exact
`BEST` and `LAST` choices never cross roles silently; retention also skips an ambiguous or absent
role instead of guessing.

The experiment executor owns an exclusive suite activity lock while training workers are active. A
final aggregator takes a shared activity lease plus the aggregation lock; retention takes exclusive
activity, aggregation and retention locks in that fixed order. This prevents a second LambdaForge
process from publishing a completion receipt or pruning files while a suite is training. The locks
are OS-owned, have a configurable timeout and release on normal or abrupt process exit.

A cooperative stop after fit is recorded as `interrupted`, not `ok`, and `test_after_fit` is skipped.
That run remains retryable and cannot contribute to a retention eligibility receipt. See the
[artifact-retention guide](../experiments/retention/README.md) for checkpoint roles, receipts,
transaction recovery and path protections.

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

## Optional tracking adapters

`lambdaforge.tracking` exposes explicit `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` and `WeightsAndBiasesTrackingLogger` targets. They use the same
`trainer.logger` position as every custom Lightning logger and may be combined in a non-empty list:

```yaml
trainer:
  logger:
    - target: lambdaforge.tracking.TensorBoardTrackingLogger
      params:
        save_dir: ./tracking/tensorboard
        name: local-study
    - target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
      params:
        project: local-study
        offline: true
        save_dir: ./tracking/wandb
        log_model: false
  write_epoch_metrics_csv: true
```

Set `task.params.logging.logger: true` when task losses and metrics should reach the provider;
choosing a provider does not override this publication policy. `EpochStats` runtime scalars use
Lightning's logger path independently.

Install the smallest matching extra—`lambdaforge[mlflow]`, `lambdaforge[tensorboard]` or
`lambdaforge[wandb]`—or `lambdaforge[tracking]` for all three. Imports remain lazy; constructing an
adapter without its SDK raises `TrackingDependencyError` with the exact installation hint.

`TaskLoggingConfig.logger` governs task losses/metrics, while framework callbacks such as
`EpochStats` also publish runtime scalars and a provider may collect additional metadata according
to its settings. Keep credentials out of YAML, leave `log_model: false` unless remote checkpoint
copies are intentional and retain `write_epoch_metrics_csv: true` for a provider-neutral local
record. LambdaForge retention cannot delete uploaded/provider-owned artifacts. The
[tracking guide](../tracking/README.md) contains the complete constructors, MLflow local/remote,
TensorBoard local and W&B offline/online examples, privacy cautions and official provider links.

## Concurrent jobs

`TrainingOrchestrator.run` launches every supplied `TrainingJob`; `run_scheduled` instead consumes a
fixed slot pool and therefore provides the concurrency bound. Jobs use `torch.multiprocessing` with
`spawn`, required for safe CUDA initialisation. GPU restrictions are staged for process creation and
set inside each child before CUDA use; the parent value is restored immediately after every spawn.

`TrainingJob.devices` and scheduler slots share one immutable device contract:

| Value | Meaning |
|---|---|
| `None` | Inherit the complete parent-visible CUDA set unchanged. |
| `[]` or `()` | Explicit CPU mode: the child receives `CUDA_VISIBLE_DEVICES=""`. |
| A non-empty integer sequence | Logical positions inside the parent's visible set, or physical IDs if the parent variable is absent. |

`run_scheduled` intentionally uses the selected slot instead of `TrainingJob.devices`; this is what
makes the slot pool, rather than a global counter, the resource authority.

Assignments reject strings, booleans, negative/fractional indices and duplicates. `ExecutionConfig`
also rejects boolean, fractional and non-finite resource values, requires finite non-negative
`grace_seconds`, positive job/device counts and consistent non-empty GPU groups for parallel/DDP.
Optional CPU limits use `null` to preserve the environment; zero is valid only for DataLoader
workers. `TrainingOrchestrator` itself requires finite non-negative `grace_seconds` and finite
positive `poll_seconds`.

Each child may receive independent values for PyTorch intra/inter-op threads, BLAS/OpenMP variables
and CPU affinity. `DataLoader` workers inherit affinity and install `GuardedWorkerInit`, which first
applies framework protection and then invokes the user's initializer.

Avoid closures, lambdas and interactive-notebook-only callables in `TrainingJob`: everything sent to
a spawned process must be importable and pickle-safe. Entry scripts must use the conventional
`if __name__ == "__main__":` guard on Windows.

## Shutdown and cleanup

`TrainingOrchestrator.request_stop()` is the public idempotent cancellation entry point. With
`manage_signals=True` (the default), temporary SIGINT/SIGTERM handlers and SIGBREAK where available
record a lock-free local request; the ordinary orchestration loop then calls that method and child
guards relay cooperative signals to the shared event from normal thread context. This avoids taking
`multiprocessing.Event` locks inside a signal handler, which can deadlock if the signal interrupted
the same lock. `StopEventCallback` checks the event at train/validation/test batch boundaries. The
orchestrator waits for `grace_seconds`, recursively terminates surviving trees, kills remaining
processes and restores signal handlers and parent affinity. Joins and recursive termination use
shared monotonic deadlines, so the configured wait is not multiplied by the number of jobs; a
survivor after bounded escalation raises `RuntimeError`.

Python only permits installing signal handlers in its main thread. A host that invokes
`run`/`run_scheduled` from a secondary thread must construct
`TrainingOrchestrator(manage_signals=False)` and call `request_stop()` from its own shutdown hook.
Leaving `manage_signals=True` in a secondary thread fails before launching work and restores any
partially installed state. `manage_signals=False` transfers handler ownership to the host; it does
not remove the shared-event cancellation path.

On Windows, every spawned root is assigned to a Job Object configured with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Initialisation failure emits `RuntimeWarning` and is retained
in `process_isolation_warnings` while psutil cleanup stays active. Assignment failure on an active
Job Object terminates the new worker and raises instead of silently weakening containment.

Every worker also receives the launcher's exact `expected_parent_pid`. Linux requests a kernel
parent-death signal with `prctl`; all POSIX workers compare the observed parent before and after
installation and run a lightweight watchdog for later reparenting. This closes the race where the
launcher dies during child startup. `DataLoader` workers inherit the applicable guard and resource
limits. Cleanup lives in `finally` paths so a Python exception does not bypass it.

The guarantee covers framework-owned process trees. A third-party executable that deliberately
becomes an unrelated daemon is outside that tree and needs its own stop integration.
`TrainingOrchestrator` is stateful and not re-entrant: never overlap `run` and `run_scheduled` calls
on the same instance, and prefer one instance per independently owned run.

## Lifecycle verification

Hosted CI creates a real launcher/worker/nested-descendant tree on Ubuntu and Windows for CPython
3.10-3.14. On POSIX it sends a real `killpg(SIGINT)` to the isolated process group. On Windows an
external request makes the launcher raise targeted Python SIGBREAK because native console events
are group-wide and would also reach unrelated numerical-runtime processes in the test group. The
suite therefore validates the Python SIGBREAK handler, not native CTRL_C/CTRL_BREAK group delivery.

A separate scenario abruptly terminates the launcher, waits for every recorded identity to
disappear and asserts that no temporary process artifact remains. Unit coverage additionally checks
`request_stop()` launch suppression, secondary-thread signal policy, exact-parent mismatch,
`None`-versus-empty CUDA semantics, strict time/resource validation and bounded cleanup. Emergency
cleanup prevents a failed assertion from leaving workers behind.

The opt-in self-hosted CUDA workflow additionally runs one real GPU epoch through
`ExperimentRunner` and the public YAML contract, synchronises the device, checks the generated
metrics/environment artifacts and records CUDA hardware evidence. Its existence is not a CUDA
guarantee: the workflow must complete successfully on a compatible runner. Native Windows
console-group control delivery, multi-GPU/DDP, machine/container death, same-instance re-entrancy and
detached external daemons remain outside this automated contract. See
[Continuous integration](../../../README.md#continuous-integration) for the complete CI contract.

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
    model_input_keys: {left: left_features, right: right_features}
    optimizer_group_kwargs: {backbone: {lr: 0.0001}, head: {lr: 0.001}}
    logging: {loss_prog_bar: false, metric_prog_bar: true, logger: true}
data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train}
trainer:
  logger:
    - target: my_project.logging.ProjectLogger
      params: {project: demo}
    - plugin: {kind: logger, name: jsonl_logger}
      params: {path: metrics.jsonl}
  write_epoch_metrics_csv: true
  checkpoint_monitor: val_selection_score
  checkpoint_mode: max
callbacks:
  - plugin: {kind: callback, name: artifact_marker}
    params: {filename: finished.txt}
```

Models may be ordinary `torch.nn.Module` objects. Losses and metrics subclass the LambdaForge
contracts so mixed-precision and DDP behaviour stay explicit. Nested `target`/`ref` specifications
also work in parameters. The automated suite runs this same pattern end to end with external-style
model, loss, metric, dataset, logger and callback classes.

- Subclass or replace `LightningTask` for non-mapping batches, multiple optimizers or manual
  optimisation. Ordinary multi-input models are already supported by `model_input_keys`.
- Supply a custom Lightning data module when dataset preparation/splitting belongs in the run.
- Supply reusable datasets, callbacks and loggers through installed plugins or local YAML `target`
  specifications. Plugin authors should inherit callback/logger bases from
  `lambdaforge.integrations.Lightning`; see the [plugin guide](../plugins/README.md).
- Select the built-in optional provider targets from `lambdaforge.tracking` when MLflow,
  TensorBoard or W&B is sufficient; use a project target/plugin for a different logger contract.
- Replace `runner.target` when a different training backend is required; the built-in experiment
  runner expects compatible `fit` and `test` methods.
- Keep process-sensitive changes covered by spawned-process tests and validate interruption on the
  target operating system and GPU host.
