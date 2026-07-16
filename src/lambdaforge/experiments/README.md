# LambdaForge experiment system

[Repository guide](../../../README.md) · [Español](README.es.md)

This package turns one trusted YAML document into reproducible variant/seed runs, schedules them and
reduces their on-disk results. It contains no task-specific model or dataset logic.

## Contents

- [Main objects](#main-objects)
- [Lifecycle](#lifecycle)
- [Expansion rules](#expansion-rules)
- [Execution](#execution)
- [Completion and resume](#completion-and-resume)
- [Artifacts and aggregation](#artifacts-and-aggregation)
- [Loading](#loading)
- [Extension boundaries](#extension-boundaries)

## Main objects

| Object | Responsibility |
|---|---|
| `Experiment` | Public handle for expand, run, aggregate and load operations. |
| `ExperimentConfig` | Own YAML loading, dotted paths, validation and suite expansion. |
| `ObjectFactory` | Recursively resolve `target` and `ref` object specifications. |
| `ExecutionConfig` | Validate resource settings and create logical GPU slots. |
| `ExperimentExecutor` | Select sequential, parallel or DDP execution. |
| `ExperimentRunner` | Materialise and run one configuration and write its result. |
| `ExperimentAggregator` | Read completed runs and create statistics, CSVs and plots. |
| `RunLoader` | Find run directories and reconstruct checkpointed models. |

Supporting classes such as `ExperimentWorker`, `StdIOCapture`, `TeeStream`, `CheckpointChoice` and
the status enums each live in their own module.

## Lifecycle

```text
YAML → ExperimentConfig → expanded variant/seed configs
     → ExecutionConfig → process/device slots
     → ExperimentRunner → config, log, metrics, checkpoints, result
     → ExperimentAggregator → cross-seed tables and plots
     → RunLoader → reconstructed model
```

Use the high-level object unless writing an integration:

```python
from lambdaforge import Experiment

experiment = Experiment.from_yaml("experiment.yaml")
for run in experiment.expand():
    print(run["experiment"]["variant"], run["experiment"]["seed"])
results = experiment.run(dry_run=True)
```

## Expansion rules

`experiment.seeds` accepts a scalar or list. `sweep.grid` maps dotted paths to non-empty value lists;
all grid dimensions form a Cartesian product. `sweep.include_base` controls whether the unmodified
configuration is included. Every `sweep.ablations` entry adds a named set of dotted-path overrides.

Expansion uses deep copies: one run cannot mutate another. Experiment names must be non-empty and
the final `(variant, seed)` identities must be unique. `lambdaforge inspect` prints the concrete
configurations without executing imported objects.

## Execution

`sequential` stays in the caller. `parallel` schedules one spawned process per independent run on
fixed one-GPU slots. `ddp` schedules a run on each group of `devices_per_job` GPUs and patches its
Lightning strategy to DDP. CLI overrides have precedence over YAML, which has precedence over
defaults.

The executor deliberately uses pickle-safe worker objects and the `spawn` start method. GPU indices
are logical relative to the parent `CUDA_VISIBLE_DEVICES`; CPU/thread/worker limits are patched into
each run rather than mutating the parent environment.

See the [training process guide](../training/README.md) for shutdown guarantees and caveats.

## Completion and resume

Machine-readable run states are `ok`, `failed`, `dry_run`, `interrupted` and `unknown`.

A run is complete only when:

1. `result.json` has status `ok`;
2. the result's selected checkpoint exists when checkpointing is required; and
3. every path in `experiment.required_artifacts` exists inside the run directory.

With `rerun_completed: false`, complete runs are skipped. With `resume: true`, incomplete runs use a
usable last checkpoint when one exists. Failure handling writes a terminal failure result and the
suite can be launched again without discarding successful seeds. Required artifacts must be relative
and are entirely project-defined.

## Artifacts and aggregation

The run directory contains the materialised `config.yaml`, `hparams.json`, captured `train.log`,
`metrics.csv`, checkpoints, `result.json` and custom artifacts. Exact paths are derived from
experiment name, variant slug and seed.

`ExperimentAggregator.write` can reconstruct suite reports from disk. It emits per-variant terminal
and epoch summaries, wide/long CSV representations, seed statistics, pairwise directional tests,
Benjamini-Hochberg q-values and optional plots. Plot failure is recorded without losing numeric
tables. `lambdaforge aggregate --no-plots` is suitable for headless minimal environments.

The aggregate statistics are exploratory. The code reports sample sizes and incomplete variants so
missing seeds are visible; study-specific inference decisions remain the researcher's responsibility.

## Loading

```python
experiment = Experiment.from_yaml("experiment.yaml")
model = experiment.load_model(seed=7, variant="base", which="auto")
```

`CheckpointChoice` offers `best`, `last` and `auto`. `RunLoader` validates the run, imports the model
from its materialised object specification, loads a direct state mapping or strips Lightning's
`model.` prefix, then returns the model in evaluation mode.

## Extension boundaries

- Configure a custom `data.datamodule.target`, `task.target` or `runner.target` rather than forking
  the experiment engine.
- Models, losses and `train_metrics`, `val_metrics` and `test_metrics` accept recursively built
  `target`/`ref` specifications. The backward-compatible `metrics` key supplies all unspecified
  stages.
- Top-level `callbacks` and nested loggers, collators and other objects use the same object syntax.
- Checkpoint and early-stopping monitors and their `min`/`max` modes are explicit trainer settings;
  when omitted, the first validation metric and its declared direction are used.
- A custom runner must preserve compatible `fit` and `test` methods if it is used by
  `ExperimentRunner`.
- Treat YAML as trusted code because imported targets can execute arbitrary Python.
- Import public classes from `lambdaforge.experiments`; file names are implementation details.

The lifecycle classes remain together deliberately: unlike metrics and callbacks, their contracts
are tightly coupled and splitting them would create several tiny packages without an independent
public purpose. Revisit that boundary when a genuinely separate backend or storage family appears.
