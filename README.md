# LambdaForge

[Español](README.es.md) · English

LambdaForge is SimpleLambda's object-oriented framework for reproducible machine-learning
training. It combines PyTorch, Lightning and a YAML experiment engine behind one stable Python
package, so a research project can focus on its dataset and task instead of rebuilding training
loops, metric logging, seed sweeps, checkpoint loading, plots and multi-GPU process scheduling.

> **Status:** `0.1.0`, usable but pre-1.0. The public namespaces documented below are the intended
> API; compatibility is not yet guaranteed between minor releases. The repository does not yet
> contain a licence file, so redistribution terms still need to be chosen by SimpleLambda.

## Contents

- [What LambdaForge provides](#what-lambdaforge-provides)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Public API](#public-api)
- [Architecture](#architecture)
- [YAML experiment reference](#yaml-experiment-reference)
- [Execution and process safety](#execution-and-process-safety)
- [Outputs, resume and loading](#outputs-resume-and-loading)
- [Built-in components](#built-in-components)
- [Extension contracts](#extension-contracts)
- [Review findings](#review-findings)
- [Development and verification](#development-and-verification)
- [Current limitations](#current-limitations)
- [Documentation map](#documentation-map)
- [Proposed roadmap](#proposed-roadmap-not-implemented)

## What LambdaForge provides

- A generic Lightning training task for mapping-shaped batches, one or more losses and independent
  train/validation/test metrics.
- Object construction from trusted YAML using fully qualified `target` and `ref` paths.
- Cartesian parameter grids, named ablations and repeatable seed expansion.
- Sequential runs, multiple independent trainings per GPU, multi-GPU scheduling and one DDP job
  per device group.
- Cooperative cancellation, forced process-tree cleanup, Windows Job Objects, Linux parent-death
  guards and guarded PyTorch `DataLoader` workers.
- Materialised configurations, logs, dense epoch CSV files, checkpoints, result manifests,
  cross-seed summaries, statistical comparisons and plots.
- Reusable MLP, 2D CNN, ECMP graph model, activations, normalisations, pooling operators, distances,
  binary/multiclass metrics and regression metrics.
- A small facade (`LambdaForge`), an object API (`Experiment`) and a CLI (`lambdaforge`).

LambdaForge is task-agnostic at the configuration and orchestration layers. A user project supplies
the domain-specific `Dataset`, optional collator and, when the default mapping contract is not
enough, its own model, task, data module or runner.

## Installation

Python 3.10 or newer is required. From a clone, create an environment and install the project in
editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS activation is `source .venv/bin/activate`. Runtime dependencies include PyTorch,
Lightning, TorchMetrics, NumPy, Matplotlib, PyYAML, psutil and threadpoolctl. `pywin32` is installed
only on Windows. CUDA itself must match the installed PyTorch build; LambdaForge does not install a
CUDA driver.

The package prefers `lightning.pytorch` and retains runtime compatibility with the legacy
`pytorch_lightning` import name.

## Quick start

Copy [the complete example](examples/experiment.yaml), replace the three `your_project.*` paths,
then inspect the expanded suite before starting any process:

```powershell
lambdaforge inspect examples\experiment.yaml
lambdaforge run examples\experiment.yaml --dry-run
lambdaforge run examples\experiment.yaml
```

The equivalent Python API is:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("examples/experiment.yaml")
expanded_runs = experiment.expand()
results = experiment.run()
```

Rebuild aggregate files without retraining:

```powershell
lambdaforge aggregate examples\experiment.yaml
```

CLI execution flags override only the corresponding YAML resource fields:

```powershell
lambdaforge run experiment.yaml --mode parallel --gpus 0,1 --jobs-per-gpu 2
lambdaforge run experiment.yaml --mode ddp --gpus 0,1 --devices-per-job 2
```

## Public API

The supported entry points are deliberately narrow:

| Entry point | Purpose |
|---|---|
| `from lambdaforge import LambdaForge` | Load, run or construct objects through the facade. |
| `from lambdaforge import Experiment` | Inspect, execute, aggregate and load one experiment suite. |
| `lambdaforge.nn` | Models and the YAML component registry. |
| `lambdaforge.metrics` | Base metric contract and built-in metrics. |
| `lambdaforge.training` | Task, runner, configuration and process orchestration. |
| `lambdaforge.experiments` | Lower-level configuration, scheduling, aggregation and loading. |
| `python -m lambdaforge` / `lambdaforge` | CLI front end to the same object API. |

`LambdaForge.build(spec)` exposes the generic object factory:

```python
model = LambdaForge.build({
    "target": "lambdaforge.nn.models.MLP",
    "params": {"in_features": 32, "out_features": 1, "hidden": [64, 32]},
})
```

Import from these namespaces instead of relying on file locations. Internal modules may move while
the public imports remain stable.

## Architecture

```text
LambdaForge/
├── examples/                     # runnable configuration templates
├── src/lambdaforge/
│   ├── LambdaForge.py            # single discoverable facade
│   ├── cli/                      # command-line object
│   ├── experiments/              # YAML, sweeps, execution, artifacts, aggregation
│   ├── integrations/             # third-party compatibility adapters
│   ├── metrics/                  # metric contracts; binary/multiclass/regression families
│   ├── nn/                       # models, losses and neural components
│   └── training/                 # Lightning core plus callbacks/data/orchestration
├── tests/                        # unit, process and real training smoke tests
└── pyproject.toml                # package and tool configuration
```

The implementation follows the project's Java-influenced object philosophy:

- each implementation `.py` contains one class;
- reusable behaviour lives on objects, class methods or static methods rather than module-level
  utility functions;
- `__init__.py` and `__main__.py` are packaging entry points and are the intentional exceptions;
- enums replace closed sets of internal magic strings;
- YAML keys and fully qualified import paths remain strings because they are external protocol
  boundaries;
- responsibilities stay separated even when that means several small files.

PEP 8 normally favours short lowercase module names; matching class and module names is therefore an
intentional LambdaForge convention, enforced for consistency rather than presented as universal
Python style. Stable lowercase package namespaces and public re-exports keep that choice out of most
consumer imports.

Subpackages are introduced for stable conceptual boundaries, not merely after an arbitrary file
count. Classification is divided into `binary` and `multiclass`, while training separates
`callbacks`, `data` and `orchestration`. `nn.pooling` remains flat despite its size because all of its
classes implement one closely related contract; splitting it into tiny technique-based folders would
make comparison and discovery harder. Public imports are re-exported from `__init__.py`, so these
physical decisions do not leak into normal user code.

The previous root-level `models`, `metrics`, `distances`, `training` and `experiments` trees are now
one installable `src/lambdaforge` package. Imports no longer depend on an unrelated package called
`core` being present on `sys.path`.

## YAML experiment reference

The example file is the canonical template. Top-level blocks are:

| Block | Required | Meaning |
|---|---:|---|
| `experiment` | yes | Name, output path, seeds, resume and completion policy. |
| `data` | yes | Train/validation/test objects and data-module configuration. |
| `model` | yes | Model object specification. |
| `losses` | yes | One or more loss objects. |
| `metrics` | no | Backwards-compatible metrics shared by stages unless overridden. |
| `train_metrics`, `val_metrics`, `test_metrics` | no | Explicit split-specific metric lists. |
| `optimizer` | no | Optimizer reference and parameters; defaults to AdamW. |
| `scheduler` | no | Scheduler reference, parameters and optional Lightning metadata. |
| `task` | no | Custom task target or default `LightningTask` parameters. |
| `trainer` | no | `LightningTrainConfig` fields and advanced `trainer_kwargs`. |
| `runner` | no | Custom runner target, callbacks or runner parameters. |
| `callbacks` | no | Additional callback objects constructed from YAML. |
| `sweep` | no | Base inclusion, Cartesian grid and named ablations. |
| `execution` | no | Sequential/parallel/DDP resources. |

### Object specifications

`target` imports a callable, recursively builds its parameters and calls it:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    activation: gelu
```

`ref` imports an object without calling it, which is useful for optimizers and collators:

```yaml
optimizer:
  ref: torch.optim.AdamW
  params: {lr: 0.001}
```

Object specifications can be nested inside dictionaries and lists. They execute Python imports and
therefore YAML files must be trusted; this is configuration, not a sandbox.

### Experiment and sweep

```yaml
experiment:
  name: study_name
  output_root: runs/experiments
  seeds: [7, 17, 27]
  resume: true
  rerun_completed: false
  test_after_fit: true
  required_artifacts: [predictions.csv]

sweep:
  include_base: true
  grid:
    model.params.hidden: [[128, 64], [256, 128]]
    optimizer.params.lr: [0.001, 0.0003]
  ablations:
    - name: no_dropout
      set: {model.params.dropout: 0.0}
```

Grid keys are dotted paths and their Cartesian product is materialised for every seed. Ablations are
additional named configurations. Empty grid value lists and duplicate `(variant, seed)` runs are
rejected before training.

With `resume: true`, an incomplete run may resume from its last checkpoint. With
`rerun_completed: false`, a successful run is skipped only if `result.json`, the selected checkpoint
and every relative `required_artifacts` path exist. Required artifacts are generic paths; the
framework contains no domain-specific prediction-file assumption.

### Trainer and data-loader escape hatches

Common Trainer fields are explicit and validated. Future or uncommon Lightning options go under
`trainer_kwargs`:

```yaml
trainer:
  max_epochs: 100
  precision: bf16-mixed
  checkpoint_policy: last_and_best   # none, last, best, last_and_best, all
  checkpoint_monitor: val_auroc
  checkpoint_mode: max
  logger: csv                        # none, csv, lightning_csv, or object spec
  write_epoch_metrics_csv: true      # canonical input for reports/aggregation
  epoch_metrics_include: ["train_*", "val_*", "epoch_time_s"]
  epoch_console_exclude: ["*_loss_binary_cross_entropy_with_logits"]
  trainer_kwargs:
    limit_train_batches: 1.0
    enable_model_summary: true
```

`trainer.logger` may itself be a nested `target` specification for any compatible Lightning logger.
The same applies to top-level `callbacks`. `write_epoch_metrics_csv` keeps LambdaForge's canonical
dense artifact independently of that external logger; disable it only when aggregation is not
needed. The CSV and console table accept shell-style include/exclude patterns, while
`checkpoint_monitor`, `early_stopping_monitor` and their explicit `min`/`max` modes remove dependence
on metric-list ordering.

Task-level publication is independently configurable:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_key: x
    logging:
      log_total_loss: true
      log_individual_losses: true
      loss_prog_bar: false
      metric_prog_bar: true
      logger: false                    # true with an external/Lightning logger
```

The metric lists determine which values are computed. `TaskLoggingConfig` determines which loss
values are published and what reaches the progress bar/logger. `MetricAlias` gives two instances of
the same metric type distinct log names, for example accuracy at two thresholds.

Likewise `LightningDataModule` accepts shared and split-specific `dataloader_kwargs`. Keys controlled
by LambdaForge (`dataset`, `shuffle`, `worker_init_fn`, and the explicit constructor fields) cannot
be overridden through these dictionaries, preventing contradictory configuration.

## Execution and process safety

| Mode | Scheduling |
|---|---|
| `sequential` | Runs execute in the caller, one after another. GPU slot fields are ignored. |
| `parallel` | Each run is a spawned process using one logical GPU; `jobs_per_gpu` permits concurrent independent jobs per GPU. |
| `ddp` | Each run receives a group of `devices_per_job` logical GPUs and Lightning runs DDP inside that process. |

GPU numbers are logical positions relative to the parent's `CUDA_VISIBLE_DEVICES`. If the parent has
`CUDA_VISIBLE_DEVICES=4,7`, a job requesting `[1]` sees physical GPU `7` as its local GPU `0`.
LambdaForge never rewrites the parent variable.

CPU thread, inter-op thread, affinity and data-loader worker limits are applied per job. Use
conservative values when several trainings share a GPU: process concurrency multiplies CPU and RAM
usage even if GPU memory fits.

Process cleanup is layered:

1. SIGINT/SIGTERM sets a shared stop event.
2. Lightning callbacks stop at a batch boundary.
3. The orchestrator waits for the configured grace period.
4. Remaining descendant trees are terminated recursively with psutil.
5. Windows jobs are placed in a kill-on-close Job Object; Linux workers install a parent-death
   signal where the platform supports it.
6. `DataLoader` workers install the same guard and thread limits.

This protects framework-created processes and descendants that remain in the same OS process tree.
No library can guarantee cleanup of a third-party program that deliberately detaches itself into an
unrelated service, so external launchers still need their own lifecycle contract.

DDP metrics synchronise their accumulated state before computing non-linear values such as AUROC,
F1 and correlations. Custom metrics used with DDP must implement the distributed-state contract;
LambdaForge raises an error instead of silently averaging invalid per-rank scalars.

## Outputs, resume and loading

Each concrete run has its own directory beneath `<output_root>/<experiment>/<variant>/<seed>/` and
can contain:

- `config.yaml`: fully materialised configuration;
- `hparams.json`: compact hyperparameter summary;
- `train.log`: captured stdout/stderr;
- `metrics.csv`: one dense row per epoch;
- `checkpoints/`: files governed by `checkpoint_policy`;
- `result.json`: terminal status, paths, duration and best/final metrics;
- project-defined required artifacts.

The suite aggregate area contains per-variant epoch CSVs, seed statistics, pairwise comparisons,
Benjamini-Hochberg directional q-values and optional PNG plots. Aggregation can be regenerated from
disk and does not require model reconstruction.

Load a checkpointed model through the suite:

```python
experiment = LambdaForge.experiment("experiment.yaml")
model = experiment.load_model(seed=17, variant="base", which="best")
```

`which` accepts `best`, `last` or `auto`. Loading validates checkpoint presence and understands both
bare model state dictionaries and Lightning keys prefixed with `model.`. A model class still needs
to be importable from the materialised configuration.

## Built-in components

- Models: `Model`, `MLP`, `CNN2D`, `BatchedKNN`, `ECMP`.
- Activations: ELU, GELU, Identity, LeakyReLU, ReLU, Sigmoid, SiLU and Tanh.
- Normalisations: BatchNorm (1D/2D/3D), IdentityNorm, LayerNorm and RMSNorm.
- Pooling: attention, gated/multi-head gated attention, auto-pool, fractional/top-k, log-sum-exp,
  max, mean, min, moments, noisy-or, probability GeM, softmax and sum.
- Distances: Euclidean and squared Euclidean.
- Losses: weighted base contract and binary cross-entropy with logits.
- Binary metrics: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, Cohen's
  kappa, AUROC and AUPRC.
- Multiclass metrics: accuracy, balanced accuracy, F1, AUROC and AUPRC.
- Regression metrics: MAE, MSE, RMSE, R², Pearson and Spearman correlation, plus `MeanMetric`.

Short activation and normalisation names are case-insensitive in YAML. Custom aliases can be added
through `ComponentRegistry`; Python model constructors also accept compatible classes directly.
See the [neural-components guide](src/lambdaforge/nn/README.md) and
[metrics guide](src/lambdaforge/metrics/README.md) for contracts and shapes.

## Extension contracts

### Model

Subclass `torch.nn.Module` or `lambdaforge.nn.models.Model`. Implement `forward(*args, **kwargs)`.
The base `predict` helper switches to evaluation/inference mode and restores the prior training
state. For the default task, either return a tensor (wrapped under `model_output_key`) or a mapping.

### Loss

Subclass `Loss` and implement `forward(outputs, batch) -> Tensor`. Give every loss a stable `name`
and use mapping keys rather than task assumptions. Multiple losses are summed after their configured
weights are applied.

### Metric

Implement `update`, `compute` and `reset`. For DDP, also expose `distributed_state` and
`merge_distributed_state`, or deliberately use a framework metric. Metric instances are deep-copied
per stage so train, validation and test state never leaks.

Every stage requires unique metric names. Wrap a metric in `MetricAlias` when the same class is used
more than once with different parameters. Explicit stage lists avoid deepcopy requirements for a
project metric that owns a non-copyable external resource.

### Data and task

The default `LightningTask` expects mapping batches. `model_input_key` selects the model input and
loss/metric objects select their own target/output keys. For tuples, multiple model inputs,
generative tasks or unusual optimizer flows, provide a custom `task.target`; all other experiment,
process and artifact machinery remains reusable.

### Runner

A custom runner must provide compatible `fit` and `test` methods. Configure it through
`runner.target`; its parameters are recursively object-built. Extra callbacks can therefore also be
declared as YAML object specifications.

All these extension objects are recursively constructed by `ObjectFactory`: models may be any
`torch.nn.Module`; losses subclass `Loss`; metrics subclass `Metric`; loggers and callbacks implement
their Lightning contracts. Custom batch structures, multiple optimizers or a different backend are
handled by replacing `task.target` or `runner.target`, without changing the experiment engine.

## Review findings

The repository-wide review found and resolved the following structural or correctness risks:

| Finding in the previous layout | Resolution |
|---|---|
| Root folders were not an installable package and imports referenced external `core`/stale `coreold` namespaces. | Added `pyproject.toml`, a `src/lambdaforge` package and absolute self-contained imports. |
| Models, distances, experiments, metrics and training competed as root-level entry points. | Introduced the `LambdaForge`/`Experiment` API and four cohesive public subpackages. |
| Large modules contained many loose functions or several unrelated classes. | Converted behaviour to collaborator objects and split every implementation class into its own file. |
| `training` and `metrics.classification` had become visually dense. | Split only along stable contracts (`callbacks`, `orchestration`, `binary`, `multiclass`) while preserving public imports; cohesive families such as `pooling` remain flat. |
| Closed choices and component names relied on repeated string literals. | Added enums and `ComponentRegistry`; strings remain only at YAML/serialization boundaries. |
| Advanced Trainer/DataLoader settings required source edits. | Added validated `trainer_kwargs` and shared/per-split `dataloader_kwargs` forwarding. |
| One shared metric list and implicit monitor order limited experiment control. | Added per-stage metric lists, aliases, explicit monitor modes, loss publication policy and CSV/console filters. |
| Selecting a custom Lightning logger removed the canonical epoch CSV. | Separated external logger selection from `write_epoch_metrics_csv`, so reporting remains available by default. |
| DDP could average already-computed AUROC/F1/correlation scalars, which is mathematically invalid. | Metrics now gather and merge their state before computing; unsupported custom metrics fail explicitly. |
| `CNN2D` selected 1D BatchNorm by default for NCHW tensors. | Its built-in default now creates `BatchNorm2d`. |
| `Model.predict` did not guarantee restoration of the previous training mode. | Inference is guarded by `try/finally` and restores the original state. |
| `test_after_fit` asked a new Trainer for an unknown `best` checkpoint and omitted its stop event. | It now uses the actual fitted checkpoint when available, otherwise current weights, and preserves cancellation. |
| Forced process termination focused on root workers and was fragile on Windows. | Added recursive psutil cleanup, Job Objects, parent-death guards and guarded data workers. |
| Completion assumed a domain-specific prediction artifact. | Replaced it with generic relative `required_artifacts`. |
| Documentation described obsolete modules/scripts and there was no root guide. | Replaced it with linked English/Spanish guides whose claims are checked against current code. |

The principal unresolved product gap is dataset memory/caching rather than training execution. It is
called out explicitly below and proposed as roadmap work instead of being represented as finished.

## Development and verification

```powershell
ruff format --check src tests
ruff check src tests
mypy src\lambdaforge
pytest -q
```

The current suite covers configuration expansion, object construction, model validation, metrics,
aggregation, spawned process scheduling, the structural POO rules, a real one-epoch Lightning CPU
fit and external model/loss/metric/logger/callback construction from YAML. Tests intentionally avoid
claiming CUDA coverage on a machine where CUDA was not exercised. Changes to scheduling or cleanup
should additionally be validated on the target multi-GPU host and interrupted manually at least
once.

Every source module and class has a docstring. The repository audit also checks class/module name
matching, the one-class-per-file rule and absence of module-level helper functions in implementation
modules.

## Current limitations

- LambdaForge wraps PyTorch datasets and data loaders, but it does **not** yet implement dataset
  caching, memory mapping, streaming or a RAM budget. Dataset storage and sample loading remain the
  user project's responsibility.
- Lightning is the only built-in training backend.
- The default task assumes mapping-shaped supervised batches; other tasks need a custom task object.
- YAML validation is structural and constructor-driven, not yet backed by a published JSON Schema.
- Curve metrics accumulate predictions on CPU, which may consume substantial RAM on very large
  validation sets.
- Statistical summaries are useful exploratory tools, not a substitute for a study-specific
  statistical protocol. Some confidence intervals use normal approximations.
- No remote experiment tracker, distributed job scheduler, hyperparameter optimiser or artifact
  store is bundled.
- Windows and CPU process behaviour is covered locally; real multi-GPU and abrupt-interruption tests
  remain environment-dependent.

## Documentation map

- [Experiment system](src/lambdaforge/experiments/README.md) · [Español](src/lambdaforge/experiments/README.es.md)
- [Training and processes](src/lambdaforge/training/README.md) · [Español](src/lambdaforge/training/README.es.md)
- [Neural components](src/lambdaforge/nn/README.md) · [Español](src/lambdaforge/nn/README.es.md)
- [Metrics](src/lambdaforge/metrics/README.md) · [Español](src/lambdaforge/metrics/README.es.md)
- [Complete YAML example](examples/experiment.yaml)

Each sub-guide links back here and to its translation. Class docstrings are the most precise source
for individual constructor arguments.

## Proposed roadmap (not implemented)

The following refinements have high value relative to their likely implementation cost. They are
proposals only; none is presented as current functionality.

1. **Published JSON Schema and `lambdaforge validate`** (small): catch unknown keys, import failures,
   invalid dotted sweep paths and resource contradictions without materialising run directories.
2. **Environment manifest** (small): save Python/platform, package versions, CUDA/cuDNN, GPU names and
   optional Git commit/diff metadata beside every result.
3. **`DatasetCache` object** (medium): optional bounded RAM LRU plus disk/memory-map adapters, with
   explicit cache keys and hit/miss statistics. This would close the largest gap between the stated
   vision and current code.
4. **Entry-point plugin discovery** (small/medium): let external packages register models, metrics and
   aliases without editing LambdaForge.
5. **Streaming curve metrics** (medium): histogram/quantile approximations for AUROC/AUPRC to cap RAM.
6. **Typed result/manifest objects** (small): replace internal free-form result dictionaries while
   keeping JSON compatibility.
7. **Stronger comparison methods** (small): bootstrap confidence intervals and paired Wilcoxon tests,
   selected explicitly in YAML.
8. **CI matrix and interruption tests** (small/medium): Python 3.10–3.13, Windows/Linux, modern and
   legacy Lightning imports, plus child/grandchild termination tests.
9. **Optional tracking adapters** (medium): objects for MLflow, TensorBoard or Weights & Biases behind
   the existing logger/runner extension boundary, without making any service mandatory.
