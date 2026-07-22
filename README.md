<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" type="image/svg+xml" srcset="icons/lambdaforge-light.svg">
    <source media="(prefers-color-scheme: dark)" type="image/png" srcset="icons/lambdaforge-light.png">
    <source media="(prefers-color-scheme: light)" type="image/svg+xml" srcset="icons/lambdaforge-dark.svg">
    <source media="(prefers-color-scheme: light)" type="image/png" srcset="icons/lambdaforge-dark.png">
    <img src="icons/lambdaforge-dark.png" width="140" alt="LambdaForge logo">
  </picture>
</p>

# LambdaForge

[Español](README.es.md) · English

LambdaForge is SimpleLambda's object-oriented framework for reproducible machine-learning
training. It combines PyTorch, Lightning and a YAML experiment engine behind one stable Python
package, so a research project can focus on its dataset and task instead of rebuilding training
loops, metric logging, seed sweeps, checkpoint loading, plots and multi-GPU process scheduling.

> **Status:** `0.2.0`, usable but pre-1.0. The public namespaces documented below are the intended
> API; compatibility is not yet guaranteed between minor releases. The repository does not yet
> contain a licence file, so redistribution terms still need to be chosen by SimpleLambda.

## Contents

- [What LambdaForge provides](#what-lambdaforge-provides)
- [Installation](#installation)
- [Integrating into another project](#integrating-into-another-project)
- [Why AGENTS.md exists](#why-agentsmd-exists)
- [Quick start](#quick-start)
- [Public API](#public-api)
- [Architecture](#architecture)
- [YAML experiment reference](#yaml-experiment-reference)
- [Configuration migrations](#configuration-migrations)
- [Execution and process safety](#execution-and-process-safety)
- [Outputs, resume and loading](#outputs-resume-and-loading)
- [Artifact retention](#artifact-retention)
- [Built-in components](#built-in-components)
  - [Advanced graph and equivariant models](#advanced-graph-and-equivariant-models)
- [Extension contracts](#extension-contracts)
- [Review findings](#review-findings)
- [Development and verification](#development-and-verification)
  - [Continuous integration](#continuous-integration)
- [Current limitations](#current-limitations)
- [Documentation map](#documentation-map)
- [Roadmap](#roadmap)

## What LambdaForge provides

- A generic Lightning training task for mapping-shaped batches, one or more losses and independent
  train/validation/test metrics.
- Object construction from trusted YAML using fully qualified `target`/`ref` paths or installed,
  contract-checked entry-point plugins, including reusable datasets, callbacks and loggers.
- Draft 2020-12 schema validation of structure, expansion, resources and imports before execution.
- Versioned, preview-first configuration migrations with round-trip YAML and explicit atomic output.
- Cartesian parameter grids, named ablations and repeatable seed expansion.
- Sequential runs, multiple independent trainings per GPU, multi-GPU scheduling and one DDP job
  per device group.
- Cooperative cancellation, forced process-tree cleanup, Windows Job Objects, Linux parent-death
  guards and guarded PyTorch `DataLoader` workers.
- Materialised configurations, environment and run-scoped loaded-plugin provenance, logs, dense
  epoch CSV files, checkpoints, result manifests, cross-seed summaries, statistical comparisons
  and plots.
- Preview-first artifact retention with role-aware checkpoint selection, verified streaming ZIPs,
  explicit pruning rules, completion receipts and crash-recoverable transactions.
- Reusable MLP, 2D CNN, ECMP graph model, activations, normalisations, pooling operators, distances,
  binary/multiclass metrics and regression metrics.
- Bounded process-local dataset caching, multiprocess-coordinated disk/mmap quotas, explicit
  dataset/transform fingerprints, checksum or HMAC records and a safe NumPy/Torch codec, all
  selectable through the same recursive YAML object syntax.
- Fixed-memory binary and multiclass AUROC/AUPRC alternatives whose histogram resolution, averaging
  and logits policy are explicit experiment parameters.
- Optional, lazily loaded MLflow, TensorBoard and Weights & Biases logger adapters, selectable alone
  or together without adding a tracking service to the base installation.
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
Lightning, TorchMetrics, NumPy, Matplotlib, PyYAML, ruamel.yaml, JSON Schema, psutil and
threadpoolctl. `pywin32` is installed only on Windows. CUDA itself must match the installed PyTorch
build; LambdaForge does not install a CUDA driver.

The package prefers `lightning.pytorch` and retains runtime compatibility with the legacy
`pytorch_lightning` import name.

Tracking providers remain optional. Install `lambdaforge[mlflow]`,
`lambdaforge[tensorboard]`, `lambdaforge[wandb]` or the combined `lambdaforge[tracking]` extra; see
the [tracking guide](src/lambdaforge/tracking/README.md) before enabling remote publication.

## Integrating into another project

LambdaForge is an installable library, not a source tree that must be copied into each study. Give
the consuming project its own virtual environment and install both projects into that environment.
For local framework development:

```bash
cd /path/to/my-research-project
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

The final `pip install -e .` is important: it makes paths such as
`my_project.models.ProjectModel` importable when LambdaForge resolves YAML. A normal consumer layout
is:

```text
my-research-project/
├── pyproject.toml
├── experiments/baseline.yaml
├── src/my_project/
│   ├── datasets.py
│   ├── losses.py
│   └── models.py
└── tests/
```

For a stable/offline installation, build a wheel in the LambdaForge checkout and install that
immutable artifact instead of an editable path:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-0.2.0-py3-none-any.whl
```

Let the consumer project's lock file or constraints select a PyTorch build compatible with its
driver before installing LambdaForge; the framework accepts an already installed compatible
`torch`. Verify the target environment rather than assuming that `nvidia-smi` implies a CUDA-enabled
Python wheel:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
lambdaforge validate experiments/baseline.yaml
lambdaforge run experiments/baseline.yaml --dry-run
```

Do not copy `src/lambdaforge`, share LambdaForge's `.venv`, or patch `PYTHONPATH`: those approaches
hide dependency and import errors and make a published experiment difficult to reproduce. For
several independent projects, use a wheel/version per environment. For reusable third-party
extensions, publish entry-point plugins; for one project, ordinary installed `my_project.*` targets
are simpler. The [extension contracts](#extension-contracts) show both routes.

## Why AGENTS.md exists

An AI coding agent should not need to read hundreds of implementation modules and every specialist
README before it can configure a model or add a loss. That approach consumes context and money,
increases the chance that an early constraint is forgotten, and encourages the agent to infer APIs
from internal files that are not stable.

[AGENTS.md](AGENTS.md) is therefore the framework's single, token-efficient operational manual. It
contains the complete capability map, the supported public boundaries, installation and YAML
workflow, extension recipes, result-publication rules, verification commands and a small routing
table for the rare case that deeper detail is needed. The intended agent workflow is:

1. Read `AGENTS.md` once.
2. Select an existing public object or extension contract from its catalogue.
3. Inspect only that object's signature/docstring or the one owner guide named by the routing table.
4. Validate and test through the documented public commands.

This does not replace the bilingual READMEs for humans or the precise class docstrings. It is a
compressed index and safety contract that prevents repository-wide context loading. Agents working
in this checkout discover it automatically; when LambdaForge is consumed from another workspace,
give the agent this file explicitly or reference its repository path from that project's own
`AGENTS.md`. Wheels also install the same source file under `share/lambdaforge/AGENTS.md`; obtain
its exact environment path without importing the framework with:

```bash
python -c "from importlib.metadata import distribution; print(distribution('lambdaforge').locate_file('share/lambdaforge/AGENTS.md'))"
```

## Quick start

Copy [the complete example](examples/experiment.yaml), replace its `your_project.*` paths,
then validate and inspect the expanded suite before starting any process:

```powershell
lambdaforge validate examples\experiment.yaml
lambdaforge inspect examples\experiment.yaml
lambdaforge run examples\experiment.yaml --dry-run
lambdaforge run examples\experiment.yaml
```

The equivalent Python API is:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("examples/experiment.yaml")
report = experiment.validate()
expanded_runs = experiment.expand()
results = experiment.run()
print(results[0].status, results[0]["status"])
```

Rebuild aggregate files without retraining:

```powershell
lambdaforge aggregate examples\experiment.yaml
lambdaforge retain examples\experiment.yaml          # read-only plan
lambdaforge results examples\experiment.yaml --write-index
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
| `from lambdaforge import RunResult, AggregateResult` | Typed immutable results with legacy dict/JSON compatibility. |
| `from lambdaforge import ResultCatalog, ResultRecord` | Identity-aware discovery and explicit selection of attempt history. |
| `from lambdaforge import ArtifactRetentionPlan, ArtifactRetentionResult` | Typed immutable retention previews and outcomes. |
| `lambdaforge.data` | Dataset adapters and bounded cache objects. |
| `lambdaforge.nn` | Models and the YAML component registry. |
| `lambdaforge.metrics` | Base metric contract and built-in metrics. |
| `lambdaforge.plugins` | Lazy discovery, run usage sessions, descriptors and resolution errors. |
| `lambdaforge.integrations` | Stable Lightning compatibility object for plugin authors. |
| `lambdaforge.tracking` | Lazy optional MLflow, TensorBoard and Weights & Biases logger adapters. |
| `lambdaforge.training` | Task, runner, configuration and process orchestration. |
| `lambdaforge.experiments` | Lower-level configuration, migrations, scheduling, aggregation and loading. |
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

`LambdaForge.validate(path)` and `Experiment.validate(check_imports=True)` return an immutable
`ValidationReport`. The equivalent CLI supports `--json` for automation and `--no-imports` for
template files whose external project is not installed yet. Import checking never instantiates
configured objects, but importing a Python module can execute its top-level code, so configurations
must still be trusted.

`LambdaForge.preview_migration(path)` returns an immutable, non-writing
`ExperimentConfigMigrationResult`. Migration objects, exact Schema versions, catalogs and preview
formats are public from `lambdaforge.experiments`.

`LambdaForge.preview_retention(path)` and `Experiment.preview_retention()` build a strictly
read-only plan. Mutation is always explicit through `LambdaForge.apply_retention(path)`,
`Experiment.apply_retention()` or the CLI command `lambdaforge retain ... --apply`; the configured
`retention.mode: apply` may also run only after a successful final aggregation has published a
current completion receipt.

Installed plugin metadata can be inspected without importing provider modules:

```powershell
lambdaforge plugins
lambdaforge plugins --kind metric --json
```

## Architecture

```text
LambdaForge/
├── .github/workflows/             # hosted CPU CI plus opt-in self-hosted CUDA
├── examples/                     # runnable configuration templates
├── src/lambdaforge/
│   ├── EnvironmentManifest.py     # typed run provenance
│   ├── LambdaForge.py            # single discoverable facade
│   ├── cli/                      # command-line object
│   ├── data/                     # safe adapters and bounded cache backends
│   ├── experiments/              # YAML, sweeps, execution, aggregation, retention
│   ├── integrations/             # third-party compatibility adapters
│   ├── metrics/                  # metric contracts; binary/multiclass/regression families
│   ├── nn/                       # models, losses and neural components
│   ├── plugins/                  # lazy installed-package extension discovery
│   ├── runtime/                  # shared cross-process filesystem locks
│   ├── schemas/                  # packaged experiment JSON Schema
│   ├── tracking/                 # optional provider logger adapters and dependency guards
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
| `schema_version` | yes | Exact quoted Schema compatibility version; current value is `"1.1"`. Historical 1.0 and unversioned files migrate forward before validation. |
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
| `aggregation` | no | Cross-seed confidence intervals, paired tests and reliability thresholds. |
| `retention` | no | Preview/apply policy for checkpoint roles, verified archives and explicit pruning rules. |

The packaged [JSON Schema](src/lambdaforge/schemas/experiment.schema.json) rejects unknown
framework-owned keys while `metadata` and `extensions` remain explicit task-defined escape hatches.
`lambdaforge validate` additionally checks sweep expansion, resource contradictions and every
`target`/`ref`/plugin import without constructing objects or creating output directories.

The canonical Schema requires `schema_version: "1.1"`. Schema 1.0 remains packaged for exact
historical validation. Files that omitted the field follow the deterministic
`unversioned -> 1.0 -> 1.1` chain; expanded and materialised configurations contain the current
version. `UnversionedToV1Migration` declares 1.0 and `ExperimentV1ToV1_1Migration` adds the
optional retention surface without enabling it.

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

`plugin` resolves a named class published by an installed distribution. The explicit `kind` avoids
context-dependent magic strings and lets LambdaForge validate the class contract before creating a
fresh instance:

```yaml
model:
  plugin: {kind: model, name: acme_encoder}
  params: {hidden_features: 128}

val_metrics:
  - plugin: {kind: metric, name: calibrated_auc}
    params: {pred_key: probabilities}

data:
  train:
    plugin: {kind: dataset, name: acme_records}
callbacks:
  - plugin: {kind: callback, name: artifact_marker}
trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
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

### Cross-seed statistical comparisons

Comparisons are paired by common seed. LambdaForge defines `delta = variant - baseline` and flips
its sign for metrics whose declared mode is `min`, so a positive `improvement` always favours the
variant. The complete nested YAML contract is:

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile  # normal or bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon              # sign or wilcoxon
      alternative: two_sided  # two_sided, greater, less, observed_direction
      calculation: auto             # auto, exact, asymptotic
      zero_method: wilcox            # wilcox, pratt, zsplit
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12             # null disables pre-ranking rounding
```

Omitting `aggregation` preserves the historical protocol: a 95% normal interval, the exact paired
sign test with `observed_direction`, `alpha: 0.05`, target power `0.80` and at least three pairs
before a verdict. Percentile bootstrap uses a deterministic stream derived from its base seed and
the comparison identity; batching bounds the transient resampling-index matrix while retaining
`O(resamples)` means for quantiles. Wilcoxon `auto` uses deterministic exact rank enumeration up to
`exact_max_pairs` and a normal approximation above it. Its zero conventions, alternatives,
calculation modes, artifact fields and Python objects are detailed in the
[statistical comparison guide](src/lambdaforge/experiments/statistics/README.md).

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

## Configuration migrations

Preview legacy normalisation before validation or execution:

```powershell
lambdaforge migrate legacy.yaml                 # unified diff, no writes
lambdaforge migrate legacy.yaml --format yaml   # complete resulting YAML
lambdaforge migrate legacy.yaml --format json   # stable result envelope
lambdaforge migrate legacy.yaml --check         # 1 if migration is required, otherwise 0
```

Persistence is always explicit and always targets a different path:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
lambdaforge validate experiment.v1_1.yaml
```

An existing destination requires `--force`; even then the source cannot be overwritten.
`--target-version` accepts an exact `MAJOR.MINOR` string and defaults to the current packaged
Schema. `--format` controls standard output, while `--output` always writes complete YAML
atomically. The default publication is no-clobber even with concurrent writers; `--force` switches
only a distinct destination to atomic replacement.

The current deterministic chain is `unversioned -> 1.0 -> 1.1`. The compatibility-only
`UnversionedToV1Migration` inserts the historical 1.0 declaration.
`ExperimentV1ToV1_1Migration` then advances that valid mapping to 1.1; because the new
`retention` block is optional and defaults to disabled, this step does not opt old experiments into
artifact mutation. Both exact Schemas remain packaged. The preview preserves comments, order,
quotes, anchors and newlines where the structural change permits, never imports user targets or
plugins, and never constructs experiment objects. Normal configuration loading applies the full
chain in memory without editing the source.

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.changed, preview.source_version, preview.target_version)
print(preview.render(MigrationPreviewFormat.DIFF))
```

See the [configuration migration guide](src/lambdaforge/experiments/migrations/README.md) for the
CLI exit-code contract, atomic-write guarantees, full object API, failure modes and the procedure
for adding a future Schema step.

## Execution and process safety

| Mode | Scheduling |
|---|---|
| `sequential` | Runs execute in the caller, one after another. GPU slot fields are ignored. |
| `parallel` | Each run is a spawned process using one logical GPU; `jobs_per_gpu` permits concurrent independent jobs per GPU. |
| `ddp` | Each run receives a group of `devices_per_job` logical GPUs and Lightning runs DDP inside that process. |

GPU numbers are logical positions relative to the parent's `CUDA_VISIBLE_DEVICES`. If the parent has
`CUDA_VISIBLE_DEVICES=4,7`, a job requesting `[1]` sees physical GPU `7` as its local GPU `0`.
The direct object API makes inheritance and CPU-only execution unambiguous:

| `TrainingJob.devices` / scheduler slot | Child visibility |
|---|---|
| `None` | Inherit the complete parent-visible CUDA set unchanged. |
| `[]` or `()` | Set `CUDA_VISIBLE_DEVICES=""` and hide CUDA explicitly. |
| `[0]`, `[1]`, ... | Restrict the child to those logical positions; without a parent variable they are treated as physical IDs. |

Device assignments are frozen and reject strings, booleans, negative/fractional indices and
duplicates. LambdaForge may stage a restricted value while spawning a child, but restores the
parent environment immediately and does not leave `CUDA_VISIBLE_DEVICES` altered.

`ExecutionConfig` applies the same strictness before scheduling: parallel/DDP require a non-empty
GPU list, DDP groups must divide it exactly, counts must be integral and finite, and
`grace_seconds` must be a finite non-negative real number. Booleans are never accepted as numbers;
optional CPU limits use `null` for inheritance, must otherwise be positive, and only
`dataloader_num_workers_per_job` permits zero.

CPU thread, inter-op thread, affinity and data-loader worker limits are applied per job. Use
conservative values when several trainings share a GPU: process concurrency multiplies CPU and RAM
usage even if GPU memory fits.

Process cleanup is layered:

1. Public `TrainingOrchestrator.request_stop()` or managed SIGINT/SIGTERM/SIGBREAK sets the shared
   stop event idempotently; SIGBREAK is installed only where the platform exposes it.
2. Lightning callbacks stop at a train/validation/test batch boundary.
3. The orchestrator waits for `grace_seconds` using shared monotonic deadlines rather than granting
   the full timeout once per process.
4. Remaining descendant trees are terminated recursively with psutil, then killed if necessary;
   any survivor after the bounded escalation is reported as a `RuntimeError`.
5. Each child receives the launcher's exact `expected_parent_pid`. Linux adds `prctl` parent-death
   delivery and POSIX workers also verify/watch that PID, closing the reparenting race.
6. Windows jobs use a kill-on-close Job Object; `DataLoader` workers install the same descendant
   guard and thread limits.

Signal management is explicit. The default `manage_signals=True` is valid only in Python's main
thread and restores the previous handlers after the run. An embedding that runs the orchestrator in
a secondary thread must use `manage_signals=False` and call `request_stop()` from its own lifecycle
hook. Disabling handler ownership does not disable cooperative cancellation.

Windows isolation does not degrade silently. If a native Job Object cannot be initialised,
LambdaForge emits a `RuntimeWarning`, records the detail in `process_isolation_warnings` and retains
portable recursive cleanup. If an active Job Object cannot accept a new worker, that worker is
terminated and the run raises instead of continuing with weaker isolation.

This protects framework-created processes and descendants that remain in the same OS process tree.
No library can guarantee cleanup of a third-party program that deliberately detaches itself into an
unrelated service, so external launchers still need their own lifecycle contract. A
`TrainingOrchestrator` is stateful and not re-entrant: do not overlap `run`/`run_scheduled` calls on
the same object, and prefer one instance per independently owned run.

DDP metrics synchronise their accumulated state before computing non-linear values such as AUROC,
F1 and correlations. Custom metrics used with DDP must implement the distributed-state contract;
LambdaForge raises an error instead of silently averaging invalid per-rank scalars.

## Outputs, resume and loading

Each concrete run has its own directory beneath `<output_root>/<experiment>/<variant>/<seed>/` and
can contain:

- `config.yaml`: fully materialised configuration;
- `environment.json`: UTC timestamp, Python/platform, core package versions, CUDA/cuDNN, visible GPU
  properties, selected CUDA variables, Git state and the distribution/version/group/value of every
  entry-point plugin successfully resolved by this run;
- `hparams.json`: compact hyperparameter summary;
- `train.log`: captured stdout/stderr;
- `metrics.csv`: one dense row per epoch;
- `checkpoints/`: files governed by `checkpoint_policy`;
- `result.json`: terminal status, attempt ID, scientific configuration fingerprint, UTC boundaries,
  paths, duration and best/final metrics;
- `.lambdaforge/attempts/result-*.json`: immutable terminal history retired before a retry;
- project-defined required artifacts.

The suite aggregate area contains per-variant epoch CSVs, seed statistics,
`baseline_comparisons.csv`, `reliability.json`, Benjamini-Hochberg q-values over the selected paired
test and optional PNG plots. The comparison CSV records the fully selected interval/test,
calculation status, effective sample sizes and bootstrap seed provenance. Historical
`ci95_improvement_*` and `p_value_sign_*` columns remain available beside the method-neutral fields.
Aggregation can be regenerated from disk and does not require model reconstruction.

Completion is identity-aware, not just path-aware. LambdaForge fingerprints the expanded
scientific configuration while excluding storage, retry, execution, aggregation and retention
controls. A changed model/data/loss/trainer configuration is therefore never skipped as an old
success and never resumes its incompatible checkpoint. Legacy results are matched against their
materialised `config.yaml` when possible and gain full identity metadata on archival.

Audit current and historical attempts before selecting values for a report or paper:

```bash
lambdaforge results experiment.yaml
lambdaforge results experiment.yaml --duplicates
lambdaforge results experiment.yaml --json --write-index
lambdaforge results experiment.yaml --fail-on-ambiguous
```

`--fail-on-ambiguous` returns 2 when one fingerprint has more than one successful attempt, making
the ambiguity enforceable in CI. `--write-index` atomically publishes
`.lambdaforge/result-index.json`; it is an index, not a second source of truth. Python exposes the
same fresh filesystem scan:

```python
records = experiment.results()                 # includes archived attempts
duplicates = experiment.result_catalog().duplicate_groups()
chosen = experiment.result_catalog().select(attempt_id="20260722T...")
```

Never choose an arbitrary “latest” directory for publication. Record the explicit `attempt_id`,
`config_fingerprint`, seed, variant, checkpoint role and aggregate artifact in the paper's evidence
manifest. Multiple successful attempts are retained and reported as ambiguous until the researcher
makes that selection; LambdaForge does not silently decide which favourable result is canonical.

Load a checkpointed model through the suite:

```python
experiment = LambdaForge.experiment("experiment.yaml")
model = experiment.load_model(seed=17, variant="base", which="best")
```

`which` accepts `best`, `last` or `auto`. `auto` resolves best, then last, then the latest safe
local checkpoint; exact `best` and `last` requests never silently cross roles. Loading validates
checkpoint presence and understands both bare model state dictionaries and Lightning keys prefixed
with `model.`. A model class still needs to be importable from the materialised configuration.

## Artifact retention

Schema 1.1 adds an optional, strict `retention` block. The safe starting point is preview mode:

```yaml
retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect: [reports/**, predictions/final.json]
  rules:
    - action: compress
      include: [artifacts/intermediate/**]
      exclude: []
      min_size_bytes: 1048576
      compression: {only_if_smaller: true}
    - action: prune
      include: [scratch/**]
      exclude: []
      min_size_bytes: 0
  archive: {name: artifacts.zip, compression_level: 6}
  lock_timeout_seconds: 60
```

Omitting this block is `disabled` and preserves every artifact. `preview` prints/plans but never
writes, archives or deletes; `apply` allows automatic retention only after a successful final
aggregation. The same boundary is explicit through the API and CLI:

```python
plan = experiment.preview_retention()
result = experiment.apply_retention()
```

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
```

Application requires a current `aggregate/aggregation_receipt.json` proving that every expected
variant/seed is terminal and successful and that committed run/aggregate fingerprints still match.
Core run files, required artifacts, protected globs, aggregate outputs, links/reparse points and
internal metadata cannot be selected by generic rules. Checkpoints use their own unambiguous
best/last policy. Compression streams into immutable per-run ZIPs, verifies member names, CRC,
sizes and SHA-256 before quarantine, and can retain incompressible sources. Pruning and compression
use a durable journal and reversible quarantine.

Training, final aggregation and retention coordinate with cross-process activity, aggregation and
retention locks in a fixed order. A crash before commit rolls back; a crash after the commit marker
finishes forward, and reapplying a committed plan is idempotent. See the
[artifact-retention guide](src/lambdaforge/experiments/retention/README.md) for the complete YAML
contract, eligibility receipt, transaction states, artifacts and limitations.

## Built-in components

- Models: configurable `MLP`/`CNN2D`; `ECMP`; GCN, GraphSAGE, GAT, GATv2, relational GCN, PNA,
  sparse GraphTransformer, EGNN and GIN stacks plus `GraphReadout`; `GradTree`, `GRANDE`,
  oblivious trees and `NODE`; RNN/LSTM/GRU,
  Transformer encoder/decoder/seq2seq, Conformer, state-space adapter and temporal convolution;
  `DeepSets`/`SetTransformer`; residual MLP, `FTTransformer`, TabNet, SAINT, AutoInt and DeepFM;
  `ResNet2D`/`ConvNeXt2D`/`MobileNetV2`, `VisionTransformer2D`, `UNet2D` and
  `FeaturePyramidNetwork2D`; autoencoders, ensembles, mixture-of-experts, multitask and siamese
  composition; VQ-VAE and Gaussian diffusion; Neural ODE/CDE, DeepONet, Fourier neural operator,
  scalar/vector tensor fields and optional higher-order equivariant adapters; and `SIREN` implicit
  representations.
- Activations: CELU, ELU, GELU, hard sigmoid/swish, Identity, LeakyReLU, Mish, PReLU, ReLU/ReLU6,
  SELU, Sigmoid, SiLU, Softplus, Softsign, SquarePlus, Tanh, Entmax15/Entmoid15, Sine/Snake and the
  dimension-changing GLU/GEGLU/SwiGLU/ReGLU family.
- Normalisations: BatchNorm and InstanceNorm (1D/2D/3D), ChannelLayerNorm, GroupNorm, IdentityNorm,
  L2Norm, LayerNorm, RMSNorm and ScaleNorm.
- Pooling: basic and smooth reductions, statistics/concatenation, learned attention, top-k and
  probability operators for dense masked sets, plus indexed sparse sum/mean/max/attention.
- Pairwise components: Euclidean, squared Euclidean, Manhattan, Minkowski, Chebyshev, cosine,
  angular and Mahalanobis distances; dot, cosine and bilinear similarities; RBF, Laplacian and
  polynomial kernels.
- Losses: binary/multiclass cross-entropy and focal objectives, MSE, MAE, Smooth L1, Huber,
  Dice/Tversky, contrastive, triplet-margin, InfoNCE and a reusable beta-VAE objective. Every
  built-in training loss reduces to a scalar and exposes mapping keys, weight and stable name.
- Encodings and regularization: sinusoidal, learned, rotary and Fourier features; DropPath,
  feature dropout and Gaussian noise.
- Data and uncertainty: `CategoricalFeatureEncoder`, `FileDataset`, `NumpyMemmapDataset`,
  `DatasetCache`, fingerprinted safe NumPy/Torch
  serialization, checksum/HMAC envelopes and coordinated disk/mmap backends. Pickle remains an
  explicit trusted-local compatibility option. `TemperatureScaler` and
  `ConformalPredictionInterval` provide held-out post-hoc uncertainty components.
- Conformance: `ArchitectureConformanceCase` and `ArchitectureConformancePack` capture provenance,
  initialization state, parameter count, output tensor and tolerances in tiny weights-only
  references, then fail on numerical/shape drift.
- Binary metrics: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, Cohen's
  kappa, exact AUROC/AUPRC and fixed-memory `StreamingBinaryAUROC`/`StreamingBinaryAUPRC`.
- Multiclass metrics: accuracy, balanced accuracy, F1, exact AUROC/AUPRC and fixed-memory
  `StreamingMulticlassAUROC`/`StreamingMulticlassAUPRC` with macro, weighted or micro reduction.
- Regression metrics: MAE, MSE, RMSE, R², Pearson and Spearman correlation, plus `MeanMetric`.

Short shape-preserving activation and normalisation names are case-insensitive in YAML. Custom
aliases can be added through `ComponentRegistry`; Python model constructors also accept compatible
classes directly. GLU-family activations remain explicit objects because they halve a dimension and
cannot safely replace an ordinary per-layer activation.
See the [neural-components guide](src/lambdaforge/nn/README.md) and
[metrics guide](src/lambdaforge/metrics/README.md) for contracts and shapes.

### Advanced graph and equivariant models

The native graph path uses ordinary PyTorch tensors and sparse directed edges; PyG/DGL and dense
adjacency are not required. For every `edge_index=(2,E)`, row zero is `source` and row one is
`destination`.

| Input | Contract |
|---|---|
| `x` | Floating node features `(N,in_channels)`. |
| `edge_index` | Integer `(2,E)` with indices in `[0,N)`; floating and Boolean indices are rejected. |
| `edge_features` | Optional real `(E,edge_channels)`, required when `edge_channels > 0` and normalized to `x` device/dtype. |
| `edge_types` | Relational-GCN integer relation ids `(E,)` in `[0,num_relations)`. |
| `coordinates` | EGNN floating coordinates `(N,D)`, `D >= 1`, with exactly the same device/dtype as `x`. |

| Family | Result, controls and primary source |
|---|---|
| `GATv2` | `(N,out_channels)` dynamic multi-head attention with `hidden_channels`, per-layer `heads`, `concatenate_heads`, `share_weights`, feature/attention dropout, negative slope, self-loop policy/fill, residual and bias; shared `edge_channels`; hidden-only activation/normalization and kwargs. `GATv2Layer.forward_with_attention` returns aligned routed edges and `(E_routed,heads)` weights for one layer. Inspired by [Brody et al.](https://arxiv.org/abs/2105.14491). |
| `RelationalGCN` | `(N,out_channels)` from `num_relations` typed transforms; `num_bases`, `aggregation` (`sum`/`mean`), `message_chunk_size`, dropout, residual, root weight and bias are scalar/per-layer; activation/normalization and kwargs are hidden-only. Messages are grouped by relation and projected in bounded chunks instead of materializing per-edge matrices. Inspired by [Schlichtkrull et al.](https://arxiv.org/abs/1703.06103). |
| `PNA` | `(N,out_channels)` using non-empty, duplicate-free mean/min/max/std `aggregators` crossed with identity/amplification/attenuation/linear/inverse-linear `scalers`. It configures edge/message widths, pre/post MLP widths, degree statistics, epsilon, dropout, activation/kwargs and bias; normalization/residual are hidden-only. `layer_kwargs` can override all layer-owned choices per layer while stack input/output/edge widths remain reserved. Inspired by [Corso et al.](https://arxiv.org/abs/2004.05718). |
| `GraphTransformer` | `(N,out_channels)` from local sparse dot-product attention. Per-layer controls are heads/concatenation, feed-forward width, activation/normalization and kwargs, three dropouts, self-loop fill, pre/post norm, residual, beta gate and bias; edge features modify keys and values. Related to [Shi et al.](https://arxiv.org/abs/2009.03509). |
| `EGNN` | `(N,out_channels)` features or a mapping of features and updated `(N,D)` coordinates. Per-layer `message_channels`, feature dropout, residual, bias and `layer_kwargs` configure message/node/coordinate MLP widths, aggregation, dropout, displacement normalization/scaling, coordinate updates/tanh and optional message attention. Hidden activation/normalization are stack policies; output keys are configurable. Inspired by [Satorras et al.](https://arxiv.org/abs/2102.09844). |

A graph stack has `L_graph = len(hidden_channels) + 1` layers. A scalar per-layer option is
broadcast; a list must have exactly `L_graph` entries. Hidden-only lists have
`len(hidden_channels)` entries. GATv2 concatenated widths must be divisible by their head counts;
R-GCN basis counts cannot exceed `num_relations`; GraphTransformer `beta=true` requires a residual
path. Attention self-loops replace existing loops and synthesize aligned edge rows rather than
duplicating topology. Empty edge lists and isolated nodes remain finite.

R-GCN's default `message_chunk_size: 65536` bounds the projected message tensor while preserving
exact sparse `sum`/`mean` results. Set a smaller positive integer for constrained devices; `None`
removes that bound and should be used only when the edge count and memory budget are known.

PNA's `average_degree = mean(in_degree)` and
`average_log_degree = mean(log(in_degree + 1))` must be computed **only on the training
split/topology**, recorded in YAML and reused unchanged for validation, test and inference. Computing
them over held-out graphs leaks topology. Both statistics and `epsilon` must be positive and finite.

`EGNN.forward` returns only node features by default. With `output_mode: mapping` it returns the
configured `feature_output_key` and `coordinate_output_key`; `LightningTask` preserves that mapping,
so each loss can select its `output_key` and each metric its documented prediction/output key
(`pred_key` for most built-ins). `forward_with_coordinates` always returns
the pair. A complete [PNA YAML example](src/lambdaforge/nn/README.md#pna-with-training-only-degree-statistics)
and [EGNN mapping example](src/lambdaforge/nn/README.md#egnn-mapping-output) document named
`model_input_keys` and the relevant per-layer list lengths.

These implementations are dependency-light native cores, not reproductions of author training
pipelines. They make no checkpoint or benchmark parity claim. Sparse attention/message routing grows
broadly as `O(EH)` rather than allocating a global `N²` attention matrix; GraphTransformer therefore
attends only over supplied edges and has no implicit global or positional encoding. EGNN is
E(n)-equivariant for scalar node/edge features and coordinate updates, not scale-equivariant and not
a higher-order vector/tensor representation.

## Extension contracts

### Model

Subclass `torch.nn.Module` or `lambdaforge.nn.models.Model`. Implement `forward(*args, **kwargs)`.
The base `predict` helper switches to evaluation/inference mode and restores the prior training
state. For the default task, either return a tensor (wrapped under `model_output_key`) or a mapping.

### Loss

Subclass `Loss` and implement `forward(outputs, batch, context=None) -> Tensor`. `LightningTask`
passes itself as `context`, while the default keeps the loss independently callable. Give every loss
a stable `name` and use mapping keys rather than task assumptions. Multiple losses are summed after
their configured weights are applied.

### Metric

Implement `update`, `compute` and `reset`. For DDP, also expose `distributed_state` and
`merge_distributed_state`, or deliberately use a framework metric. Metric instances are deep-copied
per stage so train, validation and test state never leaks.

Every stage requires unique metric names. Wrap a metric in `MetricAlias` when the same class is used
more than once with different parameters. Explicit stage lists avoid deepcopy requirements for a
project metric that owns a non-copyable external resource.

### Data and task

The default `LightningTask` expects mapping batches. `model_input_key` selects one tensor;
`model_input_keys` routes a sequence of positional inputs or a mapping of model argument names to
batch keys. Loss/metric objects select their own target/output keys. Tuple batches, multiple
optimizers, manual optimization or unusual control flow still use a custom `task.target`; all other
experiment, process and artifact machinery remains reusable.

`DatasetCache` is an optional map-style wrapper, not an implicit global cache. Its
`max_memory_bytes_per_process` budget counts retained serialized payload bytes and is paired with
`max_memory_entries`; it does not claim to cap total process RSS, live batches, prefetching or the
wrapped dataset. Worker RAM caching is off by default because every DataLoader worker owns a dataset
replica. Cache only deterministic loading/preprocessing, never random augmentation results. See the
[data guide](src/lambdaforge/data/README.md) before enabling it in parallel or DDP jobs.

### Installed plugins

External distributions can publish models, metrics, neural components, datasets, Lightning
callbacks and Lightning loggers through the canonical groups documented in the
[plugin guide](src/lambdaforge/plugins/README.md). Datasets must inherit PyTorch `Dataset`;
callbacks/loggers inherit the public `lambdaforge.integrations.Lightning` bases. Discovery reads
metadata only; resolving a selected plugin imports provider code and therefore has the same trust
boundary as `target`.
Built-ins and aliases registered explicitly in the current process keep precedence over
activation/normalisation plugins.

Each real run uses an isolated `PluginUsageSession` and atomically stores its canonically ordered
descriptors in `environment.json`. Earlier validation, sequential runs, installed-but-unused
providers and failed resolutions are excluded; cache hits and external component aliases actually
used by the run are included. Dry-runs record an empty list. See the
[provenance contract](src/lambdaforge/plugins/README.md#loaded-plugin-provenance).

### Tracking loggers

`trainer.logger` accepts the public `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` and `WeightsAndBiasesTrackingLogger` targets, or a non-empty list mixing
them with project loggers and installed logger plugins. Each adapter checks its own optional extra
only when constructed; importing LambdaForge remains provider-free. The canonical dense
`metrics.csv` is controlled separately by `write_epoch_metrics_csv`. Task losses/metrics reach the
provider only when `task.params.logging.logger` is enabled.

Provider credentials must stay outside YAML because materialized configurations are durable run
artifacts. Checkpoint upload is opt-in through `log_model` and independent of LambdaForge's local
retention transactions. See the [tracking guide](src/lambdaforge/tracking/README.md) for complete
parameters, local/remote and offline/online examples, privacy boundaries and failure behaviour.

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
| Models, distances, experiments, metrics and training competed as root-level entry points. | Introduced the `LambdaForge`/`Experiment` API and cohesive public subpackages with stable re-exports. |
| Large modules contained many loose functions or several unrelated classes. | Converted behaviour to collaborator objects and split every implementation class into its own file. |
| `training` and `metrics.classification` had become visually dense. | Split only along stable contracts (`callbacks`, `orchestration`, `binary`, `multiclass`) while preserving public imports; cohesive families such as `pooling` remain flat. |
| Closed choices and component names relied on repeated string literals. | Added enums and `ComponentRegistry`; strings remain only at YAML/serialization boundaries. |
| Advanced Trainer/DataLoader settings required source edits. | Added validated `trainer_kwargs` and shared/per-split `dataloader_kwargs` forwarding. |
| One shared metric list and implicit monitor order limited experiment control. | Added per-stage metric lists, aliases, explicit monitor modes, loss publication policy and CSV/console filters. |
| Selecting a custom Lightning logger removed the canonical epoch CSV. | Separated external logger selection from `write_epoch_metrics_csv`, so reporting remains available by default. |
| YAML mistakes were discovered late during object construction. | Published a strict Draft 2020-12 schema and added `ExperimentValidator`, `ValidationReport` and `lambdaforge validate`. |
| Runs did not record enough software/hardware/plugin provenance. | Added typed, atomically written `EnvironmentManifest` artifacts and isolated successful-plugin usage sessions for real runs; dry-runs stay import-free. |
| DDP could average already-computed AUROC/F1/correlation scalars, which is mathematically invalid. | Metrics now gather and merge their state before computing; unsupported custom metrics fail explicitly. |
| Dataset reuse had no bounded memory contract and could silently multiply across workers. | Added an opt-in per-process serialized LRU, entry/byte limits, spawn/fork isolation and explicit worker policy. |
| Persistent cache writers could race, exceed quota after a crash, reuse stale transforms or deserialize unchecked bytes. | Added OS shared/exclusive locks, immutable namespace manifests, pre-eviction, crash reconciliation, generation tokens, explicit fingerprints, verified checksum/HMAC envelopes and a bounded non-pickle NumPy/Torch codec. |
| External classes required fully qualified paths even when distributed as reusable packages. | Added lazy entry-point discovery, explicit contracts for neural objects plus datasets/callbacks/loggers, YAML integration, conflict checks and a non-importing CLI listing. |
| Exact curve metrics retained every score and target. | Added binary and multiclass histogram AUROC/AP alternatives with fixed persistent state and bounded tensor `all_reduce` synchronization. |
| The built-in neural catalog was limited to a dense network, a CNN and one message-passing model. | Added task-neutral graph, differentiable-tree, sequence, set, tabular, vision, composition and implicit-representation families plus broader component categories. |
| The default task routed only one model tensor and offered no named optimizer groups. | Added positional/named `model_input_keys` and opt-in per-group optimizer settings exposed by models. |
| `CNN2D` selected 1D BatchNorm by default for NCHW tensors. | Its built-in default now creates `BatchNorm2d`. |
| `Model.predict` did not guarantee restoration of the previous training mode. | Inference is guarded by `try/finally` and restores the original state. |
| `test_after_fit` asked a new Trainer for an unknown `best` checkpoint and omitted its stop event. | It now uses the actual fitted checkpoint when available, otherwise current weights, and preserves cancellation. |
| Forced process termination focused on root workers and was fragile on Windows. | Added recursive psutil cleanup, Job Objects, parent-death guards and guarded data workers. |
| Completion assumed a domain-specific prediction artifact. | Replaced it with generic relative `required_artifacts`. |
| Documentation described obsolete modules/scripts and there was no root guide. | Replaced it with linked English/Spanish guides whose claims are checked against current code. |

These facilities remain explicit rather than automatic: researchers choose the cache budget, plugin
provider and exact-versus-streaming metric semantics in YAML, so adding convenience does not hide
resource or scientific trade-offs.

## Development and verification

```powershell
ruff format --check src tests
ruff check src tests
mypy src\lambdaforge
pytest -q
```

The current suite covers configuration expansion, object/plugin construction, model validation,
aggregation, spawned process scheduling, the structural POO rules, a real one-epoch Lightning CPU
fit, schema/CLI validation, environment capture and external model/loss/metric/logger/callback
construction from YAML, installed dataset/callback/logger entry points, cache quotas/isolation,
HMAC corruption and substitution, spawn races,
crash recovery, mmap leases, lazy/mapped datasets, streaming metric state and retention
preview/apply, receipt, lock, ZIP-verification, rollback and concurrent-idempotence scenarios.
Plugin tests additionally cover exact distribution metadata, cache hits, validation/run isolation,
failed construction and a real `spawn` child manifest.
The process-integration tests create a real launcher/worker/descendant tree. POSIX delivers an
actual process-group `killpg(SIGINT)`; Windows asks the launcher to raise a targeted Python SIGBREAK
because a native console control event would affect the whole test group. A separate scenario
hard-terminates the launcher and verifies that every recorded descendant and temporary file is gone.
Emergency cleanup in each test also prevents a failed assertion from leaving workers behind.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs automatically for pull requests, pushes
to `main`, version tags and
manual dispatches. It grants only read access to repository contents, cancels stale runs for the
same ref, gives every job a timeout, caches pip downloads from `pyproject.toml` and retains reports
for 14 days. Its independent jobs are:

- formatting, linting and type checking on CPython 3.10;
- the complete CPU suite on Ubuntu and Windows for every currently stable supported CPython release,
  3.10 through 3.14, using the CPU-only PyTorch index and hiding CUDA explicitly;
- the complete suite with only `pytorch-lightning==2.2.*` installed, plus an assertion that the
  central adapter really selected the legacy namespace rather than modern `lightning`;
- source/wheel builds, `twine` metadata validation and an isolated wheel-content smoke test on
  CPython 3.14. JUnit reports and built distributions are uploaded as uniquely named artifacts.

[`.github/workflows/cuda.yml`](.github/workflows/cuda.yml) is deliberately manual and targets a
self-hosted runner labelled
`[self-hosted, linux, x64, cuda]`. The runner needs Actions Runner 2.327.1 or newer for the Node 24
[action versions](https://github.com/actions/setup-python/releases/tag/v6.0.0) used by these
workflows, a compatible NVIDIA driver and a CUDA-enabled PyTorch build. A preflight rejects CPU-only
or unusable installations and records the exact PyTorch, CUDA, cuDNN and device details. The
workflow then runs the `cuda`-marked one-epoch public-YAML training test and the remaining
regression suite. Merely defining, queuing or skipping this workflow is not CUDA coverage; only a
successful execution on the real runner is evidence.

Changes to GPU scheduling or cleanup should additionally be exercised on the target multi-GPU host
and interrupted manually at least once.

Every source module and class has a docstring. The repository audit also checks class/module name
matching, the one-class-per-file rule and absence of module-level helper functions in implementation
modules.

## Current limitations

- `DatasetCache` bounds retained serialized payloads per process, not total RSS. DataLoader batches,
  prefetching, pinned memory, allocator overhead and the source dataset remain outside that budget;
  enabling worker caches multiplies the configured allowance across process replicas.
- Pickle remains the compatibility default and can execute code; select the safe NumPy/Torch codec
  for supported sample trees or keep pickle strictly local and trusted. A checksum is not
  authentication; HMAC must be configured explicitly and provides no encryption.
- Fingerprints are explicit snapshots because arbitrary transform semantics cannot be inferred.
  Filesystem coordination is for cooperating local processes, not a cross-machine/NFS cache.
- Lightning is the only built-in training backend.
- The default task assumes mapping-shaped batches; it routes one or several model inputs, while
  tuple batches and manual/multiple-optimizer flows need a custom task object.
- Exact binary and multiclass curve metrics still retain predictions. Their fixed-memory streaming
  alternatives introduce binning approximation; multiclass state grows as `O(num_classes * num_bins)`.
- Entry-point discovery covers reusable neural contracts plus datasets, callbacks and loggers.
  Tasks, data modules and runners remain fully supported through `target` and intentionally have no
  dedicated group.
- Plugin provenance covers resolutions in the run process/context; user-created child processes
  require explicit IPC if their independently loaded plugins must be attributed to the parent.
- Statistical summaries are useful exploratory tools, not a substitute for a study-specific
  protocol. Normal intervals and asymptotic Wilcoxon remain approximations when explicitly selected
  or chosen by `auto` for larger paired samples.
- Schemas 1.0 and 1.1 are packaged and migration supports the deterministic
  `unversioned -> 1.0 -> 1.1` path. There is no downgrade, in-place rewrite, remote source or
  secret redaction.
- Artifact retention is local-filesystem only and currently uses ZIP/Deflate. Preview can become
  stale by design; apply replans and revalidates under locks. Remote/object stores need distinct
  lease and atomicity contracts.
- MLflow, TensorBoard and Weights & Biases tracking adapters are optional. Provider
  authentication/network/storage, remote retention and service availability remain external;
  tracker failures fail the owning run, and LambdaForge retention cannot remove uploaded artifacts.
  No distributed job scheduler, hyperparameter optimiser or provider-neutral remote artifact store
  is bundled.
- Advanced graph families are native sparse cores without paper-checkpoint/benchmark parity.
  GraphTransformer is local to `edge_index`; PNA statistics are explicit training-split inputs; EGNN
  covers E(n) scalar-feature equivariance, not scale equivariance or higher-order tensor features.
- Hosted CI covers CPU behaviour on Ubuntu/Windows and CPython 3.10-3.14, including POSIX
  process-group SIGINT, targeted Windows Python SIGBREAK and hard-launcher-death scenarios. It does
  not exercise native Windows console-group CTRL_C/CTRL_BREAK delivery. A shared
  `TrainingOrchestrator` instance is not re-entrant, and a detached external daemon is outside its
  process-tree contract. Real CUDA and multi-GPU/DDP remain host-dependent; CUDA is covered only
  after the manual self-hosted workflow has succeeded.

## Documentation map

- [Single-file agent manual](AGENTS.md)
- [Experiment system](src/lambdaforge/experiments/README.md) · [Español](src/lambdaforge/experiments/README.es.md)
- [Configuration migrations](src/lambdaforge/experiments/migrations/README.md) · [Español](src/lambdaforge/experiments/migrations/README.es.md)
- [Artifact retention](src/lambdaforge/experiments/retention/README.md) · [Español](src/lambdaforge/experiments/retention/README.es.md)
- [Statistical comparisons](src/lambdaforge/experiments/statistics/README.md) · [Español](src/lambdaforge/experiments/statistics/README.es.md)
- [Data and caching](src/lambdaforge/data/README.md) · [Español](src/lambdaforge/data/README.es.md)
- [Training and processes](src/lambdaforge/training/README.md) · [Español](src/lambdaforge/training/README.es.md)
- [Neural components](src/lambdaforge/nn/README.md) · [Español](src/lambdaforge/nn/README.es.md)
- [Metrics](src/lambdaforge/metrics/README.md) · [Español](src/lambdaforge/metrics/README.es.md)
- [Installed plugins](src/lambdaforge/plugins/README.md) · [Español](src/lambdaforge/plugins/README.es.md)
- [Optional experiment tracking](src/lambdaforge/tracking/README.md) · [Español](src/lambdaforge/tracking/README.es.md)
- [Complete YAML example](examples/experiment.yaml)

Each sub-guide links back here and to its translation. Class docstrings are the most precise source
for individual constructor arguments.

## Roadmap

Completed in this iteration: JSON Schema validation, typed environment and run/aggregate results, the extended CI
matrix, hardened `DatasetCache` plus file/mmap adapters, lazy entry-point discovery, run-isolated
loaded-plugin provenance, non-neural dataset/callback/logger contracts and binary/multiclass
streaming curve metrics, deterministic bootstrap intervals, paired Wilcoxon tests, plus the
categorized neural catalog documented above. Versioned configuration migrations now add safe
legacy normalisation, preview and explicit atomic persistence. Artifact retention now adds
completion receipts, role-aware checkpoint selection, verified streaming archives and recoverable
transactions. Optional tracking now adds lazily loaded MLflow, TensorBoard and Weights & Biases
logger objects without expanding the base dependency set. Native advanced graph support now adds
GATv2, relational GCN, PNA, local sparse GraphTransformer and scalar-feature EGNN with aligned edge
contracts and per-layer YAML configuration. The reviewed milestones and their current outcomes are:

1. **Typed result objects — completed**: terminal and aggregate objects now preserve direct
   dict/JSON compatibility, add typed attributes, version their envelopes and write atomically.
2. **Streaming multiclass curves — completed**: one-vs-rest histogram AUROC/AP requires
   `num_classes`, exposes macro/weighted/micro and per-class results, handles undefined classes
   explicitly and keeps persistent state bounded by `O(num_classes * num_bins)`.
3. **Persistent-cache hardening — completed**: canonical content/transform/configuration
   fingerprints, verified checksum/HMAC envelopes, a bounded non-pickle NumPy/Torch codec,
   immutable namespace manifests, atomic usage, generation-safe deletion and crash-recoverable
   multiprocess quotas are available through Python and recursive YAML objects.
4. **Loaded-plugin provenance — completed**: each run records the deterministic descriptor of every
   successfully resolved plugin, including cache hits and aliases, without contamination from
   discovery, validation, prior runs or parent processes; manifests are rewritten atomically on
   success and failure.
5. **Broader non-neural plugin contracts — completed**: dataset, callback and logger groups now
   enforce PyTorch/Lightning inheritance, work in their exact Schema positions, retain
   `target`/`ref` and logger-list compatibility, and participate in run provenance.
6. **Stronger comparison methods — completed**: YAML now selects deterministic bounded-memory
   percentile bootstrap or the legacy normal interval, and exact/asymptotic paired Wilcoxon or the
   legacy sign test. Typed results expose alternatives, zero handling, calculation provenance,
   effective sample sizes and explicit unavailable states while legacy aggregate columns remain.
7. **Configuration migrations — completed**: exact Schema value objects, a packaged 1.0/1.1
   catalog, immutable forward migration registry and validated
   `unversioned -> 1.0 -> 1.1` chain now back
   diff/YAML/JSON CLI previews, CI `--check`, transparent in-memory compatibility and explicit
   atomic output that cannot overwrite the source or race-clobber an unforced destination.
8. **Artifact retention policy — completed**: strict Schema 1.1 YAML selects disabled/preview/apply,
   best/last checkpoint roles, protected globs and verified streaming compression or explicit
   pruning. A fingerprinted final-aggregation receipt gates durable journal/quarantine
   transactions; ordered cross-process locks, rollback/forward recovery and immutable manifests
   make concurrent application safe and idempotent.
9. **Extended CI and interruption tests — completed**: the full CPU suite covers Ubuntu/Windows and
   CPython 3.10-3.14; strict time/device/resource tests, host-driven `request_stop()`, secondary-thread
   signal opt-out, exact-parent guards, bounded shutdown and visible Job Object degradation harden
   the process contract. An isolated job proves the `pytorch_lightning` 2.2 fallback, while real
   POSIX group SIGINT, targeted Windows SIGBREAK and hard launcher death check for residual children.
   A manual self-hosted CUDA workflow runs one public-YAML GPU epoch, records hardware evidence and
   then runs the remaining regressions without making CUDA a hosted-CI prerequisite.
10. **Optional tracking adapters — completed**: public lazy logger objects wrap MLflow,
   TensorBoard and Weights & Biases behind the existing single/list `trainer.logger` boundary.
   Separate and combined extras keep the base installation provider-free; explicit local/remote and
   offline/online parameters, opt-in checkpoint publication, actionable dependency errors and
   bilingual privacy/lifecycle guidance avoid making any service mandatory.
11. **Advanced graph layers — completed**: GATv2, relational GCN, PNA, local edge-aware
    GraphTransformer and E(n)-equivariant EGNN are dependency-light native stacks. Directed edge,
    edge-feature/relation/coordinate contracts, per-layer configuration, attention inspection,
    training-only PNA statistics and tensor/mapping EGNN outputs are documented and tested without
    claiming author-checkpoint or benchmark parity.
12. **Vision tasks beyond encoders — completed**: configurable U-Net dense prediction, a generic
   FPN over the shared hierarchical-backbone contract, variable-resolution Vision Transformer and
   MobileNetV2-style inverted-residual stages are public through Python and recursive YAML. Patch
   remainder policy, token/feature-map outputs, odd-size decoder alignment and fine-to-coarse
   feature channels are explicit and covered by gradient tests.
13. **Broader tabular research — completed**: deterministic categorical preprocessing plus native
   TabNet, SAINT, AutoInt and DeepFM objects are public from Python/YAML. Shape, range, mask and
   gradient tests cover the dependency-light cores; conformance cases provide the tree parity path.
14. **Long-sequence models — completed**: Transformer decoder/seq2seq and Conformer are native,
   batch-first models with explicit mask contracts. `StateSpaceAdapter` integrates S4/Mamba-like
   modules without forcing their compiled kernels into the base installation.
15. **Generative and uncertainty objects — completed**: a reusable beta-VAE objective, VQ-VAE,
   linear/cosine diffusion schedules, DDPM/DDIM sampling, temperature calibration and split-conformal
   prediction intervals are composable and tested.
16. **Scientific architectures — completed at the dependency-light boundary**: fixed-step Neural
   ODE/CDE, DeepONet, a 1D Fourier neural operator and native E(3) scalar/vector tensor-field message
   passing have shape/numerical/equivariance tests. Optional higher-order providers use a validated
   adapter; native `l>=2` irreps and adaptive stiff solvers remain future specialized work.
17. **Architecture conformance packs — completed**: source-linked cases capture initialization,
   parameter counts, expected outputs and tolerances, persist tiny weights-only references and group
   numerical/shape/checksum parity into a CI assertion without redistributing author checkpoints.
