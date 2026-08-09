# LambdaForge agent manual

This is the single operational entry point for an agent using or modifying LambdaForge. Read this
file first. Do not crawl the repository or all READMEs. Use the routing table near the end only when
a requested detail is not available here, then inspect the public symbol's signature/docstring or
the one linked guide that owns the topic.

## What the framework is

LambdaForge is an installable, task-agnostic PyTorch/Lightning library. It provides generic tasks,
composable preprocessing, reusable neural objects and a YAML engine for validation, construction,
seed/sweep/HPO expansion, workflow DAGs, CPU/GPU/SLURM planning, training, aggregation, statistical
comparisons, tracking, result auditing, checkpoint/model operations, artifact stores/cache and
retention. A consumer project owns its datasets and domain code.

Supported Python starts at 3.10. The documented public namespaces are `lambdaforge`,
`lambdaforge.data`, `lambdaforge.nn`, `lambdaforge.metrics`, `lambdaforge.training`,
`lambdaforge.experiments`, `lambdaforge.tasks`, `lambdaforge.preprocessing`,
`lambdaforge.configuration`, `lambdaforge.workflows`, `lambdaforge.operations`, `lambdaforge.hpo`,
`lambdaforge.execution`, `lambdaforge.storage`, `lambdaforge.registry`,
`lambdaforge.observability`, `lambdaforge.reproducibility`, `lambdaforge.plugins`,
`lambdaforge.tracking` and `lambdaforge.integrations`. Do not import from private file locations.

## Fast decision path

| Need | Use |
|---|---|
| Run a configured study | `lambdaforge run experiment.yaml` |
| Run reproducible non-training work | use `kind: task`, then the same validate/inspect/run commands |
| Preprocess JSONL or file trees | start from `examples/preprocessing.yaml` |
| Add project preprocessing | implement a transform or the source/transform/sink contracts below |
| Compose preprocessing -> training | `kind: workflow`; validate/inspect/run its YAML |
| Reuse YAML fragments | `extends`/`include`; audit with `compose` and `diff` |
| Run CPU-only parallel sweep | `execution: {mode: parallel, cpu_jobs: N, cpu_cores_per_job: C}` |
| Inference/evaluation/export | configure a task from `lambdaforge.operations` |
| Random/Optuna HPO | `RandomSearch.materialize` / optional `OptunaSearch` |
| Adaptive multi-fidelity HPO | add `hpo.enabled: true`; start at `examples/adaptive-hpo.yaml` |
| Preview SLURM | `SlurmExecutionBackend.submit(..., dry_run=True)` |
| Publish/stage artifacts | `LocalArtifactStore` or optional/injected `S3ArtifactStore` |
| Query across studies | `lambdaforge registry ROOT`; `dashboard` for static HTML |
| Inspect config/target | `lambdaforge explain KIND PATH`; `lambdaforge target IMPORT` |
| Scaffold a consumer | `lambdaforge init DIRECTORY`, then rename/implement/install it |
| Validate without training | `lambdaforge validate experiment.yaml` |
| See every expanded seed/variant | `lambdaforge inspect experiment.yaml` |
| Use an old config | `lambdaforge migrate experiment.yaml` (preview first) |
| Build one YAML object in Python | `LambdaForge.build(spec)` |
| Use project-local code | installed `my_project.*` `target`/`ref` paths |
| Publish reusable extensions | `lambdaforge.<kind>` entry-point plugins |
| Audit retries/duplicates | `lambdaforge results experiment.yaml --write-index` |
| Reject ambiguous successful results | add `--fail-on-ambiguous` (exit code 2) |
| Rebuild summaries | `lambdaforge aggregate experiment.yaml` |
| Remove/compress artifacts | preview `lambdaforge retain ...`; apply only with `--apply` |
| Load a trained model | `Experiment.load_model(seed=..., variant=..., which="best")` |
| Add a model/loss/metric | follow the contracts below; reference it by importable `target` |

## Install into a consumer project

Never copy `src/lambdaforge`, share this repository's `.venv`, or patch `PYTHONPATH`. Give the
consumer its own environment and install both packages:

```bash
cd /path/to/research-project
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

The consumer should use a `src` layout and declare LambdaForge in its own `pyproject.toml`. For a
reproducible release, build/install a versioned LambdaForge wheel instead of the editable path:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-0.4.0-py3-none-any.whl
```

Let the consumer lock the correct PyTorch wheel. `nvidia-smi` only proves the driver is visible;
verify the active Python environment with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Optional extras are `hpo` (Optuna), `adaptive-hpo` (BoTorch), `s3` (boto3), `parquet`
(Pandas/PyArrow), `onnx`, individual/all `mlflow`/`tensorboard`/`wandb` tracking providers and
`dev`. Install only used providers, for example `lambdaforge[adaptive-hpo,s3]`; Sobol/random HPO,
local stores and base training do not need those extras.

## Safe experiment workflow

Use this order. Validation and dry-run are non-training checks; retention is destructive only with
the explicit apply flag.

```bash
lambdaforge validate experiments/study.yaml
lambdaforge inspect experiments/study.yaml
lambdaforge run experiments/study.yaml --dry-run
lambdaforge run experiments/study.yaml
lambdaforge results experiments/study.yaml --write-index --fail-on-ambiguous
lambdaforge aggregate experiments/study.yaml
lambdaforge retain experiments/study.yaml
# Review the plan, then only if intended:
lambdaforge retain experiments/study.yaml --apply
```

Python equivalent:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("experiments/study.yaml")
report = experiment.validate()
if not report.is_valid:
    raise ValueError(report.summary())
runs = experiment.expand()
outcomes = experiment.run()
records = experiment.results(include_archived=True)
catalog = experiment.result_catalog()
ambiguous = catalog.ambiguous_successes()
```

Generic task equivalent:

```python
from lambdaforge import LambdaForge

task = LambdaForge.task("preprocessing.yaml")
report = task.validate()
plan = task.inspect()
result = task.run()
records = task.result_catalog().records(include_archived=True)
```

## Adaptive optimization contract

Use enabled `hpo` only for training experiments and never together with `sweep`. Keep the ordinary
data/model/loss/metric/optimizer/task/trainer configuration. In `hpo.space`, map dotted scientific
paths to `float`, `int`, `categorical` or `bool`; numeric dimensions accept linear/log scale and a
dimension may use `when` conditions on an earlier one. The objective metric must be an exact column
from dense `metrics.csv` (normally `val_<metric.name>` or `val_loss`).

```yaml
hpo:
  enabled: true
  objective: {metric: val_auroc, direction: maximize}
  space:
    optimizer.params.lr: {type: float, low: 1.0e-5, high: 1.0e-2, scale: log}
    model.params.width: {type: int, low: 64, high: 512}
  initialization: {strategy: sobol, trials: auto}
  search: {strategy: bayesian}
  fidelity: {unit: epochs, strategy: adaptive_learning_curve, min: 5, max: 100, step: 5}
  seeds:
    strategy: adaptive_racing
    values: [7, 17, 27]
    confirmation_values: [101, 211]
  memory: {per_job_budget: 6GiB, headroom: 512MiB, allocator_cap: true}
  budget: {max_actions: 50, max_total_epochs: 1500}
```

`inspect`/`run --dry-run` are read-only plans. Real run actions are START_NEW, checkpoint RESUME,
ADD_SEED and CONFIRM, selected as information/cost times memory feasibility. Cumulative epoch
promotion does not recompute completed epochs; require checkpoint policy last/last_and_best/all.
Search seeds are shared and ordered; confirmation seeds must be disjoint. Install
`lambdaforge[adaptive-hpo]` for BoTorch GP/KG. Its absence/failure is recorded and falls back to
Sobol; Sobol/random need no provider. Memory budgets use bytes or KB/MB/GB/TB/KiB/MiB/GiB/TiB.
Explicit device capacities work when cluster discovery is unavailable. Never assume allocator caps
are physical isolation and never auto-reduce batch size.
If `memory.preflight: true`, `memory.probe` is required and must build a zero-argument callable that
performs a representative CUDA forward/backward/step; it runs isolated once per configured GPU.

Controller state/events/summary live at `SUITE/.lambdaforge/adaptive/STUDY_ID`; each action remains
an ordinary seed run with config, environment, metrics, checkpoint, attempts and result. Relaunch
the identical YAML to reconcile/resume; do not select or edit files manually. Read only
`docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md` if changing controller/scheduler internals.
`summary.json` includes seed usage, merged curves, memory evidence and confirmation mean/sample
deviation/standard error/normal interval plus shared-seed paired differences. Use the experiment
statistics APIs for publication-grade bootstrap or non-parametric follow-up.
The main defaults are Bayesian search after Sobol `auto=max(4,2*(dimension+1))`, adaptive curves at
5→100 epochs, one shared search seed, no confirmation seeds, conservative pruning after the minimum
budget, 50 actions and one concurrent job. `search.refresh_interval` deterministically refits/caches
the surrogate every N scored observations. Read the complete default table in the root README before
changing scientific policy.

For domain policy replacement, `hpo.components` accepts importable `target`/`params` specs with
these duck-typed boundaries:

- searcher: `propose(space, state, *, count=1) -> tuple[dict, ...]`;
- fidelity policy: `resume_candidates(state) -> tuple[AdaptiveAction, ...]`, plus optional
  `dominated(state, learning_model) -> tuple[(config_id, probability), ...]`;
- seed policy: `candidates(state, learning_model) -> tuple[AdaptiveAction, ...]`;
- learning curve model: `predict_configuration(...)` and `probability_competitive(...)`;
- cost/memory model: `predict(action, state) -> PredictiveEstimate`;
- admission controller: `assess(action, state, memory_model, *, available_bytes=...)`;
- action selector: `rank(actions, state, *, learning_model, cost_model, memory_model, admission,
  available_bytes) -> tuple[AdaptiveAction, ...]`.

Return immutable LambdaForge action/estimate values, keep ordering deterministic and do not modify
the runner. Inspect the corresponding public built-in signature only when exact types are needed.

Workflow equivalent:

```python
workflow = LambdaForge.workflow("workflow.yaml")
plan = workflow.inspect()
result = workflow.run()
```

## Generic tasks and preprocessing

Do not force preprocessing, download, inference or export into experiment Schema 1.1. Use the
independent task Schema 1.0. Existing experiment YAML has no `kind`; a task must declare
`kind: task`. The same CLI dispatches both families.

```yaml
schema_version: "1.0"
kind: task
name: normalize-records
output_root: runs/tasks
resume: true
inputs:
  - {name: raw, path: data/raw.jsonl}
required_artifacts: [processed, dataset-artifact.json]
task:
  target: lambdaforge.preprocessing.PreprocessingTask
  params:
    source:
      target: lambdaforge.preprocessing.JsonLinesSource
      params: {path: data/raw.jsonl, key_field: id}
    transforms:
      - target: lambdaforge.preprocessing.CallableTransform
        params: {function: {ref: my_project.preprocessing.normalize_record}}
    sink:
      target: lambdaforge.preprocessing.JsonDirectorySink
      params: {output_dir: processed}
```

Declare every scientific local input at top level. LambdaForge hashes file/directory contents
before planning; changed bytes create a different fingerprinted run and cannot silently reuse stale
success. Task validation checks Schema/import/plugin/constructor contracts without construction.
Inspect and dry-run return the same immutable `TaskExecutionPlan` and create no output directory.
Built-in JSONL/file-tree source paths must match a declared input or lie below a declared directory.

A generic project task implements `run(context)` and returns `TaskOutput`:

```python
from lambdaforge.tasks import ArtifactDeclaration, Task, TaskContext, TaskOutput


class ProjectTask(Task):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def run(self, context: TaskContext) -> TaskOutput:
        output = context.output_path("output.txt", create_parent=True)
        output.write_text(str(self.scale), encoding="utf-8")
        return TaskOutput(
            outputs={"path": output.name},
            metrics={"scale": self.scale},
            artifacts=[ArtifactDeclaration("output.txt")],
        )
```

Fully qualified targets may use duck typing (`run(context)` or `run()`); task entry-point plugins
must subclass `lambdaforge.tasks.Task` and publish under `lambdaforge.tasks`. A task may return a
`TaskOutput`, a mapping treated as outputs, or `None`. Artifacts must exist below the run, are
symlink/traversal checked and receive deterministic SHA-256 digests. A matching success is skipped
only while recorded artifact bytes still verify; reruns archive the prior terminal result.

For preprocessing, a `PreprocessingSource.records(context)` yields stable-key
`PreprocessingRecord`s, each `PreprocessingTransform.transform(record, context)` preserves the key,
and a `PreprocessingSink` writes/verifies/finalizes output. `PreprocessingTask` checkpoints a
per-record manifest, retries failed/missing outputs, supports `fail`/`skip` error policy and assigns
explicit shards by stable SHA-256 modulo. It writes `dataset-artifact.json` with a content-derived
dataset ID, sample/split counts, source, task fingerprint and artifact hashes. Run each shard as an
explicit node/configuration and merge with a project task when needed; implicit shard discovery and
domain-specific merging are intentionally not guessed.

## Workflows, composition and secrets

A workflow coordinates complete documents and delegates identity/resume to their existing runner:

```yaml
kind: workflow
schema_version: "1.0"
name: prepare-train
max_parallel: 2
nodes:
  prepare: {config: preprocessing.yaml}
  train:
    config: experiment.yaml
    needs: [prepare]
    bindings:
      data.train.params.manifest: ${nodes.prepare.artifacts.dataset-artifact.json}
```

References are exact `${nodes.NAME.outputs.PATH}`, `.metrics.PATH` or
`.artifacts.RUN_RELATIVE_PATH`. Cycles/unknown dependencies fail. Failed branches block descendants,
not siblings; `continue_on_failure` is opt-in. Ready-node local concurrency is bounded.

Composition order is `extends`, `include`, leaf, explicit API overrides. Mappings merge, lists
replace, `{$delete: true}` deletes, paths are source-relative and cycles fail. Interpolation permits
only `${config:path}`, `${env:NAME}` and full-value `${secret:NAME}`—never expressions. `compose`
shows redacted values/provenance; `diff` compares semantic paths. Task secrets are unwrapped only by
`ObjectFactory` and persisted as `***`. Embedded secrets and durable experiment/workflow secrets are
rejected; provider code should read credentials from its runtime environment.

## Model operations, resources and data movement

`InferenceTask`, `EvaluationTask` and `ExportTask` are generic tasks. Declare checkpoints in task
`inputs`. Shared params: `model`, `checkpoints`, optional `data`, `batch_size`, `num_workers`, input/
output key and device. Multiple inference/evaluation checkpoints average matching tensor outputs.
Evaluation adds `metrics`; export adds `example_inputs`, `format` (`torchscript`, `torch_export`,
`onnx`) or an injected exporter. Plain/Lightning state dicts load weights-only.

`RandomSearch(space, seed).materialize(base, count)` supports choice/uniform/loguniform/int and
conditional `when`. `OptunaSearch` lazily requires Optuna and wraps seeded TPE plus
`asha`/`hyperband`. Existing grids/ablations do not change.

CPU experiment parallelism omits GPUs and sets `cpu_jobs`; affinity oversubscription is rejected.
`ResourceRequest` plus `ResourcePlanner` create declared capacity-safe waves/estimates.
`LocalExecutionBackend` runs argv. `SlurmExecutionBackend` writes quoted `submit.sbatch` and submits
only with `dry_run=False`; it supports nodes/array/dependency/resources/environment/container/requeue
and never `shell=True`. `FailureClassifier`, `RetryPolicy` and `AttemptMode` keep failure and
resume/restart/retry/fork semantics distinct.

`ArtifactReference` is `(store,key,sha256,size,media_type)`. `LocalArtifactStore` is atomic;
`S3ArtifactStore` uses an injected client or boto3. `DistributedArtifactCache` uses shared-filesystem
per-key leases, verifies stages and repairs corruption. The base store exposes no deletion.

## Registry, observability and reproducibility

`ExperimentRegistry` reads `ResultCatalog` plus snapshots, never a second DB. `RegistryQuery`
filters status/name/tags/metadata/fingerprint and exports JSON/CSV/optional Parquet.
`ExperimentComparator` reports statistics/effects/config diffs; `ReportBuilder` writes factual
Markdown/HTML plus interval plot; `LocalDashboard` is static/read-only.

Tasks write `events.jsonl`. `EventLogger` appends bounded locked JSONL; `ResourceMonitor` samples
CPU/RSS/threads/CUDA/throughput; `TorchProfilerAdapter` is finite. `ReproducibilityProfile` provides
fast/repeatable/strict, scientific/infrastructure fingerprints and seed application. `SeedDeriver`
creates stable scoped seeds; `EnvironmentExporter` writes pip/Conda/container snapshots.

## Minimal YAML contract

Start from `examples/experiment.yaml`; Schema `1.1` is current. YAML is trusted code because imports
and plugins can execute Python. A compact study looks like this:

```yaml
schema_version: "1.1"
experiment:
  name: baseline
  output_root: runs/experiments
  seeds: [7, 17, 27]
  resume: true
  rerun_completed: false
data:
  train: {target: my_project.data.TrainDataset, params: {split: train}}
  val: {target: my_project.data.TrainDataset, params: {split: val}}
  datamodule:
    target: lambdaforge.training.data.LightningDataModule
    params: {batch_size: 64, num_workers: 4}
model:
  target: my_project.models.ProjectModel
  params: {in_features: 32, out_features: 1}
losses:
  - target: lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss
    params: {output_key: logits, target_key: target}
val_metrics:
  - target: lambdaforge.metrics.BinaryAUROC
    params: {pred_key: logits, target_key: target}
optimizer: {ref: torch.optim.AdamW, params: {lr: 0.001}}
task:
  target: lambdaforge.training.LightningTask
  params: {model_input_key: x, model_output_key: logits}
trainer:
  max_epochs: 50
  accelerator: auto
  devices: auto
  checkpoint_policy: last_and_best
  checkpoint_monitor: val_auroc
  checkpoint_mode: max
execution: {mode: sequential}
```

Recursive object syntax is consistent everywhere:

- `target: package.module.Class` constructs a class with optional `params`.
- `ref: package.module.object` imports a callable/value without constructing it unless `params`
  are supplied.
- `plugin: {kind: model, name: published_name}` resolves an installed entry point; `params` may sit
  beside `plugin`.
- Nested specs inside lists/mappings are built recursively.

The default task requires mapping batches. `model_input_key: x` calls `model(batch["x"])`.
`model_input_keys: [x, edge_index]` passes positional arguments. A mapping such as
`model_input_keys: {features: x, graph: edge_index}` passes named arguments. A tensor model return
is wrapped under `model_output_key`; mapping returns are preserved.

Sweeps use `sweep.grid` dotted paths and named `sweep.ablations`; expanded runs are deterministic.
Execution modes are `sequential`, `parallel` (independent jobs per GPU), and `ddp` (one job over a
device group). Tracking logger specs accept built-in CSV, one logger, or a list. MLflow,
TensorBoard and W&B require their optional extras and explicit publication/privacy choices.

## Extension contracts

Project-local classes are normally simplest. Put them under the consumer's installed package and
use a fully qualified YAML `target`. Constructors must accept only serializable configuration.

Custom model:

```python
from typing import Any
import torch
from torch import Tensor, nn
from lambdaforge.nn.models import Model


class ProjectModel(Model):
    output_schema: dict[str, Any] = {"logits": "Tensor[B, C]"}

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.head = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {"logits": self.head(x)}
```

Any `nn.Module` works, but `Model` also supplies `predict`, parameter counting, freeze/unfreeze and
named optimizer groups. Override `parameter_groups()` only with stable, disjoint parameter sets.

Custom loss:

```python
from collections.abc import Mapping
from typing import Any
import torch
from lambdaforge.nn.losses import Loss


class ProjectLoss(Loss):
    def __init__(self, output_key: str = "logits", target_key: str = "target") -> None:
        super().__init__(name="project_loss", weight=1.0)
        self.output_key, self.target_key = output_key, target_key

    def forward(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any], context: object | None = None
    ) -> torch.Tensor:
        del context
        return self.weight * torch.nn.functional.mse_loss(
            outputs[self.output_key], batch[self.target_key]
        )
```

Return one scalar tensor, preserve its graph and apply `self.weight` in `forward`. `Loss` verifies
scalar output and upcasts unsafe reduced-precision inputs by default. Set
`supports_reduced_precision = True` only after proving numerical stability.

Custom metric (the distributed methods make it DDP-safe):

```python
from collections.abc import Mapping
from typing import Any
import torch
from lambdaforge.metrics import Metric


class MeanAbsoluteSignal(Metric):
    def __init__(self, output_key: str = "logits") -> None:
        super().__init__(name="mean_absolute_signal", higher_is_better=False)
        self.output_key = output_key
        self.reset()

    def update(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any], context: object | None = None
    ) -> None:
        del batch, context
        values = torch.as_tensor(outputs[self.output_key]).detach().abs()
        self.total += float(values.sum().cpu())
        self.count += values.numel()

    def compute(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def reset(self) -> None:
        self.total, self.count = 0.0, 0

    def distributed_state(self) -> dict[str, float | int]:
        return {"total": self.total, "count": self.count}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        self.total += float(state["total"])
        self.count += int(state["count"])
```

Metrics must detach state, implement reset/update/compute, and implement mergeable state for DDP.
Datasets inherit `torch.utils.data.Dataset`; callbacks and loggers inherit the Lightning bases
exported through `lambdaforge.integrations`. A custom task is only needed when mapping batch routing
and the standard loss/metric lifecycle are insufficient.

For reusable installed extensions, publish classes under these entry-point groups:
`lambdaforge.models`, `.losses`, `.metrics`, `.datasets`, `.callbacks`, `.loggers`, `.activations`,
`.normalizations`, `.distances`, `.pooling`, `.similarities`, `.kernels`, `.encodings`, and
`.regularization`, plus `lambdaforge.tasks`. List metadata without importing plugin modules using
`lambdaforge plugins`.

## Capability catalogue

- Data: `FileDataset`, `NumpyMemmapDataset`, deterministic `CategoricalFeatureEncoder`; bounded
  memory/disk/mmap `DatasetCache`, explicit dataset/transform fingerprints, safe NumPy/Torch codec,
  checksum/HMAC integrity, immutable namespaces and coordinated multiprocess quotas.
- Core/dense/composition: `MLP`, `CNN2D`, `ECMP`, `AutoEncoder`, `VariationalAutoEncoder`,
  `EnsembleModel`, `MultiTaskModel`, `MixtureOfExperts`, `SiameseModel`.
- Tabular/trees: `ResidualMLP`, `FTTransformer`, `TabNet`, `SAINT`, `AutoInt`, `DeepFM`,
  `ObliviousDecisionTree`, `NODE`, `GradTree`, `GRANDE`.
- Sequence/set: RNN/GRU/LSTM, TCN, Transformer encoder/decoder/seq2seq, Conformer,
  `StateSpaceAdapter`, Deep Sets and Set Transformer.
- Vision: CNN, ResNet, ConvNeXt, MobileNetV2, variable-resolution ViT, U-Net and generic FPN over
  the `HierarchicalBackbone2D` contract.
- Graph/geometric: KNN/scatter/readout primitives; GCN, GIN, GraphSAGE, GAT, GATv2, R-GCN, PNA,
  local edge-aware GraphTransformer, scalar-coordinate EGNN, native E(3) `l=0/l=1`
  `TensorFieldNetwork`, and an injected higher-order provider adapter.
- Generative/implicit/scientific: VQ-VAE, linear/cosine diffusion schedule with DDPM/DDIM,
  SIREN, fixed-step Neural ODE/CDE, DeepONet and 1D FNO.
- Uncertainty/conformance: held-out temperature scaling, split-conformal regression intervals, and
  source-linked architecture reference cases/packs with strict shapes, counts, checksums and
  weights-only checkpoints.
- Losses: BCE-with-logits, cross entropy, binary/multiclass focal, MSE, MAE, Huber, Smooth L1,
  Dice, Tversky, contrastive, triplet margin, InfoNCE and beta-VAE.
- Metrics: binary accuracy/balanced accuracy/precision/recall/specificity/F1/MCC/Cohen kappa,
  exact or fixed-memory AUROC/AUPRC; multiclass accuracy/balanced accuracy/F1 and exact/streaming
  one-vs-rest AUROC/AUPRC; MAE/MSE/RMSE/R2/Pearson/Spearman and running mean.
- Components: common/gated/sparse activations; batch/layer/group/instance/RMS/scale/L2 norms;
  dense masked and sparse indexed mean/sum/min/max/top-k/GeM/log-sum-exp/statistics/attention
  pooling; Fourier/learned/sinusoidal/rotary encodings; Euclidean/squared/angular/cosine/
  Manhattan/Chebyshev/Minkowski/Mahalanobis distances; dot/cosine/bilinear similarities;
  RBF/Laplacian/polynomial kernels; drop-path, feature dropout and Gaussian noise.
- Experiment system: strict Schema validation, versioned migrations, recursive object factory,
  seeds/grids/ablations, sequential/parallel/DDP execution, cancellation and process-tree cleanup,
  environment/plugin provenance, epoch CSVs/checkpoints, typed results, aggregation/plots,
  paired sign/Wilcoxon tests, normal/bootstrap intervals, power estimates, result catalog and
  preview-first transactional retention.
- Generic work: independent task Schema, immutable plans, facade/CLI dispatch, atomic results,
  environment/plugin provenance, attempt history, content-addressed inputs and verified artifacts.
- Preprocessing: JSONL/file-tree sources, callable transforms, atomic JSON-directory sink,
  resumable record manifests, deterministic shards and versioned `DatasetArtifact` identity.
- Workflow/config: task/experiment DAG, output bindings, branch isolation, local concurrency;
  include/extends/merge/delete, config/env/secret interpolation, redaction, provenance and diff.
- Operations/HPO: weights-only inference/evaluation/ensembles; TorchScript/torch.export/ONNX/custom
  export; finite random/Optuna plus persistent asynchronous adaptive multi-fidelity optimization.
- Execution/storage: CPU/GPU/DDP slots, portable resource packing, local/SLURM backends, failure/
  retry semantics; immutable local/shared/S3 artifact stores and lease-coordinated staging cache.
- Analysis/operations: catalog-backed registry and exports, comparison/report/dashboard, JSONL
  events, resource/profiler adapters, reproducibility profiles/seeds/environment exports.

For an exact constructor, inspect only that public symbol:

```bash
python -c "import inspect; from lambdaforge.nn.models import TabNet; print(inspect.signature(TabNet)); print(TabNet.__doc__)"
```

## Results and publication discipline

Every new terminal attempt has an `attempt_id`, UTC start/finish times and a
`config_fingerprint`. The SHA-256 fingerprint identifies the expanded scientific configuration;
operational fields such as output location, resume, execution, aggregation and retention do not
change it. Code/environment provenance remains in `environment.json`, so record a clean Git commit
or an explicit revision in configuration `extensions` for publication.

On retry, the previous terminal `result.json` is preserved under
`.lambdaforge/attempts/result-<attempt_id>.json`; it is metadata history, not a copied checkpoint.
A completed result is skipped and a checkpoint is resumed only when its scientific identity matches
the current materialized config. Changed science cannot silently reuse stale state.

Never choose a paper result by modification time, directory glob order, or “latest”. Before making
a table or figure:

1. Run `lambdaforge results CONFIG --write-index --fail-on-ambiguous`.
2. Require the intended fingerprint, variant and seed set.
3. If successful attempts are ambiguous, choose an explicit `attempt_id` and document why.
4. Read metrics and checkpoint paths from that `ResultRecord`; retain the index with the analysis.
5. Aggregate only after selection/audit. Never edit terminal result JSON manually.

Programmatic exact selection is `experiment.result_catalog().select(attempt_id="...")`. Use
`records(status="ok", fingerprint="...")` for filtered automation. The catalog is local-filesystem
based; remote stores must first materialize a coherent tree.

Generic task attempts follow the same selection discipline. Use
`LambdaForge.task(CONFIG).result_catalog()` and choose an explicit attempt/fingerprint; never select
preprocessing output by modification time. `dataset-artifact.json` identifies the processed
dataset, while `result.json` identifies the execution attempt that produced it.

## Repository modification rules

- Preserve the object-oriented layout: normally one public class/enum per same-named module,
  constructor validation, explicit types/docstrings, immutable result/config value objects, stable
  public re-exports and no new module-level utility functions.
- Keep base dependencies provider-neutral and dependency-light. Integrate compiled/specialized
  ecosystems through checked adapters or optional extras.
- Keep YAML construction recursive, schema-validated and backward compatible. Update the packaged
  Schema, example, public exports, English/Spanish owner README and this manual when an API changes.
- Keep only source, tests, reviewed examples/assets, Schemas, docs, CI and packaging metadata in
  Git. Never add `.env*`, environments, `.lambdaforge`, `runs`, provider outputs, caches, wheels,
  dashboards or SLURM logs. Do not broadly ignore scientific extensions; review `git status
  --short` and `git status --ignored --short` before handoff.
- Add focused unit tests for validation, shape/value contracts, gradients and failure paths. Add a
  YAML/public-import test for constructible public objects. Never claim paper/checkpoint parity
  without a pinned, reviewed external reference.
- Before handoff run:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src/lambdaforge
.venv/bin/pytest -q
python -m pip wheel . --no-deps --wheel-dir /tmp/lambdaforge-wheel-check
```

CUDA tests must run, not skip, on a CUDA-enabled development host. Diagnose the Python torch build
separately from the NVIDIA driver. Hosted CI intentionally remains CPU-compatible.

## Targeted documentation routes

Use these only for depth required by the current task:

- `README.md` / `README.es.md`: installation, YAML surface, architecture, limits and roadmap.
- `docs/ARCHITECTURE.md`: exact class collaboration, ownership, invariants and extension decisions.
- `CHANGELOG.md`, `docs/GOVERNANCE.md`, `SECURITY.md`: history, compatibility and threat model.
- `examples/experiment.yaml`: complete current Schema example.
- `examples/preprocessing.yaml`, `examples/workflow.yaml`, `examples/adaptive-hpo.yaml`: runnable
  task, DAG and adaptive-optimization starting points.
- `src/lambdaforge/experiments/README.md`: lifecycle, validation, outputs, result history,
  aggregation, statistics, checkpoint loading and retention.
- `src/lambdaforge/nn/README.md`: exact neural contracts and examples by model family.
- `src/lambdaforge/training/README.md`: task routing, trainer and process/device behaviour.
- `src/lambdaforge/data/README.md`: datasets, serialization, fingerprints and caching.
- `src/lambdaforge/tasks/README.md`: generic task Schema, contract, planning, results and artifacts.
- `src/lambdaforge/preprocessing/README.md`: built-ins, custom contracts, resume, shards and dataset identity.
- `src/lambdaforge/metrics/README.md`: metric semantics, streaming error and DDP state.
- `src/lambdaforge/plugins/README.md`: entry-point publication, contracts and provenance.
- `src/lambdaforge/tracking/README.md`: provider setup, privacy and lifecycle.
- `docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md`: adaptive identity, controller/backend ownership,
  recovery and failure semantics.

Known intentional boundaries: each task attempt and workflow runner are local; the SLURM backend
submits explicit plans but does not discover cluster topology or translate a whole DAG automatically.
Distributed cache leases require a coherent shared filesystem; remote destructive lifecycle remains
provider-owned. The dashboard is static/read-only. Adaptive HPO schedules local independent trials,
not DDP actions or direct SLURM jobs; finite adapters do not silently schedule science. No pretrained
weights, native optimized S4/Mamba kernels, native `l>=2` irreps, adaptive
stiff ODE solver, graph sampling or compiled sparse kernels. The integrated README roadmap records
the closed 1–30 implementation status and the owner licence decision still required for release.
