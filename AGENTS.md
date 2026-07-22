# LambdaForge agent manual

This is the single operational entry point for an agent using or modifying LambdaForge. Read this
file first. Do not crawl the repository or all READMEs. Use the routing table near the end only when
a requested detail is not available here, then inspect the public symbol's signature/docstring or
the one linked guide that owns the topic.

## What the framework is

LambdaForge is an installable, task-agnostic PyTorch/Lightning library. It provides reusable neural
objects plus a YAML engine for validation, construction, seed/sweep expansion, training,
multi-process GPU scheduling, aggregation, statistical comparisons, tracking, result auditing,
checkpoint loading and artifact retention. A consumer project owns its datasets and domain code.

Supported Python starts at 3.10. The documented public namespaces are `lambdaforge`,
`lambdaforge.data`, `lambdaforge.nn`, `lambdaforge.metrics`, `lambdaforge.training`,
`lambdaforge.experiments`, `lambdaforge.plugins`, `lambdaforge.tracking` and
`lambdaforge.integrations`. Do not import from private file locations when a public re-export exists.

## Fast decision path

| Need | Use |
|---|---|
| Run a configured study | `lambdaforge run experiment.yaml` |
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
python -m pip install dist/lambdaforge-0.2.0-py3-none-any.whl
```

Let the consumer lock the correct PyTorch wheel. `nvidia-smi` only proves the driver is visible;
verify the active Python environment with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

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

    def forward(self, outputs: Mapping[str, Any], batch: Mapping[str, Any],
                context: object | None = None) -> torch.Tensor:
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

    def update(self, outputs: Mapping[str, Any], batch: Mapping[str, Any],
               context: object | None = None) -> None:
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
`.regularization`. List metadata without importing plugin modules using `lambdaforge plugins`.

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

## Repository modification rules

- Preserve the object-oriented layout: normally one public class/enum per same-named module,
  constructor validation, explicit types/docstrings, immutable result/config value objects, stable
  public re-exports and no new module-level utility functions.
- Keep base dependencies provider-neutral and dependency-light. Integrate compiled/specialized
  ecosystems through checked adapters or optional extras.
- Keep YAML construction recursive, schema-validated and backward compatible. Update the packaged
  Schema, example, public exports, English/Spanish owner README and this manual when an API changes.
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
- `examples/experiment.yaml`: complete current Schema example.
- `src/lambdaforge/experiments/README.md`: lifecycle, validation, outputs, result history,
  aggregation, statistics, checkpoint loading and retention.
- `src/lambdaforge/nn/README.md`: exact neural contracts and examples by model family.
- `src/lambdaforge/training/README.md`: task routing, trainer and process/device behaviour.
- `src/lambdaforge/data/README.md`: datasets, serialization, fingerprints and caching.
- `src/lambdaforge/metrics/README.md`: metric semantics, streaming error and DDP state.
- `src/lambdaforge/plugins/README.md`: entry-point publication, contracts and provenance.
- `src/lambdaforge/tracking/README.md`: provider setup, privacy and lifecycle.

Known intentional boundaries: no cluster scheduler, generic HPO engine or provider-neutral remote
artifact store; no pretrained weights; no native optimized S4/Mamba kernels, native `l>=2` irreps
or adaptive stiff ODE solver; graph sampling and compiled sparse kernels remain external. Adapters
make those ecosystems injectable without destabilizing the base installation.
