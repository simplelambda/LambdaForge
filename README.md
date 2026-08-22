<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="icons/lambdaforge-light.svg"><source media="(prefers-color-scheme: light)" srcset="icons/lambdaforge-dark.svg"><img src="icons/lambdaforge-dark.png" width="140" alt="LambdaForge logo"></picture>
</p>

# LambdaForge

**Write the research in Python. Describe its execution in YAML.**

LambdaForge 0.11 is an installable framework for reproducible scientific computing on a laptop,
through SSH or with SLURM. Your project owns its models, data and algorithms. LambdaForge handles
validation, input staging, resources, environments, CUDA/PyTorch compatibility, durable jobs,
metrics, artifacts, immutable datasets, seeds, parameter studies, retries, monitoring and cleanup.

```text
ordinary project function + small YAML
                  ↓
      validated materialized Work
                  ↓
     local / SSH / SLURM execution
                  ↓
 result + metrics + artifacts + provenance
```

The normal interface has three ideas: `run` names an installed Python callable, `with` supplies its
keyword arguments, and `resources` states the real reservation. Internal Task, Workflow and
Experiment schemas still provide strict execution and backward compatibility, but a new project
does not need to author them.

> **Status:** pre-1.0. The repository has no licence file, so redistribution terms have not yet
> been granted.

## 1. Install in your research project

Python 3.10 or newer is required. Use the consumer project's environment; never copy LambdaForge's
source, share this repository's `.venv`, or modify `PYTHONPATH`.

```bash
cd /path/to/my-research
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .                # makes my_project.* importable
python -m pip check
lf --version
```

For a released/offline workflow, build and install a wheel instead of an editable checkout:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-*.whl
```

The consumer dependency should allow this release, for example
`lambdaforge[adaptive-hpo]>=0.11,<0.12`. The consumer selects the PyTorch build appropriate for its
hardware. Useful optional extras include `hpo`, `adaptive-hpo`, `s3`, `parquet`, `onnx`, `viz`,
tracking providers, `cluster-password` and `dev`.

Or generate a complete minimal consumer:

```bash
lf init my-research --template minimal
cd my-research
python -m pip install -e .
lf run experiments/task.yaml
```

## 2. First Work

Project Python (`src/my_project/research.py`):

```python
from pathlib import Path
import lambdaforge as lf

def analyse(source: Path, scale: float = 1.0) -> dict[str, float]:
    value = float(source.read_text()) * scale
    report = lf.current().run_dir / "report.txt"
    report.write_text(str(value))
    lf.metric("score", value, step=0)
    lf.artifact("report", report, role="report", media_type="text/plain")
    return {"score": value}
```

Execution YAML (`experiments/analysis.yaml`):

```yaml
name: analysis
run: my_project.research.analyse
with:
  source:
    file: data/value.txt
  scale: 2.0
resources:
  cpu: 2
  memory: 1GiB
  time: 10m
```

Run the safe progression:

```bash
lf validate experiments/analysis.yaml       # structure, import and signature; no work
lf inspect experiments/analysis.yaml         # scientific execution plan; no work
lf inspect experiments/analysis.yaml --resolved  # advanced: strict internal IR
lf run experiments/analysis.yaml --dry-run   # placement/submission plan; no work
lf run experiments/analysis.yaml
```

An ordinary string remains a string. Only exact `{file: PATH}` and `{dataset: NAME@VERSION}` values
are resolved, fingerprinted and staged. Small files (up to 10 MiB) enter a remote bundle
automatically. Large data should be an immutable managed dataset or an explicit institutional
location/transfer; LambdaForge refuses to hide hundreds of gigabytes in a control bundle.

The function return value becomes the final structured output. `lf.metric()` appends durable
history to `metrics.jsonl` and keeps the last value for comparison. `lf.artifact()` accepts a file
or directory below the run root and records its checksum, semantic role and metadata. Calls to
`lf.current`, `lf.metric`, `lf.artifact` or `lf.publish_dataset` outside an active run raise a clear
`RuntimeError`.

## 3. Training, seeds and parameter studies

Training remains normal project code. Use PyTorch, Lightning or another library inside the
function and emit the evidence LambdaForge needs:

```python
def train(dataset, hidden_dim: int, dropout: float, seed: int):
    model = ...
    for epoch in range(100):
        value = ...
        lf.metric("val_auprc", value, step=epoch)
    lf.artifact("best_model", "best.ckpt", role="checkpoint")
    return {"best_val_auprc": value}
```

```yaml
name: wisdom-v1
run: wisdom.training.train
with:
  dataset:
    dataset: wisdom-dna@1
  hidden_dim: 128
  dropout: 0.2
seeds: [7, 17, 27]
objective: {metric: val_auprc, mode: max}
resources: {gpu: 1, cpu: 4, memory: 32GiB}
```

Each seed is a distinct scientific run. LambdaForge seeds Python, NumPy and PyTorch and injects
`seed` only when the function declares it. A finite or bounded parameter study is equally small:

```yaml
name: wisdom-v1-hpo
run: wisdom.training.train
with:
  dataset: {dataset: wisdom-dna@1}
search:
  hidden_dim: {values: [64, 128, 256]}
  dropout: {range: [0.0, 0.5]}
trials: 20
objective: {metric: val_auprc, mode: max}
resources: {gpu: 1, cpu: 4}
```

Finite `values` are expanded exactly. A `range` uses the existing deterministic HPO sampler;
`trials` bounds the design. The Workflow result records the exact trial metrics and best observed
objective. The established advanced adaptive, cost/memory-aware multi-fidelity HPO configuration
remains supported for projects that need racing, pruning and checkpoint continuation.

## 4. Sequential and parallel work

Steps are sequential by default. A `parallel` group runs only after the previous level completes;
the following step waits for every branch.

```yaml
name: complete-study
resources: {cpu: 4}
steps:
  - name: prepare
    run: project.prepare
  - parallel:
      - {name: model-a, run: project.train_a, resources: {gpu: 1, cpu: 4}}
      - {name: model-b, run: project.train_b, resources: {gpu: 1, cpu: 4}}
  - name: compare
    run: project.compare
```

Document resources are per-run defaults; a step's mapping overrides only named fields. LambdaForge
derives the fixed outer request conservatively from the resources of concurrently runnable DAG
levels. In strict compatibility Workflow YAML, an explicit top-level mapping retains its historical
meaning as an exact fixed allocation. CLI flags are one-off overrides, not a second declaration.

The advanced class escape hatch is explicit and avoids constructor/method guessing:

```yaml
run:
  class: project.training.Trainer
  init: {model_size: 128}
  method: train
  with: {epochs: 100}
```

## 5. Publishing and consuming datasets

Dataset creation is Python work, while DatasetVersion remains a strict lifecycle domain:

```python
def build(source):
    def members():
        for row in read_rows(source):
            path = write_member(row, lf.current().run_dir)
            yield {
                "id": row["id"],
                "split": row["split"],
                "targets": {"label": row["label"]},
                "assets": {"features": path},
            }
    return lf.publish_dataset("wisdom-dna", "1", members(), metadata={"source": "public"})
```

Publication streams the index, copies only declared run-owned assets into staging, checks member
IDs, paths and checksums, derives a path-independent content identity, atomically renames the
complete version and then registers its placement. An existing `name@version` with different bytes
is refused. Existing DatasetArtifact v1/v2 and registry records remain readable.

```bash
lf datasets list
lf datasets show wisdom-dna@1 --on atlas
lf datasets members wisdom-dna@1
lf datasets verify wisdom-dna@1 --on atlas
lf datasets delete wisdom-dna@1 --on atlas       # preview
lf datasets delete wisdom-dna@1 --on atlas --apply
```

Legacy `kind: dataset` recipes and `lf datasets build` remain compatibility paths, delegated to the
same runners and publication contracts. New code uses `lf run` plus `lf.publish_dataset`.

## 6. Clusters, Work and monitoring

```bash
lf clusters add atlas --host atlas-login --scheduler slurm --workspace /scratch/me/lf
lf doctor --on atlas
lf clusters bootstrap atlas --dry-run
lf clusters bootstrap atlas
lf run experiments/analysis.yaml --on atlas
lf top
```

Remote submission returns a durable local handle quickly; environment preparation, bounded bundle
transfer and scheduler contact continue in a detached controller. OpenSSH reuses a private
ControlMaster. Managed environments are immutable user-space installations; LambdaForge never
changes drivers, the system CUDA toolkit, shell startup files or system Python.

Work is the normal human view. It groups low-level retry attempts by name, scientific identity and
cluster. The same active Work on the same cluster is refused unless `--allow-duplicate` is
explicit; another cluster is allowed.

```bash
lf status                         # all Work and cluster state
lf show analysis                  # semantic Work details
lf logs analysis --follow         # lifecycle plus scientific stdout/stderr
lf cancel analysis
lf retry analysis
lf top                            # interactive view; overview --json is the machine source
lf jobs list --all                # advanced low-level attempts
```

Failures have stable categories, exit codes, repair commands and a redacted diagnostic record.
Use `--json` for automation, `--verbose` for operational detail and `--debug` only for traceback
internals.

## 7. Safe cleanup

Destructive commands are preview-first:

```bash
lf delete analysis                # exact terminal Work attempts and owned job roots
lf delete analysis --apply
lf clean --on atlas               # reconstructible cache only
lf clean --on atlas --apply
```

Work deletion refuses active executions and preserves published datasets, shared environments,
caches and every other Work. A tiny deletion receipt makes retries idempotent without retaining
scientific outputs. `clean` excludes scientific datasets, results and active references.
Dataset deletion remains a separate exact-version operation because scientific data must never be
mistaken for disposable execution state.

## 8. Documentation and compatibility

- [Complete manual](docs/MANUAL.md): concepts, every CLI, advanced APIs and architecture.
- [Agent operating guide](AGENTS.md): a low-token map so coding agents do not crawl the repository.
- [Release history](CHANGELOG.md) and [security model](SECURITY.md).

Strict pre-0.11 Task, Workflow, Experiment and DatasetRecipe YAML remains accepted. It is the
advanced compatibility surface, not the recommended authoring style. Use public imports from
`lambdaforge` or documented domain namespaces; do not import private implementation files.
