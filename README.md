# LambdaForge

[Español](README.es.md) · English

LambdaForge is a managed scientific runtime for reproducible Python research. You write a normal
Python class, LambdaForge supplies execution context, and the same YAML can run locally, through
SSH, or through SLURM. It records inputs, code, parameters, resources, metrics, outputs, artifacts,
datasets, attempts, logs and environment provenance without putting infrastructure in scientific
code.

## Contents

1. [Install](#install)
2. [First Work](#first-work)
3. [Runtime services](#runtime-services)
4. [Managed scientific infrastructure](#managed-scientific-infrastructure)
5. [Clustering](#clustering)
6. [Reusable neural models](#reusable-neural-models)
7. [Composition and adaptive experiments](#composition-and-adaptive-experiments)
8. [Observe and operate](#observe-and-operate)

## Install

Use a virtual environment owned by the research project. Install a release wheel in reproducible
projects, or an editable checkout while developing LambdaForge:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.13.0
python -m pip install -e .
python -m pip check
lf --version
```

`lf init my-study` creates a complete installable example. LambdaForge requires Python 3.10 or
newer. A consumer project owns its scientific dependencies and its PyTorch build; managed cluster
bootstrap resolves a compatible remote Python/Torch environment without changing system Python,
drivers or CUDA.

Projects that also need native executables can declare them once in `pyproject.toml`; ordinary
pip-only projects do not need Conda or any additional configuration:

```toml
[tool.lambdaforge.environment]
manager = "conda"
file = "environment.yml"
required_executables = ["mmseqs", "foldseek"]
```

```yml
channels: [conda-forge, bioconda]
dependencies: [python=3.11, pip, mmseqs2, foldseek]
```

Prepare that exact consumer project with `lf clusters bootstrap gpu --project . --dry-run`, review
the detected packages/platform/connectivity, then omit `--dry-run`. LambdaForge uses its verified
micromamba, creates one immutable Conda prefix, installs the exact LambdaForge and consumer wheels
with that prefix's Python, verifies the tools and records package/build/channel provenance. A later
`lf run ... --on gpu` discovers the same project declaration automatically. Inside the Work,
`self.tools.require("mmseqs", version_args=["version"])` only verifies/resolves the prepared tool;
it never installs software during scientific execution. See the manual for exact offline locks.

## First Work

A YAML file executes only a class inheriting `lambdaforge.Work`. `run()` is its single lifecycle
method and its annotated Python signature is the source of parameter names, defaults and types.

```python
from pathlib import Path

import lambdaforge as lf


class CurateDNA(lf.Work):
    """Curate source records and keep a reproducible report."""

    def run(self, public_sources: Path, workers: int = 8) -> dict[str, int]:
        records = public_sources.read_text(encoding="utf-8").splitlines()
        curated = self.map(records, str.strip, workers=workers, executor="thread")
        report = self.outputs.file("report", filename="report.txt", role="report")
        report.write_text("\n".join(curated))
        self.metrics.log("accepted", len(curated))
        return {"accepted": len(curated)}
```

```yaml
name: wisdom-dna-curate
run: wisdom.preprocessing.CurateDNA
with:
  public_sources:
    file: ../data/dna/public-sources.json
  workers: 8
resources:
  cpu: 8
  memory: 8GiB
  time: 2h
```

Only `{file: ...}`, `{dataset: ...}` and `{from: step.output}` have special meaning. A plain YAML
string is always a string. Small typed files/directories are hashed and automatically staged for
remote work. Large project inputs may instead use an explicit remote project mirror; stable shared
corpora should normally be published/materialized as immutable datasets rather than copied into
every bundle.

Inputs and outputs deliberately use different APIs. `with: {source: {file: ...}}` declares bytes
that must exist *before* execution, so LambdaForge can hash and stage them. A Work creates new
results through `self.outputs`; no YAML output path is required. Managed outputs stay attached to
the Attempt and can additionally publish a copy to an explicit researcher path, as explained
below.

Validate and understand the Work before executing it:

```bash
lf validate experiments/curate.yaml
lf explain experiments/curate.yaml
lf run experiments/curate.yaml --dry-run
lf run experiments/curate.yaml
lf run experiments/curate.yaml --on gpu-cluster
```

For a cluster that already has a partial project mirror, configure its absolute remote root once:

```bash
lf clusters set gpu-cluster project_root /scratch/USER/WISDOM
lf doctor --on gpu-cluster
```

The same `../data/dna/design` marker then means `PROJECT/data/dna/design` locally and
`/scratch/USER/WISDOM/data/dna/design` remotely. Inputs up to the 10 MiB bundle limit remain
automatic immutable snapshots. A larger input is never copied implicitly: it must be inside the
local project, the matching remote path must already exist, and LambdaForge compares kind, byte
count and SHA-256 before submission and again in the worker. Missing or stale content fails safely.
Synchronizing the mirror with the site's recommended transfer service remains an explicit
researcher operation.
For hundreds of GB/TB, prefer a managed dataset: exact mirror verification also reads all bytes,
while a dataset gives reusable content identity and placements.

Submission is asynchronous by default on every target, including `local`: the terminal returns
after a durable preparation record is created, and `lf top`/`lf logs` reconnect to it. Use
`--wait-for-submit` only when the caller deliberately wants to wait for preparation and scheduler
acknowledgement. `--dry-run` remains read-only and runs directly because it starts no Job.

## Runtime services

Inside `run()`, immutable planning information lives on `self.config`, `self.inputs`,
`self.resources`, `self.seed`, `self.trial`, `self.source_dir` and `self.resuming`. Managed mutable
services are `self.outputs`, `self.metrics`, `self.checkpoints`, `self.cache`, `self.tools` and
`self.progress`.
`self.run_dir` owns durable attempt files and `self.temp_dir` is deleted after the attempt.

- `self.outputs.value(name, value)` stores small JSON evidence.
- `self.outputs.file/directory(...)` allocates, verifies, hashes and automatically registers durable
  artifacts. `publish_to=...` publishes the verified result to a researcher-owned path. Once the
  whole Execution is safely recorded, redundant internal bytes are removed by default;
  `retain_internal=True` deliberately keeps both copies. `artifact(...)` remains an advanced
  importer for an existing path.
- `self.outputs.dataset(name=..., version=..., members=...)` streams and atomically publishes an
  immutable dataset version.
- `self.metrics.log(...)` and `log_many(...)` append scalar histories used by results and search
  objectives.
- `self.log("message", level="info")` emits an immediately visible timestamped log line.
- `self.checkpoints.file(...)` atomically builds/reuses a validated file;
  `save_json/load_json/exists` manages small sequential state shared by compatible Attempts.
- `self.cache.put(key, content)` and `get(key, default=None)` are the ordinary cache API for bytes,
  text and strict JSON. `file/fetch(...)` handles large or library-produced files;
  `cache.path` is only the advanced raw-layout escape hatch.
- `self.tools.require/run(...)` resolves and executes external tools without a shell, streams output,
  scopes thread variables and records version provenance.
- `self.map(...)` provides bounded ordered intra-job concurrency and retries without persistence.
  `self.resume_map(..., key=...)` explicitly adds per-item JSON checkpoints and managed cache
  dependencies. Existing `map(..., key=...)` calls retain that resumable behavior.

The return value is the primary result; named outputs are separate. An unhandled exception, invalid
artifact, failed dataset publication or failed result persistence makes the Attempt fail.
Ordinary `print()` and Python `logging` also reach scheduler logs. `print(..., flush=True)` is enough
for live messages; prefer `self.log()` when a consistent timestamp and severity are useful, and use
`self.progress.update(...)` for numeric completion rather than repeatedly printing percentages.

## Managed scientific infrastructure

Use cache for reproducible bytes that can always be rebuilt, checkpoints for state needed to resume
the same Run, and outputs for durable scientific evidence:

```python
# Small reconstructible values: no paths, codecs or callbacks are required.
self.cache.put("curation-summary", {"accepted": 1842, "version": 3})
summary = self.cache.get("curation-summary")

# Ordinary parallel work has no cache or resume side effect.
normalized = self.map(records, normalize, workers=8)
```

Use the file and resumable APIs only when the operation actually needs them:

```python
from pathlib import Path


def run(self, identifiers: list[dict[str, str]], workers: int = 8):
    service = self.cache.rate_limit("archive", requests_per_second=4)

    def prepare(item: dict[str, str]):
        return self.cache.fetch(
            f"https://archive.example/{item['id']}.json.gz",
            key=f"archive/{item['id']}.json",
            retries=4,
            decompress="gzip",
            rate_limit=service,
            validate=lambda file: file.size_bytes > 0,
        )

    files = self.resume_map(
        identifiers,
        prepare,
        key="id",
        workers=workers,
        executor="thread",
        name="archive-downloads",
        retries=2,
    )
    index = self.checkpoints.file(
        "indexes/archive.tsv",
        build=lambda target: build_index(files, target),
        validate=is_valid_index,
    )
    report = self.outputs.file(
        "report",
        filename="report.json",
        role="report",
        media_type="application/json",
        publish_to="results/archive-report.json",
    )
    report.write_json({"files": len(files), "index_sha256": index.sha256})
    evidence = self.outputs.directory("evidence", role="evidence")
    write_figures(Path(evidence), files)
    return {"files": len(files)}
```

`ManagedFile` implements `os.PathLike`, `str(file)`, `open`, `read_text` and `read_bytes`, while
exposing immutable `key`, `sha256`, `size_bytes` and metadata. Cache publication uses a per-key
cross-process lock, a temporary sibling, validation, `fsync`, SHA-256 and atomic replacement.
`cache.fetch` adds bounded timeout, exponential retry, optional gzip decoding and a thread-safe
Work-local rate limit. Truncated HTTP/chunked/gzip transfers and transient server responses retry
from a clean unpublished temporary; permanent 4xx errors do not loop, and partial bytes are never
registered. No absolute cache path enters resumable-map state or scientific identity. `lf clean`
previews removal of these reconstructible entries and `--apply` removes them only while no Work
holds the cache lease.

Relative `publish_to` is resolved from the directory containing the authored YAML. With the cluster
`project_root` above, `publish_to="../data/report.json"` therefore publishes below the equivalent
remote project directory instead of an internal Job hash path. Without `project_root`, a remote
relative publication is rejected; use an explicit absolute remote path. The managed artifact
is registered before compaction; publication happens only after `run()` returns successfully, is
atomic per destination and refuses existing different content unless `overwrite=True`. The result
envelope keeps name, hash, size and published path, but the Job-owned duplicate is removed only
after that external copy is re-verified. Set `retain_internal=True` only when two copies are
intentional. Failed or interrupted Attempts discard partial managed `artifacts/` while retaining
logs, metrics, provenance, checkpoints and failure records. Unpublished successful artifacts stay
internal because they have no other durable copy. Direct writes through the advanced `run_dir`
escape are not guessed or deleted. LambdaForge never pretends a remote path is controller-local or
silently transfers large output trees back.

External command boilerplate follows the same rule:

```python
tool = self.tools.require("mmseqs", version_args=["version"])
result = self.tools.run(
    [tool, "easy-search", str(query), str(database), str(output)],
    name="MMseqs2",
    threads=self.resources.cpu,
    cwd=self.temp_dir,
)
```

Arguments are an argv sequence and never pass through `shell=True`. Stdout/stderr are streamed to
Work logs and retained with a bounded capture; nonzero exit raises an error containing the command
result. Thread limits affect only the child environment. Required/executed tool paths and explicit
version probes live once in `environment.json`.

## Clustering

Install the optional mature backend and use the uniform Python contract:

```bash
python -m pip install "lambdaforge[clustering]==0.13.0"
```

```python
import lambdaforge as lf

result = lf.clustering.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    distance="euclidean",
    threads=8,
).cluster(features)

self.metrics.log("clusters", result.n_clusters)
```

`KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN` and `Agglomerative` all implement
`Clusterer.cluster(X) -> ClusteringResult` for finite NumPy arrays or detached PyTorch tensors with
shape `[N, F]`. Results consistently expose immutable labels, cluster/noise counts and diagnostics;
centers/inertia/probabilities exist only when the algorithm provides them. The backend is
scikit-learn >=1.3 and is imported only when clustering runs.

Scientific compatibility is explicit: KMeans variants accept only Euclidean/squared-Euclidean
semantics; Ward linkage requires Euclidean distance; density and non-Ward algorithms accept their
declared native metrics. An existing `lambdaforge.nn.distances.Distance` object uses a precomputed
matrix only for algorithms that support it, after an explicit configurable O(N²) memory guard.
LambdaForge never standardizes, imputes, reduces dimensions or interprets cluster stability for the
researcher. `adjusted_rand_index`, `silhouette_score` and `stability` only produce evidence.

## Reusable neural models

LambdaForge already includes the common model families; project code imports concrete classes from
`lambdaforge.nn.models` and keeps domain-specific heads/data handling explicit. The catalog includes
MLP and CNN2D; GCN, GAT/GATv2, GIN, GraphSAGE, PNA, relational GCN, graph transformers, EGNN and
tensor-field networks; RNN/GRU/LSTM, temporal convolution, Transformer and Conformer models;
DeepSets/SetTransformer; residual MLP, FT-Transformer, TabNet, SAINT, AutoInt and DeepFM; ResNet,
U-Net, MobileNet, ConvNeXt, ViT and feature pyramids; autoencoders, mixture/ensemble/multitask
composition, diffusion/VQ-VAE; neural ODE/CDE, DeepONet, FNO and SIREN. Losses, metrics,
activations, normalizations, pooling, distances, kernels and uncertainty helpers live in their
matching `lambdaforge.nn` namespaces. The manual gives the family/import map; LambdaForge does not
add a vague generic `GNN` alias because graph topology, aggregation and equivariance are scientific
choices.

## Composition and adaptive experiments

YAML supports a sequence plus explicit parallel groups. The next level waits for the complete prior
level. Each member of a parallel group uses an isolated spawned process inside the enclosing Job;
seed/search Runs for one member remain serial so the declared resource request stays absolute. A
reference requires one unambiguous producing Run.

```yaml
name: study
steps:
  - name: prepare
    run: project.Prepare
  - parallel:
      - name: model-a
        run: project.TrainA
        with:
          data: {from: prepare.dataset}
      - name: model-b
        run: project.TrainB
        with:
          data: {from: prepare.dataset}
  - name: compare
    run: project.Compare
```

`seeds: [7, 17, 27]` creates independent Runs and exposes each value as `self.seed`. `search`
expands finite values or reproducible numeric ranges; `objective: {metric: val_score, mode: max}`
ranks the exact metric logged by the Work. With an objective and several seeds, the default strategy
is adaptive successive halving: every candidate first receives `min_seeds`, only the most promising
fraction receives more seeds, and ranking uses a conservative mean/standard-error score instead of
the luckiest seed. Use `strategy: exhaustive` when every variant/seed pair is required.

One scheduler Job owns the fixed resource reservation. Adaptive Runs are independent spawned
processes inside it. `resources.gpu` is the number of GPUs reserved for the whole study and
`runs_per_gpu` is the explicit packing factor:

```yaml
name: adaptive-training
run: my_project.Training
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  strategy: adaptive
  trials: 40
  min_seeds: 1
  reduction_factor: 2
  runs_per_gpu: 4
  early_stopping: {enabled: true, min_step: 5}
  learning_rate: {range: [0.00001, 0.003], scale: log}
  hidden_dim: {values: [64, 128, 256]}
objective: {metric: val_auprc, mode: max}
resources:
  gpu: 2
  gpu_memory: 16GiB  # bound for each independent Run
  cpu: 16            # total reservation, divided among active Runs
  memory: 32GiB      # total reservation, divided among active Runs
```

This permits at most eight simultaneous trainings, four on each of two GPUs. Six GPUs with
`runs_per_gpu: 2` permit twelve; `runs_per_gpu: 1` gives one-Run-per-GPU isolation. Packing above
one requires `gpu_memory`; LambdaForge checks `runs_per_gpu × gpu_memory` against currently free
memory before launching children. CPU-only adaptive studies can bound concurrency with
`max_parallel`. Repeated `self.metrics.log("val_auprc", value, step=epoch)` observations enable
in-training pruning; `LightningRunner` bridges its validation metric and stop request automatically.
A custom loop must log the objective with `step=` and return at a safe checkpoint boundary when
`self.stop_requested` becomes true. With only a final metric, seed allocation remains adaptive but
the current training cannot stop early.

Concurrent training does not require reading one interleaved Job stream. A study is not a special
Work type: any normal Work declaring `search` or multiple `seeds` is marked as a study during local
validation, so `lf top` exposes its study screen even while remote preparation is still running.
It drills down as `Work → Trial (parameter combination) → Seed Run → live dashboard`. The Trial
screen shows which combinations are pending, active, promoted, eliminated or complete. The Run
screen isolates that process's parameters and log, follows it automatically, and renders bounded
learning curves plus latest scalar, epoch-time and validation-time observations. `LightningRunner`
publishes its scalar callback metrics and timing automatically. A custom training loop uses the same
generic API:

A preprocessing Work, a multi-step workflow or a Work using `self.map()` remains an ordinary Work
unless its YAML actually declares a parameter search or repeated seeds. Enter/right therefore opens
its numbered Attempt and normal combined log. There is no manual “training type” to keep in sync.

```python
for epoch in range(epochs):
    train_loss, val_score = train_one_epoch(epoch)
    self.metrics.log("train_loss", train_loss, step=epoch + 1)
    self.metrics.log("val_auprc", val_score, step=epoch + 1)
    self.progress.update(epoch + 1, epochs, message="training")
    self.log(f"epoch {epoch + 1} completed")  # optional human narration
```

`print`, Python logging and `self.log()` remain process-local evidence; the Run dashboard therefore
shows only the selected seed even when many children run simultaneously. Automation reads the same
facts through `lf overview --json`, `lf show WORK --json`,
`lf show WORK --run trial-00001-seed-4 --json` and
`lf logs WORK --run trial-00001-seed-4 --tail 300`. The live index stores only compact state and
scalar JSONL; it references existing Run logs/results and down-samples curves when reading them, so
it does not duplicate checkpoints, models or large outputs.

GPU admission belongs to the cluster profile, not scientific YAML. `gpu_access.mode=auto` uses
SLURM allocation on SLURM clusters and conservative exclusive LambdaForge leases on direct hosts.
Use `shared` only on a deliberately permissive direct host, or `command` with an argv prefix when
the site requires a claim wrapper:

```bash
lf clusters set free-gpu gpu_access.mode shared
lf clusters set claimed-gpu gpu_access '{mode: command, command_prefix: [gpu, run, --]}'
```

The shared mode admits external occupancy while LambdaForge Jobs still coordinate with each other.
Because occupancy can change after preflight, `gpu_memory` reduces risk but cannot turn a shared
site into hard isolation. Resource fields in YAML remain the absolute outer scheduler request, not
capacity hints.

## Observe and operate

```bash
lf top
lf overview --json
lf show WORK
lf logs WORK --follow
lf cancel WORK
lf retry WORK
lf jobs list                 # advanced scheduler/process view
lf jobs clear                # preview terminal-history cleanup
lf jobs clear --apply
lf results list
lf datasets list
lf clean                     # preview only
lf clean --apply
```

`lf top` leads with clusters and semantic Works and does not put long operational Job IDs in the
researcher's primary path. Up/down selects; Enter/right drills into a study's Trials, seed Runs,
curves and isolated live log. A non-study Work drills into numbered scheduler Attempts and then its
complete log; press `a` from a study to inspect those outer Attempts. Open logs refresh
automatically, follow the end by default and preserve manual scroll. Left returns. Job IDs remain
available through `lf jobs ...` and in `lf overview --json` for automation and low-level diagnosis.
Press `d` to permanently delete the selected terminal Work/Job after confirmation; `D` clears all
terminal history while always preserving active Jobs. Both operations remove exact owned
workspaces and local history, never published datasets or shared caches/environments. The same
whole-history operation is machine-accessible through preview-first `lf jobs clear [--apply]`.
`lf cancel WORK` and `x` on a Work are semantic cancellation: every active scheduler Job grouped
under that Work is contacted, and each direct supervisor stops the verified process group plus
reparented/session-owning processes carrying its unique Job identity. Cancellation succeeds only
after no such process remains. Cancelling one explicitly selected Attempt or `lf jobs cancel JOB`
keeps the narrower low-level meaning. SLURM cancellation delegates the complete allocation to its
configured scheduler command. Repeating `lf cancel WORK` also reconciles direct Jobs already marked
cancelled, which safely cleans orphan workers left by older LambdaForge releases.
Failed Work logs append the structured scientific exception even when `--tail` removed its original
stdout lines. `--verbose`/`--debug` includes the persisted traceback and `--json` returns structured
failure fields plus the exact result path. Human cluster bootstrap prints phases and periodic
liveness updates to stderr; JSON output remains clean.
Normal execution reuses a verified successful scientific definition. `retry` creates a new Attempt
of the same Run, checkpoints make it resumable, and `--rerun` deliberately creates a new Execution.
Cleanup is preview-first. `lf clean` includes reconstructible cache and hash-verified redundant or
partial managed artifacts from terminal Attempts, but never published datasets, results,
checkpoints, active work or an unpublished successful output. Superseded managed environments are removed automatically after a verified
replacement becomes active, except environments referenced by live Jobs. This preserves immutable
reuse without accumulating one multi-gigabyte prefix per historical dependency identity.

For clusters, datasets, search, result metadata, cleanup ownership and the complete CLI, read
[the manual](docs/MANUAL.md) ([Spanish](docs/MANUAL.es.md)). Agent-assisted projects should give
their agent [AGENTS.md](AGENTS.md)
as the concise operational contract; this avoids costly repository crawling and prevents invented
APIs.
