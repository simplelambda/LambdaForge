# LambdaForge

[Español](README.es.md) · English

LambdaForge is a managed scientific runtime for reproducible Python research. You write a normal
Python class, LambdaForge supplies execution context, and the same YAML can run locally, through
SSH, or through SLURM. It records inputs, code, parameters, resources, metrics, outputs, artifacts,
datasets, attempts, logs and environment provenance without putting infrastructure in scientific
code.

## Install

Use a virtual environment owned by the research project. Install a release wheel in reproducible
projects, or an editable checkout while developing LambdaForge:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.12.0
python -m pip install -e .
python -m pip check
lf --version
```

`lf init my-study` creates a complete installable example. LambdaForge requires Python 3.10 or
newer. A consumer project owns its scientific dependencies and its PyTorch build; managed cluster
bootstrap resolves a compatible remote Python/Torch environment without changing system Python,
drivers or CUDA.

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
string is always a string. Small typed files are hashed and automatically staged for remote work;
large data should be published/materialized as an immutable dataset instead of being copied into
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
  artifacts. `publish_to=...` optionally copies the verified result to a researcher-owned path;
  `artifact(...)` remains an advanced importer for an existing path.
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
Work-local rate limit. No absolute cache path enters resumable-map state or scientific identity. `lf clean`
previews removal of these reconstructible entries and `--apply` removes them only while no Work
holds the cache lease.

`publish_to` is resolved relative to `self.source_dir`, or accepts an explicit absolute path. The
managed artifact remains the authoritative result; publication happens only after `run()` returns
successfully, is atomic per destination and refuses an existing different path unless
`overwrite=True`. On a remote Job, paths belong to the remote execution host: use an absolute path
on persistent cluster storage for a durable remote copy. LambdaForge never pretends a remote path
is a controller-local path or silently transfers large output trees.

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
python -m pip install "lambdaforge[clustering]==0.12.0"
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

## Composition and experiments

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
ranks the exact metric logged by the Work, averaging all seeds of the same variant. Resource fields
in YAML are the scheduler request, not capacity hints.

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
researcher's primary path. Up/down selects; Enter/right drills from Work to numbered Attempts and
then complete logs, while cluster detail also uses Work/Attempt labels. Left returns. Job IDs remain
available through `lf jobs ...` and in `lf overview --json` for automation and low-level diagnosis.
Press `d` to permanently delete the selected terminal Work/Job after confirmation; `D` clears all
terminal history while always preserving active Jobs. Both operations remove exact owned
workspaces and local history, never published datasets or shared caches/environments. The same
whole-history operation is machine-accessible through preview-first `lf jobs clear [--apply]`.
Normal execution reuses a verified successful scientific definition. `retry` creates a new Attempt
of the same Run, checkpoints make it resumable, and `--rerun` deliberately creates a new Execution.
Cleanup is preview-first and never treats published datasets, results or checkpoints as
reconstructible cache.

For clusters, datasets, search, result metadata, cleanup ownership and the complete CLI, read
[the manual](docs/MANUAL.md) ([Spanish](docs/MANUAL.es.md)). Agent-assisted projects should give
their agent [AGENTS.md](AGENTS.md)
as the concise operational contract; this avoids costly repository crawling and prevents invented
APIs.
