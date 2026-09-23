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
9. [Study Analysis](#study-analysis)
10. [Research Console](#research-console)

## Install

Use a virtual environment owned by the research project. Install a release wheel in reproducible
projects, or an editable checkout while developing LambdaForge:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.15.0
python -m pip install -e .
python -m pip check
lf --version
```

`lf init my-study` creates a complete installable example. LambdaForge requires Python 3.10 or
newer. A consumer project owns its scientific dependencies and its PyTorch build; managed cluster
bootstrap resolves a compatible remote Python/Torch environment without changing system Python,
drivers or CUDA.

CLI help is available both in conventional and command-first forms. These are equivalent and exit
successfully, including when `CommandLineInterface.main()` is embedded by another application:

```bash
lf --help
lf help
lf run --help
lf help clusters add
```

Run bare `lf` in an interactive terminal to open the Research Console. Its Clusters screen adds and
edits profiles with contextual explanations, secure credential handling and explicit test,
bootstrap and doctor steps. Scripts continue to use `lf clusters add/set/unset/credentials/test`
directly. `lf` with redirected input/output, or `lf --json` without a command, prints CLI help and
never tries to start a full-screen application. The former `lf top`, `lf clusters setup` and
`lf clusters modify` entry points were removed in 0.14 so there is only one interactive UI.

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

Working on several projects? Run `lf` inside each project's directory (or a subdirectory).
`lf project --json` shows the selected root and ID. Virtual environments choose Python dependencies;
the nearest `pyproject.toml` chooses the LambdaForge project. Jobs, Works, Studies and recent YAMLs
are filtered by that project; local results and dataset indexes stay below its `.lambdaforge`.
Cluster profiles and credentials can be shared. New remote jobs, registry state and caches live
under `<workspace>/.lambdaforge/projects/<project-id>/`. GPU/resource leases remain shared, so two
projects still compete for the same hardware safely.

`lf init` writes a stable ID automatically. In an existing project, configure one **before the
first submission** if the same remote namespace must survive checkout moves:

```toml
[tool.lambdaforge]
project_id = "my-research-project"
```

Otherwise LambdaForge derives a readable ID from the resolved local root; identical folder/package
names at different locations remain separate. Reusing an explicit ID deliberately shares the remote
namespace. Use a project-local `lambdaforge.clusters.yaml` to override `project_root` for that
project's remote mirror without repeating host/authentication fields. See
[project isolation and upgrade behavior](docs/MANUAL.md#18-project-isolation).

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
count and SHA-256 before submission and again in the worker. Directory SHA-256 uses a versioned
canonical tree identity: path components are Unicode-normalized and byte-sorted, `/` is the
logical separator, record boundaries and empty directories are explicit, and host timestamps or
permissions are excluded. It therefore does not depend on `find`, shell `sort`, locale or directory
creation order. Missing, changing or stale content fails safely.
Synchronizing the mirror with the site's recommended transfer service remains an explicit
researcher operation.
For hundreds of GB/TB, prefer a managed dataset: exact mirror verification also reads all bytes,
while a dataset gives reusable content identity and placements.

Submission is asynchronous by default on every target, including `local`: the terminal returns
after a durable preparation record is created, and the Research Console/`lf logs` reconnect to it. Use
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
python -m pip install "lambdaforge[clustering]==0.15.0"
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

The recommended YAML describes scientific intent and the external reservation; LambdaForge resolves
the internal policy. Omitted settings are not uncontrolled: they select a versioned automatic
policy which is persisted before execution and can be inspected without launching anything:

```bash
lf config resolve experiments/train.yaml
lf config resolve experiments/train.yaml --format yaml
lf seeds --count 20
lf seeds --role confirmation --count 10
```

The minimal adaptive form is:

```yaml
run: my_project.Training
with:
  dataset: {dataset: wisdom-dna@5}
  epochs: 500
  patience: 30
search:
  space:
    hidden_dim: [64, 128, 256]
    layers: [1, 2, 3]
    dropout: {range: [0.0, 0.5]}
    learning_rate: {range: [0.00001, 0.003], scale: log}
objective: val_auprc
resources:
  gpu: 3
  time: 168h
```

The filename supplies `name` when omitted. Defaults are `goal: balanced`, one required initial
replicate per proposed candidate, geometry-derived startup, automatic sampler/pruning, incremental
deterministic candidate generation, adaptive extra replication, disjoint fresh confirmation,
scientific convergence, and automatic packing/global concurrency. `resources.gpu` stays explicit
because it reserves externally owned hardware. Known objectives such as `val_auprc`, `accuracy`,
`balanced_accuracy`, `f1`, `auroc`, `mcc`, `kappa` and `val_loss` have registered direction (and a
finite range where defined); custom metrics still require `{metric: ..., mode: max|min}`.

Every project has domain-separated, versioned `replicate[n]` and `confirmation[n]` seed streams.
Candidates share replicate ordinals for paired comparisons; confirmation never reuses selection
evidence. Runs persist the actual integer, ordinal, role, namespace and generator version.
`seeds: [4, 7, 32]` is an exact finite override; `replicates: 10` selects the first ten project
ordinals. The automatic stream is prefix-stable and provides more than two billion collision-free
32-bit-compatible values per role.

A sweep fixes its cells. Without a replication count it opens complete shared-seed blocks
sequentially and uses simultaneous time-uniform paired confidence sequences for the primary stop
decision—not repeatedly inspected fixed-sample intervals. HPO pruning and per-cell seed racing are
disabled. A missing permanent cell remains visibly incomplete. Practical equivalence is possible
only when the researcher authored `objective.practical_margin`.

```yaml
run: my_project.Training
sweep:
  space:
    optimization_profile: [baseline, warmup, clipping, ema, swa]
objective: val_auprc
resources: {gpu: 3, time: 72h}
```

Use `sweep.replicates: 10` for exactly ten shared blocks. Advanced caps and overrides live under
`search.budget`, `search.replication`, `search.pruning`, `search.stop` and top-level `execution`.
`search.goal` is `optimize`, `balanced` (default), or `understand`; it changes how the same planner
values optimization versus unresolved supported questions. A hard budget is a safety ceiling, not
evidence of convergence. Automatic stopping persists its state and exact reason; execution may be
complete while scientific status remains unresolved.

Adaptive GPU admission refills from the same scientific queue on both Run completion and newly
usable capacity: deferred initial candidates/seeds must not leave an idle granted GPU waiting for
another Run to finish. Limits are ceilings, not guaranteed speedups. Every adaptive child caps native
CPU thread pools to its CPU share (not the whole Job allocation). Custom Work loops reporting
`self.metrics.log(name, value, step=epoch)` also publish bounded resource-progress evidence; no
Lightning callback is required. Compare epoch, training and validation times before raising packing:
free VRAM does not imply spare CPU or faster aggregate throughput.

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
expands finite values or reproducible numeric ranges; `objective` defines either one logged metric
or an explicit fixed-range composite utility. Whenever `search` has an `objective`, omitting `strategy`
enables the complete safe adaptive policy: Sobol startup, result-dependent proposals, probabilistic
seed racing, curve pruning, convergence detection and fresh-seed confirmation. A scrambled Sobol
pool supplies reproducible candidates. When `startup_trials` is omitted, LambdaForge derives an
`InitialDesignPlan` from the authored `ParameterSpace`: it greedily preserves the pool's useful
main-effect/curvature rank, covers available categorical levels, conditional active/inactive
states and numeric low/interior/high support, then uses deterministic D-optimal/maximin
tie-breaking. Its anchor count never depends on GPUs, `runs_per_gpu` or `max_parallel`. An explicit
`startup_trials` remains an authoritative anchor budget, while retaining the same geometric
selection. After that evidence starts arriving, observed results choose later candidates.
`sampler: auto` uses optional
BoTorch mixed-GP qLogNEI when `lambdaforge[adaptive-hpo]` is installed and enough evidence exists,
with a deterministic k-NN fallback for missing dependencies or numerical failures. Only proposed
Trials appear in the Research Console. The GP sees all encoded dimensions jointly, including conditional
activity and categorical choices, and qLogNEI incorporates the standard error observed across
seeds. Fidelity is explicit: each `(candidate, exact rung)` becomes a separate observation,
aggregated only across seeds that reached that same cumulative budget. Numeric spaces use
BoTorch's multi-fidelity GP; mixed spaces retain normalized fidelity in the joint mixed GP; the
k-NN fallback also weights exact-rung evidence. LambdaForge never labels a mean of heterogeneous
seed fidelities as a full-budget result. Per-parameter HPO panels are descriptive marginal
explanations, not the decision model. The maintained architecture is a fidelity-aware surrogate
plus an external cost-aware action scheduler, not a claim of one universal multi-fidelity Bayesian
acquisition over every mixed/conditional space.

The HPO decision authority is always one auditable scalar utility. The compact legacy form is
`objective: {metric: val_score, mode: max}`. When scientific quality genuinely depends on several
metrics, define fixed ranges and weights instead of letting observed candidates move the scale:

```text
objective:
  aggregation: geometric
  metrics:
    val_auprc: {mode: max, weight: 3, range: [0.0, 1.0]}
    val_balanced_accuracy: {mode: max, weight: 1, range: [0.0, 1.0]}
```

`weighted_mean`, `geometric` and `chebyshev` are available. Weights are normalized once; values
outside a declared range are clipped. All components must be logged at the same integer `step`:
LambdaForge never constructs a utility from unrelated best epochs or from independent latest
values. The resulting utility governs candidate proposals, seed racing, pruning, fidelity,
confirmation and final ranking. Raw components remain visible, and the Research Console marks the current
non-dominated Pareto set as a diagnostic only; Pareto status does not silently replace the utility.

If a high utility can be scientifically
misleading on its own, declare explicit outcome guardrails instead of expecting LambdaForge to
guess which other metrics matter. Each bound is evaluated at the objective's best checkpoint.
Candidate feasibility aggregates those same-checkpoint values across completed seeds with explicit
`seed_aggregation: mean` (legacy default), `worst`, or `lcb`; `lcb` also accepts `confidence` and
fails closed until repeated seed evidence can estimate uncertainty. A metric may simultaneously be
a composite-utility component and a hard constraint. Missing guardrail evidence fails closed, and
infeasible candidates remain visible but cannot guide the surrogate or win selection:

```yaml
name: guarded-training
run: my_project.Training
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55, seed_aggregation: worst}
    val_kappa: {min: 0.05, seed_aggregation: lcb, confidence: 0.95}
```

This is constrained single-objective optimization, not an implicit multi-objective compromise.
Use thresholds that express the actual scientific validity of a Run; do not add every logged
metric merely because it exists.

Beyond the authored `replication.minimum`, seed counts are probability-driven rather than equal or
fixed by a halving rung. Once a candidate is proposed, that minimum is a real evidence obligation:
its distinct seed identities may wait for resources but cannot be silently superseded. A
performance-pruned seed counts as attempted censored evidence, not as a full response; an
infrastructure failure counts only after its retry policy is exhausted. Additional seeds remain
adaptive. Shared seed order enables paired differences; an optional seed is useful while its
probability of being within the objective's `practical_margin` of the incumbent is sufficiently
high. The final
winner uses a conservative search bound and then disjoint confirmation evidence over a frozen
posterior/practically competitive contender set. By default LambdaForge starts each candidate with
`replicate[0]`, requests further ordinals only when useful and draws shared fresh evidence from the
project confirmation stream. Legacy `confirmation_seeds` and `confirmation_top_k` remain advanced
explicit overrides.
Confirmation Runs never receive performance pruning or opportunistic preemption. If a required
seed fails or cannot run within the global budget, `summary.confirmation.status` is `incomplete`,
`confirmation_incomplete` is true and no lucky subset of surviving confirmation seeds is selected.

HPO is deliberately easy to replace with a fixed experimental design. A `sweep` fixes every
authored combination. Explicit seeds/replicates make their Cartesian product required evidence;
otherwise complete shared-seed blocks are opened sequentially. ARI may reorder, wait, pack,
checkpoint and recover those Runs, but scientific replanning cannot discard them. Work-internal
training patience remains active; it is distinct from HPO pruning. A numeric grid needs `points`,
so it remains finite and deterministic. `search.strategy: exhaustive` is a compatible alias for
the same design and dispatcher, not a serial implementation.

```yaml
name: optimizer-sweep
run: my_project.Training
seeds: [7, 17]
sweep:
  space:
    optimizer: [adamw, sgd]
    momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
    learning_rate: {range: [0.00001, 0.001], points: 5, scale: log}
  reference: {optimizer: adamw}
execution:
  runs_per_gpu: auto
  max_parallel: auto
```

One scheduler Job owns the fixed resource reservation. Every child Run is an independent spawned
process inside it. `resources.gpu` is the number of GPUs reserved for the whole study, while the
top-level `execution` block is the canonical operational policy. Fixed sweeps and adaptive search
share the same ARI resource dispatcher:

```yaml
name: adaptive-training
run: my_project.Training
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  budget: {candidates: 40, runs: 180}
  proposal_pool_size: 640
  replication:
    minimum: 1
    confirmation: [1001, 1002, 1003]
  pruning:
    enabled: true
    min_step: 5
    confirmations: 2
    probability_threshold: 0.05
  # Optional exact scientific-anchor budget; omitted means geometry-derived, never hardware-derived.
  startup_trials: 10
  seed_racing: {probability_threshold: 0.1}
  confirmation_top_k: 2
  sampler: auto
  # Optional: stop after 8 full-fidelity results without a 0.0005 record gain.
  # Legacy record-streak controls; automatic scientific convergence is the normal default.
  convergence_patience: 8
  min_improvement: 0.0005
  space:
    learning_rate: {range: [0.00001, 0.003], scale: log}
    hidden_dim: [64, 128, 256]
execution:
  runs_per_gpu: auto  # or a positive integer hard cap per GPU
  max_parallel: auto  # or a positive integer hard global cap
  failure_retries: 1
  max_time: 12h
objective: {metric: val_auprc, mode: max, practical_margin: 0.002}
resources:
  gpu: 2
  gpu_memory: 16GiB  # minimum currently-free VRAM required before starting a Run
  cpu: 16            # total reservation; stable share per possible concurrent Run
  memory: 32GiB      # total reservation; Run shares never exceed this outer limit
```

Study execution has four deliberately separate authorities. **Study design** states the authored
space and fixed or adaptive protocol. Its **evidence plan** records required and optional logical
candidate/seed/fidelity identities. **Scientific planning** selects additional adaptive evidence.
**Resource planning** decides which actions safely fit now. **Dispatch** creates the isolated
process. A scientifically required identity may therefore wait for
resources without being cancelled. If a hard learned lower bound proves it impossible on every
allocated device, LambdaForge replaces it with the closest candidate preserving its coverage
obligations and records `REPLACE_STARTUP_ANCHOR`. Extra physical capacity does not enlarge the
protected design: before enough outcomes exist it runs replannable `OPPORTUNISTIC_COVERAGE` points.

The deterministic candidate generator materializes a bounded prefix and expands that exact
Sobol/mixed-space sequence only when scientific planning asks for more candidates. Existing IDs
and mappings never change. Advanced `proposal_pool_size` controls this computational window, not a
scientific stopping rule. Lightweight candidate/seed specifications contain only their own values;
live explanatory analysis uses cached geometry and a bounded rotating shortlist.

`budget.candidates` (`trials` in legacy YAML) limits distinct proposed configurations, not the
evidence already owed for them. Reaching it stops candidate 41; it does not cancel required
minimum seeds, shared seeds, fidelity promotions, scientific continuations or confirmation.
When no candidate budget is authored, versioned scientific convergence controls expansion; sparse
coverage or low confidence alone is never called convergence. `max_runs` and `max_time` remain
explicit global ceilings. `convergence_patience` is a legacy opt-in record-streak policy: a
positive value can stop proposing after that many completed full-fidelity results fail to improve
the incumbent by more than `min_improvement`; it is not a multidimensional coverage test. The final
controller event records whether execution ended because of candidate, Run or time budget,
exhausted proposal pool, or this explicit convergence policy. Pruning and seed racing still save
compute without reducing the authored candidate budget. Coverage records distinguish **search
coverage** (a region was attempted or scientifically pruned) from **response coverage** (a
comparable completed response exists), including matched-context diversity. The controller can
request `COVER_PARAMETER_VALUE` or `COVER_INTERACTION_CELL` when a conclusion is confounded by
narrow context support. It does not enforce per-value quotas; the value of another probe falls as
diverse evidence resolves the question.

`PERFORMANCE_PRUNE` means that continuing a Run is not worthwhile for finding the optimum; it does
not mean the partial curve contains no evidence. If that same censored Run later becomes the most
cost-effective way to answer an unresolved parameter or interaction question, LambdaForge may
record `SCIENTIFIC_CONTINUATION` and resume the same Trial and seed from its durable checkpoint as a
new Attempt. It does not consume another `trials` slot, bypasses competitive-performance pruning,
and still consumes Run/time budget. The original prune remains visible and valid.

Terminal telemetry keeps execution, design completion and scientific resolution separate. A
complete 80/80 sweep may legitimately be `status: completed`, `design_status: complete` and
`scientific_status: unresolved`. Conversely, a time-limited 25/80 sweep is `incomplete`. Before
closing, every remaining identity is reconciled, so a terminal Study never retains queued or active
Runs. Exports distinguish observed from evidence-complete candidates and report required completed,
missing and failed cells plus the evidence completion fraction.

Fixed shared-seed sweeps use paired differences and resample whole seed blocks. A missing cell
reduces paired support and is never replaced with a different seed. Analysis keeps the
point-estimate leader separate from an `ExactScientificConclusion`: without
`objective.practical_margin`, unstable winner identity yields `NO_CLEAR_PREFERENCE` or
`UNRESOLVED`, never invented equivalence. With a margin on the objective (or normalized composite
utility) scale, stable evidence may instead support `PRACTICALLY_EQUIVALENT`.

With `auto`, ARI may grow packing independently on each GPU until physical VRAM, host resources,
throughput or site policy says to wait. Replacing it with `runs_per_gpu: 4` would impose **at most**
four Runs per device; it would not demand four fixed slots. Each candidate receives a learned future-memory
distribution from compatible terminal and live right-censored Runs. Admission uses physical device
memory as authority and approximates future headroom as

$$
H_g(t)=C_g-E_g(t)-\sum_i M_{i,\mathrm{future}}(t).
$$

Here $C_g$ is usable GPU capacity, $E_g$ is occupancy not attributed to active LambdaForge Runs,
and $M_{i,\mathrm{future}}=m_i(t)+R_i(t)$ combines the running maximum with the distribution of
memory that may still appear. The planner chooses the least-slack safe device (best fit), so a
large and a small Run can coexist without treating every hyperparameter configuration as the same
size. LambdaForge begins conservatively but does not wait for complete Runs: phase, progress,
change points, allocator peaks and durable checkpoints make active trajectories provisional
evidence that compatible GPUs can use immediately.

Small new allocator maxima are not automatically treated as continued material growth. LambdaForge
separates jitter, allocator drift, monotonic allocation and abrupt peaks, then updates a weak
survival prior after meaningful progress or phase transitions—not after every NVML poll. Changing
the monitoring frequency therefore does not manufacture confidence. A bounded trajectory remains
available for display, while versioned incremental sufficient statistics preserve all progress
cycles in a long Run. Its weighted future distribution retains rare tails (including sub-percent
mass) and composes candidate/resident uncertainty using exact small-support integration. Allocation
phases are learned per compatible Work family: validation or checkpoint risk is retained when
history supports it, but a generic Work is never forced to exhibit a fixed training lifecycle.
The displayed `RAMPING`/`PROVISIONALLY_STABLE` state is an explanation; placement consumes the
continuous phase hazard and weighted residual distribution.

`BASELINE_ADMISSION` is exactly the first protected Run placed on an otherwise idle granted GPU;
it does not consume an exploration lane or pretend that 1→2 packing has been tested. Empty granted
GPUs receive these baselines before any occupied GPU receives co-located work.
`SAFE_ADMISSION` fits after uncertainty and known OOM bounds are considered.
`EXPLORATORY_ADMISSION` is a checkpoint-aware 1→2→3 packing step whose expected scientific progress
and resource information exceed rollback and interference cost. With at least two interchangeable
GPUs, one remains at protected baseline concurrency while an equivalent unvalidated experiment
runs on at most one sibling. A provisional success promotes the packing frontier before terminal epochs; a
later OOM invalidates it. Thus a long cold start cannot remain at one Run per GPU merely because no
training has finished. If the current bounded scientific frontier is resource-blocked, each newly
evaluated unseen alternative may trigger another bounded request; an unchanged frontier that yields
no new identity is recorded and not polled in a loop.

Waiting is also a decision with a cost. While useful work is pending and physically usable VRAM is
idle, LambdaForge integrates idle fraction × normalized scientific-value rate. This *wait regret*
closes each physical time segment at its previous rate, survives controller restarts and is not
discounted merely because the frontier, hazard or checkpoint changed. It resets only when the
opportunity disappears or is satisfied; hard memory/site/run-cap and measured negative-throughput
constraints always win. The scientific controller supplies explicit rank, normalized value,
uncertainty and expected-cost semantics, so its raw surrogate score is never resource currency.
Unknown duration remains explicitly uncertain and is estimated from same/near history or live step
rates—never replaced by one second. Checkpoint comparison accounts for each resident separately:
non-checkpointable rollback remains, learned checkpoint cost is charged, and admission waits for a
durable checkpoint acknowledgement rather than trusting the request file.

`gpu_memory` is optional. When present it remains the user's minimum safety floor for each new Run;
the effective commitment is the larger of that floor, the learned upper envelope and any known OOM
lower bound. It is never multiplied by active Runs and is not a hard limiter inside consumer code.
When omitted, automatic prediction is used. `runs_per_gpu` remains a hard per-device ceiling and
`max_parallel` a hard global ceiling. A candidate that cannot fit now is `RESOURCE_BLOCKED`, not
failed or pruned, and is reconsidered when memory changes. A lower-value candidate may safely
backfill if it can finish before a high-value heavy candidate's expected window. Best-fit placement
and a bounded ranked frontier avoid random retries and preserve room for heavy work. If one
extension is evaluated but remains blocked, the controller may request another bounded unseen
alternative from the same scientific design policy without waiting for a terminal Run. Exact
action identities stop requests as soon as that policy returns no new alternative; it never
generates random candidates until one happens to fit.

CPU, RAM and storage retain a conservative stable per-Run share derived from the hard global
concurrency ceiling. They are not temporarily over-promised to the first cold-start Runs, because
those Runs cannot later be resized safely as packing grows. Unused host capacity remains available
to the operating system, while `self.resources` consistently reports the share a Run may rely on.

LambdaForge probes every granted GPU, reacts to external occupancy, staggers same-device launches
and updates active future envelopes from bounded trajectories for the full Run, including periods
where every dispatch slot is occupied and the temporary queue is empty. Physical peaks, allocator
diagnostics, duration/time-to-peak, censored early termination and OOM lower bounds are persisted.
Admission uses the stricter of live physical free VRAM and predicted future headroom, so new
external pressure cannot be hidden. Worker PIDs and CUDA descendants are matched to NVML process
memory when available; child allocator heartbeats add phase, step and short-peak evidence. Fallback
attribution is labelled inferred and never becomes exact evidence.
The bounded active-evidence snapshot is atomic: after a controller restart it may inform
uncertainty as stale provisional evidence, but LambdaForge never pretends its old PIDs are alive.
Terminal observations supersede it and successful shutdown clears it.
An OOM without reliable candidate attribution constrains the failed resident-set placement rather
than inventing a candidate size. The same or a dominated experiment is not repeated, but
`heavy+heavy` never globally forbids `heavy+small`. An exploratory OOM becomes resource recovery: a
new Attempt of the same logical Run resumes a valid checkpoint. It is never a bad scientific
objective or a new Trial. Measured co-location throughput can also stop extra packing even when
VRAM fits, because the goal is useful scientific work per wall-clock time rather than full memory.
Performance-pruned Runs still train the resource model when their forward/backward/optimizer/
validation phases and per-process measurements make the memory profile complete; their scientific
objective remains censored. Resource diagnostics persist changed `RESOURCE_WAIT`,
`RESOURCE_EXPLORE`, checkpoint, promotion, invalidation and recovery decisions with P(fit), peak
hazard, rollback, wait regret and an explicit rejection reason, without emitting a record per poll.

When several GPUs can accept one globally available Run, the resource planner uses the current
ranked scientific frontier and device state instead of preferring GPU index zero. While eligible
scientific actions and candidate/Run budget remain, every terminal event triggers replanning.
Physical readiness is also an event: if the executable dispatch queue becomes empty while a granted
GPU has admissible capacity, the dispatcher requests one bounded action frontier immediately
instead of waiting for an unrelated Run to finish. The scientific controller still chooses the
action and the resource planner still decides whether it is safe or worth one exploration step.
Each packed Run owns a fresh spawned process which exits as
soon as the Run finishes; LambdaForge does not reuse an idle CUDA worker because its surviving
device context could retain VRAM and deadlock queued Runs. One full GPU therefore never fails the complete study, and a
GPU that fits two of four configured slots runs two. Only a threshold larger than the total memory
of every allocated GPU is rejected as impossible. CPU-only adaptive studies can bound concurrency
with `max_parallel`. The memory observer runs in a short-lived child process, so it does not leave
an idle CUDA context on each device and never consumes one of the `runs_per_gpu` scientific slots.
If a child nevertheless raises CUDA OOM, only that Run becomes `retrying`: its failed Attempt and
censored resource evidence are retained. Healthy Runs and other GPUs continue. Retry recovery is
bounded by `failure_retries`; an OOM that repeats under a genuinely improved placement becomes
honest terminal evidence rather than an infinite retry loop.
Repeated `self.metrics.log("val_auprc", value, step=epoch)` observations enable
probability-based pruning. A local trend projects each active curve beyond the common observed step;
a stop is requested only when practical competitiveness falls below the configured probability,
which protects improving slow starters better than dropping a fixed bottom fraction.
`LightningRunner` bridges its validation metric and stop request automatically.
A custom loop must log the objective with `step=` and return at a safe checkpoint boundary when
`self.stop_requested` becomes true. With only a final metric, seed allocation remains adaptive but
the current training cannot stop early.

For trainings that can resume cumulatively, optional `fidelity` avoids giving every candidate the
full epoch budget. Units belong to the Work; LambdaForge never guesses that an arbitrary integer
means epochs. The current allocation is exposed as `self.fidelity`. A custom Work must checkpoint
at `target` and resume from `current`; `LightningRunner` applies these values to `max_epochs` and a
managed `last.ckpt` automatically:

```yaml
name: multi-fidelity-training
run: my_project.Training
search:
  trials: 40
  fidelity: {min: 5, max: 100, reduction_factor: 3}
  learning_rate: {range: [0.00001, 0.003], scale: log}
objective: {metric: val_auprc, mode: max}
```

Only statistically competitive configurations are promoted through cumulative budgets. Final
confirmation always uses the maximum. The controller writes compact `hpo-control/state.json` and
append-only `hpo-control/decisions.jsonl` evidence for every `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`,
fallback, convergence and confirmation decision; result summaries link both files.

Scheduling is event-driven. Every terminal Run causes the free slot to reconsider `START_NEW`,
`ADD_SEED`, `PROMOTE_FIDELITY` and `RESUME_PREEMPTED`; each alternative has a compact
`controller_value`, incremental-cost estimate and resulting priority in
`hpo-control/decisions.jsonl`. This value is an auditable common-scale heuristic, not Shannon
information gain or a calibrated probability. Seed actions use relative standard-error reduction,
new-region actions use bounded coverage/sparsity terms, and promotions use remaining-fidelity
uncertainty, divided by observed incremental wall time. Undispatched non-anchor actions form a
provisional dispatch buffer: new evidence may replace one at zero scientific compute cost and
records `CANCEL_PLANNED_DISPATCH` or `CANCEL_SCIENTIFIC_ACTION` according to whether only the
dispatch plan or the scientific action was abandoned. A protected initial anchor instead records
`DEFER_STARTUP_ANCHOR` and remains scientific debt.
Startup is a protected space-filling design, not a barrier:
model-directed work may begin while slower startup Runs remain active. Pending candidates condition
the surrogate at their actual target fidelity, queued identities prevent duplicate seeds, and the
controller may legitimately wait when no action has positive scientific value. Fidelity is an explicit model input, so partial and
full-budget observations are not mixed as equivalent. A candidate-level Beta/k-NN survival model
uses completed versus performance-pruned evidence with uncertainty; resource failures and
scheduler pauses are neutral, and a pruned partial curve is never converted into a fake objective.
An already-running action is paused only when it has an owned checkpoint, has run for a minimum
period and a newly available action beats both its original priority and continuation value by a
50% hysteresis margin. The request is cooperative at a safe boundary, never a process kill;
confirmation and unscored startup coverage are immune. `PREEMPT`, `PAUSE` and
`RESUME_PREEMPTED` records preserve the old/new priorities and reason without treating the pause
as negative model evidence.

Adaptive HPO has two simultaneous goals: find strong configurations and learn how the authored
search space behaves. LambdaForge therefore maintains two different quantities. **Optimization
opportunity** (`O`) is the remaining practical improvement predicted by the joint surrogate;
**scientific uncertainty** (`K`) is the unresolved entropy of explicit parameter and pairwise
interaction questions. Both are normalized from current evidence, converted into automatic
weights, and divided by observed incremental cost when actions compete. There is no trial-count
phase switch and no extra YAML confidence knob: optimization can regain priority whenever a new
observation makes practical improvement plausible again.

Every parameter question ends in one structured conclusion—preferred value/region, practically
equivalent, weak preference, flat, context-dependent, no clear preference, or unresolved. Pairwise
questions distinguish material/weak interaction, additive evidence and unresolved evidence. Here
**confidence** means the fraction of deterministic candidate-level, shared-seed-aware evidence
realizations that reproduce that exact displayed conclusion. It is not coverage, effect size, a
p-value or a frequentist confidence interval. Consequently, sufficient evidence can support both
“flat” and “context-dependent” with high confidence. Coverage and missing conditional support are
reported separately, and all effects remain predictive/descriptive rather than causal.

The **practical optimal region** contains observed configurations whose uncertainty-aware regret is
compatible with the authored equivalence margin. It reports which parameters are constrained and
which remain flexible inside that region. Outside startup, the controller may issue a
`DESIGNED_PROBE`: a valid unobserved candidate chosen to resolve a parameter or interaction, with a
matched counterfactual preferred when the conditional space permits it. Its decision record names
the question, predicted performance value, expected uncertainty reduction, match quality, cost and
alternatives. Seed racing uses pooled within-candidate variance only—never between-candidate
spread—replicates the incumbent when its uncertainty limits comparisons, and prefers an unused
shared seed when that improves paired incumbent/challenger evidence. Clearly inferior candidates
may therefore stop after one seed while useful comparisons receive more.

Every adaptive Run owns a separate worker process, on CPU and GPU. A normal exception fails only
that Run. A lost/killed worker or CUDA allocation OOM is retried up to `failure_retries` times
(default 1) as a new Attempt, reusing compatible checkpoints; another identical failure becomes
terminal instead of looping. Application errors such as invalid tensors or data are not retried
blindly. Other candidates keep running, although the final Work correctly remains failed if any
Run exhausts recovery.

Adaptive Study workspaces expose a dedicated **HPO** tab. Its parameter table compares every
hyperparameter with the candidate-level objective and separates the authored domain, actually
observed domain, best supported region, importance and conservative confidence. Opening a row
shows its predictive response with model-uncertainty bounds, observed support, empirical
dispersion and its measured joint predictive gains with every other parameter. The controller
area separately shows the actual latest
`START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, fallback or confirmation decision. These are exploratory
marginal associations, not causal claims; the joint multivariate sampler remains authoritative.
The console renders a separate `SURROGATE BELIEF` block from the actual last sampler refresh
(backend, target fidelity, predicted region and uncertainty), never presenting a marginal tendency
as GP/k-NN belief.
Select a parameter and press Enter/right to open its response and pairwise
relationship panel. The latter reports leave-one-out joint predictive gain over the better
one-parameter predictor in objective-standard-deviation units; it is a visual diagnostic of where
joint structure may matter, not causal interaction importance. Explicit objective guardrails and
infeasible counts appear in the console. A response view is available from two comparable
candidates; pairwise coverage is also shown from two and predictive gain starts at three. These
small-sample panels remain explicitly low confidence. The observer recomputes old bounded analysis
snapshots locally, so an already-running remote Work gains the newer view without restarting. A
running Study does not need a final Execution result: its persisted live HPO snapshot supplies
parameter rows, observed response/support and numeric plus verbal confidence until the final
analysis replaces it. The
same bounded response points, relationship matrix and
controller evidence are available to automation under `work.items[].study.hpo_analysis`,
`.controller` and the retained `.surrogate_belief`.

Terminal charts stay compact and truthful: categorical dimensions retain labels such as `true` and
`false`, and clicking a point or bar reports its exact coordinates. The terminal response chart
omits a visually ambiguous pseudo-band; uncertainty remains numeric in the table. **Interactive
HTML** exports an offline Plotly report with exact hover values and real shaded uncertainty, plus
pairwise heatmaps and numeric 3D surfaces. A Run report is a responsive dashboard: search its
grouped metric catalogue, collapse categories, compare raw/normalized curves, latest values,
same-epoch correlations or an arbitrary X/Y metric relationship, and choose an accessible
diverging heatmap palette. Drag the metric/sidebar separator or the lower edge of analysis panels
to resize them. `Organize` creates local categories and reassigns metrics without renaming them;
`My charts` saves reusable curve, normalized, snapshot or X/Y views from the current selection.
The same generated file remembers sizes, categories, charts, selections, active view and colours;
regenerating it deliberately starts with fresh preferences. This optional
renderer requires
`lambdaforge[analysis-report]` and writes only after an explicit user action.

Threshold-dependent metrics such as F1, balanced accuracy, Cohen's kappa, accuracy, precision and
recall are consumed exactly as the Work logs them. LambdaForge never searches a classification
threshold, chooses a different threshold per candidate or treats them as semantically equivalent
to threshold-free ranking metrics such as AUROC/AUPRC. Threshold policy belongs to the Work's
evaluation protocol and must stay comparable across candidates.

Concurrent training does not require reading one interleaved Job stream. A study is not a special
Work type: any normal Work declaring `search` or multiple `seeds` is marked as a study during local
validation, so the Research Console exposes its study screen while remote preparation is running.
It drills down as `Work → Trial (parameter combination) → Seed Run → live dashboard`. The Trial
screen shows which combinations are pending, active, promoted, eliminated or complete. The Run
screen also names its assigned GPU index (and machine output retains the exact grant token),
isolates that process's parameters and log, follows it automatically, and renders bounded
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
it does not duplicate checkpoints, models or large outputs. Open a seed's **Artifacts** tab, or run
`lf show WORK --run KEY`, to list each finalized managed artifact with its preferred usable path,
role, media type and size. The JSON form also includes SHA-256, managed/published paths, retention
and metadata. Published-only outputs point to `publish_to`; otherwise the path is inside the Run.

GPU admission belongs to the cluster profile, not scientific YAML. `gpu_access.mode=auto` uses
SLURM allocation on SLURM clusters and conservative exclusive LambdaForge leases on direct hosts.
Use `shared` only on a deliberately permissive direct host, or `command` with an argv prefix when
the site requires a claim wrapper. A self-contained wrapper such as CITIUS `gpu exec` is the
simplest policy for one GPU because its reservation lifetime equals the Work command:

```bash
lf clusters set free-gpu gpu_access.mode shared
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec]}'
```

For a multi-GPU CITIUS Work, or whenever the site requires a persistent claim, configure claim and
release atomically. This is also the reproducible alternative to making a manual claim in an
unrelated login shell. The only claim placeholder is `{gpu_count}` and it expands from the maximum
GPU count declared by the YAML:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

Claiming happens in the durable background submission, so waiting for a GPU does not block the
local terminal. Direct supervisors execute the paired release after success, failure or
cancellation, and also attempt it when scheduler submission fails. Persistent claim commands are
rejected for SLURM profiles, where allocation must belong to SLURM or to a self-contained wrapper.
`gpu exec` receives the immutable managed Python by absolute path, so it does not need to preserve
a shell/Conda activation.

For `command` and `scheduler` access, the scientific process must inherit
`CUDA_VISIBLE_DEVICES`. LambdaForge treats its entries as opaque granted tokens (indices, UUIDs or
MIG UUIDs), may narrow a child to one of those tokens, and never broadens the list. A missing or
duplicated allocation fails before training instead of falling back to physical GPU indices. A
scheduler grant remains exact. An adaptive Study behind a command launcher may receive fewer GPUs
than its requested ceiling; it then reduces concurrency to the non-empty inherited token list and
reports the reduction. Thus one granted GPU is used as one, two are used as two, surplus tokens are
ignored, and a broken wrapper can never silently expose an ungranted physical GPU. A GPU-requesting
Work also forces CUDA environment resolution: launcher failure cannot silently create a CPU-only
managed environment.

Command allocations may change while a long Study is running. LambdaForge obtains current opaque
tokens from `gpu_access.visibility_command`; for the usual `[gpu, exec]` prefix it derives
`[gpu, env]` automatically. A token can only disappear or be restored from the original inherited
grant—new physical identifiers are never accepted. On shrink, only the verified child Run on each
revoked token is stopped and requeued with its logical identity/checkpoint; unaffected Runs keep
going. If the ownership probe is temporarily unavailable, existing Runs are left intact but no new
Run is admitted. A custom visibility command must print its current comma-separated tokens on its
first non-empty line.

The shared mode admits external occupancy while LambdaForge Jobs still coordinate with each other.
Admission messages in the Run log show launches and periodic waits with current free memory. Since
unrelated occupancy and a Run's allocations can still change after admission, `gpu_memory` reduces
risk but cannot turn a shared site into hard isolation or enforce a hard process limit; consumer
code must provide a conservative peak requirement. Resource fields in YAML remain the absolute
outer scheduler request, except that per-Run `gpu_memory` supplies this inner admission threshold.

## Observe and operate

```bash
lf                           # interactive Research Console
lf overview --json
lf show WORK
lf logs WORK --follow
lf cancel WORK
lf retry WORK
lf jobs list                 # advanced scheduler/process view
lf jobs clear                # preview terminal-history cleanup
lf jobs clear --apply
lf results list
lf export SUCCEEDED_STUDY --output ./exports
lf datasets list
lf clean                     # preview only
lf clean --apply
```

The Research Console is the human interface for live Work, Studies, Clusters, Datasets and Results.
Its Overview answers what is running, waiting or unhealthy; contextual screens progressively reveal
Attempts, Runs, logs, resource admission, study evidence and completed results. `Ctrl+P` opens the
fuzzy command palette, Enter opens a selected row, Esc goes back and `?` explains the current
context. Slow filesystem/provider work runs outside the UI event loop; a transient provider outage
keeps the last successful snapshot visibly stale instead of turning it into scientific failure.
Only the first read uses a blocking loading view. Later probes keep the existing screen readable
and show the age of its last successful update at the bottom; Work state rows and bounded Work logs
refresh live without issuing overlapping requests. **Run Work** treats validation and durable
enqueueing as one visible operation, so choosing a recent YAML cannot stop silently between them.
Destructive actions require an explicit confirmation and retain the same preview-first ownership
rules as their CLI counterparts. Automation must use the stable `--json` commands rather than
parsing the full-screen interface.

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

`LightningRunner` records scalar curves, `epoch_time_s`, `validation_time_s`, peak live tensor
allocation (`gpu_mem_mb`) and the PyTorch allocator cache (`gpu_reserved_mb` and
`gpu_peak_reserved_mb`). Reserved memory is reusable cache, not another LambdaForge reservation or
evidence that live tensors consume that amount. HPO packing remains governed by the explicit
per-Run `resources.gpu_memory` admission threshold and driver-reported free memory. It fills only
currently safe slots and keeps the remaining Runs queued. A finished packed Run exits its dedicated
worker process, releasing the complete CUDA context before that slot is admitted again. Choose
which collected curves the Research Console renders without discarding the others:

```python
config = lf.training.LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_metric_display_names={"val_balanced_accuracy": "Balanced accuracy"},
    epoch_chart_exclude=["*_aux"],
)
```

`epoch_console_include`/`epoch_console_exclude` control the per-epoch human log table;
`epoch_chart_include`/`epoch_chart_exclude` independently control the interactive curves. Changing
a four-curve page resets every reused chart to automatic limits, so metrics with very different
scales are immediately visible; manual pan/zoom remains available within the page.
`epoch_metric_display_names` optionally changes only Research Console labels: raw metric keys,
objective lookup and persisted scientific identity remain untouched. Common names are humanized
automatically (`val_auprc` becomes `AUPRC`, `train_loss` becomes `Train loss`, and the internal
composite utility is displayed as `Composite selection score`).
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

## Study Analysis

Adaptive search answers what to run next; Study Analysis answers what the completed evidence
supports. A terminal study automatically writes a versioned, atomic `analysis.json`. The same core
can be run or refreshed explicitly:

```bash
lf results analyze EXECUTION
lf results analyze EXECUTION --recompute
lf results analyze EXECUTION --json
python -m pip install "lambdaforge[analysis-report]==0.15.0"
lf results report EXECUTION --output study-report.html
lf results replay EXECUTION --policy ari-v3.1
lf results replay EXECUTION --policy ari-v2-compat --json
lf export STUDY_OR_WORK --output ./exports
```

`lf export` resolves one unambiguous semantic Work in the current project and downloads its newest
successful Attempt, whether it ran locally or on a configured cluster. `--output` names a local
parent directory; LambdaForge creates `NAME--EXECUTION_ID/` atomically and refuses to overwrite an
earlier export. The package contains the immutable Execution envelope, submitted YAML, Run logs and
metrics, complete Study/controller evidence, HPO analysis JSON, recorded resource replay, retained
checkpoints/weights and finalized published artifacts. It always contains the self-contained
`reports/study-analysis.html`; `lambdaforge[analysis-report]` enables the full interactive Plotly
dashboard, while the base install emits a structured HTML/JSON fallback and records a warning.
`manifest.json` lists the
SHA-256 and byte size of every exported file. Shared datasets, environments, reconstructible cache
and the staged source bundle remain referenced by provenance instead of being duplicated. The
Research Console exposes the same operation as **Export Study…** and opens a local directory
browser; it calls this domain service rather than a shell command.

Resource replay reads the Study's versioned scheduler trace, never terminal prose. It is factual
until the selected compatibility policy first disagrees; every later metric is explicitly marked
as trace-conditioned simulation, and branch-specific terminal outcomes become `null` rather than
being attributed to the wrong policy. `recorded`, `ari-v3-compat` and `ari-v2-compat` support audit and
comparison without keeping obsolete schedulers in production. New Studies report time to each
concurrency level, idle capacity, useful/scientific rate, throughput, OOM/rollback/checkpoint cost,
starvation, P(fit) calibration and peak/time-to-envelope errors when the evidence exists. An older
Study without the trace remains readable but cannot claim counterfactual replay.

All HPO components share one authored `ParameterSpace`. In particular, `scale: log` uses logarithmic
coordinates in Sobol generation, Bayesian and dependency-light samplers, resource similarity,
scientific design and final analysis; integer, categorical and conditional inactivity semantics
are likewise identical. Candidate dictionaries and scientific identities are unchanged.

The analysis keeps four objective concepts separate. `current_observed_objective` is the latest
step at which every required component was present; `best_observed_objective` is the best complete
observation so far; `final_objective` exists only for a terminal full-fidelity Run; and
`selection_objective` is the seed-aware candidate quantity used for selection. A pruned Run keeps
its partial curve and best observation as censored evidence, but never receives a fabricated final
score. Missing composite components are reported explicitly.

`analysis.json` contains candidate/seed uncertainty, paired same-seed comparisons, separate
screening and confirmation evidence, candidate-level leave-one-candidate-out surrogate validation,
global functional and observed-top-region importance, adjusted response curves, pair interactions,
observed marginal/joint resolution, boundary saturation, pruning quality, constraints and two
distinct Pareto views. One seed is explicitly insufficient for empirical winner stability;
model-based seed uncertainty is reported separately when it was persisted. Reliability combines
support, cross-validation quality, coverage and extrapolation instead of sample count alone.
Effects are observational predictive summaries, not causal claims. Confirmation reuses the study's
scientific equivalence margin. Per-comparable-Run resource efficiency is never conflated with total
HPO controller spend. Live analysis is `PROVISIONAL`; only terminal evidence is `FINAL`.

The optional Plotly report is self-contained and works offline. Its Study dashboard separates
overview, candidate evidence and two-Trial comparison, parameter responses and exact-value
summaries, pairwise heatmap/3D interactions, coverage, resource efficiency and findings. Metric,
parameter, pair, plot-kind and colour controls use only persisted `analysis.json` evidence; the
browser does not fit another model or change HPO conclusions. Its resizable panels and `My charts`
view can persist custom bar/line/scatter plots whose axes are Trial, authored parameters, objective
components or resource observations. Preferences are local to that
generated HTML. Plotly is not a base dependency: without the extra, execution, JSON analysis and
the Research Console remain fully functional.

Resource admission is evidence too. Each study persists the current GPU/CPU/RAM admission state,
including active capacity, queued Runs, per-GPU free/required VRAM and the concrete wait reason.
`runs_per_gpu` is a ceiling rather than a demand: with two suitable GPUs and five slots each, ten
independent Runs may be admitted; unavailable slots wait and are reconsidered after a staggered
probe instead of failing the study. If that short-lived CUDA probe fails transiently, admission
stops but already-running science continues; observation is retried and its actual bounded stderr
is retained. A persistent loss of the grant fails only after no active Run remains.

## Research Console

Bare `lf` opens the Textual 8.2 Research Console on a TTY. The base installation includes
`textual-plot`, a native Textual plotting widget; resource history and learning curves therefore
have readable ticks, high-resolution Braille lines, colour keys and keyboard/mouse pan/zoom without
LambdaForge maintaining a second plotting engine. Overview is a dashboard with three
separate Cluster, ordinary Work and Study panels—Studies are never mixed into or demoted to Work
when they fail. Selecting a cluster shows live CPU/RAM/GPU history, total capacity, exact requested
resources and personal observed usage when the provider can measure it. The exact selected entity
and panel survive every asynchronous refresh. Progress is concise text, not serialized JSON. The
six primary destinations are Overview, Work, Studies, Clusters, Datasets
and Results. Enter/right opens the selected entity; Esc/left or the visible **Back to …** button
returns exactly one level; Home returns to the root; and every non-current breadcrumb segment is
clickable. Remote Study and Seed views show a centered loading state until their persisted snapshot
arrives, and show an explicit retrieval error rather than pretending an empty table is evidence.
Collection screens use deliberately shallow read models. Overview performs one inventory pass per
direct provider (or only active-job status reads for schedulers without inventory), reads only local
Dataset registry counts and never transfers every Study candidate/Run index;
Work and Studies omit resource and Dataset probes altogether. Opening one Study fetches its bounded
candidate/Run index, while HPO analysis, complete controller history and logs are loaded only when
their tabs are opened. Epoch curves, artifacts and isolated logs are requested only after opening
one seed. `lf overview --json` follows the same compact contract; use `lf show WORK` and
`lf show WORK --run KEY --json` for successive detail levels.
The worker writes a compact `study/interactive.json` separately from its authoritative rich
summary; legacy oversized summaries are projected on the execution host, and the complete action
history is transferred in bounded JSONL pages only after opening Action history. Live redraws retain
the selected row, table/log offsets and any manual chart viewport.
The central scientific route is:

```text
Studies → Study → Trials → Trial → Seeds → Seed → Curves / Epochs / Logs
```

Seed views keep current, best, final and selection evidence distinct; show GPU/resource evidence;
page native charts four at a time with mouse-friendly previous/next controls and a numbered page
selector (`n`/`p` remain shortcuts); mark the selected epoch in red and the best epoch in green; and
place every recorded scalar directly in the horizontally scrollable epoch table. Pruned Runs retain
a `†` censored best observation without a fabricated final
score. Study Analysis has Summary, Parameters, Interactions, Coverage, Seeds, Pareto and Findings
views over the same `analysis.json` used by CLI and HTML reports. Coverage uses summary cards, an
exact marginal chart and a table of observed versus authored levels/ranges, bins and edge support;
it describes sampled candidates rather than claiming that an unobserved proposal pool was covered.
Charts expose exact values on click. Contextual `?` controls explain objective, confidence,
reliability, top-region effects, predictive gain, coverage and stop semantics. Contextual
**Interactive HTML** controls open self-contained high-resolution Plotly reports when the optional
analysis extra is installed. A seed export is a responsive dashboard rather than one overloaded
plot: it initially selects at most four objective/validation metrics, groups and filters the full
metric list, and updates raw curves, normalized trend comparison, latest-value bars, same-epoch
correlations and an exact statistics table from one selector. Best/selected epoch markers remain
explicit; **Select all** is deliberate rather than the default. Study reports, parameter
response/interaction reports and bounded cluster resource histories use their matching views.
Seed stability is summarized in cards, scientific/resource Pareto evidence in separate exact
tables, and findings in a severity/reliability table with a readable recommendation preview.
HPO response and interaction surfaces are labelled with—and always use—the Study's declared
selection objective. The UI does not offer an arbitrary metric switch that could be mistaken for
the controller model; components of a composite objective remain available in the scientific
Pareto table.

The Study Overview combines state, candidate/Run counts, the current leader and compute time with
objective-by-trial and Run-state charts plus the leader's parameters. Exact state counts are printed
below the bars, so small groups remain readable at large scales. The **HPO** tab has separate
summary cards, a parameter evidence table and the complete persisted controller action history.
That history always uses Action / Trial / Reason columns; opening an action reveals the exact
persisted thresholds, priorities and evidence used by the controller. New decisions use a separate
append-only history so live snapshots remain small; pre-existing Studies without it show their
available persisted tail. Trial status marks have their
own column, so `★` best, `◆` Pareto and `†` censored never obscure the trial number.

The command palette lists only operations with a real handler. Other operations remain explicitly
CLI-only until their complete form/preview/service/refresh flow exists; a name in an inventory is
not claimed as parity. Contextual Work, cluster, dataset and result actions call Python domain
services directly—never an `lf` subprocess—and destructive actions show an exact preview before
confirmation. **Run Work…** provides a YAML-only project file browser and a bounded recent list
built from existing local Job history plus newly validated MRU choices. Browsing or selecting a
recent row only fills the same YAML selector. One **Submit Work** action then validates and explains
the plan and submits only if validation succeeds, using the visibly selected cluster. Controls are
disabled with a live phase message while this atomic flow runs, so repeated clicks cannot create
accidental duplicate submissions. It uses the same durable asynchronous service as `lf run`.
Additional MRU state stores only
local paths and display metadata, never YAML contents, credentials or input data; missing files
disappear automatically. Invalid YAML remains in the dialog and is reported once with its source
file, exact line/column, bounded excerpt and a corrective hint. Passwords remain outside
configuration.

The sidebar separates one-off **Action** controls from **Browse** navigation and the **Session**
exit control. Cluster and Dataset workspaces use compact summary cards, semantic tables and bounded
on-demand sections instead of raw JSON. The Clusters list and each cluster workspace reuse the same
CPU/RAM/GPU dashboard as Overview. The cluster activity console streams bootstrap phases and
elapsed time; choose Compact/Comfortable/Large or drag its separator to give long operations more
room. The upper cluster workspace becomes an independently scrollable viewport and keeps a useful
minimum height, so enlarging output never makes its actions, tabs or evidence unreachable. Only
one operation and one transport session are active for that workspace, and periodic
resource probes pause until it completes. Applying bootstrap also requires an exact confirmation;
the dialog renders the target, changes and preserved state as readable sections instead of raw
JSON. Every mutation confirmation and modal Work preview uses the same bounded renderer. The
separate Plan action remains read-only. Hidden root screens do not run background remote loads.
**Add cluster** and **Edit** use the same complete tabbed profile editor. Identity/authentication,
backend, project/data roots, managed Python, PyTorch/CUDA policy, cache retention, SSH connection
reuse and every `gpu_access` mode/prefix/claim/release field are editable directly. Uncommon
OpenSSH and SLURM dialect mappings remain available in a validated Advanced YAML tab instead of
being silently discarded. Saving reconstructs `ClusterProfile`, so invalid backend/GPU policy
combinations remain visible in the dialog and are never written. Password values are deliberately
not profile fields: only `keyring:`/`env:` references are edited there, while **Credentials** owns
the secret itself.
A failed Study whose live telemetry cannot be reached still
opens as a Study and keeps its terminal logs; missing telemetry is shown as missing evidence, never
as a console crash. **Cancel Study** is available from that degraded view as soon as the semantic
Work exists: it stops every active Attempt and descendant Run even before the worker publishes its
first Study snapshot. Periodic provider failures stay in the view as stale/error status and never
produce a repeating notification stream. **Delete Study History…** previews and then removes the
exact terminal Study/Attempt state using its `work_id`; active Studies must first be cancelled, and
published datasets, shared environments and unrelated same-name executions remain untouched.
If a provider cannot confirm a Job beyond `unknown`, deletion is still available as an explicit
history-only operation: it forgets the local record but preserves the unverified remote process and
workspace. Reconnect and cancel first when remote computation may still be active.

Dataset Summary loads exact split and primary-target counts from the logical index without walking
large asset trees. Opening Members automatically fetches one bounded page. Physical Stats and
checksum-based Integrity stay explicit inside their own tabs because they may be expensive. The
always-visible **Delete DatasetVersion…** action reports preview/apply progress immediately,
previews every registered placement and requires confirmation before deleting a whole
DatasetVersion. Repeated activation is blocked while an operation is active. A manually removed placement converges as stale registry cleanup; it does not make
the action fail or leave an empty logical version in the browser. Pruned Runs use “not final ·
pruned” or “not observed” for expected missing evidence instead of the misleading “unavailable”.

New remote dataset publications use the current project's effective root, including
`<configured-dataset-root>/projects/<project-id>`. A verified older placement may remain usable at
its explicitly registered unscoped path; it is durable evidence, not an implicit global lookup.
The final hash directory is the content identity. If preprocessing changes any asset bytes, publish
a new dataset version rather than reusing `NAME@VERSION`. Use `lf datasets reconcile NAME@VERSION
--on CLUSTER` to preview index-only repair and `--apply` only after review; an existing or
unreachable conflicting placement is never removed automatically. LambdaForge does not silently
relay a large remote dataset through the controller: remote-to-remote placement uses the site's
durable transfer facility, followed by `reconcile` after the exact manifest-backed directory is in
place.

Work names are display labels, not identities. The same authored Work may run locally and on one or
more clusters at the same time; tables and destructive actions use the exact `work_id`, so equal
names remain distinct and cancellation/deletion cannot silently target another execution.

The red **Exit LambdaForge** control is always visible in the sidebar; `q` exits from root views.

The CLI remains authoritative for scripts and remote automation. `lf --help` lists the exact
machine-friendly commands; no-command non-TTY invocation prints that help and exits. Users upgrading
from 0.13 should replace `lf top`, `lf clusters setup` and `lf clusters modify` with bare `lf`.
