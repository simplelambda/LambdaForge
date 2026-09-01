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
python -m pip install lambdaforge==0.14.0
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
python -m pip install "lambdaforge[clustering]==0.14.0"
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
expands finite values or reproducible numeric ranges; `objective` defines either one logged metric
or an explicit fixed-range composite utility. Whenever `search` has an `objective`, omitting `strategy`
enables the complete safe adaptive policy: Sobol startup, result-dependent proposals, probabilistic
seed racing, curve pruning, convergence detection and fresh-seed confirmation. A scrambled Sobol
pool supplies reproducible, space-filling candidates;
after `startup_trials`, observed results choose later candidates. `sampler: auto` uses optional
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

Seed counts are probability-driven rather than equal or fixed by a halving rung. Shared seed order
enables paired differences; a candidate receives another seed only while its probability of being
within `seed_racing.equivalence_margin` of the incumbent is at least
`seed_racing.probability_threshold`. The final
winner uses a conservative search bound, or preferably the mean of disjoint
`confirmation_seeds` on a frozen top-K. By default LambdaForge starts with up to three authored
seeds per candidate and generates three deterministic fresh confirmation seeds; every default can
still be overridden explicitly.
Confirmation Runs never receive performance pruning or opportunistic preemption. If a required
seed fails or cannot run within the global budget, `summary.confirmation.status` is `incomplete`,
`confirmation_incomplete` is true and no lucky subset of surviving confirmation seeds is selected.

HPO is deliberately easy to disable. `strategy: exhaustive` means a literal finite sweep: every
combination of `values` and every authored seed is run, including exact finite `when` branches.
Continuous `range` dimensions cannot be exhaustive, so discretize them with `values` or use the
adaptive strategy. `trials` is a candidate budget and is therefore rejected in exhaustive mode.

```yaml
name: optimizer-sweep
run: my_project.Training
seeds: [7, 17]
search:
  strategy: exhaustive
  optimizer: {values: [adamw, sgd]}
  momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
```

One scheduler Job owns the fixed resource reservation. Adaptive Runs are independent spawned
processes inside it. `resources.gpu` is the number of GPUs reserved for the whole study and
`runs_per_gpu` is the explicit packing factor:

```yaml
name: adaptive-training
run: my_project.Training
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  trials: 40
  proposal_pool_size: 640
  min_seeds: 1
  startup_trials: 10
  seed_racing: {probability_threshold: 0.1, equivalence_margin: 0.002}
  confirmation_top_k: 2
  confirmation_seeds: [1001, 1002, 1003]
  sampler: auto
  max_runs: 180
  max_time: 12h
  convergence_patience: 8
  min_improvement: 0.0005
  runs_per_gpu: 4
  failure_retries: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2,
                   probability_threshold: 0.05, equivalence_margin: 0.002}
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
one requires `gpu_memory`, which is the minimum free-VRAM admission threshold for each new Run,
not a demand to start the maximum concurrency immediately. LambdaForge probes every allocated GPU,
launches on whichever devices currently fit, and leaves the rest of the Runs queued. It polls again
while VRAM is temporarily busy and staggers launches on the same device so the preceding process
can materialize its allocation. A conservative per-device wave budget also reserves the declared
threshold for active Runs even before their CUDA allocations become visible; it resets only when
that GPU has no LambdaForge Run left. Each packed Run owns a fresh spawned process which exits as
soon as the Run finishes; LambdaForge does not reuse an idle CUDA worker because its surviving
device context could retain VRAM and deadlock queued Runs. One full GPU therefore never fails the complete study, and a
GPU that fits two of four configured slots runs two. Only a threshold larger than the total memory
of every allocated GPU is rejected as impossible. CPU-only adaptive studies can bound concurrency
with `max_parallel`. The memory observer runs in a short-lived child process, so it does not leave
an idle CUDA context on each device and never consumes one of the `runs_per_gpu` scientific slots.
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
uncertainty, divided by observed incremental wall time. Undispatched queued actions form a
provisional dispatch buffer: new evidence may replace one at zero scientific compute cost, with a
`CANCEL_QUEUED_ACTION` record containing old/new priority and reason.
Startup is a space-filling queue, not a barrier:
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

Every adaptive Run owns a separate worker process, on CPU and GPU. A normal exception fails only
that Run. A lost/killed worker or CUDA allocation OOM is retried up to `failure_retries` times
(default 1) as a new Attempt, reusing compatible checkpoints; another identical failure becomes
terminal instead of looping. Application errors such as invalid tensors or data are not retried
blindly. Other candidates keep running, although the final Work correctly remains failed if any
Run exhausts recovery.

Adaptive study screens also expose an HPO console with `i`. It compares every detected
hyperparameter with the candidate-level objective and reports coverage, numeric direction or
possible threshold, categorical contrast, standardized effect, conservative confidence and the
most useful evidence to collect next. The header separately shows the controller's actual latest
`START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, fallback or confirmation decision. These are exploratory
marginal associations, not causal claims; the joint multivariate sampler remains authoritative.
The console renders a separate `SURROGATE BELIEF` block from the actual last sampler refresh
(backend, target fidelity, predicted region and uncertainty), never presenting a marginal tendency
as GP/k-NN belief.
Select a parameter and press Enter/right to open its binned response chart and a pairwise
relationship panel. The latter reports leave-one-out joint predictive gain over the better
one-parameter predictor in objective-standard-deviation units; it is a visual diagnostic of where
joint structure may matter, not causal interaction importance. Explicit objective guardrails and
infeasible counts appear in the console. A response view is available from two comparable
candidates; pairwise coverage is also shown from two and predictive gain starts at three. These
small-sample panels remain explicitly low confidence. The observer recomputes old bounded analysis
snapshots locally, so an already-running remote Work gains the newer view without restarting. The
same bounded response points, relationship matrix and
controller evidence are available to automation under `work.items[].study.hpo_analysis`,
`.controller` and the retained `.surrogate_belief`.

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
it does not duplicate checkpoints, models or large outputs.

GPU admission belongs to the cluster profile, not scientific YAML. `gpu_access.mode=auto` uses
SLURM allocation on SLURM clusters and conservative exclusive LambdaForge leases on direct hosts.
Use `shared` only on a deliberately permissive direct host, or `command` with an argv prefix when
the site requires a claim wrapper. A self-contained wrapper such as CITIUS `gpu exec` is the
simplest policy because its reservation lifetime equals the Work command:

```bash
lf clusters set free-gpu gpu_access.mode shared
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec]}'
```

If the site instead requires a persistent claim, configure claim and release atomically. The only
claim placeholder is `{gpu_count}` and it expands from the absolute YAML GPU request:

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
MIG UUIDs), may narrow a child to one of those tokens, and never broadens the list. A missing,
duplicated or undersized allocation fails before training instead of falling back to physical GPU
indices. Thus a broken wrapper cannot silently use a GPU that was not granted.

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
    epoch_chart_exclude=["*_aux"],
)
```

`epoch_console_include`/`epoch_console_exclude` control the per-epoch human log table;
`epoch_chart_include`/`epoch_chart_exclude` independently control the interactive curves.
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
python -m pip install "lambdaforge[analysis-report]==0.14.0"
lf results report EXECUTION --output study-report.html
```

The analysis keeps four objective concepts separate. `current_observed_objective` is the latest
step at which every required component was present; `best_observed_objective` is the best complete
observation so far; `final_objective` exists only for a terminal full-fidelity Run; and
`selection_objective` is the seed-aware candidate quantity used for selection. A pruned Run keeps
its partial curve and best observation as censored evidence, but never receives a fabricated final
score. Missing composite components are reported explicitly.

`analysis.json` contains candidate/seed uncertainty (including deterministic bootstrap intervals
when supported), paired same-seed comparisons, separate screening and confirmation evidence,
candidate-level surrogate cross-validation, global functional and top-region importance, adjusted
response curves, pair interactions/surfaces, marginal and joint coverage, boundary saturation,
candidate-pool resolution, seed/winner stability, pruning quality, constraints, component Pareto
and resource-efficiency Pareto. Sparse evidence produces an explicit reliability limitation rather
than a confident-looking claim. Effects are observational predictive summaries, not causal claims;
poor cross-validation or extrapolated regions are labelled. Live console analysis is
`PROVISIONAL`; only terminal evidence is `FINAL`.

The optional Plotly report is self-contained and works offline. Plotly is not a base dependency:
without the extra, execution, JSON analysis and the Research Console remain fully functional.

Resource admission is evidence too. Each study persists the current GPU/CPU/RAM admission state,
including active capacity, queued Runs, per-GPU free/required VRAM and the concrete wait reason.
`runs_per_gpu` is a ceiling rather than a demand: with two suitable GPUs and five slots each, ten
independent Runs may be admitted; unavailable slots wait and are reconsidered after a staggered
probe instead of failing the study.

## Research Console

Bare `lf` opens the Textual 8.2 Research Console on a TTY. Its six primary destinations are
Overview, Work, Studies, Clusters, Datasets and Results. The command palette exposes the interactive
counterpart of every CLI family while calling Python domain services directly—never spawning an
`lf` subprocess. Study views distinguish incomplete/censored evidence, show current admission and
link to the persisted analysis; Results can analyze or export the exact same evidence used by the
CLI. Cluster forms start with human choices and keep passwords out of configuration.

The CLI remains authoritative for scripts and remote automation. `lf --help` lists the exact
machine-friendly commands; no-command non-TTY invocation prints that help and exits. Users upgrading
from 0.13 should replace `lf top`, `lf clusters setup` and `lf clusters modify` with bare `lf`.
