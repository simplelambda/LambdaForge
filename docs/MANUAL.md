# LambdaForge 0.14 manual

## Contents

1. [Mental model](#1-mental-model)
2. [Installation and project layout](#2-installation-and-project-layout)
3. [Work API](#3-work-api)
4. [Managed files, cache, map, outputs and tools](#4-managed-files-cache-map-outputs-and-tools)
5. [YAML reference](#5-yaml-reference)
6. [Files and datasets](#6-files-and-datasets)
7. [Sequence, parallelism, seeds and search](#7-sequence-parallelism-seeds-and-search)
8. [Execution, identity and reuse](#8-execution-identity-and-reuse)
9. [Results and metadata](#9-results-and-metadata)
10. [Clusters and jobs](#10-clusters-and-jobs)
11. [Cleanup and safety](#11-cleanup-and-safety)
12. [CLI reference](#12-cli-reference)
13. [Clustering](#13-clustering)
14. [Reusable neural components](#14-reusable-neural-components)
15. [Architecture and extension boundaries](#15-architecture-and-extension-boundaries)
16. [Study Analysis](#16-study-analysis)
17. [Research Console](#17-research-console)
18. [Project isolation](#18-project-isolation)

## 1. Mental model

LambdaForge runs Python classes while owning infrastructure around them:

```text
Work YAML -> WorkConfig -> Execution plan -> scheduler/Job -> WorkRunner -> Work.run()
```

The researcher writes a `lambdaforge.Work` subclass and its `run()` method. YAML chooses the class,
normal arguments, resources and repetition. LambdaForge resolves external inputs, records identity,
binds services, captures output and preserves an inspectable result. A function or unrelated class
is not executable YAML.

The durable hierarchy is intentionally conceptual rather than five service layers:

- Work: the human-named operation/study in YAML;
- Execution: one invocation on a target;
- Run: one scientific seed/search member;
- Attempt: one try at completing that Run;
- Job: the scheduler/process used to perform the Attempt.

## 2. Installation and project layout

Keep framework and consumer as independently installable packages in one project-owned environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge
python -m pip install -e .
python -m pip check
```

During framework development replace the release install with
`python -m pip install -e /absolute/path/to/LambdaForge`. Never share LambdaForge's `.venv`, patch
`PYTHONPATH` in production, or copy its source into a consumer. `lf init DIRECTORY` creates
`pyproject.toml`, `src/my_project`, a Work YAML and safe ignores.

Typical layout:

```text
project/
  pyproject.toml
  src/project_name/work.py
  experiments/study.yaml
  data/small-input.json
  .lambdaforge/          # managed state; ignored by Git
```

`lf --help`, `lf help`, `lf COMMAND --help` and `lf help COMMAND SUBCOMMAND` are equivalent
successful help routes. Bare `lf` opens the Research Console on a TTY; redirected execution and
`lf --json` print help instead. Its Clusters screen edits the same catalog and credential services
as native commands and never puts passwords in argv or YAML. The 0.13-only
`top`/`clusters setup`/`clusters modify` routes have been retired.

## 3. Work API

Subclass `Work`, use a no-argument constructor (normally no constructor at all), and implement
`run()`. Runtime properties raise outside managed execution.

| API | Mutability/lifetime | Purpose |
|---|---|---|
| `run(**parameters)` | user method, once per Attempt | scientific operation and primary JSON result |
| `name` | read-only | configured Work/step name |
| `config` | recursively read-only | class, normalized parameters and requested resources |
| `inputs` | read-only mapping | only typed file/dataset provenance and resolved paths |
| `outputs` | append-only per Attempt | managed files/directories, values, artifacts and datasets |
| `metrics` | append-only | scalar history and latest values |
| `checkpoints` | durable per Run | explicit safe resume state |
| `cache` | reconstructible | simple key/value cache plus managed file/fetch APIs |
| `tools` | execution-scoped | external executable resolution, execution and provenance |
| `progress` | replaceable snapshot | optional live completed/total/message |
| `log(message, level=...)` | append-only stream | timestamped, flushed human diagnostics |
| `resources` | read-only | requested CPU, memory, GPU, GPU memory, time, storage, processes |
| `seed` | read-only | current seed or `None` |
| `trial` | read-only | current search index/parameters or `None` |
| `run_dir` | owned per Attempt | durable files before registration |
| `temp_dir` | owned per Attempt | removed after finalization |
| `source_dir` | read-only | consumer project/source context |
| `resuming` | read-only | compatible checkpoint state existed at start |
| `map(...)` | execution helper | bounded ordered intra-job concurrency, without persistence |
| `resume_map(...)` | execution helper | explicitly keyed dependency-aware per-item resume |

`outputs.value` accepts JSON-compatible values once per name. Prefer `outputs.file/directory` for
new artifacts; `outputs.artifact` is the advanced bridge for a path already created elsewhere. It
verifies a regular file/directory, rejects symlink traversal, copies external paths under managed
ownership and records SHA-256/size/role/media type. `metrics.log` accepts finite numeric scalars and
optional step/split.
`print()`, stdout, stderr and normal Python `logging` reach Job logs. Output produced inside the
managed `run()` capture also becomes the Attempt's `work.log`. Use `print(..., flush=True)` for a
plain live message, or `self.log("message", level="info")` for a timestamped `debug`, `info`,
`warning`, `error` or `critical` line that is flushed immediately. This is operational narration;
use `metrics.log` for numeric evidence and `progress.update` for bounded live completion.

Checkpoint/cache names must stay below their roots. The next section defines their high-level APIs;
raw `path()` calls are advanced escape hatches. JSON checkpoint writes are atomic and corrupt state
is an error. Plain `map` preserves order, cancels pending futures after failure and uses thread or
spawn-process executors without creating cache/checkpoint state. `resume_map` adds unique stable
keys and safe JSON result checkpoints, never pickle. `map(..., key=...)` remains a compatibility
bridge to `resume_map` but is not the recommended spelling for new code.

## 4. Managed files, cache, map, outputs and tools

### 4.1 Storage meanings

Choose storage by lifecycle, not convenience:

| Need | Service | Survives retry | May `lf clean` remove it | Published result |
|---|---|---:|---:|---:|
| small derivable bytes/text/JSON | `cache.put/get` | normally reused | yes | no |
| derivable downloaded/computed file | `cache.file/fetch` | normally reused | yes | no |
| sequential state required to resume a Run | `checkpoints.file/save_json` | yes | no | resume evidence |
| final scientific file/tree | `outputs.file/directory` | belongs to Attempt | only a verified published duplicate | yes |
| disposable intermediate | `temp_dir` | no | automatic | no |

`cache.path`, `checkpoints.path` and direct `run_dir` writes remain explicit interoperability
escapes. They are not the ordinary pattern because their caller owns validation, atomicity and
registration.

### 4.2 ManagedFile and cache

The ordinary cache is a key/value API:

```python
self.cache.put("summary", {"accepted": 1842, "schema": 3})
summary = self.cache.get("summary")
fallback = self.cache.get("missing", {"accepted": 0})
```

`put(key, content)` accepts exact `bytes`, UTF-8 text or strict JSON values. It atomically replaces
the value for that logical key; `get(key, default=None)` restores the original supported content
type and returns the default when the integrity-checked entry is absent. Pickle and implicit object
serialization are intentionally unsupported. The cache root is selected by the execution/storage
profile; scientific code normally does not choose it. Operators can place the cache through the
cluster storage policy, while `cache.path` remains the explicit advanced escape.

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --cache-root /scratch/USER/lambdaforge-cache
```

The selected root is propagated to every Work process, including detached direct Jobs; it is not a
shell-state assumption. Local project execution defaults to the project's `.lambdaforge/cache`.

Use the managed-file API when a library needs a path or construction is expensive:

```python
structure = self.cache.file(
    f"structures/{identifier}.cif",
    build=lambda target: create_structure(identifier, target),
    validate=lambda file: file.size_bytes > 0 and valid_structure(str(file)),
)
```

`build(target: Path)` writes a real temporary destination for libraries and executables. The return
value is ignored. A cache hit checks its record and current SHA-256/size, then runs the optional
semantic validator. A miss or stale entry acquires an exclusive cross-process lock derived from the
logical key, rechecks, builds beside the final destination, validates, `fsync`s, atomically replaces
and writes the integrity record. A failed producer leaves no promoted entry. Keys are relative
portable names: absolute paths, `..` and symlink traversal are rejected.

The returned `ManagedFile` is read-only and `os.PathLike`. It supports `str(file)`, `file.path` as
an advanced escape, `file.open/read_text/read_bytes`, `file.exists`, and immutable `key`, `sha256`,
`size_bytes`, `scope` and metadata. Logical keys and content evidence cross checkpoints; physical
cache paths never become scientific identity.

```python
archive = self.cache.rate_limit("archive", requests_per_second=4.0)
record = self.cache.fetch(
    url,
    key=f"archive/{identifier}.json",
    retries=5,
    retry_backoff=0.5,
    timeout=30,
    decompress="gzip",
    validate=valid_record,
    rate_limit=archive,
)
```

`fetch` is deliberately not a generic HTTP client. It performs one HTTP(S) GET into the same cache
publication path, with bounded timeout, retries after the initial attempt, exponential backoff and
optional gzip decompression. Connection loss, incomplete chunked reads (including truncation while
decoding gzip), 408/425/429 and 5xx responses are retryable; permanent HTTP errors such as 401 or
404 fail immediately. Every attempt passes through the rate limiter and starts from an empty private
temporary file. Exhaustion reports the URL, attempt count and final cause. Only a completely read,
validated file is atomically published, so a retry never exposes partial bytes or a valid record for
them. Its rate limiter is thread-safe and shared only by one Work cache
instance; it is not a distributed rate limit across Jobs. The cache is identity-scoped and
reconstructible. `lf clean` previews each Work cache identity, and `--apply` removes it only after
obtaining an exclusive GC lock; a running Work holds the corresponding shared lease.

The dataset cache remains a different specialization: `DatasetCache` stores serialized map-style
samples through memory/disk backends and codecs. Work cache reuses the common
`CrossProcessFileLock`, atomic/fingerprint conventions and storage ownership, but cannot reuse the
dataset record envelope as a normal path-like scientific file.

### 4.3 Plain and resumable map

The common operation is deliberately unsurprising:

```python
results = self.map(rows, process, workers=8, executor="thread", retries=2)
```

Its signature is `map(items, function, *, workers=1, executor="thread", name=None, retries=0,
retry_backoff=0.5)`. It materializes the input iterable, applies the callback with bounded
concurrency, reports progress and returns results in input order. It does not write cache or
checkpoint records. `thread` is useful for I/O; `process` requires a spawn-pickleable callback and
arguments.

Choose resumability explicitly for long item-wise operations:

```python
results = self.resume_map(
    rows,
    process,
    key="record_id",                 # mapping key, dataclass/object attribute, or callable
    workers=16,
    executor="thread",              # or spawn-safe "process"
    resume=True,
    name="feature-extraction",
    validate=valid_restored_result,
    retries=2,
    retry_backoff=0.5,
)
```

The complete signature is `resume_map(items, function, *, key, workers=1, executor="thread",
resume=True, name=None, validate=None, retries=0, retry_backoff=0.5)`. Keys must be unique and
stable. A string
selects a mapping field or object attribute; a callable remains available for any explicit key.
Results preserve input order even when completion does not.

Each item record is strict JSON extended only with LambdaForge `ManagedFile` references. The codec
stores cache/checkpoint scope, key, SHA-256 and size—not an absolute path or pickle. Managed files returned by the
callback, nested in its result, or touched through `self.cache` in a sequential/thread callback are
listed as dependencies. On restore LambdaForge verifies every dependency before decoding the
result. A missing/corrupt dependency or a validator returning false makes only that item pending;
valid siblings remain restored. A validator exception is reported rather than silently hiding a
scientific validation bug. Retries apply independently to new/pending items, with exponential
backoff; exhaustion remains fail-fast. Process callbacks must be spawn-pickleable and dependencies
must be present in their item/result because execution-context cache tracking cannot cross a
process boundary.

### 4.4 Outputs and checkpoints

```python
report = self.outputs.file(
    "report",
    filename="report.json",
    role="report",
    media_type="application/json",
    metadata={"schema": 1},
    publish_to="results/report.json",
)
report.write_json(summary)

figures = self.outputs.directory("figures", role="visualization")
render_plots(Path(figures))
```

Managed file outputs support atomic `write_text`, `write_bytes`, `write_json` and
`build(lambda temporary: ...)`. Managed directories exist immediately, support safe child paths
with `/`, and are fingerprinted once at finalization. After `run()` returns successfully,
LambdaForge checks type/containment/symlinks for every declaration and registers the complete set.
If any declared output is missing or unsafe, finalization fails and no managed declaration from
that set enters the result. Do not call `outputs.artifact` again for the same managed output.

`publish_to` is optional on both `file` and `directory`. A relative destination starts at the
directory containing the authored YAML; an absolute path is explicit. For remote execution,
relative publication requires the cluster's `project_root` and uses the equivalent YAML directory
below that mirror. It never resolves below the bundle/Job hash directory. Without a mirror, use an
absolute persistent remote path; LambdaForge rejects a remote relative destination rather than
silently publishing into disposable internal storage.

LambdaForge first validates and fingerprints the Attempt-owned artifact, then publishes a
per-destination atomic copy. An existing identical copy is reused, while different content is
refused unless `overwrite=True`. Its result metadata records `published_to`, SHA-256 and size.
After the whole Execution result is persisted, LambdaForge re-hashes that destination and removes
the redundant Attempt-owned bytes by default. Pass `retain_internal=True` only when a deliberate
second physical copy is worth its storage cost. Successful outputs without `publish_to` stay under
the Attempt because they have no other copy. Failed and provider-interrupted Attempts remove
partial managed `artifacts/`, while preserving logs, metrics, results, provenance and checkpoints.
Direct `run_dir` writes and imported `outputs.artifact` paths are advanced ownership escapes, so
automatic cleanup does not guess that arbitrary bytes are disposable. LambdaForge does not
silently copy arbitrary large outputs back to the controller. Publication also refuses symbolic-link traversal, type changes and any directory
destination that contains (or is contained by) its managed source; `overwrite=True` is never
permission to replace the project or Attempt root.

```python
index = self.checkpoints.file(
    "leakage/mmseqs.tsv",
    build=lambda target: build_index(target),
    validate=valid_index,
)
```

Checkpoint files use the same safe temporary-build, integrity record and optional validation as a
cache file, but live under the Run and are never garbage-collected as reconstructible data. A retry
reuses a valid file; absent, corrupt or semantically invalid state rebuilds it atomically.
`save_json/load_json/exists` remain the concise API for small state. `--restart` removes the entire
compatible checkpoint tree before the new Attempt.

### 4.5 External tools

```python
mmseqs = self.tools.require("mmseqs", version_args=["version"])
completed = self.tools.run(
    [mmseqs, "easy-search", str(query), str(database), str(output)],
    name="MMseqs2",
    threads=self.resources.cpu,
    cwd=self.temp_dir,
    env={"PROJECT_MODE": "strict"},
    timeout=3600,
)
```

`require(executable, version_args=None, version_timeout=10)` resolves `PATH` and optionally executes
only the explicitly supplied version arguments. It returns a path-like `Tool`. `run(command, *,
name=None, threads=None, cwd=None, env=None, timeout=None, check=True)` requires an argv sequence and
never uses a shell. It streams stdout/stderr line-by-line into `work.log`, retains a bounded tail in
`ToolResult`, records duration/exit status and raises `ToolExecutionError` for nonzero status when
`check` is true. `threads=N` sets OMP, MKL, OpenBLAS and NumExpr variables only in the child. Explicit
environment overrides are also child-scoped. Tools actually required/executed and explicit version
results are written once in `environment.json`; this extends rather than duplicates environment
provenance.

## 5. YAML reference

Allowed top-level fields are `name`, `run`, `with`, `resources`, `seeds`, `search`, `objective` and
`steps`. A document defines exactly `run` or `steps`. Work steps accept the same fields except
`steps`. Unknown fields fail.

```yaml
name: evaluate
run: project.work.Evaluate
with:
  threshold: 0.5
resources:
  cpu: 4
  memory: 8GiB
  gpu: 1
  gpu_memory: 12GiB
  time: 4h
  storage: 20GiB
  processes: 2
```

Resources are absolute scheduler requests. `processes` cannot exceed CPU. Byte units support B,
KB/MB/GB/TB and KiB/MiB/GiB/TiB; time supports s/m/h/d. `lf validate` imports every Work, checks
inheritance, constructor, `run()` names/defaults/obvious types, resources, inputs, expansions and
references. It never submits. `lf explain` reports docstrings, types, configured values and defaults.

## 6. Files and datasets

```yaml
with:
  manifest: {file: ../data/manifest.json}
  corpus: {dataset: research-corpus@3}
```

A file or directory is resolved relative to YAML, checked, hashed and passed as `Path`. On remote
execution, content up to the default 10 MiB inline limit is copied into the immutable control bundle.
A larger project-owned path follows the mirror contract in section 10: the bundle preserves its
project-relative name, the worker receives the matching absolute remote path, and exact kind, byte
count and SHA-256 are checked before scheduler submission and again in the worker. Nothing is
silently transferred. A large path outside the installable project is refused; use a managed
dataset or an explicit, reviewable project/data layout.

This marker is input-only: it describes content that must exist before scheduling. A new output is
created in Python with `self.outputs.file/directory/value/dataset`; there is no second YAML output
schema to learn. Managed artifacts are the safe default. Add `publish_to` only when another program
or a researcher needs a conventional path outside managed result storage, remembering that local
and remote filesystems are different physical locations.

A dataset selector resolves an exact immutable registry version/content ID and a valid placement,
then passes a usable `Path`. Publishing is Python code:

```python
self.outputs.dataset(
    name="research-corpus",
    version="3",
    members=(
        {"id": row.id, "split": row.split, "assets": {"data": row.path}}
        for row in rows
    ),
    metadata={"species": "human"},
)
```

Members stream into canonical JSONL. The publisher copies and hashes assets, validates the index,
computes path-independent content identity, writes DatasetArtifact v2 in staging, verifies it,
atomically renames it, then registers placement. An existing name/version with different content
is refused. DatasetArtifact v1 manifests remain readable because persisted research data outlives
the authoring API.

## 7. Sequence, parallelism, seeds and search

Sequence and parallel groups are the whole composition language:

```yaml
name: comparison
steps:
  - name: prepare
    run: project.Prepare
  - parallel:
      - name: a
        run: project.TrainA
        with: {data: {from: prepare.dataset}}
      - name: b
        run: project.TrainB
        with: {data: {from: prepare.dataset}}
  - name: compare
    run: project.Compare
```

Each level waits for the prior level. A parallel group gives each Work definition its own isolated
spawned process. An exhaustive seed/search member runs serially inside its fixed allocation; an
adaptive member owns its allocation and schedules independent child Runs within it.
The enclosing scheduler Job owns the aggregate allocation and cancellation boundary. A named output
reference is valid only when its producer expands to one Run. Complex branching, conditions and
loops belong in Python.

Finite and numeric search:

```yaml
name: tune
run: project.Train
with: {epochs: 20}
seeds: [7, 17]
search:
  strategy: adaptive
  trials: 12
  proposal_pool_size: 192
  startup_trials: 10
  min_seeds: 1
  seed_racing: {probability_threshold: 0.1, equivalence_margin: 0.002}
  confirmation_top_k: 2
  confirmation_seeds: [1001, 1002, 1003]
  sampler: auto
  max_runs: 180
  max_time: 12h
  # Optional record-streak stop; omit to consume the authored candidate budget.
  convergence_patience: 8
  min_improvement: 0.0005
  max_parallel: 2
  failure_retries: 1
  learning_rate: {range: [0.0001, 0.01], scale: log}
  width: {values: [128, 256]}
objective: {metric: val_score, mode: max}
```

Finite-only exhaustive dimensions form a Cartesian product. In explicit `strategy: exhaustive`
mode this is an exact contract, not a sampling hint: finite `when` branches are enumerated, while
`range` and `trials` are rejected because a continuous interval/candidate cap cannot be exhaustive.
Replace a range with explicit `values` when a full reproducible sweep is required.
Adaptive/numeric spaces build a deterministic scrambled-Sobol pool bounded by
`proposal_pool_size` (by default up to 16 times `trials`, capped at 4096); `trials` is only the
maximum number of candidates that may execute. Conditional parameters use `when` and carry
an explicit active/inactive feature rather than a fake value. This pool is planning state, not a
list of decided Trials. It is retained once and lightweight Run specifications reference only their
own candidate and seed, so planning memory grows linearly rather than duplicating the pool per Run.
Large valid values therefore do not require weakening the YAML to avoid planner memory blow-ups.
If `startup_trials` is omitted, LambdaForge first proposes
`min(trials, max(10, safe_parallelism))` distinct space-filling points. This preserves the
historical statistical floor and fills the available first wave; it does not promise that temporary
VRAM pressure can admit every point immediately. An explicit positive `startup_trials` overrides
the automatic width.
After their objective values arrive, `sampler: auto` prefers optional BoTorch mixed-GP qLogNEI and
falls back to a dependency-light mixed k-NN acquisition if the extra is absent, observations are
still insufficient or GP fitting is numerically unsafe. `sampler: knn` forces the base backend;
`sampler: botorch` requests BoTorch but still fails safe to k-NN. Only proposed candidates appear in
the Research Console. This is a genuinely joint decision model: numeric dimensions, categorical dimensions and
conditional active/inactive indicators share one input vector, so posterior predictions can depend
on interactions. When repeated seeds provide a standard error, it is passed as observation noise;
qLogNEI then selects from the finite candidate pool while accounting for noisy observations and
every already-pending candidate. Evidence is grouped by exact cumulative rung: each
`(candidate, target, maximum)` has its own across-seed mean and standard error, and the same
candidate may produce several observations. Seed racing compares candidates only on one shared
exact rung; a heterogeneous seed mean is never relabelled with maximum, minimum or average
fidelity. A pending Run retains its actual target fidelity instead of being represented as a
fictitious full-budget observation. Fidelity is a normalized explicit input:
numeric spaces use `SingleTaskMultiFidelityGP`, mixed/categorical spaces
keep that coordinate in `MixedSingleTaskGP`, and the dependency-light k-NN fallback combines
parameter and fidelity distance conservatively. Acquisition always scores new candidates at target
fidelity. The log-EI family is used for its more stable
numerics ([BoTorch acquisition guidance](https://botorch.org/docs/optimization)).
This is deliberately described as a **fidelity-aware surrogate plus an external cost-aware action
scheduler**. LambdaForge does not claim to implement a unified multi-fidelity knowledge-gradient
acquisition for every mixed/conditional space; promotions compete with seeds and new candidates in
the provider-neutral controller.

The legacy scalar objective may include explicit outcome constraints:

```yaml
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55, seed_aggregation: worst}
    val_kappa: {min: 0.05, seed_aggregation: lcb, confidence: 0.95}
```

Each constraint accepts `min`, `max` or both. LambdaForge reads its value at the exact checkpoint
where the objective utility was best. Across completed seeds, `seed_aggregation` is `mean` by
default for backwards compatibility, `worst` tests the least favourable seed, and `lcb` applies a
two-sided normal confidence bound (`confidence`, default 0.95) conservatively to lower and/or upper
bounds. LCB with fewer than two seeds and every missing metric fail closed. The same metric may be
both a composite component and a hard constraint: feasible values still contribute continuously
to utility. Infeasible candidates remain auditable but are excluded from seed racing, surrogate
observations and final selection. LambdaForge never infers constraints from metric names.

For a deliberate graded compromise, `objective.metrics` defines a composite utility. Every
component requires `mode`, a non-negative `weight` and a fixed finite `range`; aggregations are
`weighted_mean`, `geometric` and `chebyshev`. Weights are normalized during validation and ranges
never depend on later observations. Components are combined only at the same real integer step.

```yaml
objective:
  aggregation: geometric
  metrics:
    val_auprc: {mode: max, weight: 3, range: [0.0, 1.0]}
    val_balanced_accuracy: {mode: max, weight: 1, range: [0.0, 1.0]}
  constraints:
    val_kappa: {min: 0.05}
```

That one utility governs acquisition, seed racing, pruning, fidelity, confirmation and ranking.
Constraints remain separate hard feasibility rules. Raw components, contributions and the
diagnostic Pareto front are retained; Pareto status never becomes a second hidden policy.

With an objective, omitted `strategy` means `adaptive`, even with one seed. This activates all safe
optimizations by default: Sobol startup, adaptive proposals, seed racing, curve pruning,
convergence detection, bounded failure recovery and fresh-seed confirmation. Unless authored,
`min_seeds` is one available search seed and three deterministic disjoint confirmation seeds are
generated; use `confirmation_seeds: []` only to deliberately disable final confirmation.
Search and seed evidence are interleaved. For candidate \(i\), seed outcomes estimate a
random-effects mean \(\hat\mu_i\). Seed noise is calibrated only from repeated outcomes within a
candidate, after estimating effects of seeds shared by several candidates; spread between
candidates is never relabelled as random seed noise. Before that calibration, a one-seed estimate
remains explicitly uncertain. Shared seed order
allows the paired differences

$$
d_s=Y(i,s)-Y(i^\star,s).
$$

Another seed is useful while

$$
P(\mu_i\geq\mu_{i^\star}-\epsilon\mid D)\geq\delta,
$$

where `seed_racing.equivalence_margin` is \(\epsilon\) and
`seed_racing.probability_threshold` is \(\delta\). Clearly
dominated candidates stop receiving seeds; uncertainty near the decision boundary receives them
first, divided by observed Run duration. `min_seeds` is only the initial evidence floor. Ambiguous
root `search.reduction_factor` and `search.confidence` fields are rejected: fidelity spacing,
seed-racing probability and early-stop probability have separate named owners.

Search evidence and final evidence are separate. `confirmation_top_k` freezes the best search
candidates and `confirmation_seeds` evaluates them with disjoint seeds. If confirmation is present,
only its mean selects `summary.best`; otherwise a conservative search bound prevents a lucky
one-seed candidate from defeating a better-supported candidate. `trials` is the default stopping
contract: the complete authored candidate budget is proposed unless an explicit Run/time limit is reached.
`max_runs` and `max_time` bound total spending. `convergence_patience` defaults to zero (disabled).
Setting it to a positive value explicitly opts into record-streak stopping after that many
completed, full-fidelity observations fail to improve the incumbent by more than
`min_improvement`. This simple policy is not evidence of mixed-space coverage and must not be
enabled merely because HPO confidence is low. Pruning and seed racing remain active independently.
Every terminal controller event persists whether candidate, Run or time budget, proposal-pool
exhaustion, explicit convergence or the lack of another useful action ended the Study.

One seed Run has two deliberately different objective values. `current` is the observation at its
latest reported step and is used for like-for-like curve projection and pruning. `best` is the
minimum/maximum observation over its completed curve and is the checkpoint value used for seed
racing, the surrogate and final ranking. Consequently a late overfitting epoch does not
underestimate a completed Run, while a promising early point cannot save a cooperatively pruned
partial Run. A Trial's HPO value is the mean of the per-seed best values—not the single luckiest
seed. Fresh confirmation seeds and the conservative search bound limit repeated-validation bias.
The default initial floor is one authored seed per proposed Trial. That is enough to begin broad
candidate coverage; the controller then buys the repeated, incumbent and paired evidence that is
actually valuable instead of spending three seeds indiscriminately.
Confirmation is never performance-pruned or opportunistically preempted. Missing or failed
required confirmation seeds persist `summary.confirmation.status=incomplete` and
`confirmation_incomplete=true`; no survivor-only confirmation mean becomes the selected model.

Every controller action is auditable. `hpo-control/decisions.jsonl` is an append-only sequence of
initialization, surrogate/fallback, `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, convergence, confirmation and
finish events. `hpo-control/state.json` is the compact latest replay snapshot. Both are linked from
`result.json` under `summary.adaptive_controller`; they contain scalar identities and decisions,
not checkpoints or model bytes.

The acquisition loop is event-driven rather than round-barrier based. Each terminal observation
re-evaluates `START_NEW`, `DESIGNED_PROBE`, `ADD_SEED`, `PROMOTE_FIDELITY` and
`RESUME_PREEMPTED` for every free slot. It balances two evidence-derived quantities:

- optimization opportunity \(O\): posterior practical improvement still available in the finite
  proposal pool;
- scientific uncertainty \(K\): normalized entropy of unresolved parameter and pairwise
  interaction conclusions.

Their relative magnitude supplies automatic performance/understanding weights; no authored trial
count, phase boundary or hidden YAML threshold controls the transition. Each action records its
performance value, expected reduction of question entropy, observed incremental cost and resulting
priority. These are auditable empirical-Bayes design scores—not calibrated Shannon information
gain. A designed probe may rank above a slightly better predicted candidate when its matched
counterfactual resolves materially more uncertainty per unit cost; a later increase in practical
improvement makes optimization dominant again.
Startup work is submitted gradually and can interleave with model-directed decisions. Pending
identities and exact target fidelities condition the surrogate and prevent duplicate queued seeds.
Undispatched actions form a mutable dispatch buffer. After new evidence, a stale queued action may
be replaced at zero scientific compute cost; `CANCEL_QUEUED_ACTION` records old/new priority,
replacement and reason. Each action records value, score, cost, reason and compact alternatives;
waiting is valid only when no useful in-budget action exists. A running scored action may be
preempted only after 30 seconds,
with an owned non-symlink checkpoint, and when the next useful alternative exceeds the greater of
its original action priority and current continuation value by 50%. Startup actions without a
comparable score and all confirmation Runs are protected. LambdaForge writes a cooperative stop,
waits for the Work/Lightning loop to return at a safe boundary, and later resumes the same Run and
fidelity target from its checkpoint. It never kills the worker for this optimization. The decision
log distinguishes `PREEMPT`, terminal `PAUSE`, `CONTINUE` and `RESUME_PREEMPTED`, including old/new
priorities and the evidence change. `scheduler_preempted` remains neutral evidence, never a
performance prune.

### Scientific questions, confidence and practical regions

The controller and final `StudyAnalysis` call the same `ScientificQuestionAnalyzer`; the TUI does
not fit a competing explanation model. Each parameter has exactly one current conclusion:
`PREFERRED`, `PREFERRED_REGION`, `PRACTICALLY_EQUIVALENT`, `WEAK_PREFERENCE`, `FLAT`,
`CONTEXT_DEPENDENT`, `NO_CLEAR_PREFERENCE` or `UNRESOLVED`. Each eligible pair is similarly
classified as material/weak interaction, additive or unresolved. Conditional parameters report
active and inactive support explicitly, and every generated counterfactual is validated against
the authored `when` predicates.

`confidence` is the deterministic empirical stability of that exact conclusion under bounded
candidate-level delete-d and shared-seed-aware outcome resampling. It can therefore rise for a
well-supported flat or contextual result. It is deliberately separate from search-space coverage,
effect magnitude, reliability and Monte Carlo resolution; it is not a p-value or a frequentist
95% interval. The snapshot records the resampling method/count and missing evidence so an observer
can audit why confidence is low.

When an equivalence margin is authored, the practical optimal region retains candidates whose
uncertainty-aware regret is compatible with it and reports each parameter as constrained or
flexible inside the region. Without such a margin LambdaForge does not invent a percentage-based
scientific tolerance. `DESIGNED_PROBE` targets a named unresolved parameter or interaction and
prefers a close matched comparison; its decision persists target questions, match quality,
predicted objective, uncertainty reduction, cost and rejected alternatives. The implementation is
bounded to the existing deterministic finite proposal pool and pairwise predictive questions. It
does not expand domain boundaries, claim causal effects, or replace fresh-seed confirmation.
From a source checkout, `python -m benchmarks.hpo_scientific_design` runs the deterministic CPU
regression used by the test suite: old optimization-only and scientific-design selectors receive
the same candidate budget on smooth, interaction and flat-feature landscapes. It reports simple
regret and conclusions resolved; the acceptance bound is the authored practical-regret tolerance,
not an assertion that one policy wins every synthetic landscape.

### 7.1 Exact sweeps without HPO

Use a finite exhaustive sweep when the scientific question requires every authored combination:

```yaml
seeds: [7, 17]
search:
  strategy: exhaustive
  optimizer: {values: [adamw, sgd]}
  momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
```

This runs one AdamW variant and two SGD variants for each seed: six Runs exactly. It does not use an
objective, surrogate, pruning, adaptive seed allocation or confirmation. An objective may still be
declared to summarize/rank the complete evidence, but it does not change which Runs execute.

### 7.2 Independent trainings per GPU

The YAML resource block is the fixed outer reservation. `resources.gpu` reserves GPUs for one
study; it is not repeated per child. `search.runs_per_gpu` packs independent spawned Runs onto each
reserved device, and each child sees one GPU plus its divided share of CPU/RAM/storage:

```yaml
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  strategy: adaptive
  trials: 40
  runs_per_gpu: 4
  min_seeds: 1
  early_stopping: {enabled: true, min_step: 5, confirmations: 2,
                   probability_threshold: 0.05, equivalence_margin: 0.0}
  learning_rate: {range: [0.00001, 0.003], scale: log}
  hidden_dim: {values: [64, 128, 256]}
objective: {metric: val_auprc, mode: max}
resources:
  gpu: 2
  gpu_memory: 16GiB
  cpu: 16
  memory: 32GiB
  time: 24h
```

The maximum is \(2\times4=8\) concurrent trainings. Six GPUs with `runs_per_gpu: 2` gives 12;
two GPUs with `runs_per_gpu: 1` gives 2. Packing above one requires `gpu_memory`, interpreted as a
per-Run admission threshold. Before launching each individual child LambdaForge requires

$$
\texttt{gpu\_memory}
\leq \text{currently free device memory}.
$$

`runs_per_gpu` is only the per-device maximum. LambdaForge probes every allocated device and fills
only the slots that pass this check; temporarily unavailable Runs remain queued and the controller
polls again. Launches on one GPU are separated by five seconds so a new process can materialize its
allocation before the next observation. Admission then uses the newly observed free VRAM directly;
it does not multiply `gpu_memory` by the active-Run count or reserve that amount a second time.
Eligible GPUs are ordered by current active-Run count and time since their last launch. Therefore,
when only one global slot is free, GPU index zero cannot repeatedly win and starve another granted
device. Within the remaining candidate/Run/time budgets, each terminal event replans immediately
and supplies a replacement action; low post-hoc analysis confidence is not an early-stop signal.
Every admitted Run has a fresh one-worker spawned process. It is shut down as soon
as its result or error is received, releasing the whole CUDA context; a persistent idle pool could
retain VRAM and deadlock the Runs waiting for admission. A full or smaller GPU is skipped while other usable GPUs continue. Only a
threshold above the total VRAM of every allocated device is impossible and fails as configuration.
The memory probe itself executes in a short-lived child and exits after returning JSON, so the HPO
controller does not appear as one idle CUDA process per GPU and does not consume a
`runs_per_gpu` slot. Wait/admission records are flushed to the Work log.

This dynamic admission is not a promise against unrelated processes allocating memory afterward,
nor a hard memory limiter inside consumer code. Declare a conservative peak requirement.
`max_parallel` can lower the global GPU maximum and bounds CPU-only studies.

Failure isolation follows the Run boundary. CPU and GPU adaptive Runs both use fresh one-worker
processes, so a killed worker cannot poison a shared pool or cancel unrelated candidates. A worker
lost before returning a result and CUDA OOM/allocation failures are retried as a new Attempt up to
`failure_retries` (default 1, allowed 0–3). Compatible checkpoints remain under the same Run and
are discovered by the retry. A CUDA OOM lowers only the affected device's live packing ceiling
below the concurrency observed at failure, so one constrained GPU cannot throttle its healthy
siblings. After that GPU completes one full stable turnover at the reduced ceiling, the controller
tests exactly one additional slot; normal live-free-VRAM admission still applies. Already-running
children are never killed. A repeated resource failure at safer packing becomes
terminal, and ordinary consumer
exceptions are never retried blindly because invalid data/code will not improve with repetition.
Other pending/active Runs continue. The enclosing Work is still reported failed when a Run exhausts
recovery, preserving honest scientific evidence instead of hiding a missing candidate.

### 7.3 Multi-fidelity continuation

`search.fidelity` is optional because only the consumer Work knows whether its budget is cumulative
and resumable. It must not be inferred from a parameter named `epochs`:

```yaml
search:
  trials: 40
  fidelity: {min: 5, max: 100, reduction_factor: 3}
  learning_rate: {range: [0.00001, 0.003], scale: log}
objective: {metric: val_auprc, mode: max}
```

The first Attempt receives `self.fidelity.current == 0` and `target == 5`. A promoted Run keeps the
same scientific identity/checkpoint root and receives cumulative targets 15, 45 and 100. The Work
must advance only from `current` to `target`, save resumable state through `self.checkpoints`, and
log the objective at each useful step. It must not restart the entire target budget. Confirmation
Runs always target `maximum`. `LightningRunner` implements this contract automatically by limiting
`max_epochs`, persisting a managed `last.ckpt` and passing it to the next Attempt; custom trainers
must implement the same semantics explicitly. Without `fidelity`, normal full-budget behavior is
unchanged.

### 7.4 Early stopping contract

Adaptive early stopping is cooperative. Log repeated objective observations with an integer step:

```python
self.metrics.log("val_auprc", score, step=epoch)
if self.stop_requested:
    self.checkpoints.save_json("state.json", state)
    return {"stopped": True}
```

After `early_stopping.min_step`, the controller compares active Runs at a common observed step. A
bounded local-linear posterior projects each curve a short distance forward and carries residual
plus pooled uncertainty. It requests a stop only when the probability of remaining within
`early_stopping.equivalence_margin` of the projected incumbent falls below
`early_stopping.probability_threshold`. This is
more conservative with noisy or still-improving curves than dropping a fixed fraction. By default
the condition must remain true at two distinct common steps (`confirmations: 2`); repeated polling
of the same epoch cannot satisfy it. Set `confirmations: 1` only when an intentionally aggressive
policy is worth the increased risk of reacting to one noisy validation point.
`LightningRunner` forwards its validation metric and honors requests at batch boundaries. A
final-only metric still enables adaptive seeds but cannot stop the current training early.

### 7.5 Live study observability

The outer scheduler Job is an allocation/cancellation boundary, not a reason to merge scientific
meaning. LambdaForge therefore keeps one compact study index next to that Job and one tiny state
record per internal Run. A Run's existing `work.log`, `metrics.jsonl`, `training-metrics.jsonl` and
`result.json` remain authoritative; the index references them and folds only latest scalars and
state. It never copies model checkpoints, output directories or artifact bytes.

the Research Console uses this hierarchy:

```text
Work
  └─ Trial 17: {dropout: 0.21, hidden_dim: 128, ...}
       ├─ Seed 4: running  → curves, timing, isolated live log
       └─ Seed 7: complete → curves, timing, isolated log
```

The candidate view reports proposed versus planned candidates plus
scheduled/active/completed/failed/pruned counts and promotion state. Its
table separates best objective, its seed/epoch and current objective. HPO separately uses the mean
of every completed seed's best checkpoint. Every parameter of the selected Trial is rendered below
it. The seed table includes latest epoch, best epoch, best objective, current objective and GPU
index, followed by a bounded latest-metric preview; the complete Run dashboard remains one Enter
away, and the table reserves at least three seed rows at common terminal heights. `pruned` means
the Run accepted a terminal probabilistic early-stop request; it is not failed, paused or later
resumed, and its persisted reason is shown. A Trial containing only pruned seeds is therefore
`pruned`, not `failed`. Its partial curves remain available as censored diagnostic evidence, but
they are deliberately excluded from completed seed means, surrogate fitting and marginal
objective statistics. Treating a partial best value as a full-budget observation would bias the
search. If any seed receives candidate-level performance pruning, the whole candidate remains
censored; earlier successful seeds are not averaged as a survivor-only objective. LambdaForge
nevertheless uses the information: the HPO read model reports terminal pruning
rate by parameter region. Candidate-level completed/pruned outcomes fit a joint mixed-space
survival model with a smoothed uncertainty interval; multiple seeds do not overcount one candidate.
Infrastructure/resource failures and scheduler preemption are neutral. This is censored negative
evidence, not a fabricated scalar target.

The controller also backtests the configured pruner against fully completed curves and writes the
bounded report to `hpo-control/state.json` as `pruner_calibration`. It records policy inputs,
simulated candidate prunes, Runs/epochs/GPU-seconds that would have been saved, false-prune rate,
regret, probability Brier score, curve RMSE and approximate 90% interval coverage. Tiny studies
report `null` uncertainty fields instead of invented certainty. This audit is retrospective
diagnosis; it never changes recorded objectives or invents the counterfactual result of a genuinely
pruned Run.

the Research Console consumes the same controller evidence rather than fitting a second HPO model. Its candidate
view shows composite utility mean/standard error, aggregation, same-checkpoint raw and normalized
components, weights/contributions, constraints and diagnostic Pareto membership. The HPO panel
shows active/available slots, queued/paused Runs and the latest action's score, controller-value
proxy and cost. A pruned Run expands to its persisted candidate, step/fidelity, prediction uncertainty,
incumbent, probability, threshold, margin, confirmations and curve method.

Open the Study's **HPO** tab and select a parameter with Enter/right to inspect its evidence. For
every detected hyperparameter the summary separates its authored and observed domain, supported
region, importance and reliability. The detail combines bounded candidate-level diagnostics:
numeric response and uncertainty or categorical means, empirical dispersion, coverage and
pairwise predictive gain.
It labels evidence low/medium/high with conservative sample floors and an association score, and proposes the next
comparison that would reduce ambiguity. The header separately reports the controller's actual
latest action plus comparable, provisional and censored counts. Marginal associations can be confounded by correlated parameters, conditional
activation, unequal seeds or partial fidelity, whereas the sampler reasons over the mixed
multivariate space. The console therefore says “appears associated”, never causal or guaranteed.
The controller view keeps two deliberately separate concepts. Parameter and pairwise diagnostics contain these
descriptive, non-causal summaries. `SURROGATE BELIEF` is the retained lightweight output of the
actual last GP/k-NN refresh: backend, target fidelity, pending count, leading predicted candidate or
region and posterior/predictive uncertainty. It is retained independently of the 25-event tail and
contains no pickled model. Interaction-importance methods such as functional ANOVA can summarize a mature surrogate
([fANOVA paper](https://proceedings.mlr.press/v32/hutter14.html)), but applying them to the first few
Trials would create unstable precision theatre. LambdaForge therefore keeps early explanations
auditable and marginal while its actual GP/k-NN controller stays multivariate. Enter/right on a
selected parameter opens a binned numeric response chart or categorical mean bars plus a pairwise
relationship heat table. Pairwise values are leave-one-out k-NN joint predictive gain over the
better marginal predictor, normalized by objective standard deviation. Positive gain says that the
pair helped predict held-out outcomes; it is not a causal interaction score. The bounded raw
response points and a maximum 12×12 relationship matrix make the same view available to wrappers.
Descriptive response and joint coverage begin with two observations; predictive gain begins with
three, while confidence labels retain conservative sample floors. A newer observer locally
rebuilds an older bounded snapshot from candidate telemetry, so live remote studies do not require
a restart merely to obtain a newer diagnostic view.
The console also prints declared outcome constraints and infeasible-candidate counts. The same JSON
is `overview` → `work.items[].study.hpo_analysis`; recent structured actions are in
`.controller`, bounded to 25 events, and actual sampler diagnostics in `.surrogate_belief`.

Threshold-dependent metrics—F1, balanced accuracy, Cohen's kappa, accuracy, precision and
recall—are consumed exactly as logged. LambdaForge does not optimize their classification
threshold, select a per-candidate threshold or equate them with AUROC/AUPRC. Threshold selection is
part of the Work's evaluation protocol and must remain scientifically comparable.

The Run view is split into two regions. The upper region contains the complete aligned parameter
list, live duration/timing summary and up to four learning curves. Compact previous/next buttons and
a numbered selector provide mouse control; `n` and `p` remain keyboard shortcuts. The epoch table
puts every recorded scalar in its own column and scrolls horizontally. Up/down selects an epoch,
the selection is marked red on every applicable curve, and the objective-best epoch remains a green
marker and highlighted row. Enter/right opens the same evidence vertically when that is easier to
read. Duration is computed from `started_at_utc` while active. Measured
`epoch_time_s` and `validation_time_s` replace the labelled elapsed-per-epoch estimate as soon as
telemetry arrives. Run detail refreshes outside the terminal event loop.

Curves are read with a byte bound and reduced to 10–500 representative points while retaining
endpoints and the exact objective-best epoch; the TUI requests 80. Logs are bounded for interactive refresh and no telemetry reader
mutates scientific evidence.

The Textual Research Console consumes these bounded curves without changing their evidence. Rich
terminal presentation is optional to scientific execution and machine output remains plain JSON.
Cluster/study resource details use the same persisted resource snapshots and admission reasons.

Collection and presentation are separate. `LightningRunner` retains every bounded finite scalar,
but projects may select the curves drawn by the Research Console with shell patterns:

```python
from lambdaforge.training import LightningTrainConfig

training = LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_chart_exclude=["*_aux"],
    epoch_metric_display_names={"val_balanced_accuracy": "Balanced accuracy"},
)
```

`epoch_console_include`/`epoch_console_exclude` select the per-epoch human log table; the chart
patterns independently select only TUI presentation. CSV selection remains available through
`epoch_metrics_include`/`epoch_metrics_exclude`. `epoch_metric_display_names` changes only visible
Research Console labels; metric identity, objective lookup and stored values remain unchanged.
Common validation, training and resource names are humanized automatically.

The default order gives the objective first, then `epoch_time_s`, `validation_time_s`, memory,
validation and training metrics. CUDA memory names are precise: `gpu_mem_mb` is peak live tensor
allocation (`max_memory_allocated`); `gpu_reserved_mb` is the current PyTorch caching-allocator
pool; `gpu_peak_reserved_mb` is that pool's epoch peak. Reserved memory contains live allocations
plus reusable cached blocks, so it can legitimately be much larger without being a second
scheduler reservation. Adaptive packing never derives concurrency from this diagnostic value: it
uses explicit per-Run `resources.gpu_memory` as a live admission threshold plus current driver free
memory, filling safe slots rather than requiring the maximum upfront. After each packed Run,
its dedicated process exits and releases the complete CUDA context before the slot is admitted again.

There is no training-specific execution type. Local validation marks any ordinary Work declaring
`search` or multiple `seeds` as expecting a parameter study. Consequently the study screen is
available during `preparing`; it explains that telemetry is pending until the current worker creates its
bounded index. This declaration is also present as `work.items[].study_expected` for machine
clients. `study` remains `null` until live evidence exists. Runs launched by an older worker cannot
retroactively emit per-Run telemetry and require a new execution with the updated remote runtime.
The Study workspace keeps cancellation available while `study` is still `null`; it uses the exact
semantic Work identity and therefore stops every active Attempt without depending on HPO
telemetry. Refresh failures preserve the last good snapshot and are reported inline without
recurring toast notifications.
Several workflow steps, a parallel level and `self.map()` concurrency do not by themselves form a
parameter study: those Works retain the ordinary Attempt/log navigation and create no study index.
This semantic inference avoids a presentation-only YAML switch that could contradict execution.

`LightningRunner` automatically records finite scalar callback metrics after validation. Its
`EpochStats` callback supplies epoch duration and the bridge measures validation duration. Code
using another trainer should emit generic metrics—there is no trainer-specific Work API:

```python
self.metrics.log("train_loss", train_loss, step=epoch)
self.metrics.log("val_auprc", val_auprc, step=epoch)
self.progress.update(epoch, epochs, message="training")
self.log(f"epoch {epoch} complete")  # optional, immediately flushed narration
```

`print()`, standard logging and `self.log()` are captured in the selected Run's `work.log`, so the
drill-down is readable even if the outer Job has 24 active children. `metrics.log` is structured
numeric evidence and drives curves/objectives; `progress.update` is coarse progress; logging is
human explanation. Do not encode curves by printing them.

Machine consumers use the same read model instead of scraping the TUI:

```bash
lf overview --json
lf show WORK --json
lf show WORK --run trial-00017-seed-4 --json
lf logs WORK --run trial-00017-seed-4 --tail 300
```

`work.items[].study` contains the compact catalogue and exact Run keys. `show --run` returns
structured parameters, latest values, down-sampled curves, bounded log, failure and evidence paths;
`--curve-points N` selects 10–500 points. `logs --run` emits only that Run's log. Poll JSON for a
headless UI; `--follow` is intentionally reserved for outer logs because the Research Console already provides
safe live per-Run refresh.

## 8. Execution, identity and reuse

Scientific fingerprint inputs are Work import path, consumer code identity, normalized arguments,
file content hashes, dataset content IDs, seed and trial parameters. Cluster, job/run paths,
timestamps, scheduler ID and Attempt number are operational provenance and do not change the
scientific definition.

An Execution plan is read-only and creates no state. Normal execution returns a verified successful
existing Execution instead of duplicating it. A failed Run may be tried again as the same Run with a
new numbered Attempt from its immutable submitted-configuration snapshot, even if the authored YAML
was edited later; compatible checkpoints set `self.resuming`. `--restart` discards that Run's
checkpoint tree. `--rerun` creates a deliberately distinct Execution even for the same definition.
Remote control-plane submission also refuses an active same-fingerprint/same-target duplicate unless
`--allow-duplicate` is explicit.

## 9. Results and metadata

Each Attempt writes `result.json`, `environment.json`, `work.log`, metrics JSONL, optional progress,
outputs and artifacts. An Execution writes `execution.json` and aggregate `result.json` atomically.

| Category | Fields | Consumer |
|---|---|---|
| identity | Work/class, fingerprints, Execution/Run/Attempt IDs, attempt number | lookup, grouping, retry, debug |
| source | consumer package name/version, Git commit/dirty digest or source digest, LambdaForge version | reproduction/comparison |
| parameters | normalized scalar values plus logical file/dataset identities | fingerprint and comparison |
| environment | Python/platform, critical package versions, Torch/CUDA, environment manifest | diagnosis/reproduction |
| resources | requested resources; allocated/observed only when provider measures them | scheduling/performance |
| timing | created, started, finished, duration; Job adds submitted/updated | lifecycle/monitoring |
| outcome | primary result, named outputs, latest metrics/count, artifacts, datasets, logs, failure | results/search/debug |
| attempt | status, resume flag, failure type/message/traceback/diagnostic, Job retry relation | retry/history |

The exact automatic metadata audit is:

| Durable field | Writer | Consumer/reason |
|---|---|---|
| Work name/class and normalized parameters | `WorkConfig`/`WorkRunner` | signature explanation, reproduction and comparison |
| scientific fingerprint, consumer package and code identity | `WorkRunner` | reuse, duplicate admission, reproduction and scientific grouping |
| Execution/Run/Attempt IDs and attempt number | `WorkRunner` | directory ownership, lookup, retry and debugging |
| typed-input logical source, hash/content ID and size | input resolver | provenance and identity; physical path is operational only |
| requested resources | config/resource resolver | scheduler request and result interpretation |
| created/started/finished/duration | runner and Job provider | age, queue/runtime views and performance interpretation |
| environment-manifest reference | runner | de-duplicates Python/package/Torch/platform evidence |
| primary result, outputs, metric final/count/history | Work services | result queries, live monitoring and objective ranking |
| artifacts and datasets with checksums/IDs | output publisher | fetch, verification, lineage and independent dataset lifecycle |
| status, failure and resume flag | runner | retry choice and diagnosis |
| Job ID when scheduled | control plane/supervisor | connect scientific Attempt to scheduler logs/control |
| allocated/observed usage and progress when available | supervisor/provider | `overview`, the Research Console and performance diagnosis |

YAML output declarations, user-controlled fingerprints, scheduler paths in scientific identity,
duplicate complete package dumps and fabricated allocation/usage fields were excluded because no
safe consumer requires them.

Machine-specific staged paths are recorded as operational evidence but never replace logical input
identity. There are no duplicated YAML output declarations, arbitrary package dumps, fabricated
resource measurements or user-editable fingerprints. `lf results list/show/compare` reads the
Execution result envelopes rather than filesystem guesses about scientific meaning. Comparison
reports count/mean/min/max for each available scalar metric; `--metric NAME --mode min|max` adds an
explicit mean-based ranking without guessing metric direction.

## 10. Clusters and jobs

Humans configure the same profile in bare `lf` → Clusters. Common fields appear first and focused
options provide contextual meaning/risk; advanced SSH, environment, path and GPU policy fields stay
progressively disclosed. Save, test, bootstrap and doctor remain separate explicit actions. The
console calls catalog and credential services directly and never stores passwords in the profile.
Automation uses the native commands:

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --project-root /scratch/USER/my-project
lf clusters bootstrap gpu --dry-run
lf clusters bootstrap gpu
lf doctor --on gpu
lf run experiments/train.yaml --on gpu
```

For an existing profile, the equivalent update is:

```bash
lf clusters set gpu project_root /scratch/USER/my-project
lf clusters show gpu
```

GPU admission is configured once per cluster:

| `gpu_access.mode` | Meaning |
|---|---|
| `auto` | `scheduler` for SLURM; `exclusive` for a direct process host |
| `scheduler` | the batch scheduler owns allocation and isolation |
| `exclusive` | direct Jobs wait for a LambdaForge lease and avoid observed external compute use |
| `shared` | direct Jobs coordinate with each other but may enter an externally occupied GPU |
| `command` | no direct lease; prepend an explicit site claim/launcher argv |

```bash
lf clusters set free-host gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

The command form is an argv sequence and never invokes a shell. Use it when a center requires a
wrapper around every GPU command. At CITIUS, `gpu exec` is the preferred self-contained gpuctl
form for one GPU because allocation lifetime matches the command. For a multi-GPU Work, or if a
site requires a persistent claim, pair claim and release atomically:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

`{gpu_count}` is the only interpolation and comes from the absolute YAML GPU request. Claiming runs
inside durable background submission; the direct supervisor releases after success, failure or
cancellation, and submission failure also attempts release. Persistent claims are rejected for
SLURM profiles. For command/scheduler access, the child must inherit `CUDA_VISIBLE_DEVICES` from the
site. LambdaForge treats indices/UUIDs/MIG UUIDs as opaque grants, only narrows them per Run and
never replaces or broadens them. Missing or duplicate grants fail closed and scheduler allocations
remain exact. Adaptive Studies behind command launchers may scale down to fewer inherited tokens;
their parallelism and memory probes use that exact reduced set. The managed Python executable is
absolute, so no shell/Conda activation must survive the wrapper. `shared` is
an explicit risk choice for permissive hosts, not a hidden default.

`workspace` and `project_root` are deliberately different. `workspace` is LambdaForge-owned state:
bundle cache, environments, Job directories and logs may be garbage-collected according to their
ownership. `project_root` is a researcher-owned persistent partial mirror of the local directory
containing `pyproject.toml`; LambdaForge never synchronizes or deletes it. It must be an absolute,
non-root SSH path. `lf doctor` checks that a configured mirror directory exists.

Path mapping preserves the authored layout. If the YAML is `PROJECT/experiments/design.yaml`, then
`{file: ../data/dna/design}` maps to `REMOTE_PROJECT/data/dna/design`; a relative
`publish_to="../data/results.json"` maps the same way. Absolute `publish_to` remains an explicit
remote path and is not constrained to the mirror.

There are three intentional data paths:

1. Small input (up to 10 MiB): LambdaForge snapshots and transfers it in the bundle.
2. Larger mirrored project input: the researcher/site transfer service places it at the matching
   remote relative path; LambdaForge reads both sides and requires exact kind/size/SHA-256 before
   submitting, so missing, stale, partial and symlinked data fail closed.
3. Large reusable corpus: publish/materialize a managed dataset. This avoids re-hashing a TB-scale
   mirror on every submission and gives immutable identity plus per-cluster placements.

The mirror is intentionally not a hidden `rsync`. Copying hundreds of GB during `lf run` would make
submission latency, quota use and network cost surprising, and many sites require a DTN or another
approved transfer mechanism. Synchronize outside LambdaForge, then run `lf doctor` and `lf run`.

Human bootstrap output reports each preparation phase on stderr and emits a periodic liveness line
during a long solve/install. This does not pollute `--json`, whose stdout remains one
machine-readable document.

Every normal `lf run` validates and identifies locally, creates a durable preparation Job and
returns; this is also true for the built-in `local` target. The detached controller then prepares
the target and asks its scheduler, while the Research Console, `lf show` and `lf logs` remain authoritative.
`--dry-run` is direct and read-only. `--wait-for-submit` keeps the terminal attached through
preparation, not through scientific execution. For SSH, preparation resolves runtime, verifies
declared shared project inputs, builds/cache-checks wheels and bounded-input bundles, and stages one
mutable Job workspace.
OpenSSH uses normal keys, agent, known_hosts and ProxyJump and reuses a private ControlMaster for
its idle persistence period. Password values live only in prompt/keyring/environment providers.

Managed environments are immutable user-space installations identified by exact framework,
consumer, dependency, Python and Torch plans. Runtime discovery can reuse Conda-family Python or a
verified micromamba installation. LambdaForge never installs GPU drivers/system CUDA or silently
falls back to CPU when CUDA is required.

### Project-declared native packages

Native command-line dependencies belong to the installable consumer project, not to each Work YAML.
The optional declaration is strict:

```toml
[tool.lambdaforge.environment]
manager = "conda"
file = "environment.yml"
required_executables = ["mmseqs", "foldseek"]
```

`manager` is currently exactly `conda`; LambdaForge still uses its pinned, checksum-verified
micromamba and does not require a global Conda installation. `file` and `lockfile` are mutually
exclusive paths below the project root. `required_executables` contains bare names, never paths or
commands. `package_cache` is valid only with an explicit offline lock. Hyphenated TOML spellings of
the two array/path options are accepted for compatibility, but underscore spelling is canonical.

The supported `environment.yml` subset is intentionally small:

```yaml
name: wisdom                 # descriptive; it never selects a global named environment
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.11
  - pip
  - biopython>=1.84
  - foldseek
  - mmseqs2
```

Only `name`, `channels` and string `dependencies` are accepted. Prefixes, variables, nested `pip:`
sections and project hooks are rejected. PyTorch/CUDA packages are also rejected here because the
cluster's reviewed Torch plan owns them. Missing baseline Python, pip, CA and OpenSSL packages are
added to an online solve for the already selected Python minor. Consumer Python dependencies remain
normal `pyproject.toml` wheel dependencies.

```bash
lf clusters bootstrap gpu --project . --dry-run
lf clusters bootstrap gpu --project .
lf doctor --on gpu
lf run experiments/design.yaml --on gpu
```

Dry-run does not download a manager, solve channels, build wheels or mutate the cluster. It reports
the specification hash, requested packages, target Conda subdir, required executables and whether
first provisioning needs channel connectivity. If an exact solve receipt already exists it reports
that solve. Actual bootstrap/automatic preparation resolves on the remote platform, normalizes the
exact name/version/build/channel/subdir inventory and places it in `EnvironmentIdentity` together
with specification bytes, platform, manager version, exact framework/consumer/wheelhouse bytes,
Python requirement, Torch plan and online/offline policies.

The final environment is one prefix, not an activated base plus an opaque PATH overlay. Micromamba
creates a temporary Conda prefix from the exact solve; that prefix's Python then installs Torch and
the exact wheels. LambdaForge runs `pip check`, rechecks Conda inventory, TLS, framework/Torch/CUDA
and every required executable before writing a receipt and atomically renaming the prefix. The
receipt records executable path/version plus its owning Conda package, package version, build,
channel and subdir. Concurrent builders share a bounded per-identity lock; only a complete verified
prefix is reusable. `ToolService` searches the active Python prefix first, so `self.tools.require()`
remains a check rather than an installer.

Installed inventory verification enriches `micromamba list --json` with the regular records under
the prefix's `conda-meta`; this preserves the real `linux-*` versus `noarch` subdir even with manager
versions that omit it from list output. Missing metadata is an environment error, while a genuine
difference is reported as a bounded package/field summary rather than a dump of the whole prefix.

For native offline provisioning use a Conda `@EXPLICIT` lock with one target platform and SHA-256
on every URL, plus the exact package bytes in a project-owned cache:

```toml
[tool.lambdaforge.environment]
manager = "conda"
lockfile = "locks/linux-64.explicit"
package_cache = "vendor/conda-linux-64"
required_executables = ["mmseqs", "foldseek"]
```

```text
# platform: linux-64
@EXPLICIT
https://repo.example/linux-64/python-3.11.9-h123.conda#sha256=<64 hex digits>
```

Every locked basename and checksum must match a regular cache file; links, duplicates, credentials,
queries and mismatched platforms fail before transfer. The cache is content-addressed, staged once,
used with `--offline`, and reconstructible after the immutable prefix is published. Fully offline
installation of Python wheels and Torch additionally requires the existing reviewed `wheelhouse`
cluster option. One lock is platform-specific; keep separate lock/cache pairs for `linux-64`,
`linux-aarch64` or `linux-ppc64le`. `environment: existing` deliberately keeps its manual contract
and does not install the project declaration.

Every submitted Job receives its own mutable workspace copied from the immutable bundle cache;
direct SSH jobs never execute in the shared cache. The embedded controller-side code identity keeps
local and staged scientific fingerprints equal. On a remote Work, `source_dir` is this staged,
installable consumer source context rather than the controller's nonexistent physical path.

the Research Console is the semantic interactive view. Up/down traverse clusters and Work rows as one list.
For a study, Enter or right arrow drills through Trials, seed Runs and the live Run dashboard; `a`
opens its outer numbered scheduler Attempts. Other Works drill directly into numbered Attempts and
full logs. A terminal study without a published study index falls back to its outer Attempts, which
preserves ordinary output and the terminal error even when failure preceded telemetry publication.
Within a failed seed Run, `e` expands type/message/phase/result path/traceback; in a full Attempt log
the same key toggles persisted traceback detail. Left goes back. Every open log refreshes
automatically while preserving manual scroll;
when positioned at the end it follows new output. Long operational Job IDs stay out of this primary
path and remain available
through `lf jobs` and the machine-readable overview. `self.progress.update` reaches the supervisor's durable
state and therefore this view; active SLURM Jobs read the same bounded progress snapshot from their
owned workspace. `lf overview --json` exposes equivalent machine data.
`lf jobs ...` remains the advanced authority for scheduler IDs, raw logs, cancellation, retry and
provider reconciliation. Each local Job persists its absolute provider roots, so detached
submission, later observation, logs and deletion remain independent of the caller's current
directory. Old relative records are resolved from their recorded source configuration. Provider
outage yields unknown/last-known state, not fake failure; once the provider is reachable the real
state replaces unknown and the stale reachability error is removed.

Cancellation follows the hierarchy. `lf cancel WORK` cancels every non-terminal Job grouped into
that semantic Work, attempting the remaining Jobs even if one provider call fails and then reporting
an incomplete operation. `lf jobs cancel JOB` and cancelling a selected numbered Attempt affect one
Job only. A direct supervisor first signals the verified scientific process group, then finds the
same-user descendants carrying the exact inherited `LAMBDAFORGE_JOB_ID`, including reparented
workers and processes that opened another session. It terminates, escalates after a bounded grace
period and verifies zero survivors before acknowledging cancellation. The same cleanup runs when a
main process exits, so a nominal success cannot orphan trainer/dataloader workers. This marker is
internal provenance; consumer code must not set or replace it. SLURM uses the configured cancellation
command, whose site policy must cancel the complete allocation and its steps. Cancellation is
idempotent for direct Jobs already recorded as cancelled: repeating `lf cancel WORK` reruns the
ownership verification and repairs orphan leakage from older releases.

Each Work Job also publishes a bounded structured result at its exact Job root. `lf logs WORK` and
`lf jobs logs JOB` append a `Scientific failure` section after the tailed stream when captured
output does not already contain the diagnosis. The section keeps exception type, message,
phase/location and the persisted `result.json` path; `--verbose` or `--debug` adds its traceback
unless that traceback is already visible. `--tail N` limits captured streams, never this terminal
cause. Add `--json` for structured `failure`, `failures` and `result_path` fields. `lf show` and
`lf jobs show` expose the same failure. Pre-change 0.12 workspaces are read from their bounded legacy
result location, while new Jobs use the exact Job-root copy.

## 11. Cleanup and safety

Ownership categories:

- Attempt/Execution owned: results, logs, registered artifacts, progress and exact workspaces;
- Run owned: checkpoints and Attempt history;
- durable independent: published datasets and their registry placements;
- shared immutable: managed Python environments and bundles referenced by Jobs;
- reconstructible: caches and unreferenced bundle/environment/runtime entries;
- external: source files and dataset locations never owned merely because they were referenced.

`lf delete WORK` previews. For local direct execution it removes one exact Execution tree; for
scheduled work it requires terminal Jobs and removes exact owned Job roots. Both paths write a small
deletion receipt so repeats converge. `lf datasets delete` and materialization are also
preview-first. `lf clean` plans reconstructible cache collection plus the same hash-verified
terminal-Attempt compaction used automatically at completion. It retains active references,
unpublished successful artifacts and lightweight evidence, rejects unsafe roots/symlinks and is
idempotent. This also lets upgraded installations reclaim safe bulk left by older Jobs.

Inside the Research Console, `d` confirms deletion of the selected terminal Work or numbered Attempt;
`D` confirms deletion of every terminal history entry. Active Jobs are never removed. Deletion runs
outside the terminal event loop, removes the exact provider workspace plus local Job events and
submission record, and preserves published datasets, caches, environments and unrelated Jobs.
`lf jobs clear` provides the non-interactive preview and `lf jobs clear --apply` performs the same
whole-history operation, reporting failures without discarding the affected local record.

A narrower automatic retention pass runs at terminal completion. It deletes only partial managed
artifacts from failed/interrupted Attempts and verified internal duplicates of successful
`publish_to` outputs. It preserves every lightweight fact needed by the Research Console, `lf logs`, results
and reproduction; `retention.json` records reclaimed bytes. Superseded immutable managed
environments are also pruned after a verified replacement is activated, except the active prefix
and prefixes referenced by live Jobs. Bootstrap and normal automatic preparation share this rule;
Attempt environment provenance remains after reconstructible environment bytes are collected.

## 12. CLI reference

| Command | Unique purpose |
|---|---|
| `init` | scaffold an installable Work project |
| `project` | show the current project root and storage identity |
| `validate` | complete local configuration/class/input validation |
| `explain` | signature/doc/default/resource explanation |
| `run` | the only scientific execution command |
| bare `lf`, `overview` | Textual human console and machine global control-plane view |
| `show`, `logs`, `cancel`, `retry`, `delete` | semantic Work operations |
| `jobs ...` | low-level Job control; `clear [--apply]` cleans terminal history |
| `clusters ...`, `doctor`, `resources` | scriptable target setup and diagnosis; bootstrap accepts `--project` and read-only `--dry-run` |
| `datasets ...` | inspect/verify/place/delete published dataset versions |
| `results list/show/compare/analyze/report` | query, analyze and export Work Execution evidence |
| `clean` | preview/apply safe cache and terminal-artifact compaction |

`lf help`, `lf --help`, `lf help clusters add` and conventional nested `--help` all exit zero.
All command failures go through stable diagnostic categories/exit codes; add `--json` for tooling
and `--debug` for tracebacks. Unsupported authoring commands and dataset build do not exist because
all computation is `lf run`.

## 13. Clustering

Install with `python -m pip install "lambdaforge[clustering]"`. Importing `lambdaforge` and
`lambdaforge.clustering` remains valid without scikit-learn; invoking a clusterer without the extra
returns an actionable installation error. The backend is scikit-learn >=1.3,<2 because that is the
first supported line containing its mature HDBSCAN implementation.

```python
import lambdaforge as lf

clusterer = lf.clustering.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    distance="euclidean",
    threads=8,
)
result = clusterer.cluster(features)
```

The package is outside `lambdaforge.nn`: clustering accepts general feature matrices and is not a
neural layer. It does not create another component registry. Direct concrete classes are clearer
than a name factory, and custom behavior extends the small `Clusterer` ABC directly.

All clusterers accept a finite, nonempty NumPy array or PyTorch tensor of shape `[N, F]`. Tensors are
explicitly detached and normalized through CPU because sklearn clustering is non-differentiable.
For a custom `Distance`, pairwise evaluation follows that module's parameter/buffer device and
dtype, then detaches the matrix to CPU for the backend. Every class
implements `cluster(X) -> ClusteringResult`. The immutable result has `labels`, `n_clusters`,
`noise_count`, `noise_fraction` and serializable `diagnostics`; optional `centers`, `inertia` and
`probabilities` are `None` unless the algorithm produces real values. Noise is consistently `-1`.

| Clusterer | Important parameters | Distances | Specific evidence |
|---|---|---|---|
| `KMeans` | `n_clusters`, `seed`, `n_init`, `max_iter`, `threads` | Euclidean or squared-Euclidean semantics only | centers, inertia |
| `MiniBatchKMeans` | KMeans fields plus `batch_size` | Euclidean or squared-Euclidean semantics only | centers, inertia |
| `DBSCAN` | `eps`, `min_samples`, `threads` | Euclidean, Manhattan, Minkowski, Chebyshev, cosine; `Distance` precomputed | noise |
| `HDBSCAN` | cluster size/samples/selection, `threads` | same density native set; `Distance` precomputed | noise, probabilities |
| `Agglomerative` | `n_clusters`, `linkage`, threshold | Ward: Euclidean only; other linkages use declared native/precomputed | labels |

String aliases are case-insensitive and canonical (`l1/cityblock`, `l2`, `sqeuclidean`). There is
exactly one custom distance contract: `lambdaforge.nn.distances.Distance`. Existing implementations
are `EuclideanDistance`, `SquaredEuclideanDistance`, `ManhattanDistance`, `MinkowskiDistance`,
`ChebyshevDistance`, `CosineDistance`, `AngularDistance` and `MahalanobisDistance`. Compatible known
Euclidean objects use the native route. Other objects explicitly materialize an `[N, N]` matrix
under `torch.no_grad`, validate its shape/finiteness/nonnegativity, detach/move it to CPU and pass
`metric="precomputed"`. The default `max_pairwise_bytes=512 MiB` rejects dangerous allocations;
advanced callers may choose another positive limit or `None` only after accepting O(N²) memory.

`adjusted_rand_index` and `silhouette_score` normalize common backend metrics. `stability(X,
candidates, reference=None)` runs only the explicitly supplied clusterers and returns reference
scores, the complete pairwise ARI matrix, median, minimum and results. It contains no domain
threshold. Feature scaling, missing-value policy, PCA and interpretation remain explicit project
science.

## 14. Reusable neural components

Reusable PyTorch components are independent of Work/YAML: instantiate them in project Python and
import from documented namespaces. `lambdaforge.nn.models` re-exports the public model catalog:

| Family | Public choices |
|---|---|
| general/dense | `Model`, `MLP`, `CNN2D`, `ECMP`, `BatchedKNN` |
| graph | `GCN`, `GAT`, `GATv2`, `GIN`, `GraphSAGE`, `PNA`, `RelationalGCN`, `GraphTransformer`, `EGNN`, `TensorFieldNetwork`, their public layers, `GraphReadout` |
| sequence | `RNNModel`, `GRUModel`, `LSTMModel`, `TemporalConvNet`, Transformer encoder/decoder/seq2seq, `ConformerModel`, `StateSpaceAdapter` |
| sets | `DeepSets`, `SetTransformer` |
| tabular | `ResidualMLP`, `FTTransformer`, `TabNet`, `SAINT`, `AutoInt`, `DeepFM` |
| vision | `ResNet2D`, `UNet2D`, `MobileNetV2`, `ConvNeXt2D`, `VisionTransformer2D`, `FeaturePyramidNetwork2D` and public blocks/backbone |
| composition | `AutoEncoder`, `VariationalAutoEncoder`, `SiameseModel`, `MultiTaskModel`, `MixtureOfExperts`, `EnsembleModel` |
| generative | `GaussianDiffusion`, `DiffusionSchedule`, `VectorQuantizedAutoEncoder` |
| scientific/implicit | `NeuralODE`, `NeuralCDE`, `DeepONet`, `FourierNeuralOperator1D`, `SIREN` |
| differentiable trees | `ObliviousDecisionTree`, `NODE`, `GradTree`, `GRANDE` |

The adjacent public namespaces are `lambdaforge.nn.losses`, `activations`, `normalizations`,
`pooling`, `distances`, `similarities`, `kernels`, `encodings`, `regularization` and `uncertainty`;
classic reporting metrics are under `lambdaforge.metrics`. Concrete signatures and class docstrings
are the parameter authority. For example:

```python
from lambdaforge.nn.models import GCN, MLP, ResNet2D
from lambdaforge.nn.losses import CrossEntropyLoss
```

This audit deliberately adds no `GNN` catch-all, second MLP or model factory. The existing catalog
already covers the common dense, graph, sequence, image, set and tabular baselines. A generic alias
would hide scientifically material topology, aggregation, readout, equivariance and output-shape
decisions while duplicating tested classes. Domain architectures remain ordinary project modules;
reusable missing primitives should be added only with a precise tensor contract and conformance
tests.

## 15. Architecture and extension boundaries

`WorkConfig` owns strict parsing/introspection/expansion. `WorkRunner` owns runtime binding, one
`run()` call and result finalization. `ControlPlane` owns target selection, duplicate admission,
environment/bundle preparation and scheduling. `Transport` and `Scheduler` remain provider
boundaries. Job, dataset and storage services retain their independent durable responsibilities.

The concrete ownership map is:

| Component | Owns | Why it is this component |
|---|---|---|
| `WorkConfig` | schema, Python signature validation, expansion and immutable submitted values | configuration decisions finish before scheduling |
| `WorkRunner` | Execution/Run/Attempt directories, identity, input resolution, one `run()` call and result envelope | it is the local scientific lifecycle boundary |
| `AdaptiveSearchPolicy` and WorkRunner controller | seed budgets, conservative promotion, per-GPU child slots and cooperative stops | adaptive decisions stay above individual Work science and below the one outer reservation |
| `WorkRuntime` | one bound set of services and immutable views | services must never become process-global mutable state |
| `ManagedFileStore` | safe keys, records, per-key lock, validation, fingerprint and atomic promotion | cache and checkpoint files need identical byte-lifecycle invariants |
| `WorkCache` | simple typed values, managed file/fetch/rate-limit policy and GC lease | cache lifetime differs from checkpoints even though byte publication is shared |
| `CheckpointCollection` | Run-owned JSON and file state | retry/resume state must remain outside Attempt outputs and cache GC |
| `OutputCollection` and retention operations | unique outputs, atomic publication, exact evidence and safe terminal bulk compaction | only output metadata can prove a copy is redundant; provider cleanup covers interrupted Attempts |
| `ToolService` | executable resolution, argv process lifecycle, logs, child env and used-tool ledger | scientific code chooses the command; framework owns safe execution mechanics |
| `EnvironmentManifest` | one software/hardware/Git/plugin/tool provenance document | consumers need one source of environment truth, not copied fragments |
| `SubmissionService`/`ControlPlane` | durable asynchronous preparation and target selection | terminal responsiveness and remote preparation are operational, not scientific |
| `NativeEnvironmentSpecification`/`NativeEnvironmentPlanner` | strict project declaration, platform solve and exact Conda inventory | native software belongs to deployment identity, never Work parameters or runtime hooks |
| `ManagedEnvironmentProvider` | verified temporary venv or unified Conda prefix, pip/Torch install and atomic publication | only this boundary can turn a complete dependency plan into an executable environment |
| `GpuAccessPolicy` | scheduler/exclusive/shared/claim-wrapper admission semantics | cluster ownership policy must not leak into scientific YAML or Work code |
| `StorageOperations` | exact-root GC, environment pruning, Job deletion and Attempt compaction | destructive filesystem work remains bounded, idempotent and independently auditable |
| `Transport` | connection/file transfer | SSH/local transport must not decide scheduler semantics |
| `Scheduler` | submit/observe/signal provider Jobs | direct processes and SLURM expose different authoritative lifecycle APIs |
| `DatasetPublisher`/`DatasetRegistry` | immutable dataset bytes/index and logical placements | datasets outlive one Work Attempt and require an independent lifecycle |
| `Clusterer` adapters | feature validation, capability checks and backend-normalized result | clustering is scientific Python, independent of Work and YAML execution |

An Attempt therefore proceeds in one direction: resolve typed inputs and identity; create bound
services; capture the initial environment; call project `run()` with logs redirected; validate the
primary JSON result; finalize every managed output; recapture only the used-tool extension of the
same environment manifest; remove temporaries; persist `WorkResult`. A failure skips managed-output
publication, retains failure/log evidence and never converts a partial dataset staging tree into a
published version.

Researchers extend ordinary Python: PyTorch modules, losses, metrics, optimizers, data readers and
domain helper classes are constructed inside Work code. Runtime extensions should remain cohesive
services behind `self.outputs/metrics/checkpoints/cache/tools/progress`; do not turn Work into a generic
service locator. New YAML vocabulary is justified only by a genuine researcher-controlled planning
decision, not by an internal object that can stay in Python.

## 16. Study Analysis

Study Analysis is a deterministic post-hoc layer over persisted study evidence. It does not propose
Runs and never changes the controller. A terminal study automatically writes
`EXECUTION/analysis.json`; explicit routes reuse or recompute the same document:

```bash
lf results analyze EXECUTION
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output report.html
```

The HTML route requires `lambdaforge[analysis-report]`; JSON analysis and the console do not. The
report embeds its data and Plotly runtime and makes no network request.

### 16.1 Evidence semantics

For each Run, `current_observed_objective` is the most recent step where every objective component
is present, `best_observed_objective` is the best such step, `final_objective` requires terminal
full fidelity, and `selection_objective` is candidate-level selection evidence. Incomplete
composites report missing components and the latest complete step. Pruned performance is censored:
its partial curve and best observation are retained, but no exact final score is invented.
Candidate statistics use independent full-fidelity seeds, mean/SD/SE and deterministic bootstrap
95% intervals when enough repeated evidence supports them. Empirical winner stability is
`insufficient_evidence` whenever a leading candidate has fewer than two comparable final seeds; a
one-element bootstrap is never presented as stability. Any pooled/model-based seed uncertainty is
reported in a separate labelled field. Shared seeds produce paired comparisons; screening and
fresh-seed confirmation remain separate so a lucky winner is not silently validated. Confirmation
uses the persisted scientific equivalence margin; a documented 2% fallback exists only for old
records that did not persist one.

### 16.2 Effects, coverage and reliability

Analysis fits a deterministic mixed-space predictive surrogate and validates it with actual
leave-one-candidate-out predictions—not individual epochs, duplicated seeds or metadata claiming a
different fold scheme. It reports the exact method/fold count, RMSE, MAE, rank correlation,
interval coverage and a quality label. One central min/max ordering governs winners, responses,
regions and confirmation. Exact authored `when` predicates are retained through reference design,
response and pair generation; invalid inactive/active combinations are rejected rather than
randomly sampled. A common reference design supports global functional variance importance; an
observed top-candidate distribution answers a different question about where good observed
candidates concentrate. Numeric/categorical responses and pair surfaces expose non-additivity with
uncertainty and extrapolation markers.

Coverage includes observed marginal bins/categories, conditional activation and joint resolution.
It does not claim coverage of the controller proposal pool unless that pool was actually persisted.
Boundary diagnostics distinguish possible from likely saturation. Resource telemetry separates
median intrinsic cost per comparable full-fidelity Run from total controller spend (Runs/seeds/
promotions and aggregate time); resource Pareto uses only the former. Scientific component Pareto
remains separate. Constraints, pruning audit and confirmation are retained. Findings are
deterministic, evidence-linked sentences. All effects are observational/predictive, never causal;
reliability combines support, CV quality, coverage and extrapolation, so poor validation cannot
produce a high-confidence finding merely because many rows exist.

Live documents are `provisional` and cheaper; terminal documents are `final`. An evidence
fingerprint makes repeated non-recompute calls idempotent. Writes are atomic and versioned so 0.13
study records remain analyzable when they contain sufficient objective metadata.

### 16.3 Admission evidence

Study telemetry persists structured admission alongside scientific progress: active and queued
Runs, maximum parallelism, CPU/RAM context, per-GPU free/required VRAM and the exact wait reason.
`runs_per_gpu` is a ceiling. Capacity pressure queues and polls; it is not a failed Run. Launches on
one GPU are staggered, while other GPUs continue independently. With two GPUs, five allowed slots
per device and sufficient resources, the controller may admit ten child Runs; helper probes are
short-lived and do not consume scientific slots. A transient probe error is an observation outage,
not a scientific failure: new admission pauses, active child processes remain untouched and the
probe retries. Its bounded stdout/stderr is preserved for diagnosis. A persistent outage becomes
terminal only after there are no active Runs and the bounded retry budget is exhausted.

## 17. Research Console

Bare `lf` opens the Textual Research Console only when input/output are interactive. Otherwise it
prints CLI help and exits. The only primary screens are Overview, Work, Studies, Clusters, Datasets
and Results. Enter/right pushes the selected entity workspace; Esc/left pops one level; Home returns
to the root; `?` explains scientific markers. This is a real navigation stack, not a generic JSON
modal. The 0.13 `lf top`, `clusters setup` and `clusters modify` commands are removed; scripts retain
every non-interactive command.

Overview contains three independent panels for Clusters, ordinary Work and Studies. Its tables use
short progress phrases, while the selection panel formats semantic facts rather than dumping JSON.
Cluster selection accumulates a bounded live CPU/RAM/GPU history and reports both whole-cluster
load and the personal LambdaForge share when observable; requested allocation stays distinct from
measured use. Refresh reconstructs read models without changing the focused panel or exact entity.
The centered loading state is used only until the first successful snapshot. Every later refresh
keeps that snapshot on screen, reports its age in a compact footer and marks a transient failure as
stale. The Work browser and bounded Work logs poll live with one in-flight request per view.
The same dashboard appears in the Clusters browser and cluster workspace. Terminal Studies remain
Studies and retain their last telemetry-producing Attempt; a terminal `study: null` read model is
valid and opens a degraded Study/log view while the service attempts one bounded telemetry reload.
The sidebar Exit button and root `q` binding are explicit.

Overview enters a Work, Study or cluster directly. The main scientific hierarchy is `Study → Trial
→ Seed → Epoch`: Trials can be searched/sorted and distinguish selection/current/best evidence;
Seeds distinguish latest/best/final objective and censored partial evidence; Curves draw at most
four configured metrics per page (`n`/`p`), preserve all metrics in tables, and mark selected/best/
latest epochs. Epoch detail shows every same-step scalar. Active Seed telemetry and isolated logs
refresh without resetting a deliberately selected epoch or a user's log scroll position.
Study and Seed workspaces show an explicit loading indicator during remote reads, then reveal their
content atomically or report a retrieval failure; an empty table is therefore never used as a
loading placeholder. Every previous breadcrumb segment is clickable and uses the same real screen
stack as the Back button.

Study Overview combines status, candidate/Run counts, current leader, compute time, objective
history, Run-state distribution and leading parameters. Run-state bars have their exact counts
printed below them; the chart never asks the user to estimate a small count from a large axis.
Trial marks have a separate column, so
winner/Pareto/censored symbols never cover the identifier. The HPO tab summarizes strategy,
human-readable objective, evidence and scheduler. Its parameter table separates authored and
observed domains, best supported region, importance and confidence. Opening a parameter shows its
predictive response with uncertainty bounds, observed support and dispersion, plus persisted
pairwise interaction diagnostics. Before a final Execution exists, the same panel consumes the
bounded live HPO snapshot and labels observed response plus numeric/verbal confidence explicitly;
it refreshes without waiting for Study completion. These are predictive associations, not causal
effects. Terminal charts retain categorical labels and show exact point/bar values on click. They
omit ambiguous uncertainty pseudo-lines; numeric uncertainty remains in the evidence table. An
explicit **Interactive HTML** action uses optional Plotly for hoverable learning curves, genuine
shaded uncertainty, interaction heatmaps and numeric 3D surfaces. Coverage is presented as cards,
marginal bars and exact authored/observed support tables rather than raw JSON. Contextual `?` help
defines objective, importance, reliability, confidence, joint gain, coverage and stop semantics.
The full
persisted controller sequence is an Action / Trial / Reason table; opening a row shows its complete
stored thresholds and inputs.

Study tabs also expose actual controller/scheduler/pruning/admission evidence, Resources, logs and
Analysis. Analysis uses the persisted document and has
Summary, Parameters, Interactions, Coverage, Seeds, Pareto and Findings views; it never fits a
second controller model in a widget. Work owns Executions/Attempts/logs and links Studies. Cluster
workspaces expose test/doctor/bootstrap, redacted credentials, GPU policy, resources and storage.
Dataset Summary asynchronously reads exact split and per-split primary-target counts from the
logical index. This does not recurse through large assets. Members automatically loads at most 200
records when opened. Physical Stats and Integrity expose a centered explanation and explicit action
in their own tabs because a tree walk or checksum verification may be expensive. The always-visible
DatasetVersion delete control immediately reports preview/apply state, blocks repeat activation,
previews all placements and then requires explicit confirmation.
`registered_but_missing` is a convergent stale-registration cleanup: already-absent bytes are not
treated as a filesystem failure, and the empty logical record is removed from controller and target
registries. Results reuse the same analysis evidence and can recompute/export it.

The console imports domain services directly; it never executes `lf` through a subprocess. Slow
filesystem, analysis and provider operations run in daemon workers; refreshes for the same view do
not overlap, so a slow remote probe cannot multiply SSH/provider calls. Hidden root browsers do not
load until selected. **Run Work** offers a YAML-filtered directory tree and a bounded list composed
from existing local Job history plus persistent MRU choices. Browse and recent selection both fill
one path field without launching. The single Submit action validates, explains and only then
submits to the captured visible cluster selection; its busy state disables repeat actions and
reports the active phase. Validation transitions directly into enqueueing, and failure to render a
complex bounded preview cannot suppress submission or its visible result. The MRU file contains only absolute local paths,
display names and timestamps, filters missing files, and never copies authored YAML or data. An
invalid YAML is kept in the launch dialog and shown once with source path, exact line/column,
bounded excerpt and an actionable hint. Work display names are not identifiers: simultaneous
local/remote executions use distinct `work_id` row keys, and cancel/delete passes that exact
identity to the domain service. One
shared provider factory reuses the console transport, and cluster
operations are serialized; live resource probes pause during doctor/bootstrap. The cluster activity
panel shows every bootstrap phase plus elapsed liveness, offers three preset heights and can be
resized by dragging its separator. Its upper content is an independently scrollable viewport with
a protected minimum height; an enlarged log therefore cannot hide actions, tabs or evidence beyond
recovery. Applying bootstrap requires confirmation; mutation confirmations and the modal Work plan
use bounded semantic sections instead of serialized JSON. Plan bootstrap stays read-only. A provider outage retains
the last successful snapshot with a visible stale warning.
The Add/Edit cluster modal is a complete editor for the durable `ClusterProfile`, not a reduced
setup wizard. Guided tabs own transport/authentication references, execution backend, workspace/
project/data identity, Python/environment/PyTorch policy, storage retention, SSH connection policy
and `gpu_access`. Command access exposes its argv prefix and atomic claim/release pair; the text is
parsed with `shlex` into argv and never executed through a shell. Advanced YAML round-trips
`ssh_options`, the general command prefix, scheduler options, resource mapping, scheduler
directives/commands and job-script policy. Guided fields take explicit precedence. The assembled
mapping is passed through `ClusterProfile.from_mapping` before atomic persistence, while password
values remain exclusively in the credential service.
Destructive actions show the exact target, require confirmation and preserve the underlying
preview/apply, exact-root and symlink-safe rules. Machine clients must consume CLI `--json`, not
screen text. `Ctrl+P` lists only actions with real handlers; the auditable support table explicitly
labels other routes CLI-only instead of claiming parity from a button/name. Plotly remains optional
because terminal operation uses bounded persisted JSON.

Resource history and learning curves are rendered by the declared native `textual-plot` widget,
using automatic numeric ticks, Braille-resolution lines and interactive pan/zoom. Resource charts
use a fixed five-minute window labelled `5m` through `now`; their colour key sits above the plots
instead of covering data. Seed plots retain four-metric pages and overlay the selected epoch in red
and the objective-best epoch in green. Changing page restores automatic bounds for the new metrics
instead of retaining a prior metric's pan/zoom. LambdaForge owns only bounded samples and semantic
series; it does not implement axes or
rasterization. Every nested workspace has a visible **Back to …** control in addition to keyboard
navigation. The sidebar visibly separates Action, Browse and Session controls. Cluster and Dataset details use
compact cards and bounded semantic sections rather than unformatted provider/manifest JSON.
Pruned Run/seed cells describe absent terminal evidence as `not final · pruned` and absent partial
observations as `not observed`; `unavailable` remains reserved for actual retrieval/capability gaps.

The old `LiveJobMonitor` implementation is not a public interface. It remains temporarily as a
regression oracle while advanced routes still classified `CLI-ONLY BY DESIGN` are migrated; it may
be deleted only after equivalent service-backed flows and tests exist in Textual.

## 18. Project isolation

LambdaForge selects one consumer project from the nearest `pyproject.toml`, starting at the current
directory. A virtual environment selects Python packages; it does **not** select LambdaForge state.
Use `lf project` (or `lf project --json`) before operating from an unfamiliar shell to see the exact
root and project ID.

New projects created by `lf init` receive a stable identifier. Existing projects may declare one
before their next submission:

```toml
[tool.lambdaforge]
project_id = "protein-design"
```

It must contain 1–80 letters, digits, `_` or `-`, beginning with a letter or digit. Without this
field LambdaForge derives an ID from the resolved local root, so two checkouts with the same folder
name remain isolated. Moving such a checkout changes its derived ID; add an explicit ID first if
the remote namespace must survive relocation. Giving two checkouts the same explicit ID is an
intentional request to share their LambdaForge namespace.

The ownership boundary is:

| State | Scope |
|---|---|
| user cluster profiles and credential references | shared, so a cluster is configured once |
| cooperative GPU/process leases | shared per host, so projects cannot over-allocate one device |
| controller Jobs, groups and recent YAML choices | current project only |
| local results, datasets and reconstructible cache | current project under `.lambdaforge/` |
| remote Jobs, state, datasets, environments and cache | current project namespace |

For default remote storage, new work uses
`<workspace>/.lambdaforge/projects/<project-id>/{state,cache,jobs}`. A custom storage root keeps the
same ownership boundary by adding `projects/<project-id>`. These are operational paths; scientific
identity and result reuse do not include the project ID. The shared lease root deliberately remains
outside the project namespace.

The user catalog at `~/.config/lambdaforge/clusters.yaml` remains the common source for host,
authentication and site policies. A project's `lambdaforge.clusters.yaml` is a recursive overlay,
so it can define only its remote mirror while inheriting the rest:

```yaml
clusters:
  gpu:
    project_root: /scratch/user/protein-design
```

The Research Console, `lf overview`, Work operations, Results and Dataset services all use the same
project context. A selector from another project is absent rather than actionable. This prevents a
same-name Work in two projects from becoming the same control-plane object; exact IDs remain the
identity within each project.

The upgrade is non-destructive. Legacy controller Job records are discovered only when their
recorded source YAML belongs to the current nearest project, and their recorded provider/work paths
are used to reconnect. They are not copied between projects and existing remote directories are not
moved automatically. New submissions use the namespaced layout. Profiles, credentials and shared
hardware leases require no migration.
