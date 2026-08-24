# LambdaForge 0.12 manual

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
python -m pip install lambdaforge==0.12.0
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
| final scientific file/tree | `outputs.file/directory` | belongs to Attempt | no | yes |
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
optional gzip decompression. Its rate limiter is thread-safe and shared only by one Work cache
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

`publish_to` is optional on both `file` and `directory`. A relative destination is resolved from
`self.source_dir`; an absolute path is used explicitly. LambdaForge first validates and fingerprints
the Attempt-owned artifact, then publishes a per-destination atomic copy. An existing identical
copy is reused, while different content is refused unless `overwrite=True`. The managed artifact
remains authoritative even when a publication copy exists, and its result metadata records the
physical `published_to` path. On a remote Job this is a remote-host path: use an absolute persistent
cluster path when the copy must outlive Job cleanup. LambdaForge does not silently copy arbitrary
large outputs back to the controller. Publication also refuses symbolic-link traversal, file versus
directory type changes and any directory destination that contains (or is contained by) its managed
source; `overwrite=True` is never permission to replace the project or Attempt root.

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

A file is resolved relative to YAML, checked, hashed and passed as `Path`. Remote bundles copy only
files/directories below the configured small-input limit. A larger input produces a pre-submission
error explaining that it must use durable cluster storage or a managed dataset. This makes transfer
cost deliberate.

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
spawned process; seed/search members of that definition run serially inside its fixed allocation.
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
  trials: 12
  learning_rate: {range: [0.0001, 0.01], scale: log}
  width: {values: [128, 256]}
objective: {metric: val_score, mode: max}
```

Finite-only dimensions form a Cartesian product. A numeric range uses deterministic random search
for `trials` unique variants. Every variant/seed is a Run; values are normal `run()` parameters,
`self.trial` describes the variant, and `self.seed` describes the seed. The objective is selected
only from Runs belonging to that Work and must be logged under the exact scalar metric name. When a
variant has several seeds, ranking uses their arithmetic mean and preserves the contributing Run
IDs and seeds in the Execution summary; it never selects a lucky individual seed.

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
| allocated/observed usage and progress when available | supervisor/provider | `overview`, `lf top` and performance diagnosis |

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

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work
lf clusters bootstrap gpu --dry-run
lf clusters bootstrap gpu
lf doctor --on gpu
lf run experiments/train.yaml --on gpu
```

Every normal `lf run` validates and identifies locally, creates a durable preparation Job and
returns; this is also true for the built-in `local` target. The detached controller then prepares
the target and asks its scheduler, while `lf top`, `lf show` and `lf logs` remain authoritative.
`--dry-run` is direct and read-only. `--wait-for-submit` keeps the terminal attached through
preparation, not through scientific execution. For SSH, preparation resolves runtime,
builds/cache-checks wheels and bounded-input bundles, and stages one mutable Job workspace.
OpenSSH uses normal keys, agent, known_hosts and ProxyJump and reuses a private ControlMaster for
its idle persistence period. Password values live only in prompt/keyring/environment providers.

Managed environments are immutable user-space installations identified by exact framework,
consumer, dependency, Python and Torch plans. Runtime discovery can reuse Conda-family Python or a
verified micromamba installation. LambdaForge never installs GPU drivers/system CUDA or silently
falls back to CPU when CUDA is required.

Every submitted Job receives its own mutable workspace copied from the immutable bundle cache;
direct SSH jobs never execute in the shared cache. The embedded controller-side code identity keeps
local and staged scientific fingerprints equal. On a remote Work, `source_dir` is this staged,
installable consumer source context rather than the controller's nonexistent physical path.

`lf top` is the semantic interactive view. Up/down traverse clusters and Work rows as one list;
Enter or right arrow drills forward (cluster detail, Work, numbered Attempt, then full logs), while
left arrow goes back. Long operational Job IDs stay out of this primary path and remain available
through `lf jobs` and the machine-readable overview. `self.progress.update` reaches the supervisor's durable
state and therefore this view; active SLURM Jobs read the same bounded progress snapshot from their
owned workspace. `lf overview --json` exposes equivalent machine data.
`lf jobs ...` remains the advanced authority for scheduler IDs, raw logs, cancellation, retry and
provider reconciliation. Each local Job persists its absolute provider roots, so detached
submission, later observation, logs and deletion remain independent of the caller's current
directory. Old relative records are resolved from their recorded source configuration. Provider
outage yields unknown/last-known state, not fake failure; once the provider is reachable the real
state replaces unknown and the stale reachability error is removed.

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
preview-first. `lf clean` only plans reconstructible cache collection, retains active references,
rejects unsafe roots/symlinks and is idempotent.

Inside `lf top`, `d` confirms deletion of the selected terminal Work or numbered Attempt;
`D` confirms deletion of every terminal history entry. Active Jobs are never removed. Deletion runs
outside the terminal event loop, removes the exact provider workspace plus local Job events and
submission record, and preserves published datasets, caches, environments and unrelated Jobs.
`lf jobs clear` provides the non-interactive preview and `lf jobs clear --apply` performs the same
whole-history operation, reporting failures without discarding the affected local record.

## 12. CLI reference

| Command | Unique purpose |
|---|---|
| `init` | scaffold an installable Work project |
| `validate` | complete local configuration/class/input validation |
| `explain` | signature/doc/default/resource explanation |
| `run` | the only scientific execution command |
| `top`, `overview` | human live and machine global control-plane views |
| `show`, `logs`, `cancel`, `retry`, `delete` | semantic Work operations |
| `jobs ...` | low-level Job control; `clear [--apply]` cleans terminal history |
| `clusters ...`, `doctor`, `resources` | target setup and diagnosis |
| `datasets ...` | inspect/verify/place/delete published dataset versions |
| `results list/show/compare` | query Work Execution results |
| `clean` | preview/apply shared reconstructible cache GC |

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
| `WorkRuntime` | one bound set of services and immutable views | services must never become process-global mutable state |
| `ManagedFileStore` | safe keys, records, per-key lock, validation, fingerprint and atomic promotion | cache and checkpoint files need identical byte-lifecycle invariants |
| `WorkCache` | simple typed values, managed file/fetch/rate-limit policy and GC lease | cache lifetime differs from checkpoints even though byte publication is shared |
| `CheckpointCollection` | Run-owned JSON and file state | retry/resume state must remain outside Attempt outputs and cache GC |
| `OutputCollection` | unique names, allocated outputs, artifact finalization, optional explicit copies and dataset publication handoff | success depends on complete verifiable evidence |
| `ToolService` | executable resolution, argv process lifecycle, logs, child env and used-tool ledger | scientific code chooses the command; framework owns safe execution mechanics |
| `EnvironmentManifest` | one software/hardware/Git/plugin/tool provenance document | consumers need one source of environment truth, not copied fragments |
| `SubmissionService`/`ControlPlane` | durable asynchronous preparation and target selection | terminal responsiveness and remote preparation are operational, not scientific |
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
