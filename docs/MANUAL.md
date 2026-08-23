# LambdaForge 0.12 manual

## Contents

1. [Mental model](#1-mental-model)
2. [Installation and project layout](#2-installation-and-project-layout)
3. [Work API](#3-work-api)
4. [YAML reference](#4-yaml-reference)
5. [Files and datasets](#5-files-and-datasets)
6. [Sequence, parallelism, seeds and search](#6-sequence-parallelism-seeds-and-search)
7. [Execution, identity and reuse](#7-execution-identity-and-reuse)
8. [Results and metadata](#8-results-and-metadata)
9. [Clusters and jobs](#9-clusters-and-jobs)
10. [Cleanup and safety](#10-cleanup-and-safety)
11. [CLI reference](#11-cli-reference)
12. [Architecture and extension boundaries](#12-architecture-and-extension-boundaries)

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
| `outputs` | append-only per Attempt | values, artifacts and dataset publication |
| `metrics` | append-only | scalar history and latest values |
| `checkpoints` | durable per Run | explicit safe resume state |
| `cache` | reconstructible | identity-scoped reusable computation cache |
| `progress` | replaceable snapshot | optional live completed/total/message |
| `resources` | read-only | requested CPU, memory, GPU, GPU memory, time, storage, processes |
| `seed` | read-only | current seed or `None` |
| `trial` | read-only | current search index/parameters or `None` |
| `run_dir` | owned per Attempt | durable files before registration |
| `temp_dir` | owned per Attempt | removed after finalization |
| `source_dir` | read-only | consumer project/source context |
| `resuming` | read-only | compatible checkpoint state existed at start |
| `map(...)` | execution helper | bounded ordered intra-job concurrency and JSON resume |

`outputs.value` accepts JSON-compatible values once per name. `outputs.artifact` verifies a regular
file/directory, rejects symlink traversal, copies external paths under managed ownership and records
SHA-256/size/role/media type. `metrics.log` accepts finite numeric scalars and optional step/split.
Normal `logging`, stdout and stderr remain normal Python and are captured into `work.log`.

Checkpoint/cache names must stay below their roots. JSON checkpoint writes are atomic and corrupt
state is an error. `map` requires `key`, rejects duplicates, preserves input order, cancels pending
futures after failure and uses thread or spawn-process executors. Its result checkpoints are safe
JSON, never pickle.

## 4. YAML reference

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

## 5. Files and datasets

```yaml
with:
  manifest: {file: ../data/manifest.json}
  corpus: {dataset: research-corpus@3}
```

A file is resolved relative to YAML, checked, hashed and passed as `Path`. Remote bundles copy only
files/directories below the configured small-input limit. A larger input produces a pre-submission
error explaining that it must use durable cluster storage or a managed dataset. This makes transfer
cost deliberate.

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

## 6. Sequence, parallelism, seeds and search

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

## 7. Execution, identity and reuse

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

## 8. Results and metadata

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
| outcome | primary result, named outputs, latest metrics/count, artifacts, datasets, logs, failure | results/HPO/debug |
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

## 9. Clusters and jobs

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work
lf clusters bootstrap gpu --dry-run
lf clusters bootstrap gpu
lf doctor --on gpu
lf run experiments/train.yaml --on gpu
```

Remote run validates and identifies locally, creates a durable local preparation Job, returns, then
resolves runtime, builds/cache-checks wheels and small-input bundle, stages it and asks the scheduler.
`--wait-for-submit` keeps the terminal attached through preparation. OpenSSH uses normal keys,
agent, known_hosts and ProxyJump and reuses a private ControlMaster for its idle persistence period.
Password values live only in prompt/keyring/environment providers.

Managed environments are immutable user-space installations identified by exact framework,
consumer, dependency, Python and Torch plans. Runtime discovery can reuse Conda-family Python or a
verified micromamba installation. LambdaForge never installs GPU drivers/system CUDA or silently
falls back to CPU when CUDA is required.

Every submitted Job receives its own mutable workspace copied from the immutable bundle cache;
direct SSH jobs never execute in the shared cache. The embedded controller-side code identity keeps
local and staged scientific fingerprints equal. On a remote Work, `source_dir` is this staged,
installable consumer source context rather than the controller's nonexistent physical path.

`lf top` is the semantic interactive view. `self.progress.update` reaches the supervisor's durable
state and therefore this view; active SLURM Jobs read the same bounded progress snapshot from their
owned workspace. `lf overview --json` exposes equivalent machine data.
`lf jobs ...` remains the advanced authority for scheduler IDs, raw logs, cancellation, retry and
provider reconciliation. Provider outage yields unknown/last-known state, not fake failure.

## 10. Cleanup and safety

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

## 11. CLI reference

| Command | Unique purpose |
|---|---|
| `init` | scaffold an installable Work project |
| `validate` | complete local configuration/class/input validation |
| `explain` | signature/doc/default/resource explanation |
| `run` | the only scientific execution command |
| `top`, `overview` | human live and machine global control-plane views |
| `show`, `logs`, `cancel`, `retry`, `delete` | semantic Work operations |
| `jobs ...` | low-level operational Job control |
| `clusters ...`, `doctor`, `resources` | target setup and diagnosis |
| `datasets ...` | inspect/verify/place/delete published dataset versions |
| `results list/show/compare` | query Work Execution results |
| `clean` | preview/apply shared reconstructible cache GC |

All command failures go through stable diagnostic categories/exit codes; add `--json` for tooling
and `--debug` for tracebacks. Unsupported authoring commands and dataset build do not exist because
all computation is `lf run`.

## 12. Architecture and extension boundaries

`WorkConfig` owns strict parsing/introspection/expansion. `WorkRunner` owns runtime binding, one
`run()` call and result finalization. `ControlPlane` owns target selection, duplicate admission,
environment/bundle preparation and scheduling. `Transport` and `Scheduler` remain provider
boundaries. Job, dataset and storage services retain their independent durable responsibilities.

Researchers extend ordinary Python: PyTorch modules, losses, metrics, optimizers, data readers and
domain helper classes are constructed inside Work code. Runtime extensions should remain cohesive
services behind `self.outputs/metrics/checkpoints/cache/progress`; do not turn Work into a generic
service locator. New YAML vocabulary is justified only by a genuine researcher-controlled planning
decision, not by an internal object that can stay in Python.
