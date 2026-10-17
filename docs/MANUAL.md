# LambdaForge 0.19 manual

This is the single public framework manual; detailed guides are chapters here. Internal contributor contracts and historical acceptance remain separate in `ai/`.

[Español](MANUAL.es.md) · [Contributor contracts](ai/FRAMEWORK_CONTRACTS.md)

## Contents

- [1. Mental model](#1-mental-model)
- [2. Installation and project layout](#2-installation-and-project-layout)
- [3. Work API](#3-work-api)
- [4. Managed files, cache, map, outputs and tools](#4-managed-files-cache-map-outputs-and-tools)
  - [4.1 Storage meanings](#41-storage-meanings)
  - [4.2 ManagedFile and cache](#42-managedfile-and-cache)
  - [4.3 Plain and resumable map](#43-plain-and-resumable-map)
  - [4.4 Outputs and checkpoints](#44-outputs-and-checkpoints)
  - [4.5 External tools](#45-external-tools)
- [5. YAML reference](#5-yaml-reference)
- [6. Files and datasets](#6-files-and-datasets)
  - [6.1 Reuse, reconstruction and publication recovery](#61-reuse-reconstruction-and-publication-recovery)
  - [6.2 Native compressed replication](#62-native-compressed-replication)
  - [6.3 Explicit conflict management](#63-explicit-conflict-management)
- [7. Sequence, parallelism, seeds and search](#7-sequence-parallelism-seeds-and-search)
  - [7.1 Intent-first defaults and resolved authority](#71-intent-first-defaults-and-resolved-authority)
  - [Scientific questions, confidence and practical regions](#scientific-questions-confidence-and-practical-regions)
  - [7.1 Exact sweeps without HPO](#71-exact-sweeps-without-hpo)
  - [7.2 Independent trainings per GPU](#72-independent-trainings-per-gpu)
  - [7.3 Multi-fidelity continuation](#73-multi-fidelity-continuation)
  - [7.4 Early stopping contract](#74-early-stopping-contract)
  - [7.5 Live study observability](#75-live-study-observability)
- [8. Execution, identity and reuse](#8-execution-identity-and-reuse)
  - [Recover an interrupted Study](#recover-an-interrupted-study)
- [9. Results and metadata](#9-results-and-metadata)
- [10. Clusters and jobs](#10-clusters-and-jobs)
  - [Project-declared native packages](#project-declared-native-packages)
- [11. Cleanup and safety](#11-cleanup-and-safety)
- [12. CLI reference](#12-cli-reference)
- [13. Clustering](#13-clustering)
- [14. Reusable neural components](#14-reusable-neural-components)
- [15. Architecture and extension boundaries](#15-architecture-and-extension-boundaries)
- [16. Study Analysis](#16-study-analysis)
  - [16.1 Evidence semantics](#161-evidence-semantics)
  - [16.2 Effects, coverage and reliability](#162-effects-coverage-and-reliability)
  - [16.3 Admission evidence](#163-admission-evidence)
  - [16.4 Research workspace and declarations](#164-research-workspace-and-declarations)
- [17. Research Console](#17-research-console)
- [18. Project isolation](#18-project-isolation)
- [19. Operational reliability](#19-operational-reliability)
- [20. Fused Gromov–Wasserstein distance (FGW)](#20-fused-gromovwasserstein-distance-fgw)
  - [Inputs and quick start](#inputs-and-quick-start)
  - [Arbitrary features, precomputed costs and geometry](#arbitrary-features-precomputed-costs-and-geometry)
  - [Objective, coupling and interpretation](#objective-coupling-and-interpretation)
  - [Resource bounds and multiple objects](#resource-bounds-and-multiple-objects)
- [21. Work results, historical evidence and independent products](#guide-work-results)
  - [Inspect and report](#guide-work-results-inspect-and-report)
  - [Choose an exact historical dependency](#guide-work-results-choose-an-exact-historical-dependency)
  - [Explicit promotion from an ordinary Work](#guide-work-results-explicit-promotion-from-an-ordinary-work)
  - [Portable evidence and remote availability](#guide-work-results-portable-evidence-and-remote-availability)
  - [Evidence across fixed Study extensions](#guide-work-results-evidence-across-fixed-study-extensions)
- [22. Durable scientific products: implemented foundation](#guide-products)
  - [Identities and contracts](#guide-products-identities-and-contracts)
  - [Explicit publication API](#guide-products-explicit-publication-api)
  - [Native CLI](#guide-products-native-cli)
  - [Post-Study ModelSet selection](#guide-products-post-study-modelset-selection)
  - [Native StudyDecision](#guide-products-native-studydecision)
  - [Declared post-Study publication and publication-only retry](#guide-products-declared-post-study-publication-and-publication-only-retry)
  - [Typed Work dependencies](#guide-products-typed-work-dependencies)
  - [Portable product export/import](#guide-products-portable-product-exportimport)
  - [Import a portable Study without executing it](#guide-products-import-a-portable-study-without-executing-it)
- [23. Conditional scientific Studies](#guide-conditional-studies)
  - [1. One question, one Study](#guide-conditional-studies-1-one-question-one-study)
  - [2. Activation grammar](#guide-conditional-studies-2-activation-grammar)
  - [3. Validate and understand the plan](#guide-conditional-studies-3-validate-and-understand-the-plan)
  - [4. Time and target capacity](#guide-conditional-studies-4-time-and-target-capacity)
  - [5. Identity and existing evidence](#guide-conditional-studies-5-identity-and-existing-evidence)
- [24. Incremental fixed Study extensions](#guide-study-extensions)
  - [CLI](#guide-study-extensions-cli)
  - [Regularization example](#guide-study-extensions-regularization-example)
  - [Conditional and selective additions](#guide-study-extensions-conditional-and-selective-additions)
  - [Identities, compatibility and evidence](#guide-study-extensions-identities-compatibility-and-evidence)
  - [Supported execution contracts](#guide-study-extensions-supported-execution-contracts)
  - [Cumulative analysis and Console](#guide-study-extensions-cumulative-analysis-and-console)
  - [Budgets, failures and durability](#guide-study-extensions-budgets-failures-and-durability)
  - [Verification boundary](#guide-study-extensions-verification-boundary)
- [25. Published datasets and scientific reconstruction](#guide-dataset-reconstruction)
  - [Choose the intent before computing](#guide-dataset-reconstruction-choose-the-intent-before-computing)
  - [Publish once, replicate exact content](#guide-dataset-reconstruction-publish-once-replicate-exact-content)
  - [Three separate identities](#guide-dataset-reconstruction-three-separate-identities)
  - [Compare entire sealed reconstructions](#guide-dataset-reconstruction-compare-entire-sealed-reconstructions)
  - [Publication failure without recomputation](#guide-dataset-reconstruction-publication-failure-without-recomputation)
  - [Conflicting local and remote registries](#guide-dataset-reconstruction-conflicting-local-and-remote-registries)
  - [Durable equivalence certificates](#guide-dataset-reconstruction-durable-equivalence-certificates)
  - [Required WISDOM changes (not performed by LambdaForge)](#guide-dataset-reconstruction-required-wisdom-changes-not-performed-by-lambdaforge)
- [26. Storage admission and safe cleanup](#guide-storage)
  - [Inspect and clear](#guide-storage-inspect-and-clear)
  - [Admission](#guide-storage-admission)
  - [What collection can remove](#guide-storage-what-collection-can-remove)
  - [Checkpoints and publication](#guide-storage-checkpoints-and-publication)
  - [Current boundaries](#guide-storage-current-boundaries)
- [27. Full-resolution managed project report tabs](#guide-managed-reports)
  - [Browser API and lifecycle](#guide-managed-reports-browser-api-and-lifecycle)
  - [Native remote and offline viewing](#guide-managed-reports-native-remote-and-offline-viewing)
  - [Rebuilding presentation from retained inference only](#guide-managed-reports-rebuilding-presentation-from-retained-inference-only)
- [28. Study research workspace](#guide-research-analysis)
  - [1. Open a workspace](#guide-research-analysis-1-open-a-workspace)
  - [2. Declare meaning, not execution policy](#guide-research-analysis-2-declare-meaning-not-execution-policy)
  - [3. Families and questions](#guide-research-analysis-3-families-and-questions)
  - [4. How discovery works](#guide-research-analysis-4-how-discovery-works)
  - [5. Explore and save](#guide-research-analysis-5-explore-and-save)
  - [6. Architecture, safety and limitations](#guide-research-analysis-6-architecture-safety-and-limitations)
  - [7. Implementation report](#guide-research-analysis-7-implementation-report)
  - [8. Second-round simplification report](#guide-research-analysis-8-second-round-simplification-report)
- [29. Coordinated Studies: implementation status and contracts](#guide-coordinated-studies)
  - [Status: fresh adaptive and fixed Fleet launch; recovery/transfer integration pending](#guide-coordinated-studies-status-fresh-adaptive-and-fixed-fleet-launch-recoverytransfer-integration-pending)
  - [Fleet catalog](#guide-coordinated-studies-fleet-catalog)
  - [Ownership and persistence](#guide-coordinated-studies-ownership-and-persistence)
  - [Reproducible foundation acceptance tests](#guide-coordinated-studies-reproducible-foundation-acceptance-tests)
- [30. Operational health, monitoring and deliberate recovery](#guide-operational-reliability)
  - [Inspect without starting computation](#guide-operational-reliability-inspect-without-starting-computation)
  - [Independent project monitor](#guide-operational-reliability-independent-project-monitor)
  - [Optional email, durable incidents and no polling spam](#guide-operational-reliability-optional-email-durable-incidents-and-no-polling-spam)
  - [Research Console](#guide-operational-reliability-research-console)
  - [Recovery and relocated YAML](#guide-operational-reliability-recovery-and-relocated-yaml)
  - [Managed checkpoint policy and evidence](#guide-operational-reliability-managed-checkpoint-policy-and-evidence)
  - [Emergency procedure](#guide-operational-reliability-emergency-procedure)
  - [Deliberate extension versus recovery](#guide-operational-reliability-deliberate-extension-versus-recovery)
- [31. Shared-host resource safety — implementation in progress](#guide-resource-safety)
  - [Observation is not a grant](#guide-resource-safety-observation-is-not-a-grant)
  - [Cross-Study startup](#guide-resource-safety-cross-study-startup)
  - [RAM and CPU](#guide-resource-safety-ram-and-cpu)
  - [Profiles, device circuits and diagnostic reports](#guide-resource-safety-profiles-device-circuits-and-diagnostic-reports)
  - [Site reservations and visibility](#guide-resource-safety-site-reservations-and-visibility)
  - [Remaining acceptance](#guide-resource-safety-remaining-acceptance)
- [32. Developer experience — incremental implementation](#guide-developer-experience)
  - [Start and discover](#guide-developer-experience-start-and-discover)
  - [Understand YAML before running](#guide-developer-experience-understand-yaml-before-running)
  - [Inspect, diagnose and observe](#guide-developer-experience-inspect-diagnose-and-observe)
  - [Small result tables without a full experiment export](#guide-developer-experience-small-result-tables-without-a-full-experiment-export)
  - [Agent setup without pretending a connection exists](#guide-developer-experience-agent-setup-without-pretending-a-connection-exists)
  - [Console and Skills](#guide-developer-experience-console-and-skills)
  - [Remaining scope](#guide-developer-experience-remaining-scope)
- [33. Project-pinned agent integration](#guide-integration)
  - [Quick start](#guide-integration-quick-start)
  - [MCP and clients](#guide-integration-mcp-and-clients)
  - [Local human approvals](#guide-integration-local-human-approvals)
  - [Optional dependencies and paired HTTP](#guide-integration-optional-dependencies-and-paired-http)
  - [Portable knowledge](#guide-integration-portable-knowledge)
  - [Python, GUI and sync](#guide-integration-python-gui-and-sync)

[Work results guide](#guide-work-results) covers ordinary Work evidence,
reports, exact historical dependencies and explicit compressed materialization. These reuse the runner,
result store, product registry and portable import/export without inventing Study inference.

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
| sequential state required to resume a Run | `checkpoints.file/save_json` | yes | only when no longer required, under retention policy | resume evidence |
| final scientific file/tree | `outputs.file/directory` | belongs to Attempt | only a verified published duplicate | yes |
| disposable intermediate | `temp_dir` | no | automatic | no |

`cache.path`, `checkpoints.path` and direct `run_dir` writes remain explicit interoperability
escapes. They are not the ordinary pattern because their caller owns validation, atomicity and
registration.

Checkpoints are recovery state, not permanent model outputs. `checkpoints.pin(name, reason=...)`
protects owned named state; `unpin(name)` only changes metadata. Publish models with
`outputs.from_checkpoint(name, checkpoint, release=True)` or datasets from `source_checkpoint`.
Release requires independent verified publication and successful Execution finalization.
See [Storage admission and safe cleanup](#guide-storage) for examples and retention.

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
count and SHA-256 are checked before scheduler submission and again in the worker. Directory
identity is a versioned canonical record stream: NFC logical path components are ordered by UTF-8
bytes, separators/types/lengths and empty directories are explicit, while filesystem enumeration,
locale, timestamps and permissions are irrelevant. Historical bundles retain their original hash
reader. Nothing is
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

### 6.1 Reuse, reconstruction and publication recovery

Use `lf datasets preflight NAME@VERSION --intent publish|reuse|rebuild --on CLUSTER` before costly
computation. Producing Works call `outputs.dataset_preflight(...)` first: arbitrary Python output
names cannot be inferred before `run()`. `outputs.dataset(..., intent="rebuild",
scientific_identity=...)` seals a pinned unregistered candidate; exact checksums stay unchanged.
`lf datasets compare ROOT_A ROOT_B --on CLUSTER --verifier project.module:function
--policy policy.json --output report.json` verifies all bytes before project scientific comparison.
`lf datasets publish-candidate ROOT --version NEW --on CLUSTER` previews bytes-only recovery;
`--apply` publishes without repeating calculation. The Console discovers all configured locations
and marks conflicting identities. [Dataset reconstruction](#guide-dataset-reconstruction) documents
the complete public contracts, safety, retention and required WISDOM changes.

ROOT may also be a saved pre-copy publication request (0.17.1): exact member/source checksums and
the original declaration are verified, then `--apply` copies/registers without executing Work code.
For older retained indexes, explicitly prepare the original declaration with
`DatasetPublisher.prepare_publication`; never infer missing scientific contracts or relabel bytes.

### 6.2 Native compressed replication

`lf datasets replicate NAME@VERSION --source SOURCE --destination TARGET` previews; `--apply`
copies exact bytes and registers the verified placement on the target and controller.
`--route auto|direct|relay` prefers site SSH or explicitly requires direct/compressed relay.
Site SSH never forwards controller credentials; relay uses no local disk staging. Console
**Datasets → DatasetVersion → Replicate…** uses the same service, confirmation and background
progress. Keep the session open until completion. The target needs `storage.dataset_root` and
an available LambdaForge runtime, not a scientific Work/GPU claim. Checksums, safe staging,
destination-volume admission and immutable identity remain mandatory. Conflicts fail closed;
identical destinations are reused. See [replication](#guide-dataset-reconstruction-publish-once-replicate-exact-content).

### 6.3 Explicit conflict management

Dataset conflict management uses native preview/apply: `datasets adopt NAME@VERSION --on SOURCE
--content-id ID` chooses a fully verified future controller reference; `datasets delete/remove
NAME@VERSION --on TARGET --content-id ID` selects one exact target identity even in a conflict.
Delete removes owned bytes, remove keeps them; both block active consumers and archive old indexes.
Console **Datasets → exact identity → Manage copies…** exposes the same confirmed actions.
Other copies/old Runs are unchanged; unpinned operations remain fail-closed.
See [details](#guide-dataset-reconstruction-conflicting-local-and-remote-registries).

## 7. Sequence, parallelism, seeds and search

Conditional branches belong to one Study when they answer one joint experimental question. Use
`when: {parent: {in: [a, b]}}` (AND across parents), or historical scalar equality. Inactive keys
are omitted, dependencies are order-independent, and the same geometry governs generation and
analysis. Root `resources` inherits into steps; root `with`, seeds/replicates, search/sweep,
execution, objective and analysis are rejected rather than ignored. The numbered
[conditional Study guide](#guide-conditional-studies) covers exact counts, references, canonical
identity, capacity checks and scheduler wall-time versus logical Study budgets.

### 7.1 Intent-first defaults and resolved authority

The common path needs no authored seed integers or controller tuning. With `search.space` and an
objective, omitted policy means: project `replicate` stream, minimum replication one, balanced
goal, automatic geometry-derived startup/sampler/pruning/seed racing/fresh confirmation,
incremental deterministic candidate generation, scientific convergence, and automatic internal
parallelism. The YAML filename stem is the default name. External ownership remains explicit:
`resources.gpu` is never changed to “all available GPUs”.

`lf config resolve FILE` prints the exact `ResolvedStudyConfiguration`; `--format json|yaml` is
stable for tools. The same object is persisted in `execution.json` and
`resolved-configuration.json` before computation. It includes policy versions, resolved execution
limits, initial concrete seed identity, design, goal, stop/confirmation/candidate-generation
policies, objective and hard resources. `lf seeds [--role replicate|confirmation] [--count N]`
inspects the project streams. Each result and Study Run stores seed value, ordinal, role, namespace
and version; export includes the authored YAML, resolved authority and all actual identities.

The stream is a SHA-256-derived keyed permutation over 31-bit ordinals. The confirmation role owns
the other high-bit domain, so each role has (2^{31}) collision-free, NumPy-compatible values and
the two automatic domains cannot overlap. Explicit `seeds` remain exact and finite. Top-level
`replicates: N` or `sweep.replicates: N` selects project ordinals `[0, N)`.

No candidate budget means convergence-controlled adaptive search. Its Sobol/mixed generator is
prefix-stable: a bounded computational window is materialized and extended only when the shared
scientific policy requests another region. Existing candidates never change. `goal: optimize`,
`balanced`, or `understand` adjusts the value of unresolved questions within that policy. Explicit
candidate/Run/time budgets remain hard ceilings and exhaustion is not relabelled convergence.
Stop reasons such as `CONVERGED_BALANCED`, `CANDIDATE_BUDGET`, `RUN_BUDGET`, `TIME_BUDGET`,
`SEARCH_SPACE_EXHAUSTED` and `NO_POSITIVE_VALUE_ACTION` are persisted separately from scientific
status.

An auto-replicated sweep starts every cell at `replicate[0]`. Its primary paired differences use a
simultaneous, time-uniform predictable plug-in empirical-Bernstein confidence sequence
(`paired-pm-eb-cs-v2`), whose width adapts to observed variance while retaining optional-stopping
validity. Multiplicity covers only the authored primary family: treatment versus `reference`, or
the pairwise relations needed to identify a practical top set. The controller may commit exactly
one next complete block as lookahead and orders its cells by prior duration; that block cannot alter
the decision until complete. Thus a slow cell does not leave safe GPUs idle, while at most one
extra block is spent. Point leader, exact descriptive conclusion/stability and formal sequential
state/coverage are persisted separately. A bounded objective is required; equivalence additionally
requires an authored `practical_margin`. The default family alpha is `0.05`; the advanced
`sweep.sequential_alpha` override must be in `(0,1)` and is invalid with fixed `replicates`. Sweep
HPO pruning and per-cell seed racing stay disabled.

Runtime v2 consumes the persisted acquisition prefix (`seed_metadata.ordinal` for streams,
authored order for explicit seeds), never numeric seed order. Complete later blocks cannot skip
an incomplete earlier block. Recovery restores the existing one-block commitment; previews do not
alter it. Historical v1 manifests/decisions remain inspectable and unchanged. Predictive hypotheses,
descriptive support/stability and formal resolution are separate: missing response support must not
inherit a model's stability, and stopping on an incomplete block is not a formal conclusion.
See the [correctness audit and remaining acceptance boundaries](ai/IMPLEMENTATION.md#history-scientific-audit).

Long Studies retain their immutable scientific initialization in `study/controller.json` →
`initialization`. This includes the complete objective, practical margin, authored parameter space
(types, log scales and conditional `when` rules), resolved policy and seed-stream metadata.
`recent` is only a 25-action display tail. Older Studies recover initialization once from their
append-only controller history when available; missing required metadata produces an explicit
error. Live summaries reuse the controller's persisted scientific snapshot, so opening the console
cannot reinterpret the equivalence margin or turn a log search dimension into a linear one.

Seed variability is learned from repeated candidates, never from differences between architectures.
`unresolved` means no within-candidate scale; `provisional` means a scale is identified but its
variance or contextual support remains uncertain. `calibrated` requires the sampling uncertainty
of variance to be smaller than the supported contextual coverage. A single candidate with two seeds
remains provisional. The model retains deterministic scaled-inverse-chi-square variance
realizations, conditional on the normal within-candidate approximation, and reports a sigma range,
residual degrees of freedom, shared-seed support, context coverage and possible heterogeneous noise.
The range is model uncertainty, not a distribution-free guarantee. Common seed effects remain part
of population uncertainty even when paired comparisons cancel them. The conservative predictive
fallback also retains noisy-context variance instead of treating a pooled scale as homogeneous.
The normal-model variance uncertainty follows the degrees-of-freedom relationship explained in
the [NIST statistical handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm).

Another seed competes with parameter and interaction probes using the same cost-normalized planner.
Its value includes the uncertainty reduction it can provide across unresolved scientific questions.
Contenders and under-repeated contexts can therefore justify shared seeds after initial coverage;
no universal seeds-per-candidate quota is imposed. This global calibration value decays with
additional evidence. Fresh confirmation seeds remain disjoint. A checkpointed fidelity continuation
may instead carry `PRUNER_CALIBRATION_EVIDENCE`: completing that curve can improve endpoint
calibration. These Runs are protected from competitive pruning; poor calibration still disables
strong pruning. No diagnostic metric becomes a hidden selection objective.

Resource observations distinguish an exact peak from a complete sampled physical trajectory and
an interrupted lower bound. NVML readings retain their physical high-water mark across worker exit,
alongside allocator peaks, sampling cadence and an uncertainty envelope. A completed sampled
envelope can train memory prediction without claiming an exact lifetime peak; incomplete Runs stay
censored. `exact_history_count` and `sampled_history_count` expose the distinction. The safety floor
and live free VRAM still constrain every new placement. Repeated blocked decisions are coalesced
in `hpo-control/resources/blocked-summary.json`; raw replay evidence remains factual.

Stepped metrics expose `step/sec`, while `progress.update(completed=...)` exposes progress-unit/sec.
These bounded signals feed generic throughput learning without application-specific metric names.
ARI separates memory feasibility from usefulness: it compares aggregate throughput at measured
concurrency levels, and can leave safe capacity unused when another Run would reduce evidence per
time. Configured parallelism is a ceiling, not a target process count.

Live scientific inference uses 32 evidence realizations. Human Study Analysis/report generation uses
the cached higher-precision budget (up to 256) outside the scheduling heartbeat. Both retain the
same objective, margin and authored geometry. Parameter views and HTML report the realization count,
Monte Carlo resolution, direct/censored/predictive support and seed-noise diagnostics. Stability
means agreement of a qualitative conclusion across those realizations, **not** the probability that
the conclusion is true. Sampling precision can change a close stability estimate without changing
the scientific configuration.

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
spawned process. Fixed sweeps and adaptive searches both own one outer allocation and schedule
independent child Runs through the same ARI dispatcher. Their scientific controllers differ—fixed
evidence is mandatory, adaptive evidence is selected—but physical placement is not duplicated.
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
  budget: {candidates: 12, runs: 48}
  proposal_pool_size: 192
  startup_trials: 10
  replication: {minimum: 1, confirmation: [1001, 1002, 1003]}
  pruning: {enabled: true, min_step: 5, confirmations: 2}
  seed_racing: {probability_threshold: 0.1}
  confirmation_top_k: 2
  sampler: auto
  # Optional record-streak stop; omit to consume the authored candidate budget.
  convergence_patience: 8
  min_improvement: 0.0005
  space:
    learning_rate: {range: [0.0001, 0.01], scale: log}
    width: [128, 256]
execution: {max_parallel: 2, failure_retries: 1, max_time: 12h}
objective: {metric: val_score, mode: max, practical_margin: 0.002}
```

Canonical `sweep.space` dimensions form a finite Cartesian product. Exact finite `when` branches
are enumerated; numeric ranges require an explicit deterministic `points` count. Every combination
times every authored seed becomes required evidence. `search.strategy: exhaustive` normalizes to
the same fixed design and dispatcher for compatibility. Candidate, replication, pruning and
confirmation controls are rejected because they contradict that fixed contract.
Adaptive/numeric spaces build a deterministic scrambled-Sobol pool bounded by
`proposal_pool_size` (by default up to 16 times `trials`, capped at 4096); `trials` is only the
maximum number of candidates that may execute. Conditional parameters use `when` and carry
an explicit active/inactive feature rather than a fake value. This pool is planning state, not a
list of decided Trials. It is retained once and lightweight Run specifications reference only their
own candidate and seed, so planning memory grows linearly rather than duplicating the pool per Run.
Large valid values therefore do not require weakening the YAML to avoid planner memory blow-ups.
They also do not multiply controller-critical scientific-analysis work. Live analysis builds one
reusable immutable design basis, caches mixed-space distances, uses 32 deterministic
evidence realizations and evaluates bounded representative counterfactual/factorial support.
Scientific candidate scoring rotates that bounded set on every controller decision and always
includes the optimization sampler's proposal. Consequently the full pool remains eligible over
time without a full-pool scan per question/resample blocking completed-Run collection. The
persisted evidence reports pool, reference and matching-support sizes; final post-hoc analysis may
use a denser budget because it is outside the scheduling-critical path.
If `startup_trials` is omitted, LambdaForge builds a deterministic `InitialDesignPlan`. Its design
matrix contains an intercept, numeric linear terms and supported quadratic terms, independent
categorical contrasts, and conditional activity/value terms. Greedy selection first preserves the
full pool's attainable rank, then authored discrete/numeric support, D-optimal information geometry
and maximin separation. The resulting protected anchors depend on the authored `ParameterSpace`
and `trials`, never on GPU count or execution parallelism. An explicit positive `startup_trials`
sets the exact anchor budget but retains that geometric selection.

This makes three control layers explicit: scientific planning asks what evidence should exist;
resource planning asks what can safely run now; dispatch starts the isolated process. Protected
anchors may wait or reorder without becoming cancelled. Physically spare slots can run
replannable `OPPORTUNISTIC_COVERAGE` candidates without changing the protected design. A hard
device-type impossibility replaces an anchor with a candidate preserving the same obligations and
persists `REPLACE_STARTUP_ANCHOR`.
The versioned controller state also retains completed Attempt paths and pending action identities,
so a controller restart reconstructs the same anchor debt without executing finished Runs again.
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
random-effects mean \(\hat\mu_i\). Seed noise is identified only from repeated outcomes within a
candidate, retaining common seed effects for population uncertainty; spread between
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

where the canonical `objective.practical_margin` is \(\epsilon\) and
`seed_racing.probability_threshold` is \(\delta\). Clearly
dominated candidates stop receiving purely competitive seeds; a representative repeat can still
be valuable for global noise calibration. Uncertainty near the decision boundary and unresolved
scientific questions compete per observed Run duration. `replication.minimum` (`min_seeds` in legacy YAML) is a
strict obligation once a candidate is proposed: those distinct seeds cannot be removed by
replanning. A performance-pruned seed satisfies attempted/censored evidence, but not completed
response evidence; infrastructure failures count only after retries are exhausted. Seeds beyond
the minimum remain adaptive. Ambiguous
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

`goal` changes those priorities; it is not a permission switch that discards scientific value.
Once screening and the practical contender set stabilize, LambdaForge stops proposing ordinary
optimization candidates and enters fresh-seed confirmation. Protected initial-design obligations
still complete, while otherwise idle lanes may collect the highest-value response coverage,
matched comparison or checkpoint continuation. Confirmation has priority, and only its actual
persisted completion state can satisfy convergence. A hard Run/time/physical limit remains a
distinct incomplete stop reason rather than being relabelled scientific convergence.

Curve pruning has two readiness levels. Operational readiness means completed endpoints make the
curve model identifiable; strong readiness additionally requires finite-sample residual resolution,
retrospective interval coverage, probability discrimination and a harmful-prune rate consistent
with the authored threshold. The latter alone authorizes competitive stopping. Startup anchors,
minimum required evidence, confirmation and scientific continuations are protected from it.
Search coverage records attempts, response coverage records terminal objective evidence, and
scientific coverage describes support for the exact displayed conclusion; none substitutes for
another.

Startup work is submitted gradually and can interleave with model-directed decisions. Pending
identities and exact target fidelities condition the surrogate and prevent duplicate queued seeds.
Undispatched actions form a mutable dispatch buffer. After new evidence, a stale non-anchor action
may be replaced at zero scientific compute cost; `CANCEL_SCIENTIFIC_ACTION` records why. A protected
anchor instead records `CANCEL_PLANNED_DISPATCH`/`DEFER_STARTUP_ANCHOR` and remains owed. Each action
records value, score, cost, reason and compact alternatives;
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

Performance pruning and scientific continuation answer different questions. A
`PERFORMANCE_PRUNE` says that spending more on this curve is not justified for finding the optimum;
its prefix remains censored search-coverage evidence, never a fabricated final objective. If later
matched-context analysis finds that the same logical Run is especially informative for a parameter
value or interaction, `SCIENTIFIC_CONTINUATION` resumes its durable checkpoint with the same Trial
and seed as a new Attempt. Competitive pruning is disabled for that Attempt; checkpoint/fidelity
boundaries, scheduler preemption, resources and global Run/time budgets still apply. It consumes no
new `trials` candidate slot. The Research Console displays the original prune epoch, continuation
Attempt and target question together.

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
sweep:
  space:
    optimizer: [adamw, sgd]
    momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
  reference: {optimizer: adamw}
execution: {runs_per_gpu: auto, max_parallel: auto}
```

This runs one AdamW variant and two SGD variants for each seed: six Runs exactly. It does not use an
surrogate, HPO pruning, adaptive seed allocation or winner-only confirmation. Training patience in
the Work remains valid. An objective may still summarize the evidence, but cannot change which Runs
execute. ARI retains all placement, packing, OOM recovery and checkpoint responsibilities. A
numeric dimension may use `{range: [LOW, HIGH], points: N, scale: linear|log}`; omitting `points`
is invalid. An explicit Run budget smaller than the fixed design is rejected, while a wall-time
limit may produce an honestly incomplete design.

Internally, authored YAML is normalized once into `StudyDesign`, `EvidencePlan`, `ExecutionPolicy`
and the objective policy. An evidence requirement has candidate, seed, phase, fidelity, required
flag and lifecycle state. Candidate budget only stops new configurations; required minimum seeds,
shared seeds, promotions, scientific continuations and confirmation remain eligible until completed
or bounded by Run/time policy. Terminal telemetry exposes `status`, `design_status`,
`scientific_status` and `finish_reason`, reconciles every queued identity, and reports observed and
evidence-complete candidate counts separately.

Sweep analysis is paired by exact shared seed. It reports each cell's completed/required seeds,
mean, median, dispersion and uncertainty; reference comparisons add paired differences, support,
superiority and—only when a practical margin exists—practical-equivalence probability. Resampling
selects whole seed blocks. Missing cells reduce support and remain visible; they are never imputed
with another seed. A point-estimate leader is descriptive, while the displayed exact conclusion
and its confidence always refer to the same resampled hypothesis. Complete design and unresolved
science are therefore a valid combination.

### 7.2 Independent trainings per GPU

The YAML resource block is the fixed outer reservation. `resources.gpu` reserves GPUs for one
study; it is not repeated per child. Top-level `execution.runs_per_gpu` packs spawned Runs onto each
reserved device, and each child sees one GPU plus its divided share of CPU/RAM/storage:

```yaml
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  budget: {candidates: 40}
  replication: {minimum: 1}
  pruning: {enabled: true, min_step: 5, confirmations: 2,
            probability_threshold: 0.05}
  space:
    learning_rate: {range: [0.00001, 0.003], scale: log}
    hidden_dim: [64, 128, 256]
execution:
  runs_per_gpu: auto
  max_parallel: auto
objective: {metric: val_auprc, mode: max, practical_margin: 0.0}
resources:
  gpu: 2
  gpu_memory: 16GiB
  cpu: 16
  memory: 32GiB
  time: 24h
```

CPU, RAM and scratch use a dispatch-time `HostResourceLease`. The current residents share the
aggregate Job allocation; admission recomputes commitments before every launch and never exceeds
the hard global concurrency/host ceiling. CPU is elastic: Torch/BLAS/OpenMP limits and process-tree
affinity are rebalanced when residents enter or leave, so three actual Runs can use the host instead
of each receiving `total_cpu / theoretical_max_parallel`. RAM/storage are current-resident
commitments checked against live process-tree RSS, not cold-start reservations for every possible
future Run. The persisted lease explains the exact share and provenance used by each dispatch.

The final launch reconciles a native elastic RAM share with a **fresh whole-process-tree**
measurement, including controller helpers and existing workers. A slightly stale share may shrink
to the aggregate allocation's remaining headroom; an independent explicit Run requirement is not
silently reduced. Physical/cgroup availability and the aggregate Job ceiling still block unsafe
starts. Resource telemetry exposes `launch_waits` and the actual idle-device reason when a proposed
placement fails its last launch check; a planner proposal is not proof that a worker started.

Startup allocation growth is not a permanent ramp: only real advancing progress/phase boundaries
with no further growth close that streak; repeated identical polls do not establish stability.
A stable resident may use its compatible terminal learned envelope as its commitment while its
rare full-device tail remains explicit in joint-fit risk. Cold-start/unconfirmed history cannot
invent such a bound. A fresh GPU launch revalidates the same approved safe/exploratory decision,
retaining sibling protections and its original WAIT comparison rather than resetting a single-GPU
planner and losing accumulated evidence. None of this guarantees all configured slots can fit.

A custom Work's stepped
scalar metrics publish `metric-progress.json`, a bounded per-Attempt progress projection updated
once per advancing step. Admission reads it without scanning metric history, preserves more precise
callback phases, and does not interpret a reported epoch as a durable checkpoint or proven memory
stability. Non-stepped scalars do not invent progress.

Physical-capacity events consult the same deferred initial-design and adaptive-action queue as
completion events. Protected anchors retain priority; otherwise useful initial seed coverage can
fill idle capacity even before another Run finishes. Exact candidate/seed/fidelity identities and
Run/time budgets still prevent duplicates and unbounded dispatch. This does not turn adaptive
search into an exhaustive sweep or bypass resource/throughput checks.

`auto` removes the artificial per-device/global process-count caps; ARI still derives a finite
ceiling from runnable work, candidate/Run budget, allocated GPUs, host CPU and site constraints.
An integer remains a hard ceiling: two GPUs with `runs_per_gpu: 4` permit at most eight Runs, while
six GPUs with `runs_per_gpu: 2` permit at most twelve. `gpu_memory` is an optional user safety floor.
When present, each launch requires

$$
\texttt{gpu\_memory}
\leq \text{currently free device memory}.
$$

The resource model also predicts a candidate-specific peak distribution, conservative future
envelope, time to that envelope and remaining duration from exact and censored compatible history.
The effective commitment is `max(gpu_memory, learned upper envelope, known OOM lower bound)`; when
`gpu_memory` is omitted, prediction operates automatically. Physical VRAM is the packing authority.
Admissible headroom is the stricter of live physical free VRAM and predicted future headroom; an
external process appearing after launch therefore blocks an unsafe new admission immediately.
CUDA allocated/reserved values are diagnostics and model features, not a replacement for physical
occupancy. Bounded trajectories preserve their true peak and transitions; no fixed wait-to-peak
timeout exists.

For GPU \(g\), placement reasons about \(C_g-E_g-\sum_i M_{i,future}\): usable capacity minus
dynamic external occupancy and active future distributions. Here
\(M_{i,future}(t)=m_i(t)+R_i(t)\), where the residual \(R_i\) remains non-negative and uncertain.
The residual is a weighted empirical distribution, not equiprobable extrema. A weak Jeffreys
survival prior is updated after material peak events and quiet progress cycles; robust
physical/allocator dispersion separates measurement drift from material growth. Critical phase
transitions preserve late-allocation risk until observed. Thus `RAMPING` explains state but is not
an absolute placement veto.

For an eligible GPU the planner integrates wait regret

$$
W_g(t_1)-W_g(t_0)=\int_{t_0}^{t_1} u_g(t)\,r_*(t)\,dt,
$$

where \(u_g\) is physically usable idle fraction and \(r_*\) the best normalized pending
scientific-value rate. WAIT and EXPLORE are compared until the next observed step, phase,
checkpoint or completion, not a timeout. Rollback GPU-seconds use the same opportunity rate;
resource-information value remains separate from candidate priority. Unknown duration uses
candidate/near history or live step-rate extrapolation and otherwise stays unknown. Lightning can
honor a selective checkpoint request at a safe epoch boundary when rollback alone blocks a useful
experiment.
Cold start first creates one `BASELINE_ADMISSION` on every idle granted GPU, then uses live
phase/step trajectories and checkpoints for incremental 1→2→3 experiments without waiting for
terminal peaks. A baseline is protected progress, never an unvalidated packing experiment and never
an exploration-lane consumer. `SAFE_ADMISSION` uses supported future envelopes;
`EXPLORATORY_ADMISSION` is chosen only when expected progress plus resource information exceeds
rollback/interference cost. Equivalent GPUs keep one lane at baseline concurrency and never
duplicate the same unvalidated escalation. `RESOURCE_BLOCKED` is reversible;
`RESOURCE_INFEASIBLE_ON_DEVICE_TYPE` requires an intrinsic hard lower bound above the device.

The scientific controller supplies a bounded ranked frontier. The resource planner uses best fit,
preserves larger devices for heavy work and may run a short feasible backfill while a high-value
heavy action waits for its predicted window. Learned co-location throughput can stop extra packing
when it would reduce useful scientific work per wall-clock time. Resource scarcity changes
when/where an action runs, never its scientific objective. If that complete bounded frontier is
blocked, the same scientific policy may provide further bounded unseen alternatives without waiting
for a terminal Run. Waiting alternatives share a hard bounded planning frontier (at most 16 and
never above parallelism left after active Runs). Exact identities still deduplicate requests; adding one
alternative cannot recursively create unbounded permission to propose another, and a response with
no new identity closes that dispatcher state.

LambdaForge probes every allocated device throughout active Runs—even with no queued action—and
fills only safe capacity; temporarily unavailable Runs remain queued. Same-device launches are
staggered so allocation becomes observable. It does
not multiply `gpu_memory` by active Runs or create a second hidden reservation.
Eligible GPUs are ordered by current active-Run count and time since their last launch. Therefore,
when only one global slot is free, GPU index zero cannot repeatedly win and starve another granted
device. Within the remaining candidate/Run/time budgets, each terminal event replans immediately
and supplies a replacement action; low post-hoc analysis confidence is not an early-stop signal.
Every admitted Run has a fresh one-worker spawned process. It is shut down as soon
as its result or error is received, releasing the whole CUDA context; a persistent idle pool could
retain VRAM and deadlock the Runs waiting for admission. A full or smaller GPU is skipped while
other usable GPUs continue. A declared floor, or a persisted hard OOM lower bound, above every
allocated compatible device is explicitly infeasible instead of waiting forever.
The memory probe itself executes in a short-lived child and exits after returning JSON, so the HPO
controller does not appear as one idle CUDA process per GPU and does not consume a
`runs_per_gpu` slot. Wait/admission records are flushed to the Work log.

This dynamic admission is not a hard memory limiter inside consumer code. An explicit
`gpu_memory` floor is useful for a known non-negotiable minimum; automatic mode is appropriate when
demand varies across a search. `max_parallel` remains the global hard ceiling and also bounds
CPU-only studies.

Worker PID/process-tree VRAM is attributed through NVML when available and reconciled against
physical used memory on every sample. Allocator heartbeats add phase, checkpoint and short-spike
evidence; aggregate fallback remains explicitly censored. OOM evidence separates intrinsic
resident-plus-requested bytes from a failed placement signature. Consequently a failed
`heavy+heavy` packing cannot reduce a global GPU concurrency ceiling or prohibit `heavy+small`.
Scientific termination is separate from resource completeness. Resource-relevant phases are
learned from compatible Runs; there is no universal forward/backward/optimizer/validation
checklist. A performance-pruned Run whose observed phases cover the learned allocation risk and
whose process telemetry is exact contributes an exact resource profile while its objective remains
censored. A late validation/checkpoint peak in history keeps the current profile censored until
that risk is observed or represented in its tail. Changed plan comparisons are stored
in `exploration-evaluations.jsonl`, including P(fit), hazard, wait regret and explicit reasons such
as `RUN_CAP`, `HARD_LOWER_BOUND`, `KNOWN_FAILED_PLACEMENT`, `PROTECTED_LANE`,
`THROUGHPUT_REGRESSION`, `ROLLBACK_DOMINATES` and `WAIT_CURRENTLY_BETTER`; repeated polls are
suppressed.
An atomic bounded active-evidence snapshot survives controller interruption as stale provisional
knowledge. It can warm uncertainty after restart, but is never restored as a live process or exact
peak; terminal evidence supersedes it and normal completion clears it.

ARI v3.1 separates the display trajectory from `ResourceTrajectoryStatistics`. The former remains
bounded; the latter incrementally retains progress/optimizer cycles, phase visits and maxima,
material peaks, robust growth/noise/allocator residual sketches and checkpoint cadence. Hazard is
updated by progress and allocation events rather than polling count, so duplicate monitor samples
do not create confidence. Exact weighted convolution preserves rare discrete tails in P(fit), with
a small explicit independent/shared-epistemic ensemble for multiple Runs. Subsequent placement
outcomes calibrate probability bins without contaminating the scientific objective.

WaitRegret is a versioned atomic integral. A changed frontier closes the previous segment at its
previous idle fraction and scientific-value rate; it does not erase accumulated opportunity loss.
It resets only when pending feasible opportunity or usable capacity disappears, or an admission
satisfies it. Physical lower bounds, failed dominated packings, run caps, site policy and measured
throughput regression remain absolute. Checkpoint planning computes rollback per resident, charges
learned checkpoint duration, and waits for a durable checkpoint event before treating rollback as
reduced. The next-event horizon uses learned checkpoint cadence and time *until* the next event,
never time elapsed since the last checkpoint. Repeated intervals provide a robust cadence
uncertainty; insufficient evidence remains explicitly unknown.

Failure isolation follows the Run boundary. CPU and GPU adaptive Runs both use fresh one-worker
processes, so a killed worker cannot poison a shared pool or cancel unrelated candidates. A worker
lost before returning a result and CUDA OOM/allocation failures are retried as a new Attempt up to
`failure_retries` (default 1, allowed 0–3). An exploratory resource OOM instead enters bounded
`RESOURCE_RECOVERY`; compatible checkpoints remain under the same Run and
are discovered by the retry. A CUDA OOM records attempted allocation (when reported), physical
headroom, candidate residency, external occupancy and co-runners as censored lower-bound evidence,
then updates pending predictions. Its retry cannot be admitted until headroom or another compatible
condition materially improves. Already-running children are never killed. A repeated resource
failure after a genuinely improved placement becomes terminal, and ordinary consumer
exceptions are never retried blindly because invalid data/code will not improve with repetition.
Other pending/active Runs continue. The enclosing Work is still reported failed when a Run exhausts
recovery, preserving honest scientific evidence instead of hiding a missing candidate.

Study resource evidence lives under `hpo-control/resources/`: bounded observations, OOM evidence,
admission decisions and the current commitment ledger. Compatible Work/code/environment/hardware
signatures warm-start later project Studies; incompatible records do not become exact priors. The
Research Console Resources tab renders the same read model: physical free memory, external usage,
current/future commitment, predicted headroom, P(fit), blocking reason and backfill explanation.
Final Study Analysis records resource-conditioned sampling so an under-sampled heavy region is not
described as scientifically poor.

#### Resource scheduler replay

Every changed scheduler state is recorded as a bounded versioned trace containing device state,
the scientific frontier, predictions, admissions/blocks and exploration evaluation. It references
existing observations rather than copying outputs or checkpoints. Replay is available through the
same result service used by CLI and future interfaces:

```bash
lf results replay EXECUTION --policy recorded
lf results replay EXECUTION --policy ari-v3.1 --json
lf results replay EXECUTION --policy ari-v3-compat --json
lf results replay EXECUTION --policy ari-v2-compat --json
python -m benchmarks.resource_scheduling --execution .lambdaforge/runs/WORK/EXECUTION
```

Replay is exact only while the selected policy agrees with recorded admissions. The first mismatch
is `counterfactual_divergence`; subsequent metrics are explicitly trace-conditioned simulation,
not reconstructed fact. A terminal scientific outcome or OOM from the recorded branch becomes
`null` for an alternative branch instead of being falsely reassigned. Compatibility policies
exist only in replay. Metrics cover concurrency
arrival, mean packing, idle VRAM/compute where measurable, scientific/useful action rates,
throughput, completed/pruned rates, OOM classes, rollback/checkpoint cost, heavy starvation,
resource-blocked time, false-safe/conservative behavior, P(fit) calibration and prediction error.
Missing evidence is `null`, never fabricated. The no-argument benchmark remains only a synthetic
regression; use `--execution` for quantitative trace-derived evidence.

#### Canonical parameter geometry

`ParameterSpace` is the mathematical authority for the authored search schema. It owns transformed
encoding/decoding, distance, active masks, bounds, categories, conditionals, support and boundary
position. Log dimensions are normalized in log space; integers retain ordinal topology;
categoricals are discrete; inactive conditional values have an activity bit and are not encoded as
the lowest value. Sobol, adaptive and Bayesian selection, resource similarity, scientific design,
coverage/effects and Study Analysis all consume this geometry. Persisted candidate mappings and
fingerprints remain unchanged.

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

After `search.pruning.min_step`, the controller compares active Runs at a common observed step. A
bounded local-linear posterior projects each curve a short distance forward and carries residual
plus pooled uncertainty. Pruning does not activate until at least two distinct candidates have
completed far enough to backtest prefix predictions against their real endpoints and produce a
finite curve error. Two is the minimum that identifies a comparison, not a configurable tuning
threshold; before that point every startup curve remains eligible to establish the reference.
Once calibrated, it requests a stop only when the probability of remaining within
`objective.practical_margin` of the projected incumbent falls below
`search.pruning.probability_threshold`. This is
more conservative with noisy or still-improving curves than dropping a fixed fraction. By default
the condition must remain true at two distinct common steps (`confirmations: 2`); repeated polling
of the same epoch cannot satisfy it. Set `confirmations: 1` only when an intentionally aggressive
policy is worth the increased risk of reacting to one noisy validation point.
`min_step` is only the minimum evidence boundary; it is never reused as a forecast distance. If an
authored fidelity rung exists, that rung is the decision horizon. Otherwise the projection spans
one recent observed window, and a candidate still competitive at the exact common checkpoint
cannot be stopped solely because its fitted local slope is worse. This prevents long, unsupported
extrapolations while retaining conservative early termination of repeatedly inferior curves.
`LightningRunner` forwards its validation metric and honors requests at batch boundaries. A
final-only metric still enables adaptive seeds but cannot stop the current training early.

### 7.5 Live study observability

The outer scheduler Job is an allocation/cancellation boundary, not a reason to merge scientific
meaning. LambdaForge therefore keeps one compact study index next to that Job and one tiny state
record per internal Run. A Run's existing `work.log`, `metrics.jsonl`, `training-metrics.jsonl` and
`result.json` remain authoritative; the index references them and folds only latest scalars and
state. It never copies model checkpoints, output directories or artifact bytes.
Remote reads follow the visible view, not the size of the Execution:

- Overview and the Studies list receive only counts, objective/leader and basic state/cost metadata
  from `study/overview.json`. They do not download candidate or seed lists.
- Opening one Study reads `study/interactive.json` (projection v3): Trial table cells, aggregate
  Run states and one leader's parameter preview. No seed records, curves or HPO state travel here.
- Opening one Trial reads its `study/trials/trial-XXXXX.json`: parameters and that Trial's seed
  table. Parameter filtering runs on the execution host, not by downloading every parameter set.
- Opening one seed reads only its native record, bounded curves, artifacts metadata and log.
- HPO reads precomputed `study/hpo.json`; Resources reads `study/resources.json`. Full controller
  history is paged only after an explicit Action history drill-down.
- Analysis reads the aggregate `study/analysis-panel.json`, without individual Run evidence or the
  HTML research catalogue. Final analysis is persisted by the worker. Provisional post-hoc analysis
  is computed/cached on the execution host only when requested, in a separate read process, never
  in the scheduling heartbeat. HPO itself does not refit this analysis. Explicit HTML export loads
  the complete analysis instead of substituting the lightweight console model.

Older rich summaries/indexes are projected on their execution host, never downloaded wholesale.
All Study read views use 512 KiB byte pages with generation checks; unchanged refreshes return only
a small marker. The per-response safety bound remains, but there is no aggregate 8 MiB rejection
for a legitimate large Study. Hidden parent views stop polling until they are visible again.
These are read/presentation changes; HPO decisions and scientific evidence are unchanged.
Sweep Analysis uses the persisted StudyDesign space, including conditional branches not yet
completed; it does not reconstruct those conditions from observed rows. Missing valid response
contexts are reported as insufficient evidence, not a failure of the running Study. The read
process passes that geometry explicitly to older immutable worker environments, so updating the
controller does not require restarting the Study to apply this read-side correction.

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

Scientific completion is separate from optimization screening. Stable screening stops ordinary
`OPTIMIZE` proposals; confirmation and material response, parameter and interaction coverage may
continue until no feasible action has material value or a hard budget is reached.
`remaining_information_value` combines predictive uncertainty with explicit missing-support debt,
so a confident surrogate cannot make an unsupported discrete value count as observed. The shared
`scientific_status` includes parameter questions and material, feasible interaction debt.

With a scalar objective, completed Runs may additionally expose `diagnostic_metrics`: terminal
numeric metrics common to comparable seeds, including mean, dispersion and support. Trial views
and the offline dashboard can inspect them and chart them against parameters. They remain
display-only evidence and never become objective components, constraints or HPO inputs.

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
uses physical driver memory plus the learned candidate envelope and any optional
`resources.gpu_memory` safety floor, filling safe capacity rather than requiring the maximum
upfront. After each packed Run,
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

`work.items[].study` contains only compact counts, objective and leader metadata. Open one Study
and Trial in the console to obtain its Run keys without fetching every Run. `show --run` returns
structured parameters, latest values, down-sampled curves, bounded log, failure, evidence paths and
the finalized managed-artifact inventory. Human output highlights each preferred usable artifact
path; `--json` preserves its checksum, size, role, MIME type, managed/published locations, retention
and metadata. The Research Console exposes the same inventory in the selected seed's **Artifacts**
tab. This is metadata-only inspection: it neither copies nor opens remote files. Artifacts do not
appear before successful output finalization;
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

### Recover an interrupted Study

`lf retry STUDY` and the Study workspace's **Resume Study…** button use the same recovery service.
The selector identifies a failed/cancelled/timed-out attempt on the original cluster. The service
reads an exact bounded recovery preview on that host, stages current code in a new Job and reconnects
the runner to the original owned Execution directory. It never copies weights or bulk artifacts
merely to retry: only compact console indexes and decision references move to the new Job. Original
logs, scalar files, checkpoints and evidence remain referenced in place.

Recovery resolves seeds in the original persisted namespace, not the new Job's staging directory.
Explicit values/order and automatic stream coordinates remain unchanged; missing/corrupt coordinates
or changed seed policy are refused. In the Console, Logs and Resume Study target the latest Job.
Until it publishes telemetry, prior owned Trial/Run evidence stays visible without replacing the
latest Job's lifecycle state. No YAML workaround or identity-check bypass is required.

Adaptive, repeated-seed and fixed-sweep Studies share this operation. For fixed designs, recovery
reads the immutable design and owned per-Run Attempt records, reconciles the latest outcome and
reuses successful Runs. It never fabricates an adaptive controller or requires its state file.
`lf retry STUDY --dry-run --json` exposes `recovery_plan` with `reuse_runs`, `retry_runs`,
`pending_runs`, per-cell seed/Trial/actions and physical spent budgets. A failed seed gets a new
Attempt of the same logical Run, not a fresh experiment. Runs not yet started remain required.
Recovery is distinct from training continuation: compatible checkpoints are made available, but
epoch-level continuation requires the Work's own restore logic. If absent, only that unfinished
Run restarts. Incompatible scientific inputs/designs and unsafe checkpoints are refused.

For adaptive Studies, the runner restores proposed candidates/public Trial numbers, deterministic generator expansion,
seed identities, pending actions, initial-design obligations, convergence and confirmation state.
Completed and performance-pruned outcomes remain evidence. Failed Runs and interrupted pending
actions are requeued once as new Attempts, using a compatible checkpoint when one exists and
starting fresh otherwise. A terminal result published just before a controller crash is collected
rather than repeated. Physical failures stay in controller history and spent Run/time budgets are
not reset. Aggregate status uses the latest Attempt of each logical Run so a recovered failure does
not poison success; exhausted recovery still reports failure honestly.

`lf retry STUDY --accept-code-change` explicitly acknowledges a compatible code fix: the researcher
must know that earlier metrics and checkpoint formats remain valid. The console exposes the same
unchecked-by-default acknowledgement. When the preview detects changed project code, submission
is refused before creating a recovery Job unless acknowledged; Resume remains disabled until the
checkbox is selected. The freshly staged worker checks compatibility again. This does not waive
checks on configuration, inputs, objective,
seed streams or search policy. If the fix invalidates earlier results, launch a new Study instead.
The original `execution.json` stays immutable; `current-code.json`, per-Run provenance and append-only
`recovery-history.jsonl` expose actual revisions instead of pretending the code never changed.

A cross-process controller lock prevents concurrent writers. Recovery dependencies protect original
Job workspaces from GC and individual deletion while referenced; deleting the entire terminal Work
history previews and removes the chain together. Missing/corrupt state or foreign/symlinked paths
fail closed. Adaptive Studies still require compatible `hpo-control/state.json`. Fixed Studies
require complete owned design/Attempt evidence instead. Recovery supports one Study per Execution
on the same cluster;
ordinary preparation failures with no Study state can still retry submission normally. Automatic Run
retries remain bounded infrastructure recovery; consumer exceptions are retried only by this
deliberate user action, never in an endless automatic loop.

## 9. Results and metadata

Study telemetry and Execution envelopes include a versioned `lifecycle` projection with independent
`operational`, `evidence` and `health` fields. `running / pending_recovery / degraded` is not terminal
scientific failure. `stopped / complete / degraded` retains prior physical failures after recovery.
Waiting retries are queued, not occupied process slots. Physical display history stays bounded while
cumulative counters remain available; legacy truncated history is explicitly incomplete. Older
Execution envelopes can be projected in memory without creating files or guessing missing Attempts.
See [the reform ledger](ai/IMPLEMENTATION.md#history-architectural-reform) for implemented boundaries and remaining migrations.
This does not change objective fitting, censored pruning evidence, exact identity or budget rules.
Native finalization resolves latest Attempts of each logical evidence cell before projecting the
legacy terminal label; failed physical history is not itself a failed recovered Run. Phases and
fidelities remain separate. Storage/clean previews do not create cache or lock metadata; apply
revalidates ownership and targets under locks instead of trusting an earlier preview.

[Durable products](#guide-products) documents native `lf products` publication, ModelSet selection,
StudyDecision, verified portable product bundles and typed `with` product dependencies. Metadata
resolution is read-only; actual Attempts record consumers and model bytes are verified only on
explicit access. Product catalogs are project-owned, not disposable Job or cache content. This
does not yet implement automatic dependency waiting/replanning or complete distributed Fleet export.
`lf import PACKAGE` verifies single-host exports; `--apply` registers portable evidence and sealed
products without execution. Original provenance remains intact; imported records are read-only,
not native recovery state. Console **Studies → Import Study…** (also in Products) exposes the same
verified, confirmed workflow with a directory picker. Registered imports appear in Studies and
Overview with an imported/read-only label, retaining their captured status. Trial/seed drill-down
reads local retained metrics, logs and artifacts; Analysis reads included evidence without refitting.
No imported snapshot exposes remote cancel/retry controls. Reapply an older import once to add the
local presentation indexes. Root refresh never reparses a large imported result. The verified native
aggregate `result.json` has no fixed 64 MiB ceiling; inventory size/SHA and strict JSON/provenance
validation remain mandatory. Verification and registration run off the Console event loop.

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

Use `lf export WORK --output LOCAL_PARENT` when the available evidence must leave its execution
cluster. This is different from copying the workspace manually: the exporter selects the newest
exact Attempt, records its current state, includes external finalized artifacts and writes a
checksum inventory while excluding shared runtime state. Final evidence can include analysis and
replay; active, cancelled or failed evidence is labelled as a point-in-time snapshot. Names that
identify several Works fail as ambiguous; use the `work_id` shown by `lf overview --json`.

Structured finite choices such as `values: [[0.005, 2560.0]]` are one categorical parameter
value, not two independent bounds to optimize. Coverage/effects group them by canonical JSON
without rewriting the authored parameters. Exporting their completed evidence requires neither
editing the YAML nor repeating training.

Remote export first creates a ZIP64/Deflate archive and always removes it after transfer. Local
transfer and extraction temporaries are allocated below `LOCAL_PARENT`, not system `/tmp`; ensure
that filesystem can hold the compressed archive plus one extracted package. Safe hard links avoid
duplicating extracted bytes during atomic assembly. In the Research Console export is a background
operation: leave the Study workspace freely and follow phase, elapsed time and compressed size in
the global **BACKGROUND** panel.

`--profile default` is the portable audit profile: exact scientific results/controller decisions
remain complete, metric trajectories are deterministically endpoint-preserving down-sampled when
large, and high-frequency resource streams become bounded summaries. `--profile full` retains raw
metric/resource streams. `manifest.json` lists every transformed or omitted source with its reason,
original byte count and SHA-256, so compactness cannot be mistaken for missing evidence.

## 10. Clusters and jobs

Fleet discovery: `lf fleets list/show/offers`; operator catalog controls:
`lf fleets drain/disable/enable FLEET CLUSTER [--apply]`. Observations do not grant GPUs and these
controls never stop active Jobs. `lf run CONFIG --on-fleet NAME [--dry-run]` uses a durable local
coordinator and prepared managed allocations for fresh adaptive/fixed/repeated/automatic paired Studies, with
one native planner and final analysis. Run Work offers the same Fleet selector. Live/terminal Run
show/logs resolve the exact member owner; bounded current/best scalar views never read remote
paths locally. Curves/logs remain lazy; cancel/delete operate on the whole owned family.
Native ordered scalar streams and exact central pruning support fresh adaptive HPO.
GPU offers are baseline only; checkpoint/fidelity continuation, co-location, recovery/adoption, live expansion
and distributed export remain pending and fail closed.
See [implementation status and contracts](#guide-coordinated-studies) before using fleets.
The common internal prepared executor delegates to ControlPlane/JobService and verifies immutable
environment/code identities and canonical input relocation. Host-side result observation is batched;
physical observations never authorize unallocated GPUs.

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
| `shared` | direct Jobs may share GPUs; each Study coordinates fresh check/start admission across projects |
| `command` | no direct lease; prepend an explicit site claim/launcher argv |

```bash
lf clusters set free-host gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

On a shared direct host, Studies do not exclusively reserve their entire GPU subset for the
Job's lifetime. New Runs prefer a permitted device without observed compute processes (including
those belonging to other Studies/users); otherwise the existing resource-aware best-fit policy
applies. Per-device host locks serialize the fresh VRAM check and launch, and stagger starts across
Studies. These are short admission guards, not extra VRAM reservations. External applications do
not participate in these locks, so this is not hardware isolation or an OOM guarantee. Site grants
and exclusive Job leases remain authoritative; no device outside the permitted subset is added.

Repeated-seed Studies also use the common resource dispatcher: one candidate with ten seeds has
ten independent required Runs, not a parallelism ceiling of one. `execution.max_parallel` and
`runs_per_gpu`, live memory/ARI safety and the Job's aggregate CPU/RAM allocation still bound
concurrency. Concurrent Jobs need enough **host resources**, not just spare VRAM. For example,
46 reserved cores on a 48-core host leave only two for other Jobs; a second request for 32 cores
correctly queues. Change future YAML allocations deliberately rather than oversubscribing the
host. Identical managed-environment builds wait through provisioning and reuse the verified
completion; optional obsolete-environment cleanup defers while the cache is leased by active Jobs.

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

For allocations that may shrink or grow during execution, configure
`gpu_access.visibility_command` as an argv command whose first non-empty output line is the current
comma-separated opaque token set. `[gpu, exec]` derives `[gpu, env]` automatically. LambdaForge
intersects every observation with the original inherited grant, so it can never discover or use a
new physical GPU. A revoked token stops only its verified one-Run worker; that logical Run is
requeued as a checkpoint-compatible Attempt while unaffected tokens continue. Restored original
tokens become eligible again. A temporarily unavailable visibility probe preserves healthy Runs
but fails closed for new admission.
This command must describe reservation/ownership, not merely GPUs with active processes. Study
resource telemetry exposes requested GPUs, initial tokens, current grants, admission slots and an
idle reason per device; a real contraction is persisted and logged as `GPU_ALLOCATION_SHRUNK N → M`.

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

An `unknown` record is not declared terminal. `lf delete WORK` may nevertheless preview and apply a
history-only forget operation when the state remains unverifiable: no remote process or workspace is
deleted. This resolves stale history safely without pretending cancellation; reconnect and cancel
first if the computation may still be alive.

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

Cluster **Clear storage…** uses the same preview/apply authority, confirms categories and shows
background feedback. **Clear output** is not disk cleanup. The [storage guide](#guide-storage) documents
physical-volume reservations, safety pressure, quotas/LRU, checkpoint grace/pins/publication proofs,
Fleet/SLURM fit and current boundaries. Normal probes read only filesystem/lease metadata.
`lf storage status --on CLUSTER` explicitly inventories usage. `lf storage reconcile --on CLUSTER`
reports drift; `--apply` atomically updates the diagnostic ledger, never deletes scientific bytes.
Managed publication snapshots reserve on their actual destination volume, not the Run volume.

Inside the Research Console, `d` confirms deletion of the selected terminal Work or numbered Attempt;
`D` confirms deletion of every terminal history entry. Active Jobs are never removed. Deletion runs
outside the terminal event loop, removes the exact provider workspace plus local Job events and
submission record, and preserves published datasets, caches, environments and unrelated Jobs.
`lf jobs clear` provides the non-interactive preview and `lf jobs clear --apply` performs the same
whole-history operation, reporting failures without discarding the affected local record.

A narrower automatic retention pass runs at terminal completion. It deletes partial managed
artifacts from failed/interrupted Attempts and verified internal duplicates of successful
`publish_to` outputs. It preserves every lightweight fact needed by the Research Console, `lf logs`, results
and reproduction; `retention.json` records reclaimed bytes. Superseded immutable managed
environments are also pruned after a verified replacement is activated, except the active prefix
and prefixes referenced by live/recovered Jobs. Successful unpinned checkpoints are eligible only
under the documented grace/publication policy; failed/interrupted recovery state remains.
Bootstrap and normal automatic preparation share this rule;
Attempt environment provenance remains after reconstructible environment bytes are collected.

## 12. CLI reference

| Command | Unique purpose |
|---|---|
| `init` | scaffold an installable Work project |
| `project` | show the current project root and storage identity |
| `seeds` | inspect deterministic project replicate/confirmation streams |
| `config resolve` | render the exact versioned configuration without execution |
| `config schema/explain/diff` | installed static authoring guidance and descriptive native differences |
| `help --search`, `completion` | task discovery and opt-in native-parser shell completions |
| `inspect`, `diagnose`, `watch` | compact exact-identity read-only inspection and change-only observation |
| `validate` | complete local configuration/class/input validation |
| `explain` | signature/doc/default/resource explanation |
| `run` | the only scientific execution command |
| bare `lf`, `overview` | Textual human console and machine global control-plane view |
| `show`, `logs`, `cancel`, `retry`, `delete` | semantic Work operations |
| `jobs ...` | low-level Job control; `clear [--apply]` cleans terminal history |
| `clusters ...`, `doctor`, `resources` | scriptable target setup and diagnosis; bootstrap accepts `--project` and read-only `--dry-run` |
| `datasets ...` | inspect/verify/place/delete published dataset versions |
| `results list/show/compare/analyze/report/table` | query/analyze evidence; bounded derived JSON/CSV tables |
| `clean` | preview/apply safe cache and terminal-artifact compaction |
| `storage status/reconcile` | explicit usage/drift inspection; reconcile `--apply` updates diagnostic ledger only |

`lf help`, `lf --help`, `lf help clusters add` and conventional nested `--help` all exit zero.
All command failures go through stable diagnostic categories/exit codes; add `--json` for tooling
and `--debug` for tracebacks. Unsupported authoring commands and dataset build do not exist because
all computation is `lf run`.

See [Developer experience](#guide-developer-experience) for tested authoring, shell, agent setup,
Console filtering and paged result-table examples, including the explicit unfinished boundaries.

## 13. Clustering

Strategies live in individual class modules under `clustering/algorithms/`; shared contracts
have their own class modules. Public `lambdaforge.clustering` imports remain unchanged.
Whole-object [FGW distance](#20-fused-gromovwasserstein-distance-fgw) is a separate API, not
an additional clustering strategy.

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
exactly one pointwise custom distance contract for these clusterers:
`lambdaforge.nn.distances.Distance`. Existing implementations
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

Completed explicit-seed fixed sweeps can be deliberately enlarged using the original edited YAML:
`lf studies extend JOB --config experiments/sweep.yaml --dry-run`, then the reviewed
`--apply --plan-hash HASH`. The Console **Extend Study…** uses the same service. Only new cells
execute, with immutable revisions and cumulative Runs/analysis/history under a Series ID.
Local CPU and same-placement managed CPU/GPU contracts are supported. By default, changed or
unverifiable code creates an explicitly reviewed new code/runtime revision: old evidence is kept,
but statistical inference is separated by revision. `--code-policy strict` instead requires
proven equivalence. This is not retry, adaptive-optimizer recovery, migration or Fleet expansion. See
[incremental Study extensions](#guide-study-extensions) for compatibility, budget, crash, export and
checkpoint-continuation boundaries.

Charts include individual Trial points, step/area curves, violins, horizontal bars, histograms and
empirical CDFs, with metric-labelled shared axes. Projects can add self-contained HTML tabs via
`self.outputs.html_section("proteins", title="Proteins").write_text(html_text)`; report generation
collects verified finalized artifacts, not during scheduling. Multiple Runs share a document selector.
Sandboxed offline frames isolate project JavaScript; embed assets/data (16 MiB/document, 64 MiB/report).
See [project HTML tabs](#guide-research-analysis-project-html-tabs) for the contract and standalone API.

Study HTML starts in **Overview**, with candidate states, contextual predictive importance and
suggested charts. Conditional scores stay in their branch/support table, never global responsibility.
Interpretation/questions use carousels and full findings are paged. English/Spanish is selectable in
the header; authored names and exact persisted statements are not translated. Searchable anchored
dropdowns replace control modals. **Parameters** checkboxes overlay selected curves in one chart
(automatic Y scales group substantially overlapping ranges, never known different units).
Choose lines, points, grouped bars, candidate distributions or a metric heatmap; **Arrangement**
switches between one combined chart and one panel per metric. **Y scales** also offers independent
or explicitly shared ranges. **Dispersion** hides SD or shows whiskers/shaded bands (numeric lines
only; other views use whiskers). SD is across candidate summaries, not seed uncertainty; it is absent
with fewer than two observations. Heatmap colours normalize each metric visually; hover retains
original means and missing cells stay blank. These settings survive reopening the same generated
HTML. Shared explanations are behind **ⓘ**.
There is an **Analyze / By / Compare with**
Explore view and removable metric chips. All explicit axis/chart settings remain in **Advanced
visualization options**. Configured questions are first-class inspectable cards; Metrics & health
has collapsible category paths. **Evidence** replaces the duplicated findings tab and retains
conclusions, diagnostics, seeds, pruning and raw JSON. Observed arbitrary-metric interactions
remain separate from persisted HPO model surfaces. See [Research HTML](#guide-research-analysis).

Study Analysis is a deterministic post-hoc layer over persisted study evidence. It does not propose
Runs and never changes the controller. A terminal study automatically writes
`EXECUTION/analysis.json`; explicit routes reuse or recompute the same document:

```bash
lf results analyze EXECUTION
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output report.html
lf export STUDY_OR_WORK --output ./exports
```

`lf results report` writes only one HTML view of a locally available Execution. `lf export` is the
portable evidence operation: it resolves the current project's semantic Work history, selects the
newest Attempt in any state and retrieves its bounded evidence from a local or remote provider.
The output argument is a local parent directory. Final success uses `NAME--EXECUTION_ID`; all other
states use a timestamped snapshot name. Directories are staged and renamed atomically and are never
merged or overwritten. The manifest and command output always expose the captured state.

The package contains `manifest.json` (per-file SHA-256/size inventory), `execution/` (result,
configuration, per-Run logs/scalars, decisions, checkpoints and retained artifacts), `study/`
(complete persisted Study/controller telemetry), `control-plane/` (Job lifecycle evidence),
`published-artifacts/` (explicit finalized outputs outside the Execution), and `reports/`. The
analysis JSON and recorded resource replay are attempted when the persisted evidence supports them.
An early snapshot may contain only submitted configuration and Job lifecycle files. When analysis
is applicable, `analysis-report` enables its full interactive Plotly dashboard while the base
install writes a structured exact-evidence fallback. Shared Dataset versions, managed environments, cache and the staged project tree
are deliberately excluded: their immutable references remain in provenance and blindly copying
them would make an experiment export both unsafe and unexpectedly enormous. Remote assembly rejects
links/special files, local extraction rejects traversal/links, temporary provider archives are
removed on both success and failure paths, and no partial local folder is published. Transfer uses
ZIP64/Deflate; local archive/extraction state lives below the selected output parent rather than
system `/tmp`, and same-filesystem assembly safely reuses the owned extracted bytes.

The HTML route requires `lambdaforge[analysis-report]`; JSON analysis and the console do not. The
report embeds its data and Plotly runtime and makes no network request. Run reports provide a
searchable, collapsible metric catalogue, raw/normalized/snapshot/correlation/relationship views
and selectable accessible heatmap palettes. Study reports provide tabbed Trial comparison,
parameter response and exact-value summaries, pair heatmap/3D views, coverage, resource and finding
views. Browser analysis remains descriptive over persisted evidence and never refits the HPO
model. The Run sidebar and analysis panels are drag-resizable. Run reports can create custom metric
categories and save plots from the current metric selection; Study reports save observed 2D/3D
views with candidate parameter, recorded numeric metric and resource axes. Preferences persist for that
generated file; regenerating creates a clean preference scope.

To compare metrics against hyperparameters across an adaptive search or sweep:

1. In the console open **Study → Analysis → Interactive HTML**, or generate a local report with
   `lf results report EXECUTION --output study-report.html`.
2. Open **Explore**. Expand **Advanced visualization options** to choose a plot type. For a scatter, line or bar chart, choose a parameter such as `hidden_dim`
   as **X axis** and a recorded metric such as `val_accuracy` as **Y axis**. The chart previews
   automatically as controls change; saving is optional persistence, not required to draw it.
   **Add another Y metric (2D)** offers searchable checkboxes; check each additional curve without
   Ctrl/Cmd-click. These are Trial summaries, not epoch curves.
3. For **3D scatter** or **Observed heatmap**, choose two parameters (e.g. `hidden_dim` and
   `dropout`) as X/Y and select the metric under **Z / cell metric**. **Observed 3D surface**
   requires numeric parameter axes; scatter and heatmaps also support categories/booleans with
   their actual labels. Choose a palette/reversal, name the view and select **Save chart**.
4. Leave **One point per trial** to retain distinct tested combinations. **Mean per exact X
   value + SD** groups equal X values with equal weight per Trial; dispersion is empirical SD
   across Trials, not a seed confidence interval or a controlled parameter effect. Heatmap/
   surface cells likewise average observed Trial summaries at each exact X/Y pair. Untested
   pairs stay blank, including missing levels in an authored finite domain; no model is fitted
   or missing response interpolated. Very large grids request a 3D scatter instead.
5. By default only completed evidence is displayed. Enable **Include partial / pruned
   observations** to inspect recorded partial values; their marks/hover identify censored
   evidence. Choose **Best observed objective** or **Current observed objective** explicitly
   when desired. A missing final selection remains missing even if a partial best exists.

Selection/objective-component summaries preserve their persisted semantics. Other numeric metrics
use persisted diagnostic summaries or the latest recorded values from comparable screening Runs
at one fidelity rung, not newly computed per-metric optima. Hover exposes exact values, Trial IDs
and cell support. Saved views and palettes survive reopening the same HTML in the same browser;
they do not change HPO, authored YAML or evidence. Regenerate the report to include newer results.
If a preview is empty, its inline explanation distinguishes missing parameter values, missing
final selection and metric/evidence-filter gaps. It never substitutes a partial best for selection.
In **Parameters**, choose **Y metric** to plot any recorded numeric metric and update the
exact-value table. The observed chart also works for sweeps without model response curves.
The separate collapsible **Persisted adjusted selection response** remains objective-only:
changing the diagnostic metric must not relabel the persisted objective model as an accuracy model.

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

### 16.4 Research workspace and declarations

The offline report opens **Overview**, with **Research**, **Metrics & health**, a global search
(**Ctrl/⌘ K**), family plots and **Explore**. The Python domain layer persists metric profiles,
bounded exploratory relationships, multiplicity diagnostics and inspection findings; browser code
only filters/presents recorded evidence. Declare optional YAML `analysis` or class-level
`Work.analysis_profile` for metric labels/units/lineage, families and questions. Declarations and
known registry defaults are frozen in `analysis-semantics.json` before execution and travel with
recovery/export. Schema version 8 adds `research` while retaining prior scientific fields.

Test metrics, including derived/composite components, cannot govern objective/constraints.
Provisional discovery excludes test evidence, while terminal reports may inspect it. Unknown
semantics remain explicit; discovery is not causal inference or new HPO confidence. Distinct Y
units use small multiples in Explore and distinct Y axes in Parameters; unknown units may share a
visual scale when observed ranges substantially overlap, without claiming unit equivalence. Visual
normalization is opt-in. Saved views support notes and validated
JSON import/export. [The full research guide](#guide-research-analysis) contains numbered instructions,
YAML examples, algorithm limits, statistical caveats and legacy behavior.

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
Data loading follows the navigation hierarchy. Overview makes one inventory pass per direct
provider, or reads only active-job status for schedulers without inventory, and returns compact Job
and Study projections; it neither discovers remote Dataset manifests nor embeds
candidate/Run indexes. Work and Studies collection screens use the same research-only snapshot and
do not probe resources or Datasets. Opening a Study loads its bounded candidate/Run index. Analysis,
the complete controller history and outer logs remain lazy per tab, and seed curves/artifacts/logs
are read only for the selected Run. Machine clients follow the same progression: `overview --json`,
then `show WORK`, then `show WORK --run KEY --json`.
Direct inventory isolates per-Job malformed/missing/unsafe records. Healthy siblings remain
observable, while the affected known Job retains an explicit degraded diagnostic and any saved
terminal outcome. Error entries are not ownership evidence and cannot create Jobs or authorize
recovery. Transport/root failures remain host-wide; observation never repairs persisted files.
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
The visible **Delete Study History…** control delegates to the same exact semantic Work deletion
service as automation: it previews owned terminal state, confirms once, then removes it. Active
Studies are refused until cancellation, and equal display names never broaden the target.
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

The seed **Interactive HTML** action turns those same exact persisted series into a responsive
offline analysis dashboard. It starts with no more than four objective/validation metrics, while a
searchable, grouped sidebar can opt into any recorded scalar. One selection drives raw curves,
per-metric 0–1 trend normalization, latest-value/delta bars, pairwise Pearson correlation over
shared epochs and a statistics view with latest/minimum/maximum/change/sample count. Normalization
compares shape and never changes or ranks original values; correlation is explicitly descriptive.
Best and selected epochs remain reference lines. The file embeds Plotly and all data, contacts no
service and does not become another result store.

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

This applies to `storage.dataset_root` too: an authored `/persistent/datasets` becomes
`/persistent/datasets/projects/<project-id>` for new publications. An older DatasetVersion can
still resolve from an unscoped absolute root when that exact placement was already registered and
its manifest/content identity verifies. LambdaForge preserves that durable legacy placement; it
does not infer that every project may use arbitrary bytes below the old root. The hexadecimal
directory below `NAME/VERSION/` is the content ID prefix, not a project or random run ID. Publishing
different bytes under the same `NAME@VERSION` is refused; increment the version instead.

`lf datasets reconcile NAME@VERSION --on CLUSTER` previews index-only repairs. A conflicting remote
record is removable through `--apply` only when its registered directory is proven absent; valid or
unreachable bytes remain a hard conflict. Publication failures roll back newly committed bytes, so
an immutable-version rejection cannot leave a new unregistered managed directory. Remote-to-remote
replication never falls back to relaying arbitrary dataset size through controller disk: use a
configured site transfer facility, then reconcile the exact verified destination placement.

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

## 19. Operational reliability

Use `lf health [JOB] [--refresh] --json` and `lf incidents --json` to inspect operational evidence
separately from scientific results. `lf monitor start/stop/status/config/diagnose/once/install`
controls an independent project observer; optional SMTP uses credential references and durable
deduplication. Bare `lf` exposes **Operational health…** with the same services.

Native `lf retry STUDY --config NEW_YAML --dry-run --json` verifies relocated configuration against
original scientific identity, without creating computation; apply revalidates process ownership
and pins the chosen file. Recovery request ID and admission/runtime/scientific progress are
auditable through native Job records/events. See the [operating and emergency guide](
OPERATIONAL_RELIABILITY.md) for policy, platform boundaries, optional email, recovery semantics and
the broader Fleet/live-checkpoint requirements that are still pending. Managed checkpoints offer
opt-in `configure/should_save` cadence and terminal retention, plus paged `inventory(verify=True)`
host-side checks. Availability is not proof of application restoration. Concurrent provider probes
retain last-good facts and their real ages; an audited recovery resolves origin incidents only
after fresh scientific progress, never merely on acceptance.
Loading reused completed Runs is a baseline, not new recovery progress. Use `lf jobs recovery JOB
--json [--offset N --limit N]` or Study → **Recovery history** for paged saved native events, linked
Jobs and rejected requests even offline. Console validation records an operational inspection;
CLI dry-run remains entirely read-only. A persistent failure dialog retains the request/phase and
links logs, history and the same plan, without repeated polling notifications.

## 20. Fused Gromov–Wasserstein distance (FGW)

FGW compares **two complete weighted attributed objects**, not individual nodes and not clusters.
Install the optional solver explicitly: `python -m pip install 'lambdaforge[optimal-transport]'`.
Ordinary LambdaForge imports do not load or install POT, SciPy, sklearn or Torch for this API.
The implementation delegates conditional gradient and line search to Python Optimal Transport
(POT), rather than maintaining a second numerical optimizer.

### Inputs and quick start

`lambdaforge.distances.StructuredMeasure(structure, features=None, weights=None)` represents one
object. `structure` is a finite, symmetric, nonnegative N×N intrinsic dissimilarity matrix with
zero diagonal. Triangle inequalities are not checked. It may describe Euclidean point geometry,
graph shortest paths or another explicitly chosen undirected relationship. Disconnected infinities
must be handled scientifically by the caller, not silently replaced by large constants.

Relative weights must be finite, nonnegative and have positive total mass. They are normalized
to probabilities; `total_mass` retains their original sum. This is **balanced probability OT**,
not a comparison of absolute masses. Zero-weight points are permitted. Objects may contain
different numbers of points and their geometry may have different coordinate dimensions.

```python
from lambdaforge.distances import FusedGromovWassersteinDistance, StructuredMeasure

a = StructuredMeasure.from_points(
    [[0, 0], [1, 0]], features=[[0, 1], [1, 0]], weights=[1, 3],
)
b = StructuredMeasure.from_points(
    [[0], [1.5], [3]], features=[[0, 1], [0.5, 0.5], [1, 0]], weights=[1, 1, 2],
)
solver = FusedGromovWassersteinDistance(alpha=0.5)
result = solver.solve(a, b)
value = solver(a, b)  # Scalar only; solve() additionally exposes diagnostics.
```

`from_points` constructs intrinsic SciPy distances (Euclidean by default); coordinates are **not**
implicitly copied into features. Supply features explicitly. Each feature is a numeric vector
in a shared feature space, or an arbitrary project-defined object using a custom metric below.
The executable [FGW example](../examples/fgw.py) also checks transport marginals.

### Arbitrary features, precomputed costs and geometry

For structured attributes, pass a scalar distance function through
`FusedGromovWassersteinDistance(feature_metric=lambda a, b: ...)`. The function receives one
source and one target feature, which may be dictionaries, sequences or domain objects. LambdaForge
does not interpret proteins, graphs inside attributes or application-specific scientific meaning.
Alternatively, `solver.solve(a, b, feature_distances=D)` accepts a finite nonnegative N×M matrix.
These are **unsquared distances**, squared exactly once for q=2. SciPy metric names use the same
contract; `sqeuclidean` would therefore produce fourth powers and is usually inappropriate.
For structure-only comparison (`alpha=1`), features are optional. `alpha=0` is feature-only OT.
Numeric features must be finite N×F/M×F with the same F. No automatic scaling or imputation occurs.

### Objective, coupling and interpretation

The implemented q=2 objective is:

```text
min_T (1-alpha) Σ_ij d_feature(i,j)² T_ij
    + alpha Σ_ijkl (C_source(i,k)-C_target(j,l))² T_ij T_kl
subject to T >= 0, T 1 = source.weights, Tᵀ 1 = target.weights.
```

`FGWResult.value` is this **unrooted** objective; `feature_cost` and `structure_cost` are its
unweighted components, and `transport` is the N×M node coupling, not cluster labels. `alpha` is
between zero and one. Relative units directly change the balance: any normalization belongs
explicitly in the project. `iterations`, `objective_history`, `marginal_error`, `converged` and
`backend` make numerical status inspectable. Backend transport failure or invalid marginals raise
an error; reaching `max_iter` may return `converged=False`, never a false optimality certificate.
Inspect this flag rather than treating every numerical value as a converged result.

This dense NumPy/CPU computation is non-differentiable. It is not a neural pointwise
`lambdaforge.nn.distances.Distance`, a clustering strategy, an unbalanced solver or a barycenter API.
For intermediate alpha the problem is non-convex: local convergence does **not** certify the
global minimum or a mathematical metric for arbitrary dissimilarities. Different starts can
yield different local minima. An explicit feasible N×M `initial_coupling` permits warm starts;
the default follows POT's deterministic product coupling (self-comparison uses a diagonal coupling).

### Resource bounds and multiple objects

`StructuredMeasure.from_points(..., max_structure_bytes=256*1024**2)` checks dense N² storage
before construction. `FusedGromovWassersteinDistance(max_memory_bytes=512*1024**2)` bounds a
conservative workspace estimate before cross-distance allocation, not the third-party process's
hard memory ceiling. Structural loss is contracted without an N×M×N×M tensor. Large objects still
require project-side scientifically justified reduction; no hidden subsampling changes evidence.
`solver.pairwise(objects)` returns an object×object discrepancy matrix. Both directions are solved
independently by default because local minima can differ; `symmetric=True` explicitly assumes
symmetric feature distances and mirrors the upper triangle. It does not execute clustering.

Reference: [Vayer et al., ICML 2019](https://proceedings.mlr.press/v97/titouan19a.html),
[POT FGW API](https://pythonot.github.io/gen_modules/ot.gromov.html).


<a id="guide-work-results"></a>

## 21. Work results, historical evidence and independent products

[Español](MANUAL.es.md#guide-work-results) · [Manual](MANUAL.md) · [Products](#guide-products)

A Work is a scientific execution, not necessarily a training experiment. A Study adds an authored
parameter/seed design and scientific coordination. Both use the same runner, Execution/Run/Attempt
records, managed outputs and export/import transaction. Ordinary Work reports never fit a surrogate,
invent an objective or aggregate different Attempts as if they were one observation.

<a id="guide-work-results-inspect-and-report"></a>

### Inspect and report

Open a Work in the Research Console and select **Results**. The Run/Attempt selector keeps failures,
retries and their evidence separate. Overview, Metrics, Outputs, Visualizations, Products, Logs, Resources and
Provenance load only the selected panel. Root Results lists use compact indexes; they do not fetch
scalar history, models or project HTML. Opening Metrics reads bounded curves of the selected Attempt;
opening Outputs reads descriptors, structured values and checkpoint names, not model bytes. Checkpoints
are explicitly labelled as current shared logical Run state, not archived snapshots of an old Attempt. Missing
retained files are shown as such. An in-progress Job needs its persisted Execution attestation before
deep evidence is available; Job logs and controls remain available during preparation.

Provenance includes a selectable table of exact historical-result/product inputs. Enter opens the
recorded source when available in the local catalog, never another execution with the same name.
A missing source gives an explicit export/import instruction rather than launching its producer.

```bash
lf results list --json
lf results show EXECUTION --view overview --json
lf results show EXECUTION --view metrics --run RUN_ID --attempt 1 --json
lf results show EXECUTION --view outputs --run RUN_ID --json
lf results preview-output EXECUTION OUTPUT_NAME --run RUN_ID --attempt 1 --json
lf results report EXECUTION --output work-results.html
```

In Outputs, Enter explicitly previews a selected retained regular file only when it is at most
64 KiB. The preview verifies its exact size and managed checksum; UTF-8 is shown as inert text,
including HTML. Larger files, directories and unavailable artifacts show metadata and an explanation,
without downloading their bytes. Use the report's isolated viewer to run a registered HTML visualization.

**Open HTML report** explicitly builds an offline report from saved records. It includes bounded log
tails, downsampled scalar curves, exact Attempt metadata, outputs, failure, configuration and provenance.
The metric library is embedded once, with separate Run/Attempt groups and step/observation axes. Different
Attempts are never silently combined, and cross-Attempt correlations are not presented as measured evidence.
Exports also retain the complete original logs/scalar files and retained checkpoints/artifacts. No old
scientific package is imported to display evidence. A Work without metrics or HTML still has a report.
Failed reports label partial evidence honestly; compacted failed artifacts are not reconstructed.

Register project visualizations with the existing API:

```python
viewer = self.outputs.html_section("predictions", title="Predictions")
viewer.write_text("<html><body><h1>Saved scientific view</h1></body></html>")
```

The project owns its visualization logic. LambdaForge verifies the registered file and embeds it in
an opaque-origin sandbox using the same security policy as Study reports. HTML must be self-contained;
use included/data-URL assets. External requests and privileged filesystem access remain blocked.
Limits remain 16 MiB per project document and 64 MiB per report. Reading a list never opens these files.

For full-resolution arrays beyond those HTML limits, use managed `html_bundle` and its scoped
lazy browser loader. See [managed report tabs](#guide-managed-reports) for API, remote/offline viewing,
export/import integrity and retained-inference-only regeneration.

<a id="guide-work-results-choose-an-exact-historical-dependency"></a>

### Choose an exact historical dependency

Human names are selectors, never identities. Repeating a name is allowed. One match resolves directly;
multiple matches require an exact Execution ID. The Console shows deterministic local `#n` labels with
the original identity. These labels are not portable scientific selectors. Imports retain their original
Execution and origin; they never become executable Jobs.

```bash
lf results reference previous-work --json
lf results reference EXECUTION --run RUN_ID --attempt 1 --json
lf results reference EXECUTION --product selected-models --json
lf results reference EXECUTION --product selected-models --artifact model-1 --json
```

These metadata-only commands return a typed YAML/JSON marker with exact IDs fixed internally. A later
same-name execution cannot change that marker. In **Run Work**, enter the consumer parameter name and
choose **Choose historical input…**. Select an exact catalog row, then **Use exact reference**.
The picker also exposes exact Run/Attempt selection and registered product-artifact selection.
Submit validates, explains and prepares the bound configuration without editing the authored YAML. The same
operation is available through `lf run CONFIG --input-ref 'PARAMETER=JSON_MARKER'`. Launch bindings apply
to a single Work/Study, not a composed or Fleet configuration; those declare typed inputs in their YAML.
An input changed between selection and preparation fails explicitly rather than selecting another source.

<a id="guide-work-results-resultinput-evidence-of-one-execution"></a>

#### ResultInput: evidence of one Execution

```yaml
with:
  previous:
    result:
      execution: previous-work
      # run: run-...        # required to read an individual Run of a multi-Run Execution
      # attempt: 1          # optional exact Attempt; otherwise latest is pinned by the envelope
```

The resolver adds `evidence_id`, a canonical SHA-256 of the exact original result/configuration.
The consumer receives `lambdaforge.work.ResultInput`: immutable `metadata`, `configuration`, `metrics`,
`result`, `outputs`, bounded `metric_curves()` and explicit verified `artifact(NAME)`. It is pickle-safe
for spawned workers. Its identity is part of the consumer fingerprint and recorded inputs. It accepts
finalized native/imported snapshots (including failures), never a mutable live result. Multi-Run evidence
requires explicit Run selection before reading individual metrics/results/artifacts. Deleting its owning
evidence can invalidate this dependency: it is not an autonomous model contract.

<a id="guide-work-results-productinput-durable-scientific-content"></a>

#### ProductInput: durable scientific content

Existing declarations remain supported:

```yaml
with:
  model: {product: {name: selected-models, contract: example/models:v1}}
```

For an exact published output of a known Execution:

```yaml
with:
  model:
    product:
      from: {execution: previous-study, output: selected-models}
      artifact: model-1  # optional exact registered member; not a filesystem path
      expect: {dataset: example-data@1}
```

The contract can be omitted only when the exact publication receipt declares it unambiguously. It is
still validated; expectations remain explicit. Native preparation replaces the source selector with the
sealed content ID. Direct product references work independently even after deleting the producer.
Unpublished temporary outputs are not products. Dataset inputs and same-execution `from` references
keep their existing semantics.

With `artifact` declared, `ProductInput.selected_artifact` exposes the choice and `artifact()` verifies
and opens that exact member. Requesting another member fails instead of substituting it. Without it,
the existing explicit `artifact(NAME)` API is unchanged. This is a selection within the same sealed
product and contract, not a fabricated sub-product. Its logical member name participates in the
consumer identity and survives pinning, remote/Fleet preparation and export/import.

<a id="guide-work-results-explicit-promotion-from-an-ordinary-work"></a>

### Explicit promotion from an ordinary Work

```yaml
products:
  evaluation-report:
    kind: ScientificReport
    contract: example/evaluation:v1
    scientific_meaning: {protocol: evaluated-on-fixed-test-v1}
    outputs: [report, counts]
```

Supported ordinary kinds are ScientificReport, AnalysisResult, ModelArtifact and Selection. Meaning is
a publisher declaration, not an independently proven equivalence. List registered structured values
and retained regular files explicitly. Directory members must be registered individually; a directory
is not implicitly guessed as a model. Promotion currently requires one logical Run with a successful
latest Attempt; multi-Run model selection continues using the existing ModelSet declaration.

Finalization seals immutable independent bytes with the native registry and locks. Scientific results
are persisted first. A storage/publication failure leaves successful scientific evidence unchanged and
preserves the outputs needed by `lf products finalize EXECUTION --apply`; that retries only publication.
Concurrent retries reuse the exact publication. Retiring the producer does not remove sealed products.

<a id="guide-work-results-portable-evidence-and-remote-availability"></a>

### Portable evidence and remote availability

```bash
lf export WORK_OR_EXECUTION --output ./exports
lf import ./exports/EXPORTED_PACKAGE --json
lf import ./exports/EXPORTED_PACKAGE --apply
```

Works use the existing native version-2 export/import format, with additive `execution_kind` metadata.
Older Study packages remain readable. Original records stay in `portable/`; local placement/indexes
stay separate. **Results** and imported **Work** entries open the same explorer without contacting the
original cluster. Re-export of an import verifies and preserves its original package/provenance.

Resolving an ID does not materialize its bytes elsewhere:

```bash
lf products materialize EXACT_CONTENT_ID --on cluster
lf products materialize EXACT_CONTENT_ID --on cluster --apply
lf results materialize EXECUTION --on cluster
lf results materialize EXECUTION --on cluster --apply
```

Console product/result explorers provide **Materialize…**, destination selection, preview and explicit
confirmation. Preview reads source metadata only. Apply uses a compressed ZIP64 package and native
safe extraction/import, verifies original content/evidence identity and reuses an exact existing copy.
No producer is launched. Dependencies must already be available on every supported target, including
Fleet members. Fleet preparation includes pinned historical dependencies in its native input stratum;
each member rechecks them before executing its concrete Runs. Existing distributed dataset/continuation
gates remain unchanged. Preparation/worker checks fail closed if absent. A remote-only source should first be
explicitly exported/imported into the controller catalog; this operation is not an implicit inter-cluster
orchestration service. Destination Python must provide the current LambdaForge runtime and sufficient
staging/durable space. Source scratch is project-owned, not the system temporary partition. SSH credentials,
reachability and site permissions remain external requirements.

See [the executable example](../examples/work_results/README.md) for Study → selected ModelSet →
independent visualization Work → custom HTML → export/import, without repeating training.

<a id="guide-work-results-evidence-across-fixed-study-extensions"></a>

### Evidence across fixed Study extensions

An opt-in Study series references exact native Execution/Run/Attempt records across immutable
design revisions; it never clones old results into a delta. `studies show/analyze/report` and the
Console series workspace provide cumulative evidence with exploratory post-hoc interpretation.
Native Job/result deletion and GC protect these references. A series export contains independent
native Execution packages plus cumulative JSON/HTML and its journal; individual package imports
remain read-only evidence, not executable series adoption. See [Study extensions](#guide-study-extensions).


<a id="guide-products"></a>

## 22. Durable scientific products: implemented foundation

Ordinary Work output promotion, exact historical source references, ResultInput versus ProductInput,
Console selection and compressed explicit materialization are described in [Work results](#guide-work-results).

[Español](MANUAL.es.md#guide-products)

This is the working **product model/catalog/transport and explicit Work-input layer**, not the
completed Study orchestration system. Native Study decisions, local model selection, typed Work
dependencies and actual consumer records are available. A single native Study may declare automatic
post-Study decision/model publication. Native single-host Study export/import includes published
products. The Console includes a metadata/audit browser and confirmed Study import.
Dependency waiting/replanning and complete Fleet export remain pending.
See the [reform ledger](ai/IMPLEMENTATION.md#history-architectural-reform). Existing Work execution/recovery and exact Dataset
publication remain unchanged.

<a id="guide-products-identities-and-contracts"></a>

### Identities and contracts

`lambdaforge.products` exports `ProductContract`, `ProductArtifact`, `StudyProduct`,
`ProductRegistry`, `ProductBundle`, `SelectionPolicy`, `ModelSelection`, `select_models`,
`build_study_decision`, `ProductRequirement`, `ProductInput`, `ProductPublication` and
`publish_declared_products`. A product has:

- an explicit versioned contract, such as `project/report:v1`, and an exact set of scientific fields;
- `scientific_id`, computed from kind, contract declaration and those declared meanings;
- `content_id`, computed from the canonical payload/meaning and exact artifact manifests;
- immutable original producer provenance, with separately appended attestations for later producers;
- a physical catalog placement that does not alter either identity.

The producer's full configuration may be provenance; it is **not** the consumer compatibility key.
Names are immutable aliases. The same contract identifier cannot acquire another field declaration;
use a new contract version. Changing scientific objective, input/label semantics, design or selection
rule must change the declared meaning. Nothing automatically guesses exclusions inside NPZ, model
checkpoints or arbitrary formats. Equal declared scientific meaning does not certify equivalence of
different bytes.

Every artifact is an independent regular file with a raw SHA-256 and size. Publication copies and
verifies it before committing. It never links to a disposable Attempt, removes the source, or scans
unselected files. Metadata reads are bounded at 512 KiB each. Listing/showing does not open large
artifacts; `verify` is explicit. Missing reads and previews do not create directories or locks.

<a id="guide-products-explicit-publication-api"></a>

### Explicit publication API

Use this API from a postprocessing/operator script, not as a hidden replacement runner. Native
post-Study declaration and explicit post-Study model selection are available below.

```python
from lambdaforge.products import ProductContract, ProductRegistry, StudyProduct

contract = ProductContract("my_project/count-report:v1", ("sources", "labels", "method"))
report = StudyProduct(
    name="benchmark-counts-v1",
    kind="ScientificReport",
    contract=contract,
    payload={"train": 317, "validation": 272},
    scientific_meaning={
        "sources": {"dataset": "sha256:<exact-published-content>"},
        "labels": {"positive": "binding", "partitions": "audited-v1"},
        "method": "count-audited-members-v1",
    },
    producer={
        "execution_id": "<actual-original-execution>",
        "evidence_fingerprint": "<actual-persisted-evidence-fingerprint>",
    },
)
registry = ProductRegistry()  # Current pyproject project/.lambdaforge/products
plan = registry.publish(report)  # Read-only; does not publish.
registry.publish(report, apply=True)
resolved = registry.resolve(
    "benchmark-counts-v1", contract=contract,
    scientific_expectations={"method": "count-audited-members-v1"},
)
```

Placeholders in the example must be replaced by actual evidence; the generic envelope does not
derive or certify that evidence. Kind labels such as `StudyDecision`/`ModelSet` do not automatically
select models or prove their payload semantics. The project owns its explicit declaration.

For artifacts, pass `ProductArtifact(name, relative_path, raw_sha256, size_bytes, role)` descriptors
and `files={artifact_name: source_path}` to `publish`. `artifact_path(selector, name)` verifies the
selected promoted bytes by default before returning their path. Do not substitute the framework's
historical filename-aware managed-file fingerprint for the artifact's raw byte SHA-256.

<a id="guide-products-native-cli"></a>

### Native CLI

```bash
lf products list --json
lf products show benchmark-counts-v1 --json
lf products provenance benchmark-counts-v1 --json
lf products consumers benchmark-counts-v1 --json
lf products verify benchmark-counts-v1 --json
```

All operations accept an explicit `--root DIR`; otherwise use the current project's catalog or
`LAMBDAFORGE_PRODUCT_ROOT`. Metadata list/provenance support `--offset`/`--limit` pages. Scientific
code should not guess remote Job paths. Workers receive a project-owned product root from the
existing control plane, separate from disposable Jobs and caches. The remote default is
`STATE_ROOT/products` for the scoped project; local execution uses `PROJECT/.lambdaforge/products`.
Unsafe remote roots nested in the Job or cache root are rejected.

An operator may serialize `product.to_dict()` and publish it through
`lf products publish MANIFEST.json --file ARTIFACT=SOURCE [--apply]`. This is an advanced explicit
declaration, not automatic selection and not an application-specific workaround. Preview checks
source shape/size; apply checks exact bytes. A corrupt source never becomes a valid published
product. A conflicting immutable name requires another name/version, never an overwrite flag.

<a id="guide-products-post-study-modelset-selection"></a>

### Post-Study ModelSet selection

Register a scored model snapshot from the ordinary Work lifecycle:

```python
self.outputs.from_checkpoint(
    "model", "best.ckpt", role="model",
    metadata={"metrics": {"auprc": measured_auprc, "accuracy": measured_accuracy}, "step": epoch},
)
```

The project must supply values evaluated for **those exact weights**, not an unrelated last epoch.
This keeps an ordinary independent Attempt artifact. Promotion to a durable catalog is deliberate.
Alternatively use `outputs.file` with model role and the same explicit metadata. Directory models
and externally published artifacts are not yet supported by the local selector.

Create an ordinary selection YAML, for example `selection.yaml`:

```yaml
artifact: model
rank_by: auprc
mode: max
group_by: [hidden_dim]  # Omit for a global top_k.
top_k: 1
constraints:
  accuracy: {min: 0.6}  # Same registered snapshot, never latest Run metrics.
tie_policy: stable
```

```bash
lf products select EXECUTION_ID --name best-per-width --contract my_project/models:v1 \
  --policy selection.yaml
# After reviewing, repeat with --apply.
```

Use `--results-root DIR` for an explicit local ResultStore and `--root DIR` for the product catalog.
The same operation is exposed by `select_models(source, execution_dir, policy, name=..., contract=...)`.
It selects latest valid Attempts, excludes failed/pruned/partial-fidelity outcomes, applies explicit
same-snapshot constraints and deterministically ranks each exact conditional group. Inactive and
explicit null parameter values are distinct. `tie_policy: include_equivalent` optionally includes
models within an authored `practical_margin` of the top-k boundary; it is not a confidence claim.

Selection reads/hashes **selected model files only**, so even preview may perform substantial
local I/O for large selected weights; it creates no files/locks. Apply promotes exact independent
copies. Source evidence/failure history stays unchanged. The ModelSet records chosen logical Runs,
Attempts, seeds, parameters, snapshot metrics/steps, grouping, rationale, content checksums and
producer evidence. A failed unrelated Run does not invalidate completed eligible snapshots.
The ranking is explicitly **among eligible observed snapshots**, not proof of a global optimum,
fresh-seed confirmation or a redecision by HPO. Missing snapshot-bound metrics fail with guidance;
the selector never guesses that Run best/last metrics refer to arbitrary checkpoint bytes.

Remote/Fleet selection and publication inside composed `steps` remain pending. The local selector
never reaches outside its verified owned Execution to guess model paths.

<a id="guide-products-native-studydecision"></a>

### Native StudyDecision

```bash
lf products decide EXECUTION_ID --name pooling-decision --contract my_project/pooling:v1
# Review the native decision, then repeat with --apply.
```

This seals the native finalized Study selection, objective/design/input meanings, selected
Run/Attempt identities and evidence fingerprint. It does not rank candidates again or fit HPO.
Selection and scientific resolution remain distinct: missing final Analysis leaves explicit
unresolved questions, and incomplete fresh-seed confirmation cannot become a survivor-only
selection. A failed auxiliary Run does not invalidate an otherwise complete eligible selection.

If `analysis.json` exists, it must match the exact finalized Execution/evidence identity; stale,
provisional or mismatched conclusions are rejected with guidance to recompute Analysis explicitly.
No scalar histories, checkpoints or model bytes are read for this metadata operation. The Python
routes are `ResultStore.decision(...)` and `build_study_decision(...)`.

<a id="guide-products-declared-post-study-publication-and-publication-only-retry"></a>

### Declared post-Study publication and publication-only retry

A single Study (`search`, `sweep` or repeated `seeds`) may declare native products alongside its
ordinary Work fields:

```yaml
products:
  pooling-decision:
    kind: StudyDecision
    contract: my_project/pooling:v1
  best-per-width:
    kind: ModelSet
    contract: my_project/models:v1
    select:
      artifact: model
      rank_by: auprc
      mode: max
      group_by: [hidden_dim]
      top_k: 1
```

Validation checks this closed declaration before computation. `lf explain` includes it, and the
Execution records its expected products before Runs start. Finalization persists training evidence
first, then uses the same decision/selection/catalog services to publish independent products
before compaction. `StudyDecision` requires an objective; a ModelSet requires explicitly scored
artifacts. These are not arbitrary constructors, executable callbacks or a second runner.

Publication failure preserves every native result/checkpoint. `products.json` records pending,
published or failed products with the exception/phase; `product-publication-history.jsonl` retains
performed publication attempts. Run status/objective evidence is not rewritten as a training
failure. Inspect publication separately:

```bash
lf products status EXECUTION_ID --json       # Bounded metadata; no weight reads.
lf products finalize EXECUTION_ID --json     # Read-only preview.
lf products finalize EXECUTION_ID --apply    # Publication only; no Runs launched.
```

Publication-only apply acquires the native Execution ownership lock, then the publication/catalog
locks. Successful receipts are reused without requiring original Attempt artifacts; promoted bytes
are reverified on apply. Concurrent repetition is idempotent; corrupt receipts/evidence and changed
declarations fail closed. Immutable-name conflicts require a deliberate new product name/version,
not overwriting an existing model set. `--results-root` and `--root` select local evidence/catalog.
The CLI does not currently dispatch publication-only recovery to a remote host; execute it on the
host owning that Execution/catalog. Automatic Fleet promotion is not implemented.

<a id="guide-products-typed-work-dependencies"></a>

### Typed Work dependencies

Declare the product as an ordinary Work argument, not a producer-status condition:

```yaml
name: model-visualization
run: my_project.Visualize
with:
  models:
    product:
      name: best-per-width
      contract: my_project/models:v1
```

The project may add `expect: {FIELD: EXACT_VALUE}` to require explicit scientific meanings.
The resolver checks the declared contract/expectations and supplies a pickle-safe `ProductInput`:

```python
import lambdaforge as lf
from lambdaforge.products import ProductInput

class Visualize(lf.Work):
    def run(self, models: ProductInput) -> dict[str, int]:
        for model in models.payload["models"]:
            weights = models.artifact(model["artifact"])
            # Read these verified weights and register ordinary managed visualization outputs.
        return {"models": len(models.payload["models"])}
```

Metadata access does not hash/download model bytes. `artifact(name)` verifies that exact promoted
file before returning its read-only input path. Compatibility and the consumer's input identity
use sealed contract/science/content, not the full producer YAML hash or current producer status.
Validation/preview create nothing; actual Run Attempts append immutable consumer records, exposed
by `lf products consumers`. Cached Execution reuse does not fabricate a new consumption event.

Remote bundles pin aliases to exact content IDs and verify materialization in the destination
project catalog before staging. Import the explicit product bundle there first; LambdaForge does
not implicitly transfer large weights, expand a name to another representation or fall back to a
Job-local catalog. Missing/incompatible products fail with a diagnostic; automatic dependency
waiting or producer replanning is not yet implemented.

<a id="guide-products-portable-product-exportimport"></a>

### Portable product export/import

```bash
lf products export benchmark-counts-v1 --output ./report-export
lf products export benchmark-counts-v1 --output ./report-export --apply
lf products import ./report-export --root ./independent-products
lf products import ./report-export --root ./independent-products --apply
```

These operations transport **one product**, not the whole producer Study. Export writes a verified
directory containing `bundle.json`, `product.json`, selected file bytes and complete producer
attestations (not downstream consumer audit records). Existing destinations are protected. Import checks manifest SHA/identity, exact
artifact bytes and the complete provenance inventory; it preserves the original producer and
never launches computation. Repeated/concurrent imports of the same product are idempotent. Even
when the destination already contains the product, a corrupted import source is rejected.

Publication/import are serialized with the existing native cross-process lock. Interrupted copies
remove only their exact unpublished transaction directory; original files stay untouched. A crash
after object commit but before catalog binding may leave an unbound immutable object: repeating
the same operation verifies and completes its binding, not a silent new execution.

An exported/imported product survives deletion of its original Study. This does **not** implement
a dependency DAG, signed producer authenticity, certificate-based Dataset equivalence or complete
Fleet export.

<a id="guide-products-import-a-portable-study-without-executing-it"></a>

### Import a portable Study without executing it

```bash
lf import ./portable-study--execution-ID --json
lf import ./portable-study--execution-ID --apply --json
lf results list --json
lf products show models --json
```

The default verifies/previews; `--apply` registers evidence in the current project's ResultStore
and sealed products in its native ProductRegistry. `--results-root`/`--products-root` optionally
override placement, never scientific identity. Native version-2 exports are accepted: final,
failed/cancelled and running snapshots retain captured status. A snapshot never becomes completed
evidence. Pre-execution Job-only packages without an owned Execution are not Study imports.

Verification covers all listed files, SHA-256/size/count, absence of extra files, metadata versions,
Execution/Run/Attempt identity/ownership and product/producer attestations. Traversal, symbolic,
special or hard-linked content, duplicate inventory and corrupt incoming/existing evidence fail
closed. Concurrent imports are serialized/idempotent. A native Execution or another snapshot with
the same Execution ID is never overwritten: use a separate results root to retain multiple captures.
Checksums do not establish cryptographic authenticity of an untrusted author's scientific claims.

Original files/provenance remain intact in `portable/`; `import.json` records the new placement.
ResultStore lists the entry, relocates logical Run paths only in its log read model and reads included
Analysis without refitting. No application imports, computation, checkpoint continuation or automatic
HTML opening occur. Imported configuration cannot become native recovery state. An imported
archive's `portable/` directory is already the reusable package; it needs no new producer export.

In the Research Console, **Studies → Import Study…** (also accessible from Products and Ctrl+P)
selects a package directory, verifies it in a worker, previews its exact identity and asks for
confirmation. Registration adds a labelled read-only snapshot to Studies/Overview in the current
project. Trials/seeds use local hierarchical projections and the shared selected-Run dashboard;
retained metrics/logs/artifacts are read locally, not from the original cluster. Captured running
snapshots are not live controllers. Cancel, retry, history deletion and producer export controls
are hidden; imported archive deletion remains an explicit Results operation. HTMLs included in the
package remain accessible in its `portable/reports/` directory.

The native aggregate `execution/result.json` may exceed 64 MiB in a large valid Study. Import first
verifies its exact declared size and SHA-256 along with every file, then strictly parses/validates
that aggregate without applying the individual-metadata size ceiling. Checksums, path/ownership,
duplicate-field and non-finite-value checks are unchanged. Root-screen refresh only reads small
`import-study.json` indexes, not the aggregate. Reapply an older verified import once to build its
`import-view/` presentation files under the import lock; scientific records and the original receipt
remain untouched. Import copies retained bytes and therefore needs local space for the package and
its local index; it never downloads artifacts again.

Export includes products published through the native declaration/receipt, their independent bytes
and original producer history. Provider transfer fetches only those objects, never the whole catalog;
telemetry sampling cannot alter product bytes. Missing published products fail explicitly. Undeclared
operator publications use `products export`, not discovery by sweeping unrelated catalogs.

Archive registration commits atomically before independently locked product bindings. A later
product transaction failure leaves the verified archive available: retry the same import to finish
binding. This is not a cross-store distributed transaction. Deleting imported Study evidence preserves
its independently promoted products.


<a id="guide-conditional-studies"></a>

## 23. Conditional scientific Studies

[Español](MANUAL.es.md#guide-conditional-studies) · [Research HTML](#guide-research-analysis)

<a id="guide-conditional-studies-1-one-question-one-study"></a>

### 1. One question, one Study

Use one `sweep.space` when comparing alternatives with different applicable parameters. All cells
belong to one Study and share the same authored seeds. Use `steps` for a real workflow, not to
partition comparable experimental branches. Steps remain separate invocations, not merged Studies.

```yaml
name: optimizer-comparison
run: my_project.Compare
seeds: [4, 7, 32, 54]
sweep:
  reference: {optimizer: baseline}
  space:
    momentum:
      values: [0.8, 0.9]
      when: {optimizer: {in: [sgd, accelerated]}}
    optimizer: [baseline, sgd, accelerated]
resources: {gpu: 2, time: 24h}
execution: {max_parallel: 4, max_time: 12h}
```

This creates five candidates and twenty required Runs, not twelve candidates or one Study per
optimizer. `baseline` has no `momentum` argument; the Work signature's default applies. A reference
must select exactly one active candidate. Zero matches or multiple matches are authoring errors.

<a id="guide-conditional-studies-2-activation-grammar"></a>

### 2. Activation grammar

Each descriptor optionally has `when`. Multiple parents are combined with AND:

```yaml
when:
  optimizer: {in: [sgd, accelerated]}
  use_momentum: true
```

`{parent: value}` is the historical equality syntax; `{parent: {eq: value}}` means exactly the same
thing. Membership is explicit: `{parent: {in: [value1, value2]}}`. Membership values must be
non-empty, unique JSON scalars belonging to the parent's finite authored domain. Bare lists,
unknown operators, missing parents, cycles and out-of-domain values fail validation. No expressions,
`or`, `not` or recursive condition language are supported. Historical equality on numeric ranges
remains supported; membership requires a finite domain.

Parents may appear after children in YAML. One shared `ActivationCondition`/`ParameterSpace`
authority validates and topologically orders dependencies for sweeps, random/Sobol/adaptive
generation, encoding and analysis. Inactive keys are absent, never `null`, never multiplied and
then deduplicated. Duplicate authored sweep values are rejected. Domain order remains authored;
membership ordering does not change scientific identity. Conditional coverage uses active observed
values, not a fabricated inactive category.

<a id="guide-conditional-studies-3-validate-and-understand-the-plan"></a>

### 3. Validate and understand the plan

```bash
lf validate comparison.yaml
lf explain comparison.yaml
lf run comparison.yaml --on CLUSTER --dry-run --json
```

Validation/explanation include a bounded `preflight`: candidate count, shared seeds, required Runs,
first independent categorical branch counts, reference, requested GPUs, parallel limit, Study time
budget and scheduler wall-time. They do not print hundreds of individual identities in human
output. `--json` retains the full planning model for automation. Local validation explicitly says
target capacity has not been checked; a read-only target plan may also require an already prepared
runtime/environment. A branch summary is descriptive, not a domain-specific family API.
For adaptive searches these counts describe the current bounded proposal window (subject to the
candidate ceiling), not a promise to execute every proposal. Required replication becomes an
obligation only after a candidate is proposed. Fixed-sweep counts cover the complete authored design.

For selective extra combinations, `sweep.include` adds complete active scalar points without
expanding the base Cartesian domains. `sweep.candidates` instead selects only explicit points
inside space. Both use the same activation authority and reject inactive/missing keys. An exact
existing base point in include is reused, not duplicated. See the examples and incremental
completed-Study workflow in [Study extensions](#guide-study-extensions). Editing YAML alone does not
extend an existing Execution: review and explicitly apply **Extend Study…**, not Resume.

Root `resources` are inherited by steps unless overridden. Root `with`, `seeds`, `replicates`,
`search`, `sweep`, `execution`, `objective` and `analysis` are rejected for a steps composition:
declare them on the relevant Work step. No root experimental policy is silently ignored.
Preflight validates Work inheritance/signatures without constructing a Work or running scientific
code; it cannot detect arbitrary semantic bugs inside a consumer's `run()` implementation.

<a id="guide-conditional-studies-4-time-and-target-capacity"></a>

### 4. Time and target capacity

`resources.time` is the scheduler wall-time ceiling of a schedulable unit. Sequential levels add;
members of one parallel level contribute their maximum. Six sequential steps of 168 h request
1008 h (42 days); three sequential steps of 24 h request 72 h. This is a ceiling, not a prediction
of training duration. `execution.max_time` is the logical Study's dispatch budget and is reported
separately; it does not replace scheduler wall-time.

For a direct host, a successful live GPU UUID inventory together with an unambiguous inherited
visibility restriction can reject a request exceeding known capacity before bundle preparation,
including dry-run. Occupancy is not capacity. A failed/ambiguous probe stays unknown, never an
invented count. Scheduler login hosts are not allocation inventories. A site-command launcher's
grants use the existing runtime check: adaptive Studies may scale down to fewer granted tokens,
never broaden grants or claim additional devices during preflight. Local validation does not
contact any execution target.

<a id="guide-conditional-studies-5-identity-and-existing-evidence"></a>

### 5. Identity and existing evidence

Normalized scalar equality preserves historical serialization and identity. New membership syntax
is additive to the current schema; it requires no new YAML version or compatibility runner.
Predicates remain in Study initialization/design and therefore travel through analysis, export and
retry. Existing persisted equality documents are read, not rewritten. A changed condition, domain,
seed or objective remains a scientific change, not an automatic retry waiver. Fixed sweeps still
execute every required cell/seed without adaptive pruning or scientific replanning.

The generic acceptance fixture in `tests/work/test_conditional_membership.py` has branch counts
1, 2, 10, 16, 9 and 18: exactly 56 candidates × four authored seeds = 224 required Runs. Its unified
conditional design equals the intended union of independently described branches, without
importing or modifying any consumer project.


<a id="guide-study-extensions"></a>

## 24. Incremental fixed Study extensions

[Español](MANUAL.es.md#guide-study-extensions)

Edit the **original YAML**, then choose **Extend Study… → Review changes → Extend Study**.
The dialog lists retained Runs and new cells, target, revision and counts. Confirmation submits
only the delta and opens the cumulative Study series. No second user-authored YAML is needed.

The origin must be an owned, successful **fixed sweep with explicit seeds**, including a
conditional sweep. Grow its candidate set **or** append seeds. For a rectangular design without
explicit points, add values to one existing dimension at a time. Complete each revision before
extending again. Conditional/selective extensions compare exact active candidate dictionaries.
Reordering/inserting values is supported; removing candidates, changing science, or adding values
and seeds simultaneously is rejected. A reorder-only plan is a no-op.

<a id="guide-study-extensions-cli"></a>

### CLI

```bash
# Remote: exact original Job; placement defaults to its cluster.
lf studies extend job-EXACT_ID --config experiments/regularization.yaml --dry-run
lf studies extend job-EXACT_ID --config experiments/regularization.yaml --dry-run --json
lf studies extend job-EXACT_ID --config experiments/regularization.yaml --apply --plan-hash sha256:REVIEWED_PLAN

# Optional: refuse any unproven code/runtime equivalence instead of creating a new cohort.
lf studies extend job-EXACT_ID --config experiments/regularization.yaml --dry-run --code-policy strict

# Direct local evidence also accepts its exact Execution ID.
lf studies extend execution-EXACT_ID --config experiments/regularization.yaml --dry-run

# The receipt supplies a stable Series ID for subsequent operations.
lf studies show series-EXACT_ID --json
lf studies refresh series-EXACT_ID
lf studies history series-EXACT_ID
lf studies analyze series-EXACT_ID --json
lf studies report series-EXACT_ID --output regularization.html
lf studies export series-EXACT_ID --destination ./exports
lf studies extend series-EXACT_ID --config experiments/regularization.yaml --dry-run
```

Omitting `--dry-run` still previews only. Preview never writes revisions, reserves Jobs, consumes
seeds or downloads model/checkpoint/artifact/epoch-history bytes. Explicit typed input verification
may hash input content. Validation imports the actual Work; an agent READ/PLAN session does not
automatically authorize that import. No unrestricted MCP apply operation is exposed.

<a id="guide-study-extensions-regularization-example"></a>

### Regularization example

Use a Python-valid Work argument such as `lambda_` for scientific λ:

```yaml
name: regularization
run: project.Training
seeds: [11, 23, 47, 59, 71]
sweep:
  space:
    lambda_: [0.0, 0.1, 0.2, 0.3]
objective: {metric: score, mode: max}
```

After 20 successful Runs, change the same list to
`[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`: **20 retained + 15 new = 35**.
Then append three seeds: **35 retained + 21 new = 56**. Alternatively, adding three seeds directly
to the original four-value design gives **20 retained + 12 new = 32**.

SubmissionService executes an exact native delta, rectangular or sparse, not another runner/recovery.
LF retains a read-only `.lf-extension-UUID.yaml` snapshot beside the authored YAML to preserve
relative inputs/source context. Do not edit/remove it. Bytes and actual scientific identity are
pinned through asynchronous preparation; recents still show the authored YAML.

<a id="guide-study-extensions-conditional-and-selective-additions"></a>

### Conditional and selective additions

Adding to an existing conditional dimension expands only its active branches. To add a value
only to a particular combination of branch settings, **keep the base space unchanged** and append
complete points to `sweep.include` in the original YAML:

```yaml
sweep:
  reference: {pooling: max}
  space:
    pooling: [max, mean, lse]
    heat: {values: [false, true], when: {pooling: {in: [mean, lse]}}}
    lambda_: {values: [0.0, 0.1], when: {pooling: {in: [mean, lse]}}}
  include:
    - {pooling: lse, heat: true, lambda_: 0.2}
    - {pooling: mean, heat: false, lambda_: 0.3}
```

This adds exactly two candidates, not every combination with λ=0.2/0.3. With four unchanged
seeds, it requests eight new Runs. Use your actual Work argument names. Every included point must
provide **all active dimensions and no inactive ones**. The `max` control therefore has neither
`heat` nor `lambda_`. Inclusion is a union: a point already in the base is not run twice;
duplicate entries inside `include` are rejected. Values are finite JSON scalars, not lists of
values or branch-domain shorthand. Existing include entries must remain in subsequent revisions.

Advanced `sweep.candidates` instead executes **only** the listed complete points within the exact
declared `space`; it does not enumerate that space. It is mutually exclusive with `include`.
Native sparse deltas use this form to execute the exact set difference without adding controls,
cross-products, fake dimensions, missing-key nulls or extra seeds. Space remains the authored
conditional geometry; it does not imply every possible joint point was measured or required.
Removing old candidates or changing activation rules is still rejected. Add seeds in a separate
revision. Historical records and identities are never rewritten.

For a remote sparse delta, a retained parser must support native sparse candidates. By default,
an incompatible/missing retained runtime on a managed cluster selects normal native deployment
of the current project/framework as a new runtime cohort. The old prefix is never patched.
`--code-policy strict` instead rejects missing retained compatibility before submission.

<a id="guide-study-extensions-identities-compatibility-and-evidence"></a>

### Identities, compatibility and evidence

A series, revision, Execution, Job, Run and Attempt have distinct identities. Historical evidence
is referenced, never copied into another Execution or renumbered on disk. Candidate identity uses
exact normalized parameters, not Trial ordinal or name; `true`, `1` and `1.0` stay distinct.
References retain original Trial, Execution/Run/Attempt, seed, input identity and result checksum.
Global CodeIdentity and Run-ID algorithms are unchanged.

The default `--code-policy revisioned` no longer requires a clean original repository. Exact native
code identity or a clean same-HEAD/YAML-only proof permits compatible cumulative inference as before.
Otherwise the reviewed plan explicitly proposes **current code in a new code/runtime revision**.
Dirty origins, changed source, documentation, dependencies or HEAD do not by themselves prevent
adding new cells. The Console shows this warning and changes its confirmation to **Extend · current
code**. Approval of that exact plan authorizes its new revision, never asserts code equivalence.

Each old Run retains its exact code/Execution/Run/Attempt identity. New Runs obtain their own native
scientific identities through the ordinary deployment pipeline; they are not labelled as if the
old code executed them. Review pins current source and preparation inputs, rechecks after the
generated configuration is written and again in the detached worker. An intervening edit requires
a fresh review. Only owned checksum-attested generated YAML snapshots are excluded from that pin.
After native remote preparation, LF validates the **actual staged** Work, input content, exact
candidates/seeds, objective, metric semantics and design before provider dispatch. Its host-derived
scientific identity is recorded separately from the controller submission pin; configuration and
embedded code checksums are checked again inside the native worker. Completion/retry use that
attested Execution identity, not the controller's filesystem path or seed namespace. Metadata
readers use the recorded scientific interpreter, including native extension preflights.
Only canonical SHA-256 commitments for each complete contract field travel in the SSH argv;
the host recomputes them from native configuration. Large metric descriptions do not create
oversized exec requests. No metadata, input identity, seed or design field is truncated/ignored,
and a mismatch still names the affected fields. This does not change native scientific hashes.
Unsafe source paths and missing/corrupt scientific evidence still fail closed. Work sources outside
the project remain unsupported. `--code-policy strict` retains the earlier equivalence-only policy.

Native configuration, input and per-Run checks preserve Work class, fixed parameters, exact
file/dataset/product content, objective/direction, metric semantics, seeds and publication
contracts. Aggregate and authoritative latest Attempts must agree. Failed/pruned/incomplete
required cells must first be repaired with native retry; they are not completed evidence.

Historical Trial ordering is reconstructed from the Execution-owned `authored.yaml` snapshot,
not the edited original YAML. Its exact JSON values must equal frozen `configuration.json`;
native scientific fingerprint and per-Attempt checks still apply. This avoids alphabetically
sorted JSON mapping keys changing conditional/cartesian Trial ordinals. Metadata paging preserves
that original order explicitly. Missing/corrupt/unsafe order evidence is not an identity waiver.
Different code revisions do not become equivalent merely by fixing configuration order.

<a id="guide-study-extensions-supported-execution-contracts"></a>

### Supported execution contracts

- Local CPU: the same existing interpreter and recorded Python/framework/package/platform
  contract, without an environment or command-prefix switch.
- Single-cluster managed CPU/GPU: the same placement and its **retained native immutable
  environment**. The exact owned prefix, environment identity, readiness marker and recorded
  Python/packages/platform are checked on the host. Dependencies are not re-solved. A tiny trusted
  preflight calls the unchanged native CLI/WorkRunner inside the normal scheduler/site launcher.
  Actual granted GPU model/capability/memory type and numerical runtime are checked before
  science. GPU ordinals/counts are not hardware identity. Native grant/admission policies remain
  authoritative; CUDA visibility is never broadened.
- Changed/unverifiable code or incompatible retained runtime: the same **managed** cluster uses
  its ordinary current-project bundle, native runtime resolution, environment, scheduler and GPU
  grants. Preview does not build/install/launch it. The existing environment is not rewritten or
  silently reused with new code. Remote existing/unmanaged environments cannot deploy changed code.

This is not full filesystem attestation of arbitrary installed binaries. Strict retained execution
rejects different frameworks/runtimes/GPU types. Remote unmanaged environments,
local GPU extensions, migration, Fleet, adaptive HPO, automatic sequential replication,
structured dimensions and composed Works remain outside this contract.
They are never silently replayed as a full experiment or single-host substitute.

<a id="guide-study-extensions-cumulative-analysis-and-console"></a>

### Cumulative analysis and Console

The research catalogue groups opt-in revisions under one series. Its workspace shows cumulative
Runs/metrics/plots, revision and native Job links, and history. Linked Jobs retain normal logs,
cancellation and Resume controls; physical Jobs remain in Advanced Jobs and `lf jobs`.
Other Studies and native selectors keep their existing meaning.

`StudyExtensionService.overview()` projects exact metadata; `.analysis()` uses StudyAnalysis,
paired seeds and an independent accumulated evidence fingerprint. It does not overwrite an origin
analysis cache. New candidates selected after seeing results are **exploratory post-hoc evidence**,
not retroactive confirmation. Incomplete seed sets are not final survivor-only means.
Native descriptive coverage/boundary findings do not prove an optimum beyond observed data or
authorize automatic expansion. No second statistics engine is introduced.

If any revision has unproven code/runtime equivalence, the series remains one navigable history
but analysis is **separate per revision**. The default analysis is the latest cohort; JSON retains
`code_cohort_analyses`, and offline HTML adds **Code · revision-N** tabs. Original metric semantics
are retained. Cross-revision tests, intervals, parameter effects and shared-seed means are not
pooled. A candidate completed across two code cohorts has no fabricated combined selection mean;
its Runs still show their metrics and completed state. To evaluate the *whole design* under new
code, submit a new native full Study: extension executes only additional cells. Native HTML byte
limits still apply to embedded cohort reports.

<a id="guide-study-extensions-budgets-failures-and-durability"></a>

### Budgets, failures and durability

Historical Attempts/time across all revisions, including repaired failures, are never refunded.
A full-design cap leaves only its remainder for the delta; impossible obligations are rejected.
Delta products are not automatically promoted when the cumulative winner changes.

The owned checksummed journal `.lambdaforge/study-series/` has immutable revisions and chained
operational events. The reviewed hash binds science, evidence, code, inputs, target and budgets.
Apply revalidates it. Same accepted plan returns the same native Job; concurrent incompatible
revisions are serialized while different series dispatch independently. No-op writes nothing.

A Job ID is committed before handoff. A proven crash before native Job/request/provider creation
can reconcile to **that same ID**. Existing preparation is never blindly relaunched; missing or
corrupt authority fails closed. Refresh seals exact native completion, not scheduler success
alone. The Console refreshes the selected series. Successive preview can read an unsealed
successful Job without writing; apply reconciles that operational completion.

For a failed delta, open its Job and use **Resume Study…**, or `lf retry JOB --dry-run` followed
by native explicit retry. Only the delta's original Execution recovers. Series refresh follows
exact retry lineage, retains cost/history and counts each cell once. Epoch-level checkpoint
continuation still requires explicit consumer Work support; extension is neither retry nor rerun.

If preparation failed **before any new Execution existed**, the same Resume/retry operation
retries only the reviewed delta's preparation. The original series binding and submission pin
remain attached to the new Job; previous successful revisions are not repeated. In this case
`resumable: false` means there is no training Execution/checkpoint to resume, not that the original
full Study must be restarted. Recovery discovery ignores unrelated local project Executions.

Referenced Jobs/Executions and managed environments stay protected from deletion/GC. Series export
is a read-only collection of native Execution packages plus journal and cumulative JSON/HTML;
partial failures are preserved with a receipt. Native packages can be imported individually as
read-only evidence. Executable series adoption/migration is unsupported. Exporting a single
Execution never pretends to include its whole series.

<a id="guide-study-extensions-verification-boundary"></a>

### Verification boundary

Disposable native CPU tests cover 20→35→56, 20→32, unchanged origins, compatibility rejection,
concurrency/idempotence, missing/corrupt authority, pre-Job crash reconciliation, failed-delta
native retry and cumulative analysis. A loopback provider test creates an isolated retained
interpreter without installing dependencies, reads host-side evidence and executes only the
delta through the native entrypoint/process isolation. GPU contracts use synthetic metadata.
These are not claims of live SSH/SLURM/GPU or WISDOM acceptance.


<a id="guide-dataset-reconstruction"></a>

## 25. Published datasets and scientific reconstruction

[Español](MANUAL.es.md#guide-dataset-reconstruction) · English

A published `NAME@VERSION` identifies one immutable content ID. A scientific validation **PASS**
does not mean that a reconstruction has identical bytes, or that all members are scientifically
equivalent. LambdaForge never overwrites this version, relaxes transport checksums, rounds arrays,
or guesses which fields inside an NPZ/other format are operational.

<a id="guide-dataset-reconstruction-choose-the-intent-before-computing"></a>

### Choose the intent before computing

```bash
lf datasets preflight corpus@6 --intent publish --on gpu12 --json
lf datasets preflight corpus@6 --intent reuse --on local --json
lf datasets preflight corpus@6 --intent rebuild --on gpu16 --json
```

Preflight reads the controller and all configured remote indexes, without changing them, downloading
assets or starting a Work. It returns locations, exact content IDs, discovery failures and an
`allowed` decision. A new publication is refused if the version exists; reuse/rebuild requires an
unambiguous reference. Incomplete discovery cannot prove absence. Preflight is not a reservation:
the final publisher still checks identity under the registry lock.

1. **Reuse**: `lf datasets materialize corpus@6 --on TARGET` previews native exact placement;
   review and add `--apply`. `replicate` explicitly selects source/destination. This copies/verifies
   published content; it does not rebuild scientific arrays. Native compressed replication supports
   local/remote placements, preferring direct site SSH; see below.
2. **Rebuild**: compute an independently sealed, unregistered reconstruction and compare it.
3. **Publish**: use a fresh version for changed bytes, even after an equivalence report.
   Additional representations under the same version are not supported; there is no silent
   alternative-content placement policy.

<a id="guide-dataset-reconstruction-publish-once-replicate-exact-content"></a>

### Publish once, replicate exact content

```bash
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --json
# Review the route, size and destination, then apply:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --apply
```

In bare `lf`, open **Datasets → DatasetVersion → Replicate…**, choose endpoints, review the native
preview and confirm. The Console stays responsive; inline phases/elapsed time and compressed byte
counts (on the relay route) show progress. Keep the session open until completion. Locations updates
after registration. The target needs `storage.dataset_root` and a usable LambdaForge runtime
(existing or bootstrapped); no scientific Work or GPU claim runs. Effective roots remain project-scoped.

`--route auto` probes site SSH from source to destination with trusted host-key checking, without
prompts or agent/credential forwarding. If available, bytes travel directly between hosts.
Configure site-authorized noninteractive authentication and trusted host keys on the source host
to enable it; LambdaForge never installs keys or copies controller credentials. `--route direct`
requires that connection. `--route relay` uses existing authenticated controller transports with
a bounded **compressed in-memory stream**, not an archive staged on your computer or in `/tmp`.

Streaming tar/gzip (level 3) verifies the complete source, rejects symbolic/special entries and
unsafe archive paths, uses owned destination staging/storage admission, and checks exact content
before locked atomic promotion/registration. Both indexes expose the same content ID to typed
Dataset inputs. An identical destination is verified/reused; different existing content is refused.
Interrupted copies never register partial content. Deliberate retry reuses an already committed
exact placement or transfers again; it is not byte-offset resume. Legacy registered source paths
remain usable. Replication never rebuilds/relabels/deletes source data or automatically reconciles
conflicting versions; inspect existing divergence or publish a fresh version first.

LambdaForge cannot infer an output name/version from arbitrary Python before calling `Work.run()`.
The producing project must perform the public preflight as its first operation:

```python
def run(self, version: str, intent: str = "publish"):
    self.outputs.dataset_preflight(name="corpus", version=version, intent=intent)
    # Expensive computation only after preflight. Checkpoint results as usual.
    rows = self.compute_members()
    return self.outputs.dataset(
        name="corpus", version=version, members=rows, intent=intent,
        scientific_identity={
            "sources": {"source-id": "sha256:<exact source checksum>"},
            "selection": ["member-id"],
            "labels": {"member-id": 1},
            "configuration": {"modes": 64},
            "algorithm": "my-project/spectral-contract-v1",
        },
    )
```

The Work preflight checks its execution host's registry. Use the native preflight above to also
detect cross-host divergence before submission. `intent="rebuild"` stores the candidate in a pinned
named checkpoint collection and returns its content ID and placement path as an ordinary output
value (`publication_status="reconstructed-unregistered"`), never as a registered DatasetVersion.
`reuse` is a preflight/materialization intent, not a `members` publication option.

<a id="guide-dataset-reconstruction-three-separate-identities"></a>

### Three separate identities

- **Scientific**: explicit project `scientific_identity` with required sources, selection, labels,
  configuration and algorithm contract. The manifest/record exposes `scientific_id`. It is a
  declaration, not proof of reproducibility. Historical manifests have no scientific declaration;
  they remain readable, but non-identical content is unresolved until explicitly declared.
- **Content**: existing `dataset_id == content_id`, exact per-asset checksums, sizes and canonical
  logical index. Nothing about transport or verification is weakened. Identical content cannot
  silently acquire a different scientific declaration in an existing registry version.
- **Operational**: manifest producer/build provenance, paths, Execution/Run/Attempt, hardware and
  timestamps. These are not content identity. Putting them *inside* an asset still changes its bytes.

<a id="guide-dataset-reconstruction-compare-entire-sealed-reconstructions"></a>

### Compare entire sealed reconstructions

```bash
lf datasets compare /reference/root /candidate/root --on gpu16 \
  --verifier my_project.dataset_checks:compare_member --policy comparison-policy.json \
  --output comparison-report.json --json
```

Both roots must be on the selected host. Policy and output-report paths are on the controller;
only policy/JSON reports cross the transport, not assets. A verifier is explicitly trusted project
code imported on that host; LambdaForge does not load code specified by a dataset manifest.

Python API: `from lambdaforge.data import DatasetComparison, DatasetComparisonContext`;
`DatasetComparison.compare(left, right, verifier=callback, verifier_id="project/check-v1", policy=policy)`.
The callback receives one `DatasetComparisonContext` with validated roots, both `DatasetMember`
descriptors and the explicit policy. Return strict JSON with `equivalent: bool`,
`checked_assets: [logical_asset_names]` and optional `details`. Every changed asset must be covered.
Use each descriptor's relative asset path below its root; never infer paths from machine provenance.

Historical reference without a contract: use `--contracts contracts.json` (Python:
`scientific_contracts={exact_content_id: declaration, ...}`). Each declaration is explicitly supplied
by the project and bound to the exact verified content ID; it must reflect independently checked
source/selection/label/algorithm evidence, not an assertion that two datasets are equivalent.
The report preserves the original absent `scientific_id`, records the assertions and separate
`comparison_scientific_id`, and still requires the full member verifier. No historical manifest or
registry is edited. An assertion cannot override a conflicting persisted declaration.

Example policy (illustrative; choose tolerances from the actual scientific contract):

```json
{
  "diffusion_eigenvectors": {"method": "spectral-projector", "atol": 1e-8, "rtol": 0.0},
  "metadata_json": {"method": "exact-scientific-fields-operational-differences-reported"}
}
```

LambdaForge validates policy shape/finite nonnegative tolerances, not numerical formats. Your
verifier must implement every method and report operational differences explicitly. It must compare
embedded identifiers, labels, partitions and source bytes **exactly**, never with tolerances. Sign
flips or rotations of degenerate spectral spaces may require subspace projectors, residuals and
appropriate eigenvalue grouping, not elementwise rounding or a global `allclose`.

Before invoking the callback, LambdaForge verifies both complete datasets with their own exact IDs,
then requires equal scientific contracts, member identifiers, partitions, targets/member metadata,
asset names, target schema and global assets. Changed URI assets must be locally materialized first.
Corruption yields `invalid`; a missing declaration/verifier yields `unresolved`; scientific changes
yield `different`. A complete approved comparison yields `equivalent`, while retaining both distinct
content IDs and `byte_equal=false`. It never registers/replaces content. Byte-identical content is
`exact` unless declared scientific contracts conflict. No single-member comparison proves a dataset.

<a id="guide-dataset-reconstruction-publication-failure-without-recomputation"></a>

### Publication failure without recomputation

`outputs.dataset` preserves a fully sealed rejected candidate and prints its exact path. Failed or
interrupted publication also protects Attempt artifacts, metrics/logs/results and checkpoints from
automatic compaction. If the recovery volume cannot accept a copy, the sealed tree is retained on
its original volume and its path is reported. These retained bytes intentionally consume storage;
inspect them before explicit lifecycle deletion.

```bash
lf datasets publish-candidate /preserved/candidate --on gpu16 --version 7 --json
# Review exact source, content/scientific IDs, target and version, then:
lf datasets publish-candidate /preserved/candidate --on gpu16 --version 7 --apply
```

This verifies saved bytes and atomically publishes/registers only that candidate. It does not run
the Work or repeat calculation. Omit `--version` only when the original registration is absent or
already has the exact content/declaration. Repeated apply is idempotent; corrupt/unsafe candidates
are rejected. The original failed Attempt history remains failed: publication recovery is not a
fabricated successful scientific execution. Candidate, original published version and checkpoints
are not deleted by this command. Ordinary `lf retry` still invokes Work code; it is not this
publication-only operation. Local publication uses configured `storage.dataset_root`, or the local
registry's sibling `datasets/published`; remote publication requires a configured permanent root.

<a id="guide-dataset-reconstruction-failures-before-sealing-0171"></a>

#### Failures before sealing (0.17.1)

Before the first copy, `outputs.dataset` now stores `publication-request.json` and a streaming
`members.jsonl` with exact source checksums in its publication checkpoint collection. If that
volume cannot accept the initial metadata, it uses an owned `.publication-requests` directory on
the publication volume. A failure prints the exact request directory. The same
`lf datasets publish-candidate REQUEST_DIRECTORY [--on CLUSTER] [--apply]` handles this request:
preview checks every source byte and the declaration; apply only copies/seals/registers those
bytes under existing locks. Source mutation, unsafe paths and immutable-version conflicts fail
closed. It does not invoke a Work or rewrite the original failed Attempt as succeeded.

Older failures may have retained validated source files and their member index, but no complete
publication request. Do not invent missing scientific metadata. Explicitly restore the original
member/asset declaration, schema, metadata and provenance, then use the public preparation API:

```python
from lambdaforge.data import DatasetIndex, DatasetPublisher

# original_members must match the producer's declaration, including any extra design assets.
# An existing validated index can provide the basic members, without recalculating geometry.
original_members = (member.to_dict() for member in DatasetIndex(source / "members.jsonl"))
request = DatasetPublisher().prepare_publication(
    name, version, original_members,
    source_root=source, request_root=recovery_directory,
    build_provenance=original_provenance,
    metadata=original_metadata, target_schema=original_target_schema,
    scientific_identity=original_scientific_identity,
)
print(request)  # Use publish-candidate on this directory, preview first.
```

For WISDOM's older preprocessing failure, its retained `attempt-0001/dataset/members.jsonl`
and `dna-validation/dna-validation-report.json` are the starting evidence. Preserve its original
first-member `dataset_design` asset and publication declaration too; do not infer them from NPZ
internals. This repeats only inventory verification/publication, not geometry or annotation.

<a id="guide-dataset-reconstruction-conflicting-local-and-remote-registries"></a>

### Conflicting local and remote registries

`lf datasets list --all` and the Console Datasets screen discover every configured cluster's small
registry. Equal name/version/content merges locations; distinct content stays in separate visible
rows, marked **CONFLICT** in the Console. Listing never changes indexes and incomplete discovery is
visible. Member reads and whole-version deletion never guess an identity in a conflict.
**Datasets → exact identity row → Manage copies…** remains enabled: choose a target and
**Keep as project reference**, **Remove registration · keep files**, or **Delete managed copy**.
Every operation previews the exact hash/root and requires confirmation. REFERENCE marks the
controller reference; other divergent rows remain CONFLICT until explicitly retired.

The same native CLI operations are preview-only unless `--apply` is supplied:

```bash
lf datasets list --on gpu16 --json  # obtain the full exact content ID
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:FULL_ID_16
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:FULL_ID_16 --apply
lf datasets delete corpus@6 --on gpu12 --content-id sha256:FULL_ID_12  # preview
lf datasets delete corpus@6 --on gpu12 --content-id sha256:FULL_ID_12 --apply
# Index-only alternative, including a broken/missing copy or legacy external root:
lf datasets remove corpus@6 --on gpu12 --content-id sha256:FULL_ID_12 --apply
```

Replace the example IDs with complete observed hashes. Adoption verifies all source checksums
on its host before archiving/replacing the controller declaration. It chooses **future resolution**,
not a scientific equivalence certificate: it does not rewrite other registries, relabel bytes,
overwrite immutable publications or change previous Runs, pinned input IDs, checkpoints or provenance.
To retain a divergent representation as published evidence, use a distinct explicit version.

Exact deletion independently selects the target registry identity, even if the controller points
elsewhere. It requires a matching manifest, managed path and no active consumers. Index-only removal
does not delete bytes or free disk space and can retire a corrupt/missing physical copy; the registry
must still be readable and valid. Empty entries are retired so they do not reappear in inventory.
Both operations reject a changed hash/root, lock the target registry, and archive the previous
declaration in `dataset-registry-history/change-*.json` before mutation. Archive entries record
before/requested state, not proof a subsequent write succeeded. Unreachable/corrupt registries cannot
be treated as absent. If remote retirement succeeds but controller cleanup fails, inspect both
inventories again; never retry against an assumed identity. Historical evidence is never deleted.
Ordinary unpinned operations still fail closed; `reconcile` remains identity-preserving.

<a id="guide-dataset-reconstruction-durable-equivalence-certificates"></a>

### Durable equivalence certificates

`DatasetEquivalenceCertificate` reuses the immutable ProductRegistry as a `ScientificReport` with
the reserved `lambdaforge/dataset-equivalence:v1` contract. It runs the existing whole-dataset
comparison before sealing approval; it does not accept an arbitrary approval JSON as proof.
The certificate binds both exact content IDs, the complete scientific declaration, verifier ID,
variable policy and comparison evidence. Operational roots, comparison timestamp and producing
Execution/operation provenance are separate from scientific/content identity.

```python
from lambdaforge.data import DatasetEquivalenceCertificate
from lambdaforge.products import ProductRegistry
from my_project.dataset_checks import verify_member

certificate = DatasetEquivalenceCertificate.build(
    "/published/reference", "/sealed/reconstruction",
    name="corpus-reconstruction-check-v1",
    scientific_contract=science_declaration,  # The full explicit dataset contract.
    verifier=verify_member,
    verifier_id="my_project.verify_member:v1",
    policy=variable_specific_policy,
    producer={
        "execution_id": "<recorded producing Execution or comparison operation>",
        "evidence_fingerprint": "<recorded source/comparison identity>",
    },
)
registry = ProductRegistry()
preview = certificate.publish(registry)  # No locks/files/registry changes.
certificate.publish(registry, apply=True)
restored = DatasetEquivalenceCertificate.load(registry, certificate.product.name)
```

Unresolved, scientifically different or corrupt comparisons cannot be certified. Byte-identical
copies still need an explicit scientific declaration for a scientific certificate. Historical
manifests need the same explicit content-bound `scientific_contracts` assertions as `compare`.
Reports above the catalog's 512 KiB metadata bound fail explicitly; no evidence is truncated.
`lf products show/provenance/export/import` and Console Products inspect/transport these records.
The recorded verifier is never executed on load/import.

Certificates are trusted project assertions, not cryptographic authenticity or transitive equality.
`accepts(reference_content_id=..., candidate_content_id=..., scientific_contract=...)` tests only
that exact recorded pair and contract; it is not a current integrity check. Verify the materialized
candidate separately with `DatasetOperations.verify(root, candidate_content_id)` before use.
Never substitute the reference's content ID for the candidate's or overwrite an immutable version.
Automatic contract-aware Work YAML resolution is **not implemented yet**: ordinary typed dataset
inputs still require exact content. No certificate silently changes that behavior.

<a id="guide-dataset-reconstruction-required-wisdom-changes-not-performed-by-lambdaforge"></a>

### Required WISDOM changes (not performed by LambdaForge)

For `wisdom-dna-reduced@6`, the three reported hashes are distinct exact representations. Matching
45/47 NPZ entries and 27 tiny eigenvector differences for `10FI_Y` do not establish equivalence of
589 proteins. WISDOM should call `dataset_preflight` before geometries/annotations, expose its
publish/rebuild intent, checkpoint expensive results, declare exact sources/design/labels plus
algorithm contract, and move absolute paths/framework runtime provenance out of NPZ scientific
payloads into manifest provenance/sidecars. Relevant algorithm changes still belong in the contract.
Provide a versioned member verifier that checks all variables, protects labels/splits/source bytes,
and uses justified variable-specific tolerances and spectral invariants. Compare **all** members,
save the report, and publish a new version if bytes differ. Do not modify historical registered NPZs
in place, reuse a conflicting `@6`, or infer a universal tolerance from this one protein.


<a id="guide-storage"></a>

## 26. Storage admission and safe cleanup

[Español](MANUAL.es.md#guide-storage) · English

<a id="guide-storage-inspect-and-clear"></a>

### Inspect and clear

```bash
lf clusters storage CLUSTER --json
lf storage status --on CLUSTER --json
lf storage reconcile --on CLUSTER --json         # measure; no metadata mutation
lf storage reconcile --on CLUSTER --apply       # update diagnostic ledger only
lf clean --on CLUSTER              # preview; no deletion
lf clean --on CLUSTER --apply      # recheck and collect safe candidates
```

In the Research Console, open **Clusters → your cluster → Clear storage…**. The operation
previews reclaimable categories and protected references, requires confirmation, runs outside
the UI loop and reports pending/success/failure in the activity panel. **Clear output** only
clears the visible terminal text. Neither operation forgets scientific history. Use native
`lf delete` or `lf jobs clear` for deliberate history deletion.

Explicit storage inspection inventories owned directories. Normal resource probes use filesystem
capacity and small lease records, not recursive scans, checkpoints or dataset assets. Reports
separate category sizes from physical filesystems; apparent bytes are not physical free space.
`reconcile` compares deep measurements with `state/storage-ledger.json`, reports size/file-count
drift and can atomically update that versioned ledger. It never removes files or changes scientific
records. Overlapping categories are not additive. The ledger does not replace physical free-space
checks or existing references; a corrupt ledger is reported rather than silently overwritten.

<a id="guide-storage-admission"></a>

### Admission

```yaml
resources:
  storage: 100GiB
```

This declares a future storage commitment for the outer Job allocation, not a filesystem quota
and not its current measured usage. It does not prevent arbitrary consumer code from writing
more bytes. Cooperating LambdaForge Jobs on the same host use a shared, filesystem-specific
lease authority. Submission never combines free capacity from different volumes.

Admission requires `physical free - other active commitments - new commitment >= safety` and
available inodes when the filesystem reports them. Usage does not shrink the declared commitment;
it is released at supervisor completion or after positively verified owner death. Unreadable,
corrupt or ambiguous ownership blocks new commitments rather than granting fictitious capacity.
PID existence alone is not liveness. Storage checks the recorded birth time: a different birth
proves that the original owner no longer holds that PID; a zombie/dead process cannot use its
commitment. Those stale same-host records are reclaimed only during locked admission. Read-only
status never deletes them. A matching living lifetime retains its commitment across argv changes
(exec), while signal authorization still requires the stricter birth/command match. Inaccessible
and foreign-host identities remain protected. An existing owned allocation is not reserved again
by its workers or Fleet offer.

Ownership refusals raise `StorageOwnershipError`, not ENOSPC. Diagnostics name the lease, host
and owner; `lf storage status --json` exposes bounded ownership issues. Inspect the recorded host,
restore process-inspection access or wait for verified owner exit, then retry the publication.
Do not delete live/foreign leases or reset scientific history. Even a zero-byte reflink checks
ownership, physical safety and inodes; nested transactions have independent exact release identities.

When admission cannot proceed, direct Jobs wait before acquiring CPU/GPU leases. The state
includes requested/free/reserved/safety bytes and the reason. The same authority is available
inside scheduler-owned native workers. Safe automatic GC is attempted before continued waiting;
healthy science is not killed to reclaim disk. No Run is restarted by cleanup.

Configure the shared authored cluster profile (example):

```yaml
storage:
  state_root: /durable/lambdaforge/state
  cache_root: /scratch/lambdaforge/cache
  run_root: /scratch/lambdaforge/jobs
  dataset_root: /durable/lambdaforge/datasets
  lease_root: /durable/lambdaforge/host-leases
  cache_max_size: 500GiB
  cache_max_age: 30d
  safety:
    min_free: 20GiB
    min_free_percent: 5
  terminal_jobs:
    grace_period: 2d
```

The larger absolute/percentage floor wins. Default free-space safety is 5%; default successful
checkpoint grace is two days. Project scoping changes operational roots, not the host-wide lease
root. The console profile editor exposes these fields as well as native `clusters set/unset`.

Pressure is reported as NORMAL, SOFT_PRESSURE, HARD_PRESSURE or CRITICAL using physical free
space, safety, leases and inode exhaustion. Reservations are cooperative, not protection against
another user's writes or an external disk quota.

For SLURM, no portable scratch directive is assumed. Positive storage requests produce an
omission warning unless `resource_mapping.storage` is configured for the site, for example
`{option: tmp, value: "{storage_mib}"}`. Worker-side root admission does not imply SLURM has
reserved that space. Fleet placement requires fresh known storage fit for a positive requirement;
unknown capacity is not permission to dispatch.

<a id="guide-storage-what-collection-can-remove"></a>

### What collection can remove

Collection respects active Job, exact environment, runtime, bundle and Work-cache references.
Environment builds protect their own prefix and mutating package categories, not unrelated
Work caches. Unknown ownership stays protected. Only positively identified dead, stale same-host
build owners permit orphan temporary cleanup; foreign-host/legacy markers are retained.
A dead local controller alone is insufficient if pip/Conda still references the exact temporary
prefix. Unreadable process evidence also keeps it protected. Bootstrap uses an existing host
interpreter for ownership metadata; that interpreter need not satisfy the scientific Work version.

Automatic cache collection observes authored quota/age, pressure and proven orphan status; without
an authored quota its ceiling is 10% of the cache filesystem capacity. Reconstructible lower-cost
categories are selected before expensive environments, then by last use. Protected entries can
prevent reaching a quota; the plan reports unresolved excess instead of deleting them.
`lf clean` additionally includes inactive reconstructible Work caches. Categories include pip,
Conda/native/runtime packages, managers, environments, runtimes, bundles and temporary content.

Cache deletion records intent, renames an exact entry into owned trash, and resumes interrupted
trash deletion idempotently. It never replays an uncommitted intent against newly referenced
original content. `state/storage-gc.jsonl` records applied operational collection. Results, logs,
metrics, provenance, published datasets, researcher-owned mirrors, successful unpublished
artifacts and active/unknown Jobs are not cache candidates.

<a id="guide-storage-checkpoints-and-publication"></a>

### Checkpoints and publication

Failed/interrupted Executions retain recovery checkpoints. Terminal artifact compaction continues
to discard partial failed artifacts and verified redundant published output bytes. A **successful**
Execution may release unpinned checkpoints after its grace period. Keep durable scientific models
as registered outputs; explicitly pin resumable state that must outlive ordinary retention:

```python
self.checkpoints.pin("best/model.ckpt", reason="retain-for-follow-up")
self.checkpoints.unpin("best/model.ckpt")  # metadata only; does not immediately delete
model = self.outputs.from_checkpoint("model", "best/model.ckpt", release=True)
```

`from_checkpoint` seals an independent registered artifact and checks source/snapshot identity.
Dataset publication can read an owned checkpoint directory directly, avoiding an intermediate
Attempt copy:

```python
self.outputs.dataset(
    name="example", version="1", members=members,
    source_checkpoint="prepared-records",
    release_checkpoints=["prepared-records"],
)
```

Member asset paths are relative to that named checkpoint directory. Explicit release becomes
eligible only after successful Execution finalization and a verified independent artifact or
dataset publication; changed/missing destination bytes invalidate the proof. Copies use
copy-on-write where supported, otherwise independent regular copies—never writable hard links.
This is not a promise of zero-copy on every filesystem.
Managed snapshot copies reserve their additional bytes on the **destination** filesystem, including
external `publish_to` and dataset assets. An insufficient volume triggers one safe, same-volume cache
collection attempt before a compact space diagnostic; GC never scans or deletes that external
publication directory. Uncommitted copies retain their original checkpoints. These incremental
copy commitments are released on success/error, do not silently lower the Job's declared commitment,
and do not multiply physical payload bytes for a successful reflink. A clone checks safety/inodes;
its full-copy fallback must first reserve the payload bytes.

Cleanup takes the Execution writer lock, rechecks pins/identity and journals checkpoint deletion;
it cannot race an active Study recovery writer. `checkpoint-retention.json` records removals while
Attempt results, logs, metrics and history remain. Corrupt/missing retention evidence fails closed.

<a id="guide-storage-current-boundaries"></a>

### Current boundaries

Run-root admission and managed publication-copy reservations are implemented. Bundle staging and
dependency provisioning do not yet have separate byte reservations on their destination volumes.
Dataset index/manifest generation is not a disk quota. Managed environments
still use identity v2 with exact wheel hashes: dependency environments and independent code layers
have **not** yet been separated. Build markers are conservatively protected, but cross-host orphan
proof remains conservative. Both Python-runtime and managed-environment build markers now use
the same atomic GC handshake and bounded heartbeat, including reuse verification. Invalid complete
environment prefixes are preserved rather than replaced underneath referenced Jobs. Explicit
`lf storage reconcile` provides a diagnostic measurement ledger, not a complete per-write allocation
ledger. Do not treat cleanup as protection against every ENOSPC during preparation.


<a id="guide-managed-reports"></a>

## 27. Full-resolution managed project report tabs

[Español](MANUAL.es.md#guide-managed-reports) · [Work results](#guide-work-results)

`html_section` remains the self-contained HTML API (16 MiB/document, 64 MiB/report).
For larger retained visualizations, use `self.outputs.html_bundle`. This is one ordinary
managed directory artifact, not a new scientific store or runner. LambdaForge never selects
proteins, seeds, checkpoints, winners, points or feature channels for a project.

```python
bundle = self.outputs.html_bundle("predictions", section="proteins", title="Proteins")
bundle.entry.write_text(inspector_html)
bundle.asset("catalogue", filename="catalogue.json", media_type="application/json").write_json(rows)
bundle.asset("surface", filename="surface.f32", metadata={
    "dtype": "float32", "byte_order": "little", "shape": [point_count, 3],
}).build(write_full_surface)
```

The project declares every asset; serializers receive the managed output path. Assets are
opaque bytes, not JSON arrays hidden in HTML. Float32, NaN/Inf and explicit validity masks can
therefore remain exact. Give each file a unique stable ID and portable relative filename.
Logical IDs are ASCII; filenames are NFC Unicode and must not collide ignoring case. Absolute
paths, traversal, symlinks, special files, duplicate IDs and undeclared files are rejected.

Finalization seals `lambdaforge-report.json`: version, entry HTML, asset IDs, relative filenames,
byte sizes, native managed checksums, raw-content SHA-256, media types and project metadata.
The manifest itself is attested in the original Run artifact. The public schema is packaged as
[`report-bundle-v1.json`](../src/lambdaforge/schemas/report-bundle-v1.json).
Entry HTML remains bounded to 16 MiB and the manifest to 8 MiB; these are not asset-data limits.
The full directory is retained/exported with the ordinary artifact integrity contract.

<a id="guide-managed-reports-browser-api-and-lifecycle"></a>

### Browser API and lifecycle

Inside the project entry HTML:

```javascript
await lambdaforgeReport.ready;
const controller = new AbortController();
const bytes = await lambdaforgeReport.load('surface', {signal: controller.signal});
const positions = new Float32Array(bytes); // Consumer knows its own format/shape.
// On selection change: controller.abort(); dispose renderer/GPU buffers/references.
// On permanent shutdown: lambdaforgeReport.dispose();
```

`lambdaforgeReport.assets` exposes declared IDs, sizes, media types and metadata. The public
version is `1`. `load` resolves to an ArrayBuffer or rejects with a visible error/code;
aborts use `AbortError`. Two concurrent loads are allowed, not an unbounded queue. Parent
transport transfers ownership of buffers instead of keeping a whole-array cache. Switching
tabs/documents closes the scoped port, aborts requests and unloads the iframe. The consumer
must also dispose its own renderer and avoid accumulating old arrays.

The iframe retains its opaque origin and `connect-src 'none'`: no external URLs, parent DOM,
filesystem paths or provider credentials. A MessagePort goes only to the exact selected iframe.
The trusted parent requests only manifest IDs, verifies size/raw checksum and transfers bytes.
The server rechecks the original Run artifact and attested manifest before opening that one file.
Missing/corrupt assets are presentation errors, never invented model failure/success. A bundle
sealing error is saved as a visible named `presentation_error`; other scientific outputs survive.
An exception raised by consumer code *before* finalization remains a normal consumer exception.

<a id="guide-managed-reports-native-remote-and-offline-viewing"></a>

### Native remote and offline viewing

The Console's **Open HTML report** starts LF's read-only loopback viewer automatically for
managed bundles. Lists/HPO scheduling/heartbeat never download asset bytes. Opening the report
reads entry HTML and manifest; choosing an asset retrieves only that asset through native
provider transport, gzip-compressed and verified. No application HTTP server, CUDA launch or
training process is started. Remote reads have cancellation/deadlines and one disposable spool
per selected asset beside the generated report, not on the system `/tmp` volume.

For a generated or exported report:

```bash
lf reports serve ./reports/study-analysis.html --open
# Explicitly raise the active-view safety bound if a *single* asset exceeds 256 MiB:
lf reports serve ./reports/work-results.html --max-asset-mib 1024 --open
```

This serves only declared evidence on `127.0.0.1` with a private capability URL, exact Host and
same-origin checks. It is not a general file server. Defaults: two active binary reads, eight
connection threads, 256 MiB per selected asset and 30-minute idle shutdown. Ctrl-C closes only
the reader, never the scientific job. No points/proteins are silently truncated to meet limits.

`file://` cannot lazily fetch sibling files safely: opening it gives the native-viewer instruction.
Export/import retains the complete bundle, generates the native report and deployment sidecar,
and relocates references to package-owned bytes. Keep the entire package, not only the HTML;
moving it to another computer and running the command above needs no original cluster or model.
The sidecar is disposable presentation deployment metadata, not authoritative scientific state.

<a id="guide-managed-reports-rebuilding-presentation-from-retained-inference-only"></a>

### Rebuilding presentation from retained inference only

Use the existing public ResultInput/Work lifecycle, not hidden paths or a second evaluator.
Select an exact historical Run/Attempt via `lf results reference`, bind it as a typed `result`
input, and read an explicitly retained artifact with `previous.artifact(NAME)`. Rebuild the
bundle from those saved predictions; do not load a model or perform forward passes. See
[`managed_report.py`](../examples/managed_report.py). Missing saved predictions must fail clearly:
LF does not regenerate lost inference, substitute another seed or choose a checkpoint.

Large payload browser validation is covered with a 70 MiB selected asset and a catalogue of
206 consumer records. Scientific provenance and selection reasons remain project-authored;
native tabs do not imply those choices were made or endorsed by HPO.


<a id="guide-research-analysis"></a>

## 28. Study research workspace

[Español](MANUAL.es.md#guide-research-analysis) · [Manual](MANUAL.md#16-study-analysis)

Current application release: **0.19.0**. Analysis v8 and saved-view v1 remain compatible; the
implementation reports below retain their historical release context.

<a id="guide-research-analysis-1-open-a-workspace"></a>

### 1. Open a workspace

```bash
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output research.html
```

Install `lambdaforge[analysis-report]` for HTML; numerical analysis needs no Plotly. Remote Studies
can be exported through `lf export STUDY --output ./exports` or the console's Export action. The
report is an offline snapshot, not a live connection to the cluster. Regenerate to include new Runs.

**Overview** is the first tab and the default on a newly generated report. It shows candidate
states with exact counts, selection/confirmation context, unconditional predictive associations
and suggested views that open the exact chart. The optional score-by-trial plot is collapsed in
**Trials**, not the main overview. Reopening a file restores your last tab.

Conditional parameters are deliberately excluded from the global importance chart. Their table
shows the activation rule, observed active/total candidates, completed support, persisted model
score and reliability. That score may mix parent-branch activation with within-branch variation;
it is not a causal percentage, an additive share of responsibility or a reason to choose the branch.
Interpretation is a horizontal, keyboard/mouse-scrollable carousel. The **ⓘ** help contains shared
stability/seed-noise explanations; individual details preserve the exact original conclusion.

Use **Language / Idioma** in the header for English or Spanish. Interface labels, contextual help,
status summaries and supported chart labels switch immediately and persist for that generated
file. Authored names and saved scientific statements remain in their original language; no remote
translation service changes your evidence.

**Research** retains actionable health cards. Configured questions and the separate exploratory
inbox use horizontal carousels; the complete exploratory list is paged six findings at a time.
**Inspect** explains evidence, method, support, ranking and limits;
**Explore these observations** opens the relevant recorded values. An exploratory association is
not a new HPO conclusion or a causal effect.

**Metrics & health** searches labels, stable names, descriptions, aliases and tags. Filter by
category and sort by priority, coverage or spread. Constants, missing and hidden metrics are
initially omitted, not deleted: enable **Show constants / missing / hidden**. The metric inspector
keeps units, direction, aggregation and unavailable information explicit. Redundancy groups are
descriptive; each recorded member remains accessible. **Ctrl/⌘ K** searches metrics, parameters,
families, findings, Trials and saved views. Ordinary metric, parameter, axis, type, palette and
filter controls open an anchored searchable dropdown, not a modal. Global **Ctrl/⌘ K** search
remains a deliberate dialog with favourites and recents. **Categories** is a collapsible tree: selecting `validation` includes every descendant,
such as `validation/global` and `validation/surface/quality`.

<a id="guide-research-analysis-2-declare-meaning-not-execution-policy"></a>

### 2. Declare meaning, not execution policy

Everything works without declarations. Standard directions/ranges and simple split prefixes are
known; other semantics remain unknown. Add an optional `analysis` section to one Work YAML, or a
class-level `Work.analysis_profile` mapping with the same shape:

```yaml
analysis:
  defaults:
    - pattern: val_*
      metadata: {category: validation, split: validation, unit: ratio}
  metrics:
    val_accuracy:
      label: Accuracy
      description: Fraction of correctly classified validation examples.
      category: validation/quality
      direction: max
      range: [0, 1]
      aggregation: latest
      visibility: primary
      aliases: [accuracy]
      priority: 10
    rejected_fraction:
      category: integrity
      expected_to_vary: true
      unit: ratio
      range: [0, 1]
    transformed_accuracy:
      derived_from: [val_accuracy]
      transformation: Explicit project-defined transformation.
```

Labels never rename logged keys. Units are display semantics, not automatic conversions.
`aggregation` documents the actual evidence: `latest`, `best`, `selected_epoch`, `terminal`, `mean`
or `unspecified`. Declaring `best` does **not** calculate a missing per-metric optimum. Selection
retains its existing checkpoint/seed semantics; diagnostic summaries retain their actual values.
Unknown aggregation is warned about rather than guessed.

Optional fields also include `tags`, `role`, `phase`, `scale`, `notes`, `discovery: false`,
`practical_scale` and `visibility: normal|advanced|hidden`. A practical scale describes metric
health; it is not the objective's scientific equivalence margin. Precedence is inferred defaults,
matching pattern rules in order, class declarations, then YAML metric overrides. YAML questions
and defaults replace the corresponding class collections. Families merge by family name, with
the YAML definition replacing a same-named family; individual metric fields merge.
For a `steps` composition, put `analysis` on each Work step or class, not the composition root.
Custom `Metric` subclasses may pass `metadata=` to the base constructor and reuse their public
`.metadata` mapping in the class profile. Transient metrics are not discovered by constructing a Work.

Declarations are validated locally: unknown fields, invalid ranges/categories/directions,
ambiguous aliases/family coordinates, missing lineage parents and lineage cycles fail explicitly.
Test-split metrics and metrics derived from test evidence cannot be objective components or hard
constraints, including composite utility. Test evidence remains available for terminal post-hoc
inspection, but is excluded from provisional automatic discovery. Analysis never adds an inferred
guardrail, objective weight or pruning rule.

<a id="guide-research-analysis-3-families-and-questions"></a>

### 3. Families and questions

Group repeated measurements semantically rather than listing hundreds of unrelated names:

```yaml
analysis:
  families:
    efficiency:
      label: Quality by data fraction and strategy
      dimensions:
        strategy: {kind: categorical, values: [uniform, balanced]}
        fraction: {kind: ordered, values: [10, 25, 100]}
      template: quality_{strategy}_{fraction}
  questions:
    - {id: efficiency-view, kind: metric_family, family: efficiency}
    - id: capacity-quality
      kind: parameter_screen
      parameters: [width]
      metrics: [val_accuracy]
    - id: expected-agreement
      label: Agreement with transformed accuracy
      priority: 10
      kind: relationship
      x: val_accuracy
      y: transformed_accuracy
      expected: positive
```

Dimensions can be numeric, ordered or categorical. `members` may explicitly map each stable metric
name to its dimension coordinates instead of a template. Templates resolve at most 4096 unique
members and use bare `{dimension}` placeholders, not Python expressions or formatting commands.
Family plots retain dimension labels/order, missing values and observed support. Their
SD is dispersion across candidates, not seed confidence; incomplete family support is not a paired
intervention. Family members are never automatically collapsed as redundant.

Question kinds are `relationship`, `consistency`, `parameter_screen`, `metric_family`,
`category_summary` and `tradeoff`. Relationships/consistency require metric `x` and `y` and may
declare `expected: positive|negative|equal`; tradeoffs list at least two metrics; category summaries
name a category. Unknown required metric/family references fail validation. `optional: true`
allows a metric reference absent from this dataset and records an unavailable question, not a
fabricated result. Optional `label` and non-negative `priority` name and order the question cards;
they do not weight HPO. Questions focus exploration; they do not schedule Runs or formalize a hypothesis
test protocol. Category/family summaries and existing resource Pareto views remain descriptive.

<a id="guide-research-analysis-4-how-discovery-works"></a>

### 4. How discovery works

1. Build a candidate-level matrix from **comparable completed evidence**. Censored candidates do
   not receive exact final scores. Keep screening separate from fresh-seed confirmation and avoid
   mixing fidelity rungs. Recorded aliases map to canonical analysis names.
2. Profile every metric: finite/missing support, unique values, range, mean, median, SD, robust
   spread and logged Run/seed support. Exact constants require repeated equal observations.
   Metrics observed in fewer than half the candidates carry a `low_coverage` display warning;
   candidate/Run coverage and seed support remain numeric. Association ranking also penalizes
   the actual paired coverage rather than treating a few observations as complete evidence.
   Near-constant display filtering uses spread relative to declared range/practical scale or
   observed magnitude (`1e-6`), never an absolute epsilon that erases small-unit measurements.
3. Inspect a deterministic bounded metric set, prioritizing declared priority. Group near-identical
   rank responses (absolute Spearman at least `.995`, at least six paired candidates), except
   families. This is a navigation suggestion, not proof that two metrics are interchangeable.
4. Screen bounded parameter/metric and metric/metric relationships, interleaving both types and
   rotating parameters so the first parameter cannot monopolize the budget. Numeric parameters use the
   existing authored `ParameterSpace`, including log coordinates and categorical/conditional
   semantics. Numeric responses compare rank correlation with a leave-one-out quadratic rank
   response; categorical responses report eta-squared. Neither is a multivariate causal model.
5. Confirm at most 32 shortlisted relationships by deterministic candidate permutations and
   bootstrap sign stability. Reuse small design matrices; never build a candidate-squared distance
   matrix or enumerate the full metric Cartesian product. Unconfirmed screened tests count as
   `p=1` in Benjamini–Yekutieli adjustment. The default screen is 64 metrics/128 relationships/
   256 permutations; advanced `analysis.discovery` can adjust `max_metrics`, `max_pairs` and
   `resamples` within validated bounds, or disable discovery. All effective limits are persisted.
6. Rank findings by separate effect/support/stability components. Down-rank relationships explained
   by direct/transitive/shared metric lineage. Also retain invariance warnings, descriptive
   1.5-IQR outliers, bounded categorical-context sign reversals and existing Study findings.
   Persist ranking components and redundancy penalties. The inbox shows at most eight distinct
   evidence groups; equivalent findings remain accessible in the complete list.

Effects, support, stability, adjusted p-values and scientific confidence are different quantities.
**Supported exploration** requires enough observations and a small adjusted diagnostic p-value;
**preliminary** is not confirmed. Adaptive selection, multiple comparisons, reused data and seed
noise limit these diagnostics: BY adjustment does not make an adaptively collected Study a formal
randomized experiment. Bootstrap sign agreement is not the probability that a conclusion is true.
Outlier/context signals are explicitly inspection-only and have no fabricated confidence.
The methodology follows [SciPy's permutation guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
and [dependent-test FDR adjustment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

<a id="guide-research-analysis-5-explore-and-save"></a>

### 5. Explore and save

In **Explore**, choose **Analyze** (metric), **By** (parameter chips), optional **Compare with**
(metric chips), and the completed/partial subset. **Explore observations** uses the existing
renderer: one parameter means scatter, two mean a heatmap, and more mean parallel coordinates.
Remove any chip with ×. Ordinary metric and advanced-control changes still preview immediately.
**Advanced visualization options** holds explicit type/X/Y/Z, grouping, palette/reverse and visual
normalization. Type/palette/filter menus share the searchable dropdown. Naming, notes and saving stay visible.

- Multiple Y metrics share one scale only when declared units agree. Different/unknown units use
  small multiples; **Visual 0–1 normalization only** is opt-in and never changes scientific values.
- In **Parameters**, open **Analyze metric**, search e.g. `mean`, check a metric, then search `auroc`
  and check another without closing the dropdown. All selections stay checked across searches.
  Uncheck a metric to remove its curve; zero selections shows an explicit empty state. Arrow keys
  navigate and Space toggles a checkbox; Escape/Done/outside click closes the dropdown.
  Choose **Chart style**: lines/points, grouped bars, distributions across Trials with the same X,
  or a heatmap for many metrics. Pie charts are deliberately absent: unrelated metrics are not
  additive parts of a whole. **Arrangement** provides one combined chart or separate metric panels.
  Further native Plotly views include individual Trial points (no averaging), step and area curves,
  violin distributions with recorded points, horizontal bars, histograms and empirical CDFs.
  Histogram/CDF views pool all matching candidate summaries rather than grouping parameter values;
  use boxes/violins for that comparison. Connecting points is descriptive, not a fitted response.
  Every automatic axis is titled with its member metric names, not an anonymous scale number.
  **Y scales → Automatic** groups ranges with at least 50% overlap of the narrower interval
  (including containment); all metrics in a group must agree pairwise. Constants may share a range
  that contains them. Known different units never group automatically. Unknown units can share
  a visual scale, not a scientific meaning. Override with independent or explicitly shared scales;
  separate panels retain the chosen range grouping. Do not compare heights across independent axes.
  **Dispersion** selects hidden SD, whiskers or a shaded ±SD band on numeric lines. Other styles
  fall back to whiskers; distributions already expose the observations. SD is empirical variation
  across candidate summaries, not seed confidence, and remains absent for a single observation.
  Heatmap colours use per-metric visual min–max normalization, including a neutral colour for
  constant rows; hover shows original means and missing values stay blank. Settings persist in
  this generated HTML's browser storage; regenerating the report resets them. Explore keeps
  its unit-aware small-multiple default and optional visual normalization. Selection changes do not
  change scientific aggregation, objective policy or the persisted model response.
- Observed pair heatmaps and numeric 3D surfaces retain empty untested cells; persisted surrogate
  surfaces stay in their separately labelled model panel. 3D scatter supports categories.
- Parallel coordinates use the chosen parameters (or the existing eight-parameter default for old
  views) and one metric; missing coordinates are excluded explicitly. They do not fit new effects.
- Trials retain partial/pruned markers. Search the ledger and compare two candidates in a category-
  ordered metric/difference table. A conditional-root branch filter is available when applicable.
  The comparison starts with primary/priority metrics; search or **Show all recorded metrics**
  expands it. Unknown direction is never labelled improvement.

**Interactions** has a metric-agnostic observed panel with shared parameter/metric pickers and
heatmap/3D/surface choices. The existing predictive objective panel stays separate. **Evidence**
(formerly Findings & evidence) retains scientific conclusions, surrogate diagnostics, seed/pruning
evidence, methodology and the structured snapshot in expandable sections. Exploratory finding cards appear only in Research;
the Run metric dashboard is unchanged and continues to use epochs.
The complete audit snapshot is formatted on expansion and reuses the single serialized metric
catalog. All panels share the same parsed catalog rather than constructing independent selectors.

Save a named view with optional notes. Export/import **research-views.json** to reuse chart
specifications, not evidence. Imports validate version, fields, metric/parameter references and
size/count limits. Same-file browser preferences retain views, palettes, panel sizes, favourites
and searches; regenerating creates a new preference scope. No cloud or LLM service is required.

<a id="guide-research-analysis-project-html-tabs"></a>

#### Project HTML tabs

Domain visualizations belong to the consumer. Inside `Work.run`, declare a finalized managed
HTML output and write your application's complete document:

```python
viewer = self.outputs.html_section("predictions", section="proteins", title="Proteins")
viewer.write_text(render_predictions_html())
```

The Study's interactive HTML action, `lf results report SELECTOR --output report.html` and portable
export collect these files only when explicitly requested. A named section creates one navigation
tab next to Explore/Evidence/Resources. Multiple outputs or Runs in that section get a searchable
document selector labelled by Trial/seed/output. Shared section titles must agree. No HTML content
is fetched by overview, HPO, Analysis panels or ordinary polling; remote report reads are paged and
cached on their owning host, outside the scheduling heartbeat.

Documents are self-contained UTF-8 HTML: embed JavaScript/CSS, visualizations and data (including
prediction data) in the file; relative asset paths and external requests do not work. LambdaForge
does not generate domain logic, load NPZ files or expose Python callbacks to the browser. Each
document uses an opaque-origin `sandbox="allow-scripts allow-downloads"` iframe and a restrictive
CSP; project scripts cannot access the parent dashboard, local files, other sites or credentials.
This is presentation isolation, not a sandbox for the trusted Work's Python code. The HTML remains
interactive inside the frame. Parent preferences/navigation stay independent of project code.
Managed artifact ownership, exact size and the persisted content fingerprint are verified before
embedding. Symlinks, missing/changed content and documents exceeding 16 MiB (64 MiB total) fail
explicitly rather than silently producing empty tabs. Declaration uses the same ordinary output
lifecycle/retention; no second registry, runner or report plugin system exists.

Standalone integrations can pass documents directly without executing a Work:

```python
from lambdaforge.analysis.Report import write_html

write_html(analysis, "report.html", sections=[
    {"name": "proteins", "title": "Proteins", "label": "Trial 3 · seed 7", "html": html_text},
])
```

<a id="guide-research-analysis-6-architecture-safety-and-limitations"></a>

### 6. Architecture, safety and limitations

`MetricCatalog` owns meanings; `AnalysisProfile` validates immutable declarations;
`ResearchAnalysis`/`ResearchDiagnostics` produce deterministic read models; modular offline assets
render them. `StudyAnalysis` still owns effects, coverage, seed uncertainty, winner/confirmation and
resource evidence. Analysis document version **8** adds `research` without removing existing fields.
The obsolete, unused linear Study renderer was removed rather than maintained as a second path.

Before execution, `analysis-semantics.json` records resolved declarations, known registry defaults,
identity and versions. Child specifications reference this owned file instead of repeating its
catalog per Run. Final analysis, reload, recovery and export use frozen evidence; changing labels
does not alter the existing scientific evidence fingerprint or HPO decisions. The analysis cache
also checks the separate semantics identity. Legacy Studies without declarations receive inferred
metadata; old HTML-input documents gain a name browser but no invented retrospective findings.

Limitations are intentional: discovery is bounded rather than exhaustive; unknown semantics stay
unknown; nonlinear/context signals are coarse inspection tools, not a universal interaction model;
no automatic causal claims or recommendation to modify objective weights; no new shared-seed
confidence calculation; no generated data or imputation; no browser-side scientific fitting. The
existing multivariate HPO/Study Analysis remains the authority. Import/export covers presentation
views, not a portable live workspace or automatic HTML refresh. Exact coupled resource/performance
claims still require adequate comparable resource and seed evidence.

<a id="guide-research-analysis-7-implementation-report"></a>

### 7. Implementation report

This report records the implementation and local verification on 2026-10-03.

1. **Previous architecture:** Study Analysis already owned normalization, winner/seed evidence,
   coverage, multivariate effects and resources. HTML was chiefly a collection of independent plots.
2. **Usability causes:** flat names/selectors, weak metric semantics, unavailable/constant values
   competing for attention, single-metric navigation and missing research-level prioritization.
3. **New structures:** `MetricCatalog`, immutable `AnalysisProfile`, bounded `ResearchAnalysis`,
   inspection/ranking helpers and modular offline research presentation, integrated into that analysis.
4. **Catalog declarations:** optional `analysis.metrics/defaults`, or `Work.analysis_profile`;
   custom `Metric.metadata` is reusable without constructing scientific Work objects.
5. **Profiles:** metric/family declarations plus validated questions/discovery rules, merged from
   class and YAML and frozen independently of objective/execution policy.
6. **Families:** explicit coordinates or bounded bare-placeholder templates; ordered/categorical
   dimensions and incomplete member support remain visible in one grouped chart.
7. **Metric health:** repeated equality, relative near-constancy, missingness, candidate/Run/seed
   coverage, out-of-range and expected-variation warnings; no deletion or imputation.
8. **Redundancy:** deterministic rank-similarity groups, expandable members and representative;
   family members are protected. Ranking/inbox suppress repetitive evidence, not raw values.
9. **Discovery:** interleaved parameter/metric and metric/metric screening, numeric rank/quadratic
   response, categorical association, configured relationships and cheap inspection signals.
10. **Multiplicity:** bounded deterministic permutations plus BY adjustment across the screened
    hypotheses, including unconfirmed tests as `p=1`; limited resolution remains conservative.
11. **Reliability:** support, effect, coverage, sign stability, adjusted diagnostic p-value and
    novelty are separate recorded components. None is a probability that a finding is true.
12. **Derived metrics:** validated acyclic lineage; direct/transitive/shared structural relationships
    lose novelty. Direction and practical equivalence are not inferred from a derived formula.
13. **Test evidence:** objective/components/constraints reject test-derived evidence locally;
    provisional discovery excludes it, while terminal inspection retains recorded test values.
14. **Persistence:** `analysis-semantics.json`, separate semantic identity, reference-only child
    specifications and frozen defaults across finalization, reload, recovery and export.
15. **HTML:** Research inbox, health/browser, semantic inspectors, diverse findings and progressive
    Explore controls; removed the unused old linear Study renderer.
16. **Search:** shared word/subsequence search over names, labels, aliases, descriptions/categories,
    parameters, families, findings, Trials and saved views; keyboard, favourites and recents.
17. **Multi-metric Parameters:** shared observed renderer; add comparison metrics, separate unknown/
    different-unit scales, optionally normalize only the visual presentation.
18. **Pair/3D:** observed scatter/heatmap/surface with categorical labels and untested gaps;
    existing modelled objective surfaces remain separately labelled. Parallel coordinates are bounded.
19. **Explore:** immediate preview, X/Y/Z, multiple Y, plot type, grouping, palette, partial evidence
    and explicit normalization; advanced controls are not the opening experience.
20. **Research Views:** named saved chart specifications, notes and same-file preferences;
    bounded validated JSON import/export transports presentation, not scientific data.
21. **Added regressions:** metadata precedence/validation, leakage/lineage, immutable reload,
    namespaces, missing/censored evidence, metric health, families, deterministic/nonlinear discovery,
    large catalogs, browser search/drill-down, units, comparisons and view persistence/import.
22. **Verification:** Ruff and mypy passed; full pytest: **1118 passed**, four non-fatal Lightning
    CPU-loader warnings. Focused analysis/CUDA/documentation tests, installed-wheel CLI/scaffold/
    validation/HTML smoke and Chromium interaction/screenshots passed. A synthetic 500-candidate,
    300-metric, 15-parameter, 15-seed matrix took **9.424 s**, produced **4,809,115 bytes** and used
    only 32 expensive relationship resamples with default policy. Timing is machine-specific.
23. **Versions/schema at implementation:** application release was **0.16.0**; analysis became **8**;
    research/semantic documents start at **1**. YAML schema adds optional explicit analysis shapes.
24. **Older evidence:** no declaration is required; missing semantics are explicitly inferred.
    Existing scientific fields remain. Legacy HTML inputs get navigation but no fabricated findings.
25. **Real limits:** discovery is bounded/observational and may remain preliminary; generic
    saturation, subgroup/fairness and high-order causal models are not claimed. Existing objective
    top-region/resource/seed analyses are reused, not generalized to every metric. External-baseline
    ingestion and arbitrary point colour/size/candidate faceting are not added; the baseline is the
    selected reference Trial in a two-Trial comparison. Saved views do not form a live workspace.

No WISDOM files were changed and no real cluster was contacted during this verification.

<a id="guide-research-analysis-8-second-round-simplification-report"></a>

### 8. Second-round simplification report

This section records the focused follow-up on 2026-10-03; section 7 describes the earlier foundation.
The [conditional Study guide](#guide-conditional-studies) includes grammar and planning examples.

1. **Removed controls:** long static metric/parameter selects for ranking, Parameters and X/Y/Z,
   the second checkbox/search catalog for extra metrics, and the flat category select. Hidden values
   are renderer adapters only; there are no hidden long option lists.
2. **Preserved features:** ranking, multi-metric/unit-aware plots, observed heatmaps/3D/surfaces,
   parallel coordinates, partial/pruned markers, families, candidate comparison, persisted models,
   saved views/import/export, notes, favourites/recents and the individual Run dashboard.
3. **Tabs:** Findings & evidence becomes Evidence. Scientific conclusions/seed/pruning/surrogate
   audit stay there; exploratory cards stay in Research. Other scientific tabs retain their purpose.
4. **Picker:** one catalog/dynamic dialog, bounded results, semantic/fuzzy search, keyboard selection,
   favourites/recents; parameters show kind/domain/activation rather than bare names.
5. **Explore:** Analyze / By / Compare with / subset, removable chips and smart plot inference.
   Explicit X/Y/Z, type, grouping, palette and normalization remain optional Advanced controls.
6. **Questions:** labelled, priority-ordered cards show human status, support/reason and Inspect,
   before exploratory findings. Optional label/priority affects display, not scientific policy.
7. **Hierarchy:** collapsible category paths; selecting a parent includes all descendants.
   Metric-family dimensions remain distinct from category paths and parameter conditions.
8. **Grammar:** `when: {parent: scalar}`, `{parent: {eq: scalar}}` or
   `{parent: {in: [scalar, ...]}}`; multiple parents mean AND. Bare lists/unknown operators fail.
9. **Representation:** immutable ActivationCondition predicates inside ParameterDescriptor;
   canonical parent/membership ordering, historical scalar equality serialization, JSON persistence.
10. **Authority:** Work normalization, exhaustive sweeps, RandomSearch, ParameterSpace/Sobol/
    adaptive geometry, encode/decode/validity and analysis activation all reuse the same predicates.
11. **Membership:** finite, nonempty, unique values checked against the parent domain; no implicit
    OR, recursive expressions, domain expansion or consumer-specific branch API.
12. **Equality:** explicit eq normalizes to the old scalar form. Numeric-range equality and
    independent authored domain order retain their behavior; regressions protect old identity.
13. **Inactive keys:** absent from candidates/arguments; signature defaults still apply. No null
    placeholders, invented inactive category or serialization of irrelevant child values.
14. **Duplicates:** active topological enumeration avoids inactive Cartesian multiplication;
    repeated sweep choices fail instead of silently duplicating evidence identities.
15. **Fixture:** six generic branches yield 1+2+10+16+9+18 = **56 unique candidates** × four shared
    seeds = **224 required Runs**, equal to the intended union of independently described branches.
16. **Reference:** the existing selector must match exactly one generated candidate; zero/multiple
    matches fail. No fabricated inactive-value reference is introduced.
17. **Preflight:** validate/explain/dry-run expose design/seeds/required Runs/branches/reference,
    parallelism, GPUs and budgets. Human dry-run does not dump hundreds of Runs; JSON keeps the
    complete planning data plus target-capacity observation.
18. **Time:** scheduler wall-time and logical Study dispatch budget remain separate. Sequential
    levels sum; parallel members use their maximum. Tests cover 1008 h/42 days and mixed compositions.
19. **Capacity:** a bounded direct UUID/visibility probe rejects known impossible GPU requests before
    bundling/submission. Failed/ambiguous probes stay unknown; occupancy is not capacity. Scheduler/
    site-command grants retain their existing boundary; preflight never claims or broadens GPUs.
20. **Steps:** root with/seeds/replicates/search/sweep/execution/objective/analysis fail explicitly
    instead of being ignored. Root resources retain documented inheritance.
21. **Consumer hook:** none added. Existing class/signature/type/marker validation is reused without
    constructing Work. Arbitrary consumer semantics need consumer validation; a new callback contract
    was not necessary for these focused changes.
22. **Schema/version:** additive activation definitions and question label/priority, and rejection of
    unused composition-root policy. Implemented against 0.16.0; analysis v8 and saved-view v1 remain compatible.
    No new YAML version, runner, scientific policy or parallel condition DSL.
23. **Compatibility:** frozen historical evidence is not rewritten; scalar identity, shared seeds,
    objective/HPO/resource policies, retry/export and old HTML/saved views remain intact. Conditions
    persist through the existing StudyDesign, initialization and recovery identities.
24. **Tests:** grammar/AND/domain/cycles/order, immutable pickle/schema/identity round trips, finite
    sweeps/reference, Sobol/random/analysis geometry, steps/root/time and synthetic pre-submit capacity.
    Chromium covers 300 metrics/15 parameters, hierarchy, questions, pickers/chips, smart and previous
    chart types, comparison and saved-view reload. Ruff/mypy pass; full pytest: **1136 passed**,
    four nonfatal Lightning warnings. Final focused analysis/browser/config/docs checks: **63 passed**;
    GPU-policy/config/docs/training smoke: **34 passed**. Wheel build, isolated installation and
    scaffold/validate/dry-run smoke also pass, reusing the local environment's installed dependencies.
    Explicit local CUDA integration: **1 passed**. Numeric-parent regression ensures observations
    never redefine an authored range as a finite domain.
25. **Limits:** no expression DSL, new fitting, automatic step merging, heuristic partition lint or
    consumer-specific validation. Unreliable capacity stays unknown; membership needs a finite parent
    domain. Adaptive counts describe a bounded window, not an execution guarantee. HTML is offline;
    controls never schedule/prune Runs or mutate evidence.

Verification is local/synthetic. No consumer project was edited and no remote Work was submitted.


<a id="guide-coordinated-studies"></a>

## 29. Coordinated Studies: implementation status and contracts

[Español](MANUAL.es.md#guide-coordinated-studies)

<a id="guide-coordinated-studies-status-fresh-adaptive-and-fixed-fleet-launch-recoverytransfer-integration-pending"></a>

### Status: fresh adaptive and fixed Fleet launch; recovery/transfer integration pending

`lf run CONFIG --on-fleet FLEET` now queues one durable local coordinator for repeated Runs,
fixed sweeps, automatic paired sweeps and fresh adaptive HPO without fidelity continuation.
The Research Console's Run Work target selector exposes
the same `fleet:NAME` route. Scientific Work YAML remains unchanged. This is a **limited public
capability**, not completion of the full coordinated adaptive-HPO request.

The existing single-cluster route and independent `MultiClusterSubmissionService` remain unchanged.
Independent submissions are not coordinated HPO.

<a id="guide-coordinated-studies-implemented"></a>

#### Implemented

- One native `WorkRunner` and the original adaptive/fixed/`PairedSweepSequentialAnalyzer` planners. The
  dispatcher executes exact proposals and invokes the existing result/frontier callbacks; it
  never creates candidates, seeds or an optimizer on a worker. One Execution and final analysis.
- One persistent provider Job per member, prepared by ordinary `ControlPlane` and `JobService`.
  Its owned member runtime receives finite leased waves, reuses native isolated Run processes
  and ARI, and drains accepted work before releasing its allocation. Waves do not create new
  GPU claims or another scheduler Job.
- Fresh member heartbeats under the exact owned Job attest immutable code/environment/file-input/
  numerical identities plus actual homogeneous hardware. All five identities must match across
  members before any scientific Run is leased. Preparation placeholders are never placement evidence.
- Baseline GPU admission uses only inherited opaque tokens, native short-lived hardware/memory
  probes, per-member caps and physical headroom. No broadening of device grants. A busy wave offers
  zero spare slots: **GPU co-location and incremental ARI refill inside a Fleet member are pending**.
  Offers expire; physical `lf fleets offers` observations alone never grant dispatch capacity.
- The common prepared-provider path retains direct/site-command/SLURM policy, TLS, bundles and
  immutable managed environments. Loopback CPU and tiny real CUDA acceptance exist; this is not
  yet end-to-end SSH, site-command or SLURM acceptance.
- Typed file markers retain authored scientific parameters and verify canonical bytes/size on
  preparation and worker binding. Large inputs still require the existing project mirror.
  Distributed dataset attestation remains rejected; matching NAME@VERSION is not enough.
- Existing atomic coordinator v2 leases, submission fences, unknown ownership, quarantine,
  caps, original clock/budgets and reconciliation remain authoritative. The global placement wave
  additionally respects the native global parallelism ceiling. No lease is retried from a lost
  connection alone.
- Member Jobs remain in `lf jobs` but do not become independent semantic Works. The parent is
  shown as `fleet:NAME`; active and terminal per-Run metadata includes exact
  cluster/Job/shard/lease/Attempt. Native worker Run records project bounded latest/best scalars
  into the coordinator's existing Study view. `lf show STUDY --run KEY --json` and
  `lf logs STUDY --run KEY` read live or terminal curves/logs on the verified member owner, lazily.
  The coordinator never opens a remote path locally. Display summaries do not substitute for
  the durable scientific scalar stream below.
- Native `metrics.jsonl` and `training-metrics.jsonl` are transported incrementally in complete,
  SHA-256-verified records, at most 32 KiB per channel/Run/read, with optional lossless compression.
  Exact Study/Run/Attempt/lease/member/shard identities fence every read; byte offsets preserve
  order and deduplicate retransmission. Durable receiver cursors recover unacknowledged appends
  after interruption; missing/corrupt acknowledged data, gaps and contradictory replay fail closed.
  Workers retain their native evidence through disconnection. Logs and bulk artifacts are not
  part of this stream. A terminal result reaches the planner only after its full scalar stream.
- The same central adaptive planner consumes completed evidence and all leased/queued/running
  proposals across members. Its existing pending-aware acquisition (BoTorch qLogNEI when available,
  deterministic mixed-kNN fallback otherwise) is retained. Spare admissible capacity invokes the
  native bounded scientific frontier, not worker-side/random proposal generation. Unleased
  reprioritization preserves scientific identity and records the prior invocation/priority.
- Central pruning reuses native utility/history/calibration rules and sends durable idempotent
  exact-lease stop requests. Required/startup/confirmation evidence stays protected. Requests
  stop cooperatively; an unavailable connection is not pruning acknowledgement or terminal evidence.
  Verified local scalar mirrors feed historical calibration without opening execution-host paths.
  Checkpoint continuation/recovery is still gated, not inferred from similarly named local paths.
- `lf cancel STUDY` stops the coordinator before enumerating and cancelling exact member owners.
  Semantic deletion previews the whole owned Job family and refuses active members. Individual
  cleanup cannot remove evidence referenced by the family. Remote bulk compaction stays on its
  execution host; coordinator paths are independently owned.

<a id="guide-coordinated-studies-commands-available-now"></a>

#### Commands available now

```bash
lf run study.yaml --on-fleet research --dry-run --json  # No grant, upload or Job.
lf run study.yaml --on-fleet research                  # Durable asynchronous hand-off.
lf show STUDY --json
lf show STUDY --run trial-00001-seed-4 --json
lf logs STUDY --run trial-00001-seed-4
lf cancel STUDY --dry-run
lf cancel STUDY --apply
lf delete STUDY                                       # Preview terminal family.
lf delete STUDY --apply
```

The coordinator is currently `local`; members must be managed cluster profiles, not the built-in
local target. Dry-run validates source/signature/design, member roles and credentials, reports caps
and deferred environment/equivalence checks, and does not probe or acquire GPUs. The durable hand-off
captures operational profiles and credential references, never credential values. Ordinary
`--on` and `--on-fleet` are mutually exclusive. `--rerun`, `--restart`, hidden recovery options and
`--wait-for-submit` are rejected for this route.

<a id="guide-coordinated-studies-not-implemented-yet"></a>

#### Not implemented yet

Checkpoint stream/continuation and distributed recovery; predictive/lookahead
policy; member co-location; public pause/resume/reconcile and adoption; coordinator restart recovery;
live membership expansion/drain; dataset-placement equivalence; checkpoint/artifact transfer,
compression/resumption and distributed export; full placement/utilization TUI/HTML; StudyDecision
and dependency gating. Internal coordinator lifecycle/catalog controls are not substitutes for those
public operations. Fleet retry refuses single-cluster replay; export refuses a coordinator-only
package that would silently omit member evidence.

The request is therefore **not complete**. Do not describe this limited route as production-ready
full distributed production readiness or claim provider acceptance from loopback/fake tests.

<a id="guide-coordinated-studies-incremental-validation-2026-10-05"></a>

#### Incremental validation (2026-10-05)

Ordered stream regressions cover bounded multi-chunk reads, compression, replay, incomplete records,
receiver interruption, foreign leases and symlinks. Member regressions cover directed pruning,
idempotence, immutable commands and protected evidence. Public loopback CPU integration runs the
native adaptive planner across two prepared allocations and retains one HPO state/final analysis.
Historical pruning is also exercised using verified mirrors with deliberately absent owner paths.
These tests do not establish real remote/GPU adaptive acceptance or coordinator restart recovery.

<a id="guide-coordinated-studies-validation-of-this-implementation-pass-2026-10-04"></a>

#### Validation of this implementation pass (2026-10-04)

- `ruff check .`: passed.
- `mypy src/lambdaforge`: passed, 558 source files.
- Complete `pytest`: 1,246 passed; four Lightning warnings in CPU tests.
- A freshly built, installed wheel passed packaging smoke checks, installed-import verification,
  `run --help`, scaffolding, validation and read-only dry-run.
- Public Fleet integration exercised two loopback CPU members, including live Run metrics/logs;
  separate prepared-provider tests exercised tiny real local CUDA workloads. These are not
  real SSH/site-command/SLURM acceptance tests. WISDOM and real clusters were not modified or run.
- GitHub CI was not executed. Passing local checks does not establish the pending capabilities
  listed above.

<a id="guide-coordinated-studies-fleet-catalog"></a>

### Fleet catalog

Fleet configuration belongs in `lambdaforge.clusters.yaml`, never the scientific Work YAML:

```yaml
clusters:
  a: {transport: ssh, host: a-login, user: USER, scheduler: local}
  b: {transport: ssh, host: b-login, user: USER, scheduler: local}
  queue: {transport: ssh, host: queue-login, user: USER, scheduler: slurm}
fleets:
  research:
    coordinator: local
    members:
      - {cluster: a, max_gpus: 2, max_runs: 4}
      - {cluster: b, max_gpus: 4}
      - {cluster: queue, max_inflight_jobs: 2, required: false}
```

Cluster profiles keep their own GPU launch/claim policy, project mirror and data environment. All
references must resolve locally before any probe. User < project < explicit precedence is shared
with the existing catalog. A fleet override replaces its whole membership list; it does not merge
lists accidentally. Cluster definitions, credential references and execution profiles are retained.
Caps accept positive integers or null; online/draining/disabled are operator controls, not probes.

```bash
lf fleets list --json
lf fleets show research --json
lf fleets offers research --json         # Explicit read-only probes; not an allocation.
lf fleets drain research b               # Preview catalog change.
lf fleets drain research b --apply
lf fleets disable research b --apply     # Never signals active Jobs.
lf fleets enable research b --apply
```

Use `lf fleets --catalog PATH ...` to select an explicit catalog. Otherwise mutations go to the
project catalog. Saving a fleet does not save secrets or change Work scientific identity.

<a id="guide-coordinated-studies-ownership-and-persistence"></a>

### Ownership and persistence

The current local driver owns a project-scoped coordinator below the durable JobStore's
`fleets/PARENT_JOB/coordinator`, with independently owned native Execution metadata beside it. `coordinator.json`
is schema version 2 (with a v1 reader) and is atomically fsynced with the existing JSON publisher under the existing
cross-process file lock. `leadership()` fences the planning loop. Short state transactions never
hold a lock over network I/O. The state contains immutable initialization/budgets, Run definitions,
all leases/Attempt history, placement reasons, shard identities, member controls and quarantine
references. Separate immutable digest-named `results/` files hold accepted envelopes; `quarantine/`
holds contradictions. Missing/corrupt/version-incompatible state fails closed, never silently resets.
Restart cannot silently change coordinator authority. Readiness attestations must be explicit
booleans and proposed scientific payloads must be strict finite JSON before they are persisted.

The global Run key hashes `(study_identity, candidate, seed, phase, fidelity)` using the existing
scientific identity primitive. A lease adds Attempt number, opaque lease ID and assigned cluster.
Two drivers cannot reserve the same logical Attempt. Offers exclude locally acknowledged resident
leases; the coordinator subtracts additional unacknowledged leases to prevent repeated stale
observations from overbooking during the prepare/submit gap.

A submission-intent fence is persisted **before** the provider call. An ambiguous acknowledgement
therefore requires observation by immutable shard ID, not another submit or a new Attempt. The
executor protocol is independently idempotent by shard ID. No credentials belong in a shard.
Late submission acknowledgements attach to their exact Attempt: they cannot overwrite a completed
result, move a running Run back to queued, or attach an old Job ID to a replacement Attempt.

Fresh/healthy executor offers can permit new work, but expired offers cannot. Transport failure
marks active leases `unknown_remote`, with exponential reconnection backoff capped at five minutes.
It does not cancel workers, create failures or release scientific identity. Missing observations
also remain unknown. Lost-attempt retry requires a local scheduler/owned-process/shard authority
proof naming the exact lease and confirming no live owner, and remains bounded. Arbitrary consumer
exceptions are not retry triggers. Old shard records do not consume the replacement Attempt's Job cap.

Result envelopes must match Study, Run, Attempt, lease, target, parameters and execution equivalence.
Exact duplicated bytes are harmless; contradictory or late released-Attempt results are quarantined
with a compact diagnostic, never overwrite valid evidence. A crash between immutable result write
and atomic index write is repaired by repeated delivery. Per-Run result reads verify the digest.
Root/evidence symlinks are rejected. Overview snapshots expose compact counts, not every result.

Reservation currently conservatively consumes the `max_runs` dispatch allowance; an operator
withdrawal before the submission-intent fence refunds its unspent reservation, preserving the
Attempt audit. Wall-time is
measured from immutable coordinator creation across restart. Physical resource accounting and
reversible unstarted reservation accounting still need the production dispatcher integration.

<a id="guide-coordinated-studies-reproducible-foundation-acceptance-tests"></a>

### Reproducible foundation acceptance tests

No real cluster or WISDOM execution is needed:

```bash
python -m pytest -q tests/controlplane/test_coordinated_study.py tests/controlplane/test_fleets.py
python -m pytest -q tests/work/test_concrete_shard.py
python -m pytest -q tests/work/test_study_dispatch_boundary.py tests/controlplane/test_coordinator_pause.py
python -m pytest -q tests/work/test_coordinated_cpu_dispatch.py tests/controlplane/test_prepared_cpu_executor.py
python -m pytest -q tests/controlplane/test_shard_preparation.py tests/controlplane/test_prepared_provider_dispatch.py
python -m pytest -q tests/controlplane/test_fleet_study_service.py tests/controlplane/test_member_allocation.py
```

The fixed design test supplies 56 candidates × 4 seeds = 224 unique required Runs and fake targets
with capacities A=2, B=3, C=1. All are centrally ingested once; the exact synthetic final objectives
match a serial reference. The partition test disconnects B, loads a fresh coordinator instance,
redirects new work to A/C and reconciles one resident, one completed and one positively lost B
Attempt; only the lost Run becomes Attempt 2. Other tests cover competing drivers, ambiguous
acceptance, stale offers, caps, checkpoint locality, environment/hardware mismatch, provenance,
budgets, operator drain and immutable global identities. These validate the operational contract,
not yet end-to-end distributed Work execution or predictive optimizer quality.
The CPU shard test invokes four real spawned Work processes at concurrency two, keeps successful
and failed scientific evidence and native paths, and proves repeated delivery does not create
another Attempt. It does not submit a provider Job, grant a GPU or start an optimizer.

Prepared-provider acceptance also exercises actual bundle staging, JobService, a detached direct
supervisor, CPU children and a minimal CUDA tensor child with native ARI. Installation/runtime
resolution are fixtures; the transport is loopback, not SSH. The GPU test uses explicitly shared
local access and inherits the supervisor's grant; it skips when CUDA is unavailable. This does not
establish SLURM/site-command acceptance or managed package installation. Persistent-allocation
cases verify fresh baseline offers, exact Job reuse and drain without releasing the grant between
waves. Input identity/mutation and opaque-grant regressions are separate.

The prepared-provider tests run four fixed identities or three complete two-candidate paired
blocks over two detached direct CPU targets, ingest native outcomes centrally and produce one
final analysis. This is real
local ProcessScheduler execution, not a fake provider, but it is still an internal prepared-CPU
test. `tests/controlplane/test_fleet_study_service.py` additionally tests the product service's
real two-member allocations, single final analysis, terminal Run-owner reads, semantic cancellation,
captured asynchronous request and read-only CLI grammar/preflight. Installation remains a fixture;
the test does not pretend to be real SSH or complete adaptive Fleet acceptance. The live-read
fixture holds scientific Runs open until the coordinator has actually read their isolated log
and step-7 metric through the exact member owner. Foreign seeds/Trials/Attempts/paths are rejected;
remote display scalars cannot become completed response evidence or trigger local path reads.
Dispatcher-injection tests also exercise the
existing native adaptive planner without creating a worker-side optimizer. Pause tests cover
network partitions, completed-result ingestion, pre-submit races, restart and immutable budgets.

Fixed authored sweeps will keep performance pruning disabled. Automatic sweeps will retain complete
paired blocks; adaptive searches may prune under their existing scientific contract. Future
predictive work belongs to the current Study and its ordinary budgets: spare capacity is not
permission to launch scientifically dependent downstream work or uninformative random proposals.


<a id="guide-operational-reliability"></a>

## 30. Operational health, monitoring and deliberate recovery

[Español](MANUAL.es.md#guide-operational-reliability) · English

Logical execution state, scientific evidence and operational health are different facts.
An old `running` record is not a current process observation. `UNKNOWN` means ownership or
observation is unresolved, not that science failed or that starting a duplicate is safe.
These capabilities use the existing project JobStore, providers and Work runner.
The [acceptance ledger](ai/IMPLEMENTATION.md#history-operational-reliability-progress) lists the broader reform still pending.
The unreleased [resource safety guide](#guide-resource-safety) covers shared observation, GPU/MIG
startup coordination, cgroup RAM evidence and conservative external-reservation boundaries.

<a id="guide-operational-reliability-inspect-without-starting-computation"></a>

### Inspect without starting computation

```sh
lf health --json
lf health JOB --refresh --json
lf incidents --json
lf jobs show JOB --json
lf jobs logs JOB --tail 100
lf monitor diagnose --json
```

`health` without `--refresh` reads saved observations only. `--refresh` queries the provider;
the observer's attempt/success times remain separate from the host's heartbeat clock and from
scientific progress. A failed query preserves the last confirmed state/time and increments its
failure count. Reading the cache or rewriting a local record does not advance confirmation.
Health commands return exit code 2 when observations are unknown, stale, degraded, unreachable
or confirm lost processes; that exit code is not a scientific outcome.

Health may be `healthy`, `degraded`, `unreachable`, `stale`, `lost_confirmed` or `unknown`.
Provider errors do not become a successful empty inventory. A missing inventory entry stays
unknown until exact ownership can be verified. Direct observation checks process lifetimes and
same-user inherited Job markers, including reparented children. Inaccessible ownership evidence
fails closed. Observation itself never kills processes or edits host `state.json`.

A corrupt or unreadable individual Job record is isolated within direct-host inventory: healthy
siblings still receive fresh state/health observations. Only that known Job becomes degraded
(a saved terminal outcome remains terminal). Error rows never create new Jobs or certify loss or
recovery eligibility. Root/transport failures still invalidate the host query. A later valid
observation clears the prior query error; no manual history editing or scientific restart is needed.
Shipped read probes select definitions and helper dependencies by name from one source snapshot,
not loaded function line offsets that can become stale after an editable source update. Restart
an already-open Console/monitor after upgrading LF to load the corrected probe implementation;
this does not restart scientific Jobs or require modifying their immutable worker environments.

SLURM retains queue/accounting source, raw state, reason, exit code, node and available times.
An accounting delay is unknown; a query error remains an error unless another query supplies
positive evidence. See the official [sacct reference](https://slurm.schedmd.com/sacct.html) and
[state codes](https://slurm.schedmd.com/job_state_codes.html). `NODE_FAIL` is infrastructure evidence,
`OUT_OF_MEMORY` is resource evidence, not an objective score. Polling remains bounded.
If `squeue` times out, a bounded accounting query may still supply positive evidence. Empty or
failed accounting never legitimizes a cached RUNNING response; previous query details are cleared.

Each health read includes a native `diagnostic` with category, source, confirmation, event time,
impact and recommendation. Scheduler OOM does not by itself identify CUDA OOM; exit 137 does not
prove host OOM. Generic CUDA kernel/dtype errors remain consumer failures, not OOM retry grants.
Malformed data is degraded observation, not proof of an offline host.
An unclassified query exception likewise keeps connectivity unknown even after repeated failures;
typed connection/authentication errors remain distinct in the shared diagnostic authority.
Legacy terminal files without ownership requests remain readable but do not certify ownership
or safe recovery.

<a id="guide-operational-reliability-independent-project-monitor"></a>

### Independent project monitor

```sh
lf monitor config --file monitor.json
lf monitor start
lf monitor status --json
lf monitor once --json
lf monitor stop
```

`start` launches a detached observer of the native project catalog, not a scientific runner.
One cross-process owner lock prevents duplicate monitors. `stop` requests only that exact monitor
to exit; it never cancels Works. Status checks process lifetime and heartbeat separately and
reports repeated observation failures as degraded. Restart preserves observations, incident IDs
and notification delivery records. The monitor does not read epochs/artifact bytes, fit HPO models,
allocate GPUs or change scientific YAML. Config/state/logs are project-owned below the JobStore's
`.operational/monitor/`; incidents are compact records below `.operational/incidents/`.

Example: [monitor.json](../examples/monitor.json). Operational deadlines are seconds:

```json
{
  "interval_seconds": 60,
  "observation": {
    "freshness_seconds": 180,
    "heartbeat_seconds": 45,
    "grace_seconds": 120,
    "progress_seconds": null,
    "unreachable_failures": 2
  },
  "smtp": {"enabled": false}
}
```

Progress expectations default to disabled. Some valid Works emit no metrics for hours; silence
alone must not prove death. Heartbeat grace, provider freshness and scientific-progress expectations
are independent configurable checks. Failed observations immediately show connectivity loss;
repeated failures escalate health without repeated Console toasts.

Closing Textual does not stop the detached monitor. Reboot/logout persistence is a separate OS
service concern. Linux provides a preview-first systemd-user installer:

```sh
lf monitor install --json
lf monitor install --apply
```

The installer refuses unrelated units, enables only the project observer and does not enable user
lingering or change administrative policy. Inspect the unit and `loginctl show-user "$USER"`.
systemd's [user manager lifecycle](https://github.com/systemd/systemd/blob/main/man/org.freedesktop.login1.xml)
depends on session/lingering configuration; a detached process alone does not guarantee survival
of a site's logout cleanup. On macOS/Windows, use `lf monitor run` with launchd/Task Scheduler or
another managed service. These platform integrations and real-site logout/reboot have not been
certified by the local CPU regression tests.

<a id="guide-operational-reliability-optional-email-durable-incidents-and-no-polling-spam"></a>

### Optional email, durable incidents and no polling spam

SMTP is off by default. Configure a host, sender, recipient list and TLS mode `starttls` or `ssl`.
Credentials must be `env:NAME` or `keyring:KEY` references; never store a password in configuration.
Authenticated monitoring is noninteractive: the referenced credential must be available to the
service process. A site may use an authenticated relay or an explicitly configured unauthenticated
TLS relay. Verify access and policy with the operator.

```json
{
  "interval_seconds": 60,
  "smtp": {
    "enabled": true,
    "host": "smtp.example.org",
    "port": 587,
    "tls": "starttls",
    "sender": "research@example.org",
    "recipients": ["research@example.org"],
    "username": "research@example.org",
    "credential": "env:LF_SMTP_PASSWORD",
    "cooldown_seconds": 900
  }
}
```

```sh
lf monitor test-alert --json
lf incidents --json
```

A delivery is marked sent only after SMTP accepts it. Transport failure never changes a Job.
Automatic attempts are bounded at three per incident transition, with a persisted cooldown.
Stable incident IDs deduplicate repeated probes; escalation, reopening and resolution remain
auditable in native Job events. Resolution requires positive healthy evidence, not disappearance.
For an audited retry, origin incidents resolve only after fresh scientific progress in the exact
accepted recovery Job, after its request timestamp. Enqueue, preparation, scheduler acknowledgement
and heartbeat alone do not resolve them. The origin outcome stays unchanged; `recovered_by` links
the new Job/request and the observed progress. A failed recovery cannot resolve the origin.
Loading reused Runs establishes a baseline, not fresh progress. Changes to obligation totals,
counter resets and duplicate polls likewise do not resolve an incident; a shared completion/step/
epoch counter must advance beyond its persisted/native or positively observed runtime baseline.
Native previews retain the number of successful logical Runs, so a first observation may certify
a genuinely new completion beyond that count; rereading the same count never refreshes progression.
SMTP runs outside the incident lock; a slow relay cannot block provider commits. If escalation or
resolution occurs during delivery, the earlier acknowledgement never marks that newer transition sent.
SMTP acceptance is not an inbox receipt. Exactly-once delivery cannot be guaranteed across an
interrupted send; stable Message-ID supports receiver deduplication. No blanket mail-all policy
or implicit external messaging is enabled.

<a id="guide-operational-reliability-research-console"></a>

### Research Console

Bare `lf` → **Operational health…** (or Ctrl+P → Operational Health System) opens saved health,
incident details and monitor settings. Refresh providers is explicit; Start/Stop affect the
independent observer only. Save settings uses the same validator as `lf monitor config`.
Send test email exposes acceptance or failure inline. While an operation is pending, duplicate
activation is disabled; closing a panel does not cancel scientific work.

Monitor and Console share nonblocking project/provider observation gates. While a same-cluster probe
is in flight, readers retain native last-good records and their original clocks; receipt files are
coalescing metadata, never proof of provider freshness. Separate SLURM Jobs retain separate lanes.
An abruptly terminated observer releases its OS lock; it cannot leave a permanent probe barrier.

Overview/Work/Studies display health separately from scientific state and last confirmation.
Imported snapshots remain historical read-only evidence, with no expectation of a live host.
Do not interpret a previously completed scientific result as invalid merely because its host
cannot now be contacted.

<a id="guide-operational-reliability-recovery-and-relocated-yaml"></a>

### Recovery and relocated YAML

```sh
lf retry STUDY --dry-run --json
lf retry STUDY --config experiments/moved.yaml --dry-run --json
lf retry STUDY --config experiments/moved.yaml
# Only after explicitly reviewing whether earlier evidence/checkpoints remain valid:
lf retry STUDY --config experiments/moved.yaml --accept-code-change
```

The Console's **Resume Study…** offers the historical path, a YAML picker and Validate recovery
plan. Missing historical YAML is not a reason to replay the full experiment. A selected YAML is
compared against original persisted scientific identity after restoring the original seed namespace.
Formatting/comments/key order do not define scientific identity. Parameters, input bytes, dataset
identity, seeds, objectives, design and publication contracts must still match. Relative inputs
are resolved from the selected YAML: move it only with correct exact bindings. Code-change
acknowledgement is not an identity-check waiver.

For single-host Studies launched through a provider command such as `gpu exec`, recovery
uses the **new** inherited `CUDA_VISIBLE_DEVICES` grant, not the previous Job's GPU list.
If the unchanged YAML requests three GPUs and the provider grants only one, adaptive, repeated
and fixed-sweep Studies dispatch only onto that GPU; `runs_per_gpu` is applied to the current
device count and `max_parallel` remains a ceiling. Historical completed Runs, checkpoints,
controller decisions, costs and the original requested-resource record are retained. No YAML
change is needed merely because the grant became smaller. An explicit empty/disabled CUDA
visibility never falls back to physical indices; a GPU Study fails closed with an actionable
grant diagnostic. This is not a promise to rewrite an application's own multi-GPU/DDP topology.
SLURM/exclusive scheduler allocations retain their exact request contract: an insufficient
allocation is queued or rejected safely, never supplemented with ungranted GPUs.

Recovery dialogs keep the compatibility checkbox, current blocking reason and confirmation
controls outside the scrollable plan. Clicking Resume without a required acknowledgement or
validated YAML explains why nothing was submitted; it never implies approval. Validation
disables submission until its result arrives, including when the checkbox is toggled meanwhile.
Once explicitly confirmed, recovery is submitted at most once and its outcome appears in the
workspace or a persistent failure dialog.

The selected file hash is pinned from preview to preparation; editing it invalidates the plan.
Preview creates no Job, consumes no seed/budget, reserves no GPUs and downloads no checkpoint bytes.
CLI `--dry-run` also leaves the saved operational audit unchanged. Explicit Console **Validate
recovery plan** records an operational inspection request/outcome, but never changes scientific
evidence or launches computation.
Apply rechecks provider ownership under the native recovery lock; unreachable/ambiguous live
ownership blocks duplication. A fresh, positively verified lost process can be recovered without
falsely rewriting its scientific outcome to `failed`.

Direct process verification scopes unreadable processes to the exact Job, rather than treating
every same-user access denial as host-wide uncertainty. Positive creation-time evidence excludes
processes that predate the native Job; POSIX session/UID/lifetime evidence may also prove an
independent foreign-owned login session. Process names alone are never sufficient. A live
supervisor, scientific child or marked orphan still blocks recovery, as do inaccessible owners
and unreadable processes whose independence cannot be proved. Inventory scans processes once
and evaluates that snapshot separately for each Job; observation never signals processes or
rewrites persisted scientific records. Restart an already-open Console after updating the local
framework. The read-only probe also applies to older immutable remote environments without
installing into them.

After a host reboot, the direct probe compares authoritative process creation times with
OS boot time. Verified pre-boot owners cannot survive the reboot; positively dated new
access processes are not their descendants. This does not rewrite cancellation/failure or
grant permission based on an empty `htop`: a matching live Job marker, unknown birth/boot
evidence, or an unresolved post-boot owner still blocks recovery. The same native probe is
used by CLI and Console. Restart an already-open Console to load the updated probe.

Accepted recovery receives a `recovery_request_id`, linked from the origin's append-only events
and new Job metadata/events. `lf jobs show NEW_JOB --json` exposes `recovery.phase`: accepted,
preparing, scheduler acknowledged, runtime observed, scientific progress observed or terminal
outcome. Admission is not progress; epoch zero is not training progress. Preparation failures retain
their phase/category and controller logs. The Console shows the request ID and observed phase.

<a id="guide-operational-reliability-saved-recovery-history-including-rejected-requests"></a>

#### Saved recovery history, including rejected requests

```sh
lf jobs recovery JOB --json
lf jobs recovery JOB --offset 100 --limit 100 --json
```

This read-only view uses native saved Jobs and append-only events, without contacting the execution
host or reading checkpoint/artifact bytes. It remains available during an outage. It links exact
retry/dependency/scientific identities, never display names, and shows missing dependencies rather
than fabricating history. Events are paged newest-first; `next_offset` selects older events.

Study workspace → **Recovery history** opens the same paged view, with linked Jobs, request IDs,
phases and event details. Apply and explicit Console inspection allocate a request ID before
validation: rejected ownership, missing state or incompatible configuration remains auditable even
when no new Job exists. Rejection leaves scientific records and completed Runs untouched.
An audit-write failure is reported rather than falsely claiming the request was durably recorded.

On recovery failure the Console keeps a persistent dialog with the phase, request ID, bounded
diagnostic and Logs / Saved history / Back to plan controls. It does not repeatedly notify on polls,
and closing the dialog is not another retry. A successful preview is not admission; only apply may
launch preparation, and it revalidates the exact plan under the existing ownership lock.

Fixed/repeated designs use owned design/Run/Attempt records; adaptive Studies require their genuine
controller state. Valid Runs remain valid, failed/interrupted Runs create new Attempts, spent
budgets remain spent, and original Execution/Attempts are never silently overwritten. Recovery of
a Study is not proof of epoch continuation: the Work must implement compatible checkpoint loading.
Managed checkpoint helpers, explicit retention pins and exact recovery transport remain the native
checkpoint authority. Native Fleet recovery remains pending and rejects single-host replay.

<a id="guide-operational-reliability-managed-checkpoint-policy-and-evidence"></a>

### Managed checkpoint policy and evidence

At an application-safe boundary, a Work may configure advisory cadence and terminal retention:

```python
self.checkpoints.configure(
    interval_steps=5,
    interval_seconds=1800,
    retention_seconds=7 * 86400,
    storage_durability="shared",
)
if self.checkpoints.should_save(step=epoch):
    self.checkpoints.save_json("state.json", algorithm_state, step=epoch)
```

No default write interval is imposed and arbitrary algorithms are never automatically serialized.
Either interval may make a save due. Only a committed write advances cadence, not reads/polling.
JSON saves preserve exact checksums. Named `checkpoints.file(..., build=..., step=...)` files also
register native evidence; version binary names or supply semantic validation when rebuilding,
because `file` still reuses valid existing content. Advanced `path()` writes remain unregistered.

New records include execution/Run/Attempt/scientific owner, write time and optional generic step.
`inventory()` pages metadata; `inventory(verify=True)` verifies bytes on the owning host, not
application compatibility. Corrupt registered JSON refuses loading; legacy unregistered JSON
remains readable without a fabricated certificate.
Registered Execution/Run bindings are checked before managed state is loaded or replaced; a foreign
owner or corrupt registration is rejected without changing payloads. Older Attempts of the same
logical Run remain usable. Code/scientific compatibility still requires native recovery validation
and explicit acknowledgement where applicable; this owner check does not waive it.
Recovery preview/Console show the last registered descriptor with **not checked** integrity,
without downloading payload or promising
epoch continuation. Historical `resumed_from_checkpoint` indicates availability at Attempt start,
not proof of application restoration; subsequent progress is separate.

Durability (`unspecified`, `ephemeral`, `persistent`, `shared`) is an operator/project declaration,
not a filesystem guarantee. Node-local bytes do not protect against node loss; verify the actual
storage policy. Configured terminal retention integrates with preview-first cleanup; failed/
interrupted evidence and pins remain protected. Increasing retention after preview is rechecked
before deletion. Live rotation, application-specific deserialization and cross-host/Fleet transfer
acceptance are not certified by this cadence/metadata API.

<a id="guide-operational-reliability-emergency-procedure"></a>

### Emergency procedure

1. Preserve the exact Job ID. Inspect `health --refresh --json`, Job details, bounded logs and
   incidents. Do not delete history, checkpoints or processes to make a stale row disappear.
2. Separate connectivity loss from proved process loss, scheduler failure and consumer failure.
   An inaccessible host may still be computing. Retry is not permission to duplicate it.
3. Restore connectivity, inspect/reconcile native evidence and preview deliberate recovery.
   Select moved YAML if necessary; inspect reuse/retry/pending Runs and code compatibility.
4. Confirm apply only when ownership and scientific identity are valid. Track the **new** Job's
   preparation, admission, heartbeat and subsequent scientific progress, retaining the origin ID.
5. Preserve diagnostic/incident/recovery events. Notify site support for infrastructure failures.
   Historical successes are reusable; arbitrary consumer exceptions are not automatically retried.

Local regression evidence is listed in the acceptance ledger. It does not certify a real SMTP
server, CUDA failure, SLURM site or systemd activation. The native local two-Study/Textual recovery
fixture passes with simulated provider/admission faults; real detached preparation and the full
multi-host chaos scenario remain outside that demonstrated acceptance.

<a id="guide-operational-reliability-deliberate-extension-versus-recovery"></a>

### Deliberate extension versus recovery

Adding fixed-sweep values/seeds uses **Extend Study**, not Resume. Its reviewed delta receives
a separate native Execution/Job; previous revisions and physical Attempt/time expenditure remain
unchanged. A failed delta still uses native `lf retry` / Console **Resume Study…** on its linked
Job, preserving its Execution and successful Runs. The series follows the exact retry lineage
without duplicating evidence. Missing Job plus existing preparation/provider state is uncertain,
not permission to resubmit. Only a proven pre-Job handoff gap can reuse the preallocated ID.
Checkpoint epoch continuation remains an explicit consumer Work contract. See
[Study extensions](#guide-study-extensions) for supported placements and safe reconciliation.

Extension review defaults to `--code-policy revisioned`: changed/dirty project code may add new
cells using normal native managed deployment, with separate code/runtime provenance and per-revision
inference. It does not relabel completed Runs as new-code evidence. Strict equivalence remains
available with `--code-policy strict`. Reviewed source is pinned through preparation, and the actual
host's staged semantic contract/Execution identity is attested before provider submission.


<a id="guide-resource-safety"></a>

## 31. Shared-host resource safety — implementation in progress

[Español](MANUAL.es.md#guide-resource-safety) · English

This is unreleased work on 0.19.0, not acceptance of the complete resource reform or an
explanation of a server reboot. No real CiTIUS connection, reservation change, stress test or
scientific computation was performed for this implementation. See the
[baseline audit and remaining work](ai/IMPLEMENTATION.md#history-resource-robustness-audit).

<a id="guide-resource-safety-observation-is-not-a-grant"></a>

### Observation is not a grant

The existing runner, supervisor and direct resource service use one same-user/host GPU cache.
Ordinary measurement uses the optional official distribution `nvidia-ml-py` (module `pynvml`);
no PyTorch import, tensor, allocation or CUDA compute context is needed. CPU installations do
not acquire a mandatory GPU dependency. Without the optional binding, the same isolated
observer batches bounded `nvidia-smi` memory/process queries; unavailable attribution stays
unknown rather than pretending a device has no processes. MIG instance memory requires a
measured instance identity; parent capacity is never substituted.

Default policy: 5 s informative TTL, 0.5 s admission TTL, 0.25 s minimum refresh interval,
5 s isolated helper deadline and failure backoff from 2 to 60 s. Concurrent misses are
single-flight. Cache hits create no helper. The native cache exposes request/query/hit/error/
timeout counts and last latency. A timed-out private helper group is killed and reaped;
unresolved ownership prevents another helper. Cache data is bounded to 1 MiB. Storage/ownership
errors fail closed. Last-known numbers remain informative, never fresh admission evidence.

The root is UID-owned/private, scoped by host and boot, and independent of Project/virtualenv:
`$HOME/.cache/lambdaforge/hosts/<host-hash>/`. An operator may explicitly choose its base using
`LAMBDAFORGE_HOST_COORDINATION_ROOT`; different projects must use the same base. Symlinks,
foreign ownership and unsafe permissions are rejected. This metadata is operational, not a
scientific identity or a second result store. Do not manually edit it during execution.

Full GPU/MIG UUID grants need no CUDA API. Legacy numeric grants require a **one-time bounded
driver identity enumeration** (`cuInit`/`cuDeviceGetUuid_v2`, without creating a compute context),
cached/pinned to the controller lifetime, boot, original visibility and CUDA ordering.
This initializes the driver for identity resolution; it is not a periodic memory probe.
CUDA ordinals are never assumed equal to NVML indices. Missing/remapped UUIDs pause admission,
not permission to substitute another GPU. After an established baseline, loss of one device's
measurement does not prevent launching on a healthy granted sibling. Last-known failed-slot
values are excluded from prediction updates/admission and visibly qualified as unavailable.

<a id="guide-resource-safety-cross-study-startup"></a>

### Cross-Study startup

Both `shared` and `command` modes coordinate starts using short per-GPU/MIG UUID locks.
The existing `GPUPlacementPlanner` remains authoritative; it rechecks fresh physical headroom
under that lock. A temporary startup commitment is persisted **before** spawning, with native
controller/worker lifetime and Job/Execution/Run/Attempt references when available. Disk errors
cannot hide an already spawned Future. Other GPUs have independent admission locks.

Peer startup demand is `max(startup commitment - observed resident bytes, 0)`. Physical free
VRAM already excludes resident bytes; do not subtract them again. The current controller's
future envelopes already cover its own Runs. Confirmed dead worker lifetimes are reconciled;
unresolved owners remain protected, and a missing worker record alone proves nothing. Orphan
reconciliation can use the native Job ownership observation; a timestamp does not prove absence.
The default 60 s startup window for an attested live worker is finite, not a permanent reservation
of `gpu_memory` per resident. Compatible learned time-to-envelope may extend it (bounded to one
hour). A peer's pending startup cannot be bypassed by baseline or exploration. Packing ceilings
and scientific decisions are unchanged. Per-device bounded waiters account for native lifetime,
priority, wait age and served launches. Older feasible Studies get an opportunity, without
forced alternation or blocking light work behind a peer that cannot fit. Stale waiters expire;
startup commitments require stronger ownership reconciliation. Cross-user fairness remains the
provider's responsibility; unrelated scientific scores are not compared.

<a id="guide-resource-safety-ram-and-cpu"></a>

### RAM and CPU

Read-only host observation reconciles physical availability with readable cgroup v2 ancestor
`memory.max`/`memory.current`; it records memory pressure and OOM events where available. It
writes no cgroup limits. Invalid current/limit evidence is unavailable, not zero. Job admission
uses the effective capacity; an impossible request is rejected before computation. Before a
GPU Run starts, host availability is checked again against its native dispatch-time RAM share.

Process memory attribution verifies PID birth and UID. RSS, PSS and USS remain distinct;
RSS sums may double-count shared pages. Only qualified PSS reduces an unmaterialized Job RAM
commitment. Unknown attribution conservatively retains that commitment. `ram_bytes=0` is
explicitly labelled unknown demand, not a claim that the Job consumes no RAM. Unresolved/exec-
changed lease owners retain CPU/RAM protection; argv mismatch is not proof of death.

Direct access leases store host/boot ownership below `hosts/<host-id>` in the configured
lease root. GPU keys and child visibility use exact GPU/MIG UUIDs, not CUDA/NVML ordinals.
Exclusive parent-device access conflicts with its MIG instances; independent instances remain
independent. Discovery runs outside the lease lock; release uses retained ownership rather
than probing a potentially changed topology. The original outer lock namespace is retained
so old supervisors cannot race new writers. Legacy unscoped/ordinal records remain protected,
including apparently dead local PIDs: shared-home records may belong to another host.
Queued Job feedback explains this ambiguity. Active older supervisors normally release their
own leases on exit; crash leftovers without host/boot evidence require explicit ownership
reconciliation, not blind deletion.

Existing native affinity/thread-pool limits and elastic Run CPU leases remain in force. No
consumer DataLoader parameter, scientific fidelity, seed, checkpoint or result was rewritten.
Readable `cpu.max`/`cpuset.cpus.effective` narrow effective capacity; fractional quota is disclosed
and bounded by its ceiling, without expanding inherited affinity. Controller overhead is
deducted before dividing aggregate Run RAM. Online learned RAM demand and process-pressure
backpressure remain incomplete.

<a id="guide-resource-safety-profiles-device-circuits-and-diagnostic-reports"></a>

### Profiles, device circuits and diagnostic reports

Memory errors/disappearing UUIDs open expiring device-observation circuits (5–120 s by default).
Suppressed NVML devices skip ordinary memory/activity queries until recovery time; identity
enumeration remains necessary. A new qualified same-UUID read closes its circuit. Existing
candidate/placement-specific OOM dominance remains separate. A single OOM does not permanently
cap concurrency; generic CUDA exceptions do not imply ECC, Xid, reset or retry permission.

Cluster-operational settings, not Work YAML:

```sh
lf clusters set CLUSTER resource_safety shared-host
lf clusters set CLUSTER resource_safety.priority 2
lf clusters set CLUSTER resource_safety.probe_timeout 3
lf resources --report --on CLUSTER --since 2026-10-13T00:00:00Z --until 2026-10-14T00:00:00Z --json
```

`balanced` preserves defaults; `throughput` uses 3 s informative TTL/20 s fairness wait;
`shared-host` uses 10 s TTL/15 s wait; `conservative` uses 120 s startup and longer device backoff.
No profile changes scientific parameters or concurrency ceilings. Overrides are validated and
propagate through native direct/remote submission. The Console's Advanced YAML can edit
`resource_safety`; existing profiles use balanced without a parallelism migration.

CPU-only Job heartbeats create no GPU helper. New operational `usage.jsonl` rotates above 2 MiB
into four compressed windows, retaining recent samples and cumulative sampled peaks in
`usage-summary.json`. Scientific metrics, logs and checkpoints are untouched. An already
oversized legacy operational log is neither read unboundedly nor silently deleted; current
state remains available and cleanup is deliberate/scoped.
Interrupted archive temporaries do not disable subsequent compaction and are never adopted
or removed as somebody else's evidence.

The sanitized saved-evidence report is also available under cluster **Resources → Saved resource
report** and the session-pinned agent method `get_resource_report` (`clusters.resource_report`).
It reads bounded project-owned Job metadata, without a provider query or asset/full-log read.
Commands, paths, arbitrary errors, credentials and other users' PIDs are excluded. It works
offline/after reboot if metadata survives. Missing facts remain unknown. Peaks are labelled
Job-lifetime sampled maxima, not exact selected-interval peaks. It does not fabricate complete
historical timelines, driver-event diagnoses or the cause of a host reboot.

<a id="guide-resource-safety-site-reservations-and-visibility"></a>

### Site reservations and visibility

For CiTIUS, use `command_prefix: [gpu, exec]`; a persistent reservation can be made manually
through the provider. Absolute managed Python is passed inside the wrapper, without relying
on shell/Conda activation. LambdaForge does not issue `gpu release` on success/failure/cancel.
The provider owns wrapper-created reservations; manual reservations remain user-owned.

Generic `claim_command`/`release_command` pairs have no verified, scoped reservation receipt.
Automatic use is rejected **before any claim**, including preview. Existing profiles remain
readable but must remove that pair. This is an explicit conservative unsupported boundary,
not an invented gpuctl ownership protocol. A reservation-scoped provider integration needs
authorized real-site validation; do not infer ownership from GPU activity or a matching UUID.

Exact `[gpu, exec]` derives `[gpu, env, --shell]`. Explicit custom `visibility_command` argv
is unchanged. The parser accepts one CSV token line or one explicit `CUDA_VISIBLE_DEVICES`
assignment, never executes output, rejects ambiguity and bounds its size. Confirmed empty
grants differ from unavailable probes. Cross-format comparison needs pinned identities;
ambiguous numerical/UUID mappings do not trigger false revocation. Grants can only shrink or
restore original tokens. Healthy siblings are not cancelled on observer failure.

Study recovery establishes a fresh grant for the new Job. In `command` mode (`gpu exec`),
an authored three-GPU Study can resume with one currently granted GPU, without editing YAML,
reusing the old device list or replaying completed evidence. Numeric `runs_per_gpu` limits
apply to the new count; the physical-memory/resource gates still apply. This is covered with
native adaptive, repeated and fixed-sweep recovery using synthetic hardware. Explicit empty/
disabled `CUDA_VISIBLE_DEVICES` fails closed in every mode, rather than inventing physical
indices. CPU-only dispatch remains valid with GPU visibility disabled. Scheduler allocations
remain exact contracts; application-specific distributed topology is not automatically rewritten.

<a id="guide-resource-safety-remaining-acceptance"></a>

### Remaining acceptance

Remaining work: phase-aware startup beyond the bounded learned window, online RAM demand for
unauthored budgets, process-pressure backpressure, complete historical interval/confirmed hardware
event reports, native reconciliation of ambiguous legacy leases and extended provider acceptance.
Direct outer leases are now host/UUID scoped and locally regression-tested; that does not
certify an actual shared-home/site deployment. Existing throughput/resource intelligence is retained, not FIFO
or a one-Run-per-GPU policy. Initial baseline establishment still requires all requested device
identities; optional-binding/MIG and real-site preemption/claim cleanup need explicit validation.
CPU synthetic gates compare static/dynamic packing and terminal/live cold start. A cache
microbenchmark measures local coordination with a fake backend, not NVML latency or real science.
Competition tests use independent native processes.

Primary references: [official NVIDIA binding](https://pypi.org/project/nvidia-ml-py/),
[CUDA visibility/order](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html),
[NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
[CiTIUS reservation manual](https://wiki.citius.gal/en:centro:servizos:servidores_de_computacion_gpgpu).


<a id="guide-developer-experience"></a>

## 32. Developer experience — incremental implementation

[Español](MANUAL.es.md#guide-developer-experience)

This development checkpoint adds native ergonomics without changing scientific identity, YAML
meaning, recovery, GPU scheduling or the uncommitted Study-extension implementation. It is **not
the completed broad UX reform**. No release number is changed. Read the acceptance ledger for
tested/unfinished boundaries.

<a id="guide-developer-experience-start-and-discover"></a>

### Start and discover

```bash
lf init my-research --template minimal
cd my-research
python -m pip install -e .
lf validate experiments/work.yaml
lf explain experiments/work.yaml
lf run experiments/work.yaml --dry-run
lf help --search "recover study"
lf help --search "recuperar estudio" --json
```

Init derives distribution, import package, YAML, README and lightweight test names from the
directory. Unicode is transliterated where possible; invalid/empty names receive a safe fallback,
numbers get a package prefix, and Python keywords/stdlib/framework shadows receive a suffix.
Old projects are not migrated. Overwrites require existing explicit `--force`; symlinks are refused.
Agent/knowledge setup and dependency installation remain separate opt-ins. The dry-run does not
execute Work.run. Generated instructions name the actual files; no checkout installation path is
invented. Pytest is needed only to run generated tests, not scientific Work execution.

Task help projects the installed argparse grammar and reuses the native bilingual discovery
vocabulary. Results contain real command usage, not guessed services. Standard `lf help`,
`lf --help`, and nested help remain supported. Misspellings suggest actual grammar choices.

Shell completions are generated from that same grammar, including subcommands, flags and enum
choices. They never load projects, query clusters, read secrets or import Torch:

```bash
source <(lf completion bash)
source <(lf completion zsh)  # after the shell's usual compinit
lf completion fish | source
```

Choose the command appropriate for your shell. Nothing edits your startup files. Dynamic cluster/
dataset names are deliberately not expanded; filename fallback is owned by the shell. Bash is
syntax-tested locally; live Zsh/Fish interaction remains an acceptance gate.

<a id="guide-developer-experience-understand-yaml-before-running"></a>

### Understand YAML before running

```bash
lf config schema
lf config schema --output work.schema.json
lf config explain experiments/work.yaml --field execution.max_parallel
lf config diff experiments/previous.yaml experiments/work.yaml --json
lf config resolve experiments/work.yaml --format json
```

Schema output is the installed schema, not another grammar. `--output` creates an explicit new
file and never overwrites it. Associate it with YAML in your editor yourself; no VS Code settings
are changed. Field explanation reads bounded YAML/schema metadata. Execution limits are normalized
by the native ExecutionPolicy, without consumer imports. `with` parameter names/types still belong
to Work.run; use existing native `lf explain`/`validate` with trusted project imports for those.

The schema now documents concurrency/budgets and matches native null/case-insensitive `auto`
concurrency values. It does not claim complete schema parity for every legacy resource alias.

Semantic diff uses WorkConfig normalization and the existing pure extension comparison's canonical
JSON helper; it does not change that extension service. It reports parameter, seed, policy, resource,
analysis and product differences. It does **not** establish recovery/extension compatibility,
dependency equivalence or byte identity of two path declarations. It imports trusted Work code,
but never calls Work.run. Native `lf explain` now distinguishes experiment totals, allocation and
admission ceilings: requested concurrency is not guaranteed future GPU availability.

<a id="guide-developer-experience-inspect-diagnose-and-observe"></a>

### Inspect, diagnose and observe

```bash
lf inspect job-EXACT --json
lf inspect execution-EXACT --kind execution
lf inspect NAME --kind work
lf diagnose job-EXACT --json
lf watch job-EXACT --refresh --interval 2 --count 10 --json
```

Exact prefixes select native identities. Ambiguous names across Work/Study/Execution/Job require
`--kind` or an exact ID. First scope does not include arbitrary Run/product/dataset selectors.
Results use compact ResultStore catalogues, native Work grouping and JobService/OperationalHealth;
no artifact/model/checkpoint bytes are opened. `--refresh` deliberately requests provider metadata
through the existing observation gate. Without it these commands show persisted observation age,
not a fictitious current provider state.

Diagnose includes native persisted diagnostics, confirmed recorded failures and recent native audit
events. It does not guess possible causes or certify recovery from FAILED/UNKNOWN. Use the reported
`lf retry JOB --dry-run --json` to inspect actual ownership/checkpoint eligibility. It never retries
automatically. Deep scientific traceback/Run analysis remains explicit through existing commands.

Watch is a foreground observer, not another permanent monitor. JSON is **NDJSON**: one valid bounded
document per changed snapshot. Unchanged observations back off up to 60 seconds; age alone is not a
new event. Ctrl-C stops only this observer, never the Job. It does not stream log bodies or detailed
per-Run resources; use existing `lf logs --follow` and saved resource reports for those.

<a id="guide-developer-experience-small-result-tables-without-a-full-experiment-export"></a>

### Small result tables without a full experiment export

```bash
lf results table execution-EXACT --metric val_accuracy --metric val_loss --json
lf results table execution-EXACT --trial 2 --columns seed,value,parameters --format csv --output scalars.csv
lf results table execution-EXACT --offset 100 --limit 100 --fingerprint sha256:PREVIOUS_PAGE
lf results table execution-EXACT --attempts all --json
```

The native `ResultStore.table` projects local retained or imported Execution metadata. By default
only the latest Attempt of each logical Run appears; `--attempts all` explicitly includes physical
history. Exact repeatable metric/Run/Trial selectors and optional columns narrow the table. Missing
values remain empty; pruned observations are `censored_snapshot`, not invented final scores.
Values are **persisted scalar snapshots**, not a claim about best epoch or selection aggregation.
Execution/Run/Attempt IDs, fingerprint, status and evidence kind remain mandatory provenance columns.
This operation reads the selected aggregate metadata, not series, log, model, checkpoint or artifact
payloads; it does not recompute StudyAnalysis or retrieve remote artifacts.

JSON carries the descriptive notice, row count, next offset and an evidence fingerprint. Supplying
that fingerprint on the next page rejects changed evidence rather than silently mixing snapshots.
CSV retains provenance on every row and prints pagination/interpretation guidance to stderr;
spreadsheet-formula strings are escaped and native secret redaction applies. Pages have at most
1000 rows and a bounded byte size; reduce optional columns for an oversized single row. Explicit
file output creates a new non-symbolic path outside native Run evidence and never overwrites it.
These tables are disposable analysis aids, not importable scientific authority. Complete native
experiment export/import remains unchanged.

<a id="guide-developer-experience-agent-setup-without-pretending-a-connection-exists"></a>

### Agent setup without pretending a connection exists

```bash
lf ai doctor
lf ai context
lf ai connect codex --json
lf ai connect codex --apply --plan-hash sha256:REVIEWED_PLAN
lf ai launch codex
```

`ai context` pages existing local YAML/AST Work metadata, clusters, knowledge status, installed
integration/SDK metadata and actual default READ/PLAN permissions. It does not import consumer
Python, read note bodies or probe clusters. Doctor adds bounded installed Codex version/help/
registry inspection. Missing or incompatible clients remain explicit and no installation occurs.

Connect uses the [official Codex MCP CLI](https://developers.openai.com/codex/mcp/), verifying the
installed add/get/list grammar before preparing a project-specific server name and the invoking
Python. Preview writes nothing. Explicit apply requires the current exact plan hash, compatible
optional SDK and native project lock; conflicting entries are refused, matching entries are
idempotent. The official client manages its own configuration; LambdaForge does not directly edit
TOML or overwrite another server. Independent non-LambdaForge client edits are not protected by
this project lock; actual-client config/comment preservation acceptance is still pending.

**Registry verification is not an MCP handshake or an agent conversation.** The result explicitly
reports that limitation. Use existing `lf ai mcp test` for the SDK test and Codex `/mcp` for actual
client discovery. SDK installation, client registration and launch are separate actions. Launch
requires TTY and existing exact registration, uses the pinned project cwd, passes no permission/
sandbox upgrades and launches no scientific work. No global Codex configuration was changed during
this implementation; registration is tested using disposable simulated client state.

<a id="guide-developer-experience-console-and-skills"></a>

### Console and Skills

Work/Study root tables have local name/state/cluster/exact-ID filters. Typing does not call the
provider. Filtering, opening and refreshing use the same filtered rows and exact entity identity.
Unchanged snapshots update freshness without clearing tables; changed snapshots retain scroll and
the existing selection logic. Removed widgets discard late presentation replies without stopping
science. Other domain screens keep their own layouts and controls.

The packaged Work and Study Skills now route through these real authoring commands and explain
design choices, missing/censored evidence, imports, explicit input types and the actual native
extension boundary. They never grant submission; consult the current native extension guide for
fixed-design, same-placement managed GPU and cumulative revision/analysis contracts.

<a id="guide-developer-experience-remaining-scope"></a>

### Remaining scope

Global paged entity search, broad table sorting/paging, complete consolidated recovery eligibility,
cross-Execution compatibility/paired comparison improvements, guided editor merging,
recipe-copy operations, rich report refinements,
larger Workspace extraction, comprehensive MCP/output-schema improvements and the other priority
Skills remain pending. No live GPU/SLURM/SSH, external agent conversation or complete backend suite
is claimed. Existing domain owners, public commands/JSON and unfinished incremental extensions
remain authoritative.


<a id="guide-integration"></a>

## 33. Project-pinned agent integration

[Español](MANUAL.es.md#guide-integration). **The full reform is not complete.**
[The acceptance ledger](ai/IMPLEMENTATION.md) identifies implemented foundations and remaining gates.
Nothing here changes Work identities, HPO, GPU grants, recovery or existing scientific records.

<a id="guide-integration-quick-start"></a>

### Quick start

```bash
lf init my-research --agents portable --knowledge markdown
# Existing pyproject project: preview first, explicit local operator apply:
lf ai --project /absolute/my-research init --agents codex --knowledge obsidian
lf ai --project /absolute/my-research init --agents codex --knowledge obsidian --apply
lf ai --project /absolute/my-research doctor
lf ai skills list
lf ai skills show lambdaforge-work
lf ai capabilities --limit 20 --json
```

Legacy init remains science-only. `--interactive` asks about agent/memory modes in a TTY.
No dependency is installed automatically. **Agents / knowledge…** in the Console uses the same
Python services, displays pending/error/success and confirms asset setup. Consumer instructions
teach public APIs, not framework internals. The root contributor guide is canonical; detailed
contracts were preserved in [FRAMEWORK_CONTRACTS.md](ai/FRAMEWORK_CONTRACTS.md). Spanish explains it.
24 packaged specialized Skills copy into `.agents/skills/`; `lf ai skills examples` exposes native
CPU fixtures for metrics, checkpoints, product promotion, HTML, repeated seeds and adaptive design.

<a id="guide-integration-mcp-and-clients"></a>

### MCP and clients

```bash
python -m pip install "lambdaforge[agents]"  # Explicit operator installation
python -m lambdaforge ai --project /absolute/my-research mcp serve
lf ai mcp test  # Actual SDK in-memory handshake; not external-client acceptance
```

The verified dependency range is official SDK 2.3. Stdio stdout is protocol-only. MCP tools,
input/output schemas, resources, prompts and version use the SDK. Tests additionally handshake
through a real stdio subprocess from a foreign cwd. Trusted consumer imports are isolated in a
fresh planning process, with timeout/cancellation cleanup and stdout directed away from protocol.

Codex mode generates `.lambdaforge/agents/codex-config.example.toml`, never edits global settings.
Review and merge its named entry into your client configuration, preserving existing entries:

```toml
[mcp_servers.lambdaforge]
command = "/absolute/my-research/.venv/bin/python"
args = ["-m", "lambdaforge", "ai", "--project", "/absolute/my-research", "mcp", "serve"]
```

Use the same command/argv with any standards-compatible stdio client; no shell wrapper needed.
Actual Codex 0.162.0-alpha.17.2 discovery and direct tool calls have been verified with an
isolated temporary client configuration, a foreign cwd and no model credentials. This is not
an LLM conversation; second vendor-client and conversational acceptance remain **pending**.
See [official Codex MCP](https://developers.openai.com/codex/mcp/) and
[Skill discovery](https://developers.openai.com/codex/skills/).

Fixed toolsets: core/work/study/results/products/datasets/clusters/operations/knowledge.
Core is always present; default core/work/knowledge. Restart the server to change toolsets.
Catalog discovery distinguishes implemented, session-enabled and unsupported. Listings are compact
pages; Study panels/logs are explicit provider reads, not recursive startup artifact downloads.
Resources identify project/capabilities/installed Work and Study contracts, project structure,
recipe inventory, exact results/products and a paged note index. Toolset-disabled resources are
not registered; no resource template accepts arbitrary paths. Thirteen short prompts cover
authoring, models, checkpointing, analysis/comparison, HTML, products, diagnostics, recovery,
evidence review, architecture decisions and next experiments. Prompts guide but never authorize.
Responses include schema/project/source/retrieval time/data/warnings/structured error. Retrieval
time is not fresh provider observation; native evidence retains its own age/state.

Default READ/PLAN does not permit submission, operation, deletion or agent writes. Validation imports
project Python and requires operator `--allow-project-code`. Isolation is **not a sandbox** against
trusted code or an agent with independent same-user shell access. Submission/recovery/deletion
remain disabled as agent actions. Exact Job cancellation requires OPERATE, operations toolset and
separate local human review. Approved knowledge edits and paired loopback HTTP are opt-ins. Notes, prompts,
model answers and `approved=true` are not authorization. No arbitrary shell or read-file tool exists.

<a id="guide-integration-local-human-approvals"></a>

### Local human approvals

Exact Job cancellation uses the same signed approval service, not a second authorization engine.
The stdio operator selects `--toolsets core,operations --permissions READ,PLAN,OPERATE`.
`prepare_job_cancellation(job_id)` reads saved metadata only, without a provider query or write;
`request_job_cancellation(job_id)` proposes but does not stop anything. The human reviews the
exact project/Job/scheduler/provider binding and scope through the same local CLI/Console below.
Then `apply_job_cancellation(request_id, plan_hash)` reobserves, revalidates ownership and invokes
native `JobService.cancel`. It affects that exact Job, not every Job in the semantic Work, and
retains scientific evidence. Unknown ownership/changed plans block apply; Fleet is explicitly
unsupported. `job_cancellation_status(request_id)` returns signed state and the saved native
outcome. A crash/failure after claiming never grants replay. Knowledge permissions/approval
cannot authorize cancellation; no MCP endpoint approves.

The v1 response envelope remains compatible. Declared payloads have operation-specific JSON
schemas validated by the facade and advertised by the SDK's public output-model contract.
Tool failures set `isError=true`, with distinct approval-required/expired/stale/conflict,
provider-uncertain/unreachable and internal-error codes. `retryable` is never permission to
repeat a mutation. Deterministic English/Spanish intent discovery supports, for example,
`lf ai capabilities --search "recuperar estudio" --limit 20 --json`; relevance is not scientific
confidence. Catalog metadata identifies native owner, permissions, preconditions, costs,
freshness, errors and actual next tools; it does not certify provider configuration.

For note actions the stdio operator enables `--permissions READ,PLAN,WRITE_PROJECT`.
Discover `agent_actions`: its knowledge subset contains `knowledge.create`, `knowledge.update`, `knowledge.link` and
`knowledge.export` (a notes-only ZIP, not scientific export).
Use `preview_agent_action`, then `request_agent_action` with exact arguments. Neither writes a note.
A human reviews that request through **Agents / knowledge… → Review agent requests…**, or:

```bash
lf ai approvals list
lf ai approvals show approval-EXACT
lf ai approvals approve approval-EXACT --plan-hash EXACT_HASH
```

The CLI requires a real local TTY and the typed full request ID. Console shows every argument and
requires the same typed confirmation. Approval does not apply the edit: the agent calls
`apply_agent_action(request_id, plan_hash)` separately, within the requested expiry (maximum
15 minutes). Exact project, arguments, target revisions and plan hash are revalidated. Signed
records, a cross-process claim and metadata audit reject forged/stale/replayed/concurrent requests.
A crash after claiming is not permission to replay. No MCP tool can approve; a note saying the human
approved grants nothing. This does not sandbox another program with unrestricted same-user shell.

<a id="guide-integration-optional-dependencies-and-paired-http"></a>

### Optional dependencies and paired HTTP

```bash
lf ai dependencies check
lf ai dependencies install                 # exact preview, no installation
lf ai dependencies install --apply         # invoking virtualenv/Conda only
lf ai dependencies install --wheelhouse wheels --apply  # offline, project-owned wheels
lf ai init --agents portable --apply --install-deps
lf ai mcp token --apply                    # private pairing file; never prints its token
lf ai mcp serve --transport http --port 8765
```

Only the declared `agents` extra is installable; no arbitrary package/command, system Python,
desktop application or cluster installation. Failed installs preserve project files/notes and
report bounded diagnostics with a failing exit status. `init --install-deps` is explicit consent.
HTTP uses the **same official SDK and services**, restricted to READ/PLAN, `127.0.0.1` and an
exact Host. Pair a trusted local client using `Authorization: Bearer TOKEN` from the private
`.lambdaforge/agents/http/token` file; never embed it in a URL. Browser clients also need an exact
operator `--origin https://your-local-client.example` allowlist. Every request rechecks the token
and project. Requests are bounded, forwarded headers untrusted and non-loopback peers refused.
This is a local capability token, **not an OAuth server**: public/network deployment is unsupported.
Stdio remains recommended and never needs this token.

<a id="guide-integration-portable-knowledge"></a>

### Portable knowledge

```bash
lf ai knowledge init --apply
lf ai knowledge create hypothesis "Does width improve score?" --author-type ai --apply
lf ai knowledge search width --kind hypothesis
lf ai knowledge read EXACT_NOTE_ID
lf ai knowledge update EXACT_NOTE_ID --revision CURRENT_SHA256 --body "Revised draft" --apply
lf ai knowledge reindex --force
lf ai knowledge reindex --rebuild  # Explicitly rebuild a corrupt derived index
lf ai knowledge audit --limit 20  # Follow the returned byte next_cursor with --cursor
lf ai knowledge export notes-snapshot.zip          # exact preview, no archive created
lf ai knowledge export notes-snapshot.zip --apply  # explicit local operator export
lf ai knowledge obsidian                 # passive hints only; never starts the app
lf ai knowledge obsidian --probe-cli --apply  # explicit version probe may start Obsidian
lf ai knowledge open                     # preview the owned README URI
lf ai knowledge open --apply             # explicit desktop request, when available
```

Markdown is canonical; SQLite FTS under `.lambdaforge/agent-knowledge/` is disposable. Eleven kinds:
decision/hypothesis/experiment/finding/concept/architecture/issue/research-question/procedure/
literature/task. Schema-1 frontmatter retains human/AI author, timestamps, project, tags, links,
status and exact evidence references. AI hypotheses are proposed, never verified conclusions.
Use Python `KnowledgeService.create(..., evidence_refs=[{"kind":"execution","id":EXACT_ID}])`;
products/datasets use SHA-256 content IDs. `verify_references` takes a native resolver; existence
does not verify interpretation. Retrieved text is untrusted data, never permission or instructions.

Search supports title/text, kind/status/tag/evidence/date filters, pages and exact backlinks.
`knowledge_related` returns explicit links/shared tags/exact evidence, not a truth score;
hypothesis/decision/open-question tools give bounded views. `knowledge_verify_references` resolves
local registered identities through native owners, never downloading artifacts or verifying a claim.
External edits refresh within a bounded five-second discovery interval, or immediate reindex.
Unchanged note bytes are not reread. A corrupt index gives an actionable rebuild diagnostic;
rebuild retains its old copy and never changes notes. `knowledge link` requires exact existing
note IDs and the current source revision. Hash revisions reject stale native writes; external editors
do not participate in the lock. Corrupt notes/duplicate IDs are excluded with warnings, not deleted.
Native create/update/link operations record intent and completion, note identity and before/after
revisions in a metadata-only journal; no note titles/bodies are copied. Interrupted intent is not
success. Byte-cursor paging never loads the full journal. External editor changes are not attributed
to a person or silently called verified; the Markdown revision remains authoritative.
The journal is `.lambdaforge/agents/knowledge-audit.jsonl`, separate from the disposable FTS cache;
rebuilding/deleting the search index never deletes history. A malformed journal requires explicit
local inspection and prevents further native edits rather than silently discarding audit evidence.

Knowledge export validates every canonical note and creates a compressed ZIP with exact bytes and
a checksum manifest. It refuses corrupt/duplicate notes, stale plans, symlinks, unsafe destinations
and existing archives; failures clean only owned temporary files, never notes. Choose an existing
project-relative parent outside the vault. The bounded snapshot includes neither scientific data
nor indexes, plugins, desktop settings or unrelated Markdown; it is not `lf export STUDY` or a
recovery package. MCP previews with `knowledge_export(destination)`; actual export uses the same
separate human approval workflow for action `knowledge.export` and binds the entire note inventory.
Obsidian mode uses the same Markdown and an optional empty .obsidian directory: no app/plugins/CLI
required, existing preferences preserved. External vault configuration and semantic/hybrid search
remain pending. Text retrieval works offline without an LLM key.

`knowledge_obsidian_status` only inspects configuration/PATH/desktop hints; installed/running/
compatible are **not** assumed. Explicit local CLI probes use bounded `obsidian version`, and
may launch the app as explained in the [official CLI documentation](https://obsidian.md/help/cli).
`knowledge open --apply` uses the [official encoded open URI](https://obsidian.md/help/uri) for
the exact owned README. Register that folder as a vault manually first. A successful opener is
only `open_requested`, not proof of an open window; headless/missing/disabled/timeout cases return
actionable hints. No app/plugins/settings are installed or changed, and no MCP desktop-launch tool
is exposed. These helpers are fixture-tested, not a claim of real Obsidian desktop acceptance.

<a id="guide-integration-python-gui-and-sync"></a>

### Python, GUI and sync

```python
from lambdaforge.agents import AgentServices, IntegrationService

service = AgentServices("/absolute/my-research")
project = service.project_info()                 # typed AgentResponse
notes = service.knowledge_search("width")        # bounded interpretations
result = service.get_result("execution-EXACT")   # native metadata-first evidence
plan = IntegrationService("/absolute/my-research").preview()
```

The facade pins its project regardless of cwd and rejects secret/traversal/symlink paths.
Widgets never own scientific records/provider logic. `examples/agent_gui.py` is an optional tiny
nonblocking Tk client, not a chatbot or vendor-specific LLM integration.
`lf ai sync` previews; `--apply` checks managed hashes and protects personalized AGENTS/Skills.
`.agents/LAMBDAFORGE.md` is a separate managed reference when custom root instructions exist.
Plans bind project/version/paths/content hashes and reject changes. Notes and global client settings
are never part of sync. This local operator command is not the future agent approval protocol.

Static model discovery now resolves public exports to installed constructor/forward signatures
and docstrings without importing Torch. Declared tensor semantics come from their owner,
not invented or inferred. Recipes use installed native examples. Product contract/member/producer/
consumer queries are metadata-only; selected Run curves/logs and cluster resources are explicit
bounded native reads. A text artifact preview is explicit, ownership/integrity checked and 64 KiB.

Tests cover protocol/stdio/resources/prompts, isolation, malformed inputs, external note edits,
revision conflicts, duplicates, assets and one deterministic native CPU Work with durable evidence.
The reference Agent Skills validator checks all 24 packages. Approved-note and loopback HTTP
protocol/security tests pass. Remaining scientific actions, semantic retrieval, complete domain
tool/resource/prompt coverage, second vendor-client E2E and full GUI
conversations remain explicit gates in the ledger.

Standards: [MCP](https://modelcontextprotocol.io/specification/latest),
[official Python SDK](https://py.sdk.modelcontextprotocol.io/),
[Agent Skills](https://agentskills.io/specification).
