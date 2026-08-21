# Changelog

All notable LambdaForge changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The repository currently has no Git
tags; the 0.1.0 and 0.2.0 entries below were reconstructed from their version commits and packaged
metadata rather than invented release numbers.

## [Unreleased]

## [0.10.1] - 2026-08-21

### Added

- Target-aware placement reconciliation with explicit `AVAILABLE`, `REGISTERED_BUT_MISSING`,
  `DISCOVERED_UNREGISTERED`, `CONFLICT`, `ABSENT` and `UNREACHABLE` states.
- Preview-first `datasets reconcile`, target-aware `datasets show --on`, bounded managed-root
  manifest discovery and degraded-discovery warnings for tolerant `datasets list --all` calls.
- Local failure-injection regressions covering stale indexes, identity conflicts, offline targets,
  corrupt registries, active consumers, path escape, preview purity and interrupted deletion.

### Changed

- `show`, physical inspection, `verify`, `materialize`, replication and deletion now derive target
  reality through one immutable-manifest-backed placement resolution path.
- `remove` is registration-only; `delete` is the strict physical operation and validates the exact
  DatasetArtifact, managed-root containment and exact-version consumers before `--apply`.
- Remote registries retain only their own placements; controller indexes merge those observations
  by immutable identity without treating either index as stronger than physical content.

### Fixed

- In target-aware operations, a stale local record with no placement can no longer mask an exact
  placement held by the target Registry, eliminating contradictory verify/materialize/delete
  answers.
- Missing, corrupt, unreadable and unreachable registry states are distinct: only a missing file is
  an empty new Registry, and target connectivity failures are never interpreted as absence.
- Physical deletion and subsequent target/controller index cleanup are idempotent, so retries safely
  converge after either registry update fails without deleting unrelated paths.

## [0.10.0] - 2026-08-21

### Added

- A research-oriented experiment read model derives revision, target, state, progress, attempts and
  executions from project YAML, durable jobs and existing results without adding another database.
- `experiments runs`, revision-aware experiment inspection, semantic `overview.work` snapshots and
  the default research-centric `lf top` view; `v` retains direct access to raw jobs.
- `lf explain CONFIG` summarizes materialized scientific intent without constructing consumer
  targets, and common Adam/AdamW/SGD optimizer authoring compiles to the strict object form.
- Frozen 0.9.2-shaped job and `wisdom-dna@1` registry fixtures protect legacy read models plus
  dataset list/show/resolution/consumption and prove compatibility reads never mutate state.

### Changed

- Active experiment submissions are deduplicated by scientific identity and target before remote
  preparation. Another cluster remains valid and `--allow-duplicate` is the explicit escape hatch.
- Dataset selectors are exact and consistent: list output is accepted by show, and unversioned
  selectors with multiple versions fail instead of choosing the most recent record.
- `--rerun` names deliberate terminal repetition (`--force` remains compatible); retry is limited
  to failed, cancelled or timed-out attempts and increments the durable attempt number.
- Configuration name, kind, identity, datasets and planned units now come from one descriptor used
  by submission, project discovery, diagnostics and research views.

### Fixed

- Async retry recovers the controller-side YAML from new metadata or the persisted 0.9.x submission
  request instead of attempting to open the remote staged configuration path locally.

## [0.9.2] - 2026-08-21

### Added

- Durable append-only job lifecycle events now preserve preparation phases, scheduler transitions
  and liveness separately from scientific output; `jobs logs --follow` reports quiet provider
  observations instead of appearing frozen.
- Workflow node start/end/blocking messages and throttled generic preprocessing checkpoints provide
  useful progress even when consumer code has no logger.
- `lf top --history SECONDS` now has a compact whole-cluster overview and an Enter-driven cluster
  detail with paired vertical history charts, prominent personal usage, cluster-filtered jobs,
  separate job runtime/age and a complete non-blocking scrollable log viewer.
- Machine snapshots expose per-job timing/usage and per-cluster current-user LambdaForge resource
  aggregates for GUI and automation consumers.

### Changed

- `lf run` is now the documented canonical entry point for dataset recipes as well as tasks,
  workflows and experiments; `datasets build` remains a selector-oriented compatibility alias.
- Workflow and dataset YAML resources now determine the real fixed scheduler reservation. An
  explicit top-level request is exact; otherwise concurrent stage/node requests are safely
  aggregated, and CLI flags remain optional explicit overrides.
- `lf top` navigation now crosses directly between the cluster and job lists with vertical keys;
  job cancellation uses a visible non-blocking confirmation instead of reading a hidden prompt.
- Repeated managed submissions reuse an immutable, previously verified Torch plan when current
  Python, architecture, driver, compute capability and CUDA policy still match; SLURM staging uses
  copy-on-write bundle clones where supported with a portable copy fallback.
- Direct supervisors measure CPU from process-time deltas and aggregate RAM/thread use over the
  complete job process tree.

### Fixed

- Dataset file assets now use the conventional SHA-256 of file bytes during publication and
  verification, matching producer contracts such as WISDOM; early v2 filename-prefixed checksums
  remain readable for compatibility.

## [0.9.1] - 2026-08-20

### Changed

- `lf top` now isolates slow provider snapshots from terminal input, keeps navigation responsive,
  scrolls the selected row into view and cancels an outstanding refresh promptly on exit.
- Successful managed bootstrap activates the verified environment before pruning superseded
  LambdaForge-owned environments; the current environment and every known live-job reference are
  retained, and concurrent environment construction defers cleanup.
- Package and runtime metadata now consume one version constant from `lambdaforge._version`; user
  documentation uses version-independent wheel commands instead of duplicating the release number.

### Fixed

- Managed online installation no longer passes a nonexistent empty wheelhouse to pip.
- Consumer wheels whose declared LambdaForge requirement excludes the controller version now fail
  locally with the exact dependency declaration and repair steps, before SSH transfer or remote
  environment creation.

## [0.9.0] - 2026-08-20

### Added

- Durable asynchronous remote submission: CLI `run` and `datasets build` persist a `preparing` job
  immediately, then detach bundle construction, bounded-input hashing/transfer, environment setup
  and scheduler submission while preserving the same job ID and actionable pre-scheduler failures.
- A dependency-light interactive `lf top` view with cluster CPU/RAM/GPU summaries, selectable live
  jobs, recent logs and confirmed cancellation. `overview --json` now exposes the versioned snapshot
  and complete job rows; `top --json --follow` provides an NDJSON stream for wrappers.
- Deterministic host CA-bundle discovery and validation for LambdaForge-managed Python runtimes,
  with explicit propagation through Conda/Micromamba, pip/Requests and scientific job commands.
  Doctor now reports system and managed Python TLS trust separately.

### Changed

- Remote CLI submission no longer holds the user's terminal through slow preparation. Use the new
  `--wait-for-submit` escape hatch when synchronous provider acknowledgement is required; dry-runs
  remain synchronous and read-only.
- Global overview reconciliation avoids a duplicate scheduler refresh and includes honest
  `preparing`, `staging`, `queued` and provider states from the same services used by the TUI and
  programmatic interfaces.

### Fixed

- Remote execution bundles now discover, hash, copy and rewrite bounded local inputs declared in
  embedded dataset-recipe tasks, matching standalone and referenced task YAML behavior. Large
  embedded inputs fail during preflight with explicit DataCatalog/DatasetVersion guidance instead
  of becoming missing paths inside a hashed remote job workspace.
- Managed Conda/Micromamba Python no longer passes bootstrap while using an incomplete CA store that
  fails ordinary HTTPS on institutional clusters whose system Python succeeds.

### Security

- Managed trust inherits only a host-selected, readable and locally validated PEM CA bundle.
  Certificate verification remains enabled; LambdaForge neither downloads arbitrary CA roots nor
  modifies system trust, `/etc` or shell startup files.

## [0.8.1] - 2026-08-17

### Fixed

- Direct remote dataset builds now launch their detached supervisor with the exact LambdaForge
  Python interpreter even when the scientific command is wrapped by dataset environment variables
  and a site `command_prefix`. This prevents GNU `env` from receiving Python's `-m` option and also
  makes durable retries preserve the interpreter recorded in the original command.

## [0.8.0] - 2026-08-15

### Added

- One structured diagnostic model and central human/JSON/debug renderer covering configuration,
  validation, environment, execution, resource, connection, authentication, data, storage,
  deliberate refusal, warning, cancellation and unexpected internal failures.
- Atomic owner-readable diagnostic records with full tracebacks, invocation/job context,
  remediation and recursive secret redaction under the user state directory.
- Root-cause-first dataset/workflow job diagnostics, coherent category exit codes and global
  `--debug`, `--verbose` and error `--json` handling.

### Changed

- CLI and control-plane boundaries now explain what ran, what was preserved, why an operation
  stopped and the exact supported next command instead of exposing raw exception headings.
- `doctor` checks expose category, reason, fix and command fields, and warn when legacy scalar
  Python configuration unexpectedly disables managed-runtime fallback.
- Dataset-root preflight, placement, immutable-content conflicts, cache-GC refusals, Python runtime
  resolution, failed jobs and preprocessing debug failures now use actionable domain context.

### Security

- Terminal, JSON and persistent diagnostics redact password/token/API-key assignments, credential
  URLs, bearer headers, private keys and structured secret fields; diagnostic files use restrictive
  permissions where supported.

## [0.7.2] - 2026-08-14

### Added

- Managed cluster Python runtime resolution with explicit `auto`, `existing` and `managed`
  strategies, bounded interpreter discovery, consumer `Requires-Python` constraints and a
  preview-only `clusters bootstrap --dry-run` plan.
- Reusable user-space Conda-family runtime prefixes and a pinned micromamba fallback staged from the
  controller after local and remote SHA-256 verification. Offline wheelhouse profiles prefetch and
  transfer the target Python package cache instead of requiring cluster internet access.

### Changed

- New managed cluster profiles default to automatic Python resolution. Legacy `python: EXECUTABLE`
  profiles retain strict existing-runtime behavior until explicitly migrated.
- Environment identity, doctor output, storage accounting and reference-aware GC now distinguish
  the Python runtime, isolated package environment and installed framework/project packages.

### Security

- Managed Python never invokes sudo, edits shell startup files, activates a global Conda
  environment or changes drivers/system CUDA. Runtime and environment publication is verified,
  locked, staged and atomic; corrupt downloads and incomplete prefixes are not activated.

## [0.7.1] - 2026-08-14

### Changed

- Consolidated fragmented bilingual and package documentation into a brief landing README, one
  canonical user/maintainer manual and a compact operational agent guide.
- Reorganized release-named tests around stable behavioral contracts and added lightweight
  documentation, example, entry-point and package-version staleness checks.
- Grouped cohesive dataset, job, workflow and task models/errors and split CLI parsing/dispatch by
  domain while preserving documented imports, YAML and command behavior.

### Removed

- Historical audit documents, closed roadmap prose, duplicated translations, internal package
  READMEs and unused detached-run state left over from earlier development iterations.

### Fixed

- Managed cluster bootstrap no longer derives the LambdaForge project root from
  `lambdaforge.__file__`, which incorrectly searched for `pyproject.toml` under a normal virtual
  environment. Editable installs use their PEP 610 source, local wheel installs reuse the original
  artifact when available, and other wheel installs are repacked deterministically from verified
  installed code and metadata without requiring a package index.
- `doctor` and managed Torch resolution now reject a reachable but unsupported remote Python below
  3.10 before CUDA selection, wheel construction or transfer, with an actionable cluster-profile
  fix instead of a later pip failure.

### Security

- Installed-distribution repacking excludes bytecode and symlinks, rebuilds wheel `RECORD` hashes
  deterministically and preserves installed package metadata rather than executing remote source.

## [0.7.0] - 2026-08-14

### Added

- First-class `kind: dataset` recipes and durable DatasetBuild jobs compiled onto the existing
  Workflow DAG, with content-addressed stage reuse, granular downstream force and build plans.
- Streaming JSONL `DatasetIndex`, generic `DatasetMember`/`DatasetAsset`, arbitrary partitions and
  targets, bounded member inspection, logical diff and explicit schema validation.
- DatasetArtifact/Record v2 with path-independent content identity separated from build provenance,
  richer lineage/global assets and compatible v1 readers.
- Registry-first `DatasetResolver`, versioned references, exact reproducibility bindings and managed
  remote-bundle resolution without duplicated DataCatalog placements.
- Real NOOP/verified atomic REPLICATE/durable BUILD materialization, atomic final publication and
  reconstructible stage-cache GC.
- `lf` entry point, root `plan`, moderate resource/action aliases, shell completion, default-cluster
  preference, friendly job selectors and terminal PLANNED dry-runs.

### Changed

- Preprocessing is again an ordinary Task by default. Dataset publication is explicit through a
  recipe, `publish_dataset`, or compatible legacy `dataset_name`.
- DataCatalog now primarily describes aliases, external data, loaders, pins and overrides;
  DatasetRegistry is authoritative for managed placements.
- Dataset stats derive members/partitions/assets from the index and project profilers can execute
  beside remote data in the exact managed consumer environment.
- Job and top output distinguish running, queued, staging and paused states and human job lists now
  include stable headers, name, type, age and resource requests.

### Security

- Dataset publication and replication use verified staging before atomic rename/registration;
  immutable aliases cannot overwrite different content. Dataset deletion and normal cache GC remain
  exact-root and preview-first.

## [0.6.0] - 2026-08-13

### Added

- Generic `PostRunAction` lifecycle with explicit checkpoint roles, per-action required/optional
  policy, content-verified shared artifacts, independent fingerprints and interruption-safe receipts.
- Detached validation outputs for project callbacks, enabling one-forward streaming diagnostics and
  ordinary `val_*` metrics usable by checkpointing and adaptive HPO.
- Serverless terminal control plane with a detached per-job `ProcessSupervisor`, durable remote
  state/logs/heartbeats/usage, safe process identity/groups, runtime timeouts and recovery inventory.
- Complete job lifecycle commands for show, pause, resume, cancel, retry, local metadata deletion,
  reconciliation and persistent independent multi-cluster job groups.
- Direct-host CPU/RAM admission, CPU affinity, cooperative GPU leases, external GPU-process
  avoidance and `CUDA_VISIBLE_DEVICES`; separate direct and SLURM resource observations.
- First-class dataset records/placements, remote discovery, automatic preprocessing registration,
  universal and explicit classification stats, verification, lineage, remove/delete and
  deterministic NOOP/REPLICATE/BUILD materialization plans.
- Project YAML discovery and unambiguous config/experiment/task operations by name, global
  overview/top, storage reports and managed-environment inspection.
- Bilingual 0.5.3-to-0.6 audit and focused control-plane, job, dataset and storage guides.

### Changed

- Training success is durably committed before post-run actions; changing or recovering final
  analysis reuses the checkpoint instead of repeating fit. Actions are rank-zero and adaptive HPO
  defaults them to confirmation trials, never multi-fidelity pauses or cancellations.
- OpenSSH now enables private idle-expiring connection multiplexing by default. Connect,
  authentication, banner, keepalive and command timeouts are independent; scientific commands have
  no implicit transport deadline.
- `scheduler: local` is now asynchronous and durable. SLURM and direct jobs share `JobService`,
  while provider errors/offline clusters remain distinct from scientific failures.
- Cluster storage separates small state, immutable bundle/environment/package cache, mutable job
  work and optional dataset roots. Remote jobs no longer write scientific output into bundle cache.
- Managed environments build transactionally in a temporary directory, verify before atomic
  publication and share a pip download cache; legacy 0.5 environments/pointers remain readable.

### Security

- Pause/cancel/resume refuse mismatched or reused PIDs by checking creation time, process group and
  command hash. Inventory only accepts LambdaForge-owned matching request/state directories.
- Dataset physical deletion and storage GC are preview-first, locked and exact-root constrained.
  GC protects active references and can never select datasets, results or retained checkpoints.

### Fixed

- Restored the declared Torch 2.1+ compatibility for optional unsigned graph index dtypes and
  legacy device-specific autocast APIs; the full type check now passes on the minimum-era API.

## [0.5.3] - 2026-08-13

### Added

- Remote `CudaCompatibilityResolver` probing of configured Python/architecture plus NVIDIA driver
  and compute capability, with verified official PyTorch channel/wheel availability.
- Explicit cluster `pytorch.channel` and `require_cuda` policy, exact resolved Torch plans in managed
  environment identity/bootstrap output, and CPU/legacy/offline overrides.

### Changed

- Managed bootstrap pins Torch from a driver-compatible official channel before installing
  LambdaForge/consumer wheels and constrains dependency resolution against incompatible upgrades.
- Automatic channels use native toolkit driver floors (rather than assuming minor compatibility
  and its PTX caveats) for CUDA 12/13, while legacy capability may use the documented cu118
  compatibility floor when an appropriate remote-Python wheel exists and passes a CUDA probe.

### Fixed

- Generic `torch>=2.1` resolution no longer installs a newest cu130 build on GPU nodes whose older
  driver cannot initialize it. Cached environments now verify exact framework/Torch and required
  CUDA availability before reuse; doctor reports GPU driver facts and fails unavailable expected
  CUDA even without a GPU-requesting config argument.

### Security

- Automatic selection fails closed when driver/capability or a compatible wheel cannot be proven;
  it never mutates drivers/system CUDA, installs forward-compatibility packages, or silently falls
  back to CPU when CUDA is required.

## [0.5.2] - 2026-08-12

### Added

- Layered user/project/explicit cluster catalogs with source/conflict inspection, portable export
  and user scope as the safe default for new remote profiles.
- Optional password SSH through hidden interactive, OS-keyring or environment-reference credential
  providers and Paramiko SSH/SFTP with strict host-key verification and bounded timeouts.
- Per-cluster `SlurmProfile` customization for validated CPU/memory/GPU/time mappings, static flags/
  repeated directives, scheduler command argv/job-ID parsing and trusted job-script hooks.
- Credential set/delete CLI, detailed scheduler dry-run previews and expanded read-only doctor checks.

### Changed

- OpenSSH remains the recommended/default transport and now also accepts explicit user/port while
  preserving aliases, keys, agent, known_hosts and ProxyJump. Legacy cluster catalogs and
  `scheduler_options` remain compatible.

### Security

- Password values cannot enter cluster serialization, command-line flags, bundles, job records,
  fingerprints or logs; known transport errors are redacted. Scheduler templates use allowlisted
  placeholders and argv, while trusted prologue/epilogue never interpolate credentials.

## [0.5.1] - 2026-08-11

### Added

- Real preprocessing workload semantics: deterministic sequential mode, bounded threads for I/O,
  spawn-safe processes for CPU transforms with parent-owned sinks/manifests, conservative auto mode,
  single-worker GPU safety and isolated N-record stage debugging.
- Friendly training aliases for top-level name, singular loss, trainer epochs and portable resources;
  experiment `DataCatalog` support for direct and nested typed dataset references, subpaths and
  physical-path-independent scientific fingerprints.
- Exact managed cluster environments from local LambdaForge/consumer wheels, content identities,
  user-space idempotent venvs, offline wheelhouses, cluster registration/bootstrap and expanded
  Python/project/PyTorch/CUDA diagnostics without driver installation.
- Persistent job filters and log following, allowlisted lightweight remote result synchronization,
  explicit logical artifact listing/fetch and stored scientific/execution/environment identities.
- Public `ResultService`, normalized `MetricSeries`, human selectors, ambiguity reporting,
  comparison and JSON/CSV/optional Parquet export.
- Public renderer-neutral `PlotSpec` and `VisualizationService` for learning/seed/sweep/HPO/resource
  views, seed-aware uncertainty, atomic Matplotlib/optional Plotly output and reproducible sidecars.
- Separate artifact inspector/visualizer/schema/validator contracts; safe bounded NPY/NPZ and
  tabular inspection/export, explicit graph/point-cloud/mesh semantics and entry-point registry.
- Dataset manifest inspection, complete bilingual cluster/result/artifact/preprocessing/architecture
  guides and a synchronized Spanish agent manual.

### Changed

- Remote bundles now carry exact locally built project/framework wheels instead of relying on a
  remotely different checkout; managed environments live below the user workspace and reuse by
  complete wheel/Python/offline identity.
- Result, plot, artifact, data and job commands delegate to reusable application services. Legacy
  `lambdaforge results SOURCE` remains compatible through the `results audit` path.
- The root documentation now uses a hierarchical index, documents simple paths before provider
  details and records the 0.5.1 roadmap as complete while explicitly deferring automatic placement
  and distributed workflow execution.
- Packaging gained `viz`, `graph` and `viz3d` optional extras, ships all paired technical/agent
  guides and includes release build/check tools in `dev`.
- Preprocessing execution controls no longer change scientific/dataset identity; sweep metric
  normalization is explicit and comparisons require metric direction before naming best/worst.

### Fixed

- Windows architecture-reference persistence no longer calls `fsync` on a read-only descriptor.
- Dynamic scheduling observes and refills a completed slot before processing unrelated simultaneous
  completions, preventing slower queued work from being delayed by completion ordering.
- Dense epoch metrics CSV rewrites now use flush/fsync plus atomic replacement, so concurrent/live
  readers never observe a partially rewritten file.
- The package CI smoke now installs the wheel with dependencies in an isolated environment and runs
  from outside the source tree; the prior `--no-deps` smoke could fail on a correct distribution.

### Security

- NumPy artifact loading disables pickle, previews/statistics are bounded, geometry roles are
  explicit, remote sync is file/size allowlisted and artifact fetch rejects paths outside the job.
- SSH retains standard host-key/agent configuration; offline bootstrap fails closed on incomplete
  wheels and no cluster operation installs NVIDIA drivers or system CUDA.

## [0.5.0] - 2026-08-11

### Added

- A beginner-facing `AuthoringConfig` layer that compiles concise task, preprocessing, workflow and
  experiment YAML into the existing strict `MaterializedConfig`; `inspect --resolved` exposes the
  exact boundary and target strings are shortened only in unambiguous object fields.
- Named task inputs/outputs through `TaskContext.input()` and `TaskContext.output()`, logical
  `DatasetReference` values, environment-aware `DataCatalog` locations and pluggable strict,
  manifest, dataset-ID and explicit-version identity providers.
- Git/distribution/explicit `CodeIdentity`, separate scientific and execution identities, and
  `explain changes` so reuse decisions are auditable without making hashes part of normal UX.
- Explicit `--force`, `--restart` and `--no-resume` lifecycle controls, bounded preprocessing
  workers/workload intent and normalized portable CPU, memory, GPU, storage, duration and process
  resource requests.
- A provider-neutral local control plane with local/SSH transports, local/SLURM schedulers,
  content-addressed execution bundles, cluster/execution profiles, persistent job records and
  reconnectable status/log/cancel/retry services.
- `doctor`, `clusters`, `jobs` and preview-first `data` commands; explicit rsync dataset replication;
  cluster-aware `run --on`/`--profile`; and expanded project scaffolding templates.

### Changed

- Preprocessing may resolve logical input/output names instead of repeating physical paths; legacy
  strict YAML and path-oriented context methods remain supported.
- SLURM resource translation now includes process count and CPU cores per process.
- Human, agent and technical documentation now explain authoring, identity, idempotency, resource
  translation, remote execution, data placement and intentional distributed-workflow limits.

### Security

- Remote submission remains explicit, SSH keeps the user's normal host-key policy, argument vectors
  replace shell fragments, large implicit input transfers fail closed and data replication requires
  `--apply`.
- Mixed-cluster DAG execution is deliberately refused until durable coordinator recovery and
  artifact transfer can be guaranteed; placement remains visible in read-only plans.

## [0.4.1] - 2026-08-10

### Changed

- Hardened adaptive HPO around a real mixed multi-fidelity surrogate: categorical Hamming
  geometry, explicit conditional activity, ordinal parameters, all observed fidelity points,
  pending evaluations, cost-aware multi-fidelity KG and named safe-numerics fallback.
- Replaced recent-slope curve extrapolation with probabilistic Bayesian curves and corrected
  hierarchical seed uncertainty to combine between-seed variance with within-seed estimation
  variance exactly once. Seed racing and pruning now prefer paired shared-seed posteriors.
- Replaced heterogeneous `improvement + uncertainty` action scoring with a documented one-step
  Gaussian moment Knowledge Gradient approximation shared by START/RESUME/ADD_SEED.
- Replaced global empirical VRAM estimates with feature-aware conservative prediction, explicit
  UNKNOWN/UNBOUNDED/KNOWN capacity states, candidate-aware smart preflight and durable censored OOM
  lower bounds. The PyTorch allocator fraction remains a final defensive ceiling.
- Added deterministic synthetic hardening tests plus real single-GPU concurrent trials,
  candidate-aware probe, isolated OOM and cumulative checkpoint-resume coverage. Multi-GPU and
  SLURM execution remain opt-in infrastructure validations rather than simulated claims.
- Reworked the human documentation around a beginner-first path that explains YAML, Schemas,
  document types, object specifications, command side effects and output interpretation before
  presenting advanced subsystem contracts.

### Fixed

- Removed the redundant `1/n` factor that made within-seed uncertainty overconfident.
- Prevented missing memory discovery and exact zero capacity from being interpreted as unlimited.
- Constrained NumPy below 2.5 while Python 3.10 remains supported, because NumPy 2.5 type stubs use
  syntax unavailable to the project's Python 3.10 static-analysis target.

## [0.4.0] - 2026-08-09

### Added

- Action-centric asynchronous adaptive HPO with Sobol initialization, optional BoTorch GP/KG,
  dependency/numerical fallback, pending-point awareness, deterministic surrogate refresh/cache and
  cost-aware acquisition.
- Real checkpoint multi-fidelity continuation, conservative learning-curve pruning, shared adaptive
  seed racing and disjoint full-budget confirmation seeds.
- Logical/empirical VRAM admission, child allocator ceilings, peak/OOM telemetry, dynamic scheduling
  and atomic replay state plus structured decision events.
- Scientific summaries with search/confirmation seed usage, merged learning curves, memory evidence,
  uncertainty statistics and shared-seed paired confirmation differences.
- Strict additive HPO Schema, adaptive inspect/dry-run/result APIs, complete example and human/agent
  documentation. Existing sweeps and finite Random/Optuna search remain compatible.
- Uniform duck-typed HPO policy extension contracts plus release-hygiene documentation and ignore
  rules for controller/provider/build output.

## [0.3.0] - 2026-08-09

### Added

- Generic `kind: task` Schema, plans, runner, typed results, content-addressed inputs, artifacts,
  attempt history and task plugins.
- Composable, resumable and deterministically sharded preprocessing with dataset manifests.
- Workflow DAGs for task/experiment nodes, output bindings, cycle detection, branch isolation,
  resume through node identities, dry-run plans and bounded local concurrency.
- Explicit YAML `extends`/`include`, recursive merge/delete, safe config/environment/secret
  interpolation, per-value provenance and semantic configuration diff.
- CPU-only parallel experiment slots, oversubscription checks, portable resource requests/plans,
  deterministic packing and runtime/storage estimates.
- Execution backend boundary, synchronous local backend and preview-first SLURM scripts with arrays,
  dependencies, resources, environment, containers, requeue and explicit submission/cancellation.
- Failure taxonomy and bounded retry policy with attempt lineage.
- Reusable inference, evaluation, ensemble prediction and TorchScript/`torch.export`/ONNX/custom
  export tasks.
- Reproducible random HPO and an optional Optuna TPE/ASHA/Hyperband adapter.
- Immutable artifact references, verified local/shared and optional S3-compatible stores, plus a
  lease-coordinated distributed staging cache with corruption recovery.
- Catalog-backed experiment registry with tag/metadata/status filters and JSON/CSV/optional Parquet
  exports, cross-experiment comparison, objective Markdown/HTML reports and a static read-only
  dashboard.
- Structured JSONL events, bounded resource monitoring, PyTorch profiler adapter, reproducibility
  profiles, stable hierarchical seeds, scientific/infrastructure fingerprints and environment
  exports.
- CLI commands `init`, `compose`, `diff`, `explain`, `target`, `registry` and `dashboard`, plus
  transparent workflow dispatch through `validate`, `inspect` and `run`.
- Technical architecture, security/release policy, integrated product roadmap and expanded human and
  agent documentation.

### Changed

- Package version and public facade advance from 0.2.0 to the 0.3.0 development target.
- `execution.mode: parallel` now accepts either GPU slots or explicit `cpu_jobs`.
- The package description now covers declarative tasks, preprocessing and training rather than only
  training experiments.

### Security

- Composition never evaluates Python expressions; secret values are redacted by default and are
  rejected from persisted workflow structure.
- SLURM/local launch uses argument vectors or quoted generated scripts and never `shell=True`.
- Artifact stores validate keys, regular-file boundaries, sizes and SHA-256 before publication or
  staging.

## [0.2.0] - 2026-07-22

### Added

- Broad reusable model, loss, metric, component, graph, vision, sequence, tabular, tree, generative,
  uncertainty and scientific architecture catalog.
- Strict versioned experiment Schema, migrations, seeds/grids/ablations, dry runs, typed results,
  result auditing, aggregation, confidence intervals, paired comparisons and power estimates.
- Checkpoint selection/loading, retry history, ambiguity detection and preview-first transactional
  artifact retention.
- Persistent dataset caching with safe codecs, checksum/HMAC integrity, locking, recovery,
  namespaces and multiprocess quotas.
- Entry-point plugins, optional MLflow/TensorBoard/W&B tracking, environment provenance, improved
  CUDA/DDP/process cleanup, bilingual documentation and the token-efficient `AGENTS.md` manual.

### Changed

- Hardened packaging, public exports, CI across supported Python/OS variants and real-CUDA
  verification.

## [0.1.0] - 2026-07-16

### Added

- Initial object-oriented infrastructure for reproducible PyTorch training and YAML experiments.

[Unreleased]: https://github.com/simplelambda/LambdaForge/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/simplelambda/LambdaForge/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/simplelambda/LambdaForge/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/simplelambda/LambdaForge/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/simplelambda/LambdaForge/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/simplelambda/LambdaForge/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/simplelambda/LambdaForge/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/simplelambda/LambdaForge/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/simplelambda/LambdaForge/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/simplelambda/LambdaForge/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/simplelambda/LambdaForge/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/simplelambda/LambdaForge/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/simplelambda/LambdaForge/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/simplelambda/LambdaForge/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/simplelambda/LambdaForge/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/simplelambda/LambdaForge/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/simplelambda/LambdaForge/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/simplelambda/LambdaForge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/simplelambda/LambdaForge/compare/510b8e8d2ebbd76eb86dfcfa6fb309d1e6d680e6...v0.3.0
[0.2.0]: https://github.com/simplelambda/LambdaForge/commit/510b8e8d2ebbd76eb86dfcfa6fb309d1e6d680e6
[0.1.0]: https://github.com/simplelambda/LambdaForge/commit/4c1ddc985b681ad88c7b8e8962f20bdccec22a49
