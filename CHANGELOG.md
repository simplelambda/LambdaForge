# Changelog

[English](CHANGELOG.md) | [Español](CHANGELOG.es.md)

All notable LambdaForge changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The repository currently has no Git
tags; the 0.1.0 and 0.2.0 entries below were reconstructed from their version commits and packaged
metadata rather than invented release numbers.

## [Unreleased]

## [0.6.0] - 2026-08-13

### Added

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

[Unreleased]: https://github.com/simplelambda/LambdaForge/compare/v0.6.0...HEAD
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
