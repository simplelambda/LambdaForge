# Changelog

All notable LambdaForge changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The repository currently has no Git
tags; the 0.1.0 and 0.2.0 entries below were reconstructed from their version commits and packaged
metadata rather than invented release numbers.

## [Unreleased]

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

[Unreleased]: https://github.com/simplelambda/LambdaForge/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/simplelambda/LambdaForge/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/simplelambda/LambdaForge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/simplelambda/LambdaForge/compare/510b8e8d2ebbd76eb86dfcfa6fb309d1e6d680e6...v0.3.0
[0.2.0]: https://github.com/simplelambda/LambdaForge/commit/510b8e8d2ebbd76eb86dfcfa6fb309d1e6d680e6
[0.1.0]: https://github.com/simplelambda/LambdaForge/commit/4c1ddc985b681ad88c7b8e8962f20bdccec22a49
