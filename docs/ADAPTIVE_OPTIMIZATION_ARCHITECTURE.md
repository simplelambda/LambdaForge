# Adaptive Experiment Optimizer: architecture and migration note

This note records the design review performed before implementing adaptive optimization. It is an
engineering document; the user-facing contract lives in the repository README and packaged Schema.

## Existing architecture reused

LambdaForge already separates `ExperimentConfig`, `ExperimentValidator`, `ExperimentRunner`,
`ExperimentExecutor` and `TrainingOrchestrator`. A materialized run owns a stable scientific
fingerprint, seed-specific directory, atomic `RunResult`, attempt history, dense `metrics.csv`, full
Lightning checkpoints and cooperative process cancellation. `LightningRunner.fit(..., ckpt_path=)`
restores model, optimizer, scheduler, precision and loop state. The orchestrator launches importable
workers with `spawn`, assigns logical CUDA devices relative to the parent-visible set and bounds
shutdown. `ResultCatalog`, aggregation and retention remain the sources of truth for terminal runs
and artifacts.

The existing `sweep.grid` and ablation expansion is deterministic but static. The small
`RandomSearch`/`OptunaSearch` APIs generate configurations but do not own partial budgets, seeds,
resources, asynchronous actions or durable controller state.

## Implemented architecture

`AdaptiveExperimentOptimizer` is an alternative orchestration path selected only by
`hpo.enabled: true`. It does not replace the experiment runner. The controller chooses typed
`AdaptiveAction` values (`START_NEW`, `RESUME`, `ADD_SEED`, `DROP`, `CONFIRM`); the training backend
materializes each accepted action as an ordinary LambdaForge run and delegates it to the existing
runner. Increasing fidelity changes the current `trainer.max_epochs`, while fingerprint
canonicalization uses the declared maximum fidelity so a later attempt can safely resume the same
configuration and seed from its last checkpoint instead of recomputing earlier epochs.

Responsibilities remain independent:

- `SearchSpace` encodes/decodes continuous, integer, categorical and conditional parameters.
- searchers propose new configurations; Sobol is dependency-free through PyTorch, random is the
  baseline, and `BoTorchSearcher` encapsulates optional GP/acquisition APIs, deterministic
  scored-observation refresh buckets, an in-process fitted-model cache and safe fallback;
- `LearningCurveModel` predicts final performance and uncertainty from all epoch observations;
- seed policies compare shared seeds and request repetitions only when ranking uncertainty warrants;
- cost and memory models learn conservative estimates from completed actions and censored OOMs;
- `ResourceAdmissionController` applies logical budgets and feasibility probabilities;
- `UtilityAwareScheduler` performs deterministic utility-aware CPU/VRAM packing;
- `AdaptiveActionSelector` compares information value per predicted cost times feasibility;
- `AdaptiveOptimizerState` is the atomic, versioned replay source for proposals, curves, pending and
  completed actions, RNG counters, drops, confirmation and resource observations;
- structured decision/event logs explain every selected and rejected alternative;
- `AdaptiveRunMaterializer`, `AdaptiveExperimentWorker` and `AdaptiveObservationReader` form the
  narrow backend bridge, so controller mathematics does not depend on Lightning objects.

Custom policies use these same duck-typed boundaries rather than concrete-class checks. In
particular, every seed policy receives `(state, learning_model)` and every fidelity policy may expose
an optional `dominated(state, learning_model)` method. This makes `hpo.components` substitution real
without forcing third-party classes to inherit LambdaForge implementations.

No component requires MIG, MPS, `nvidia-smi`, daemon configuration or physical GPU identifiers.
Configured capacities are preferred on restricted clusters. When discoverable, PyTorch reports
memory for logical visible devices. A child-only allocator fraction is a defensive ceiling, not
physical isolation. Preflight is an optional isolated task contract and learned admission never
silently changes a scientific batch size.

## Affected modules

- `lambdaforge.hpo`: adaptive domain, policies, models, controller, runner and optional BoTorch
  adapter and project-local component overrides;
- `experiments`: HPO dispatch, fidelity-safe fingerprints, validation and result integration;
- `training`: dynamic job supply and child memory instrumentation while preserving existing static
  scheduling;
- Schema/example/CLI: strict optional `hpo` block and adaptive inspect/dry-run/result envelopes;
- plugins: stable extension contracts for search, fidelity, seed and resource policies;
- documentation/tests: synthetic objectives, slow starters, seed racing, feasibility, packing,
  persistence, asynchronous dispatch and real checkpoint continuation.

## Compatibility and migration

The experiment Schema remains `1.1` because `hpo` is a wholly optional additive section. Documents
without enabled HPO follow the existing expansion/execution/aggregation path byte-for-byte. Grid
sweeps and static seeds remain supported and are not implicitly combined with adaptive search.
Existing run/result formats remain readable. Adaptive controller files live under a separate
suite-local `.lambdaforge/adaptive/` directory and reference, rather than replace, ordinary run
results and checkpoints. Infrastructure settings, live fidelity targets and controller bookkeeping
do not alter scientific identity; sampled hyperparameters, dataset/model/training semantics and the
actual seed do.

BoTorch/GPyTorch are optional dependencies behind LambdaForge interfaces. Numerical or dependency
failure produces a recorded Sobol/random feasible fallback instead of terminating the study. Python
3.10 uses the last compatible BoTorch minor; newer Python may use the current supported series.

## Runtime sequence and failure semantics

`AdaptiveExperimentOptimizer` acquires the ordinary suite activity lock, optionally runs the
consumer's representative CUDA preflight in isolated children, reconciles pending actions whose
canonical result already exists, then starts `TrainingOrchestrator.run_dynamic`. For each free slot,
the controller applies budgets and phase transitions, creates candidate actions, scores them and
atomically registers exactly one pending action before launch. The worker applies any allocator cap,
delegates training and always attempts to write peak telemetry. The parent reads canonical
`result.json`/`metrics.csv`/telemetry, completes the action and saves state before asking for more
work. Thus an observation is visible to the next decision and asynchronous completion order is part
of the recorded replay history.

A controller interruption leaves dispatched work in `pending_actions`. Relaunch first checks its
deterministic run path: terminal evidence is incorporated once, while genuinely incomplete work is
eligible for checkpoint continuation. Dependency/model failure in Bayesian proposal is a search
fallback; child OOM is censored memory evidence; other child failure is a failed observation. None
is silently relabeled as scientific pruning. Cooperative host cancellation remains a normal
orchestrator stop and persisted state stays resumable.

The current adaptive backend schedules one independent training process per logical slot. DDP
inside an adaptive action is not yet a supported resource contract because a per-rank allocator cap
and group reservation must be enforced consistently; static LambdaForge DDP remains available.
