[English](ARCHITECTURE.md) | [Español](ARCHITECTURE.es.md)

# LambdaForge architecture

This document explains how the framework fulfils its public promises and where a maintainer should
make a change. The [root guide](../README.md) teaches use; this guide explains ownership. LambdaForge
is a library and CLI, not a hosted control service. The consumer project owns domain datasets,
models and policies.

## 1. Architectural boundaries

The dependency direction is deliberately one-way:

```text
CLI / LambdaForge facade
        ↓
application services (tasks, experiments, results, artifacts, control plane)
        ↓
immutable plans, records and public contracts
        ↓
provider adapters (Lightning, SLURM, SSH, S3, Plotly, BoTorch...)
```

Value objects validate at construction and serialize explicitly. Application services coordinate
work; providers translate an already-decided operation. YAML construction is recursive but only
explicit `target`, `ref` and `plugin` values import code. Public imports come from documented
namespaces; file locations are implementation details.

## 2. Configuration and object construction

`AuthoringConfig` is the beginner-facing compiler. It accepts concise forms such as `name`, one
`loss`, `trainer.epochs`, logical datasets and portable resources, then produces a
`MaterializedConfig`. Existing strict task Schema 1.0, experiment Schema 1.1 and workflow Schema 1.0
remain the runner IR. `ConfigurationComposer` resolves `extends`, `include` and restricted
interpolation before normalization. Validators check Schema, imports and constructor signatures
without constructing user objects. `ObjectFactory` performs trusted recursive construction only at
execution time.

This split prevents convenience syntax from multiplying runner branches: new authoring aliases
belong in `AuthoringConfigNormalizer`; scientific execution changes belong in the strict object that
owns them.

## 3. Tasks, preprocessing and workflows

`TaskConfig` owns identity, resolved content inputs, output layout and lifecycle policy. `TaskRun`
creates an immutable `TaskExecutionPlan`, executes one `Task`, verifies declared artifacts and
publishes one atomic `TaskResult`. Previous terminal attempts are archived rather than overwritten.

`PreprocessingTask` is a task specialization built from a source, ordered transforms and a sink.
Stable record keys drive sharding and resume. The parent process owns the sink and manifest. I/O
work uses threads; explicitly CPU-bound work uses spawn-safe child processes for transforms; GPU
work stays in one process. This avoids forked CUDA state and concurrent manifest corruption.

`Workflow` schedules complete task/experiment documents in a DAG. It delegates identity and resume
to each node runner. It does not distribute one DAG across clusters in 0.5.3; remote submission is
one explicit schedulable unit.

## 4. Experiments and training

`ExperimentConfig` normalizes and expands seeds, grids and ablations. It also resolves only typed
`DatasetReference` markers through `DataCatalog`; ordinary strings are never guessed to be paths.
`RunFingerprint` removes operational locations and replaces physical dataset paths with logical
references plus their declared identity.

`Experiment` coordinates expansion, execution and aggregation. `ExperimentRunner` owns a single
materialized run directory and its checkpoints/result. `LightningTask` adapts mapping batches to a
model, losses and metrics. `TrainingOrchestrator` owns process scheduling and immediately refills a
slot after observing completion. `EpochMetricsCSV` publishes dense learning curves with atomic
replacement so readers never consume a partial rewrite.

## 5. Adaptive optimization

`AdaptiveExperimentOptimizer` persists controller state and turns each decision into an ordinary
checkpoint-aware experiment run. `AdaptiveExperimentController` composes search, fidelity, seed,
curve, cost, memory, admission and action-selection contracts. The scheduler executes explicit
START/RESUME/ADD_SEED actions; CONFIRM is a separate scientific phase. State and events are the
recovery authority. See [the dedicated architecture](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md).

## 6. Explicit multi-cluster runtime

`ClusterCatalog` merges named `ClusterProfile` values across user/project/explicit scopes and keeps
winning-source provenance. `ClusterAuthentication` contains only mode/reference;
`CredentialService` selects hidden interactive, environment or OS-keyring `CredentialProvider`.
`ControlPlaneFactory` keeps OpenSSH as default and constructs `PasswordSshTransport` only for the
explicit Paramiko password mode. `SlurmProfile` composes `SlurmResourceMapping` plus safe
`SchedulerCommand` values, so resources are translated once before `SlurmScheduler` writes a job
script.

`ExecutionBundleBuilder` materializes the
configuration, selects target-specific logical data locations, builds exact LambdaForge/consumer
wheels and creates a content-addressed bundle. `EnvironmentIdentity` hashes installable wheel bytes,
Python policy and offline wheelhouse contents.

`ControlPlane` stages a bundle through `Transport`, asks an `EnvironmentProvider` for an exact
interpreter and submits through `Scheduler`. `ManagedEnvironmentProvider` creates an idempotent
user-space venv; `ExistingEnvironmentProvider` only verifies a user-managed interpreter. `JobStore`
persists reconnection data and `JobService` queries the scheduler on later processes. There is no
daemon, automatic placement, custom cryptographic password file, driver installer or cross-cluster
workflow coordinator. See [cluster operations](CLUSTERS.md) and its [security model](SECURITY.md).

## 7. Results, plots and artifacts

`ResultCatalog` only discovers immutable result envelopes and ambiguity. `ResultService` adds human
selectors, normalized `MetricSeries`, comparison and tabular export. `VisualizationService` turns
selected evidence into an immutable `PlotSpec` and renders atomically outside the training loop.
The adjacent `.plot.json` is the regeneration and cache record.

`ArtifactService` coordinates separate `ArtifactInspector`, `ArtifactVisualizer`, `ArtifactSchema`
and `ArtifactValidator` contracts. Built-ins inspect NumPy and tabular formats with bounded work and
pickle disabled. Geometry requires explicit semantic roles. `RemoteResultService` synchronizes only
allowlisted small evidence; `RemoteArtifactService` fetches one explicitly named artifact.

## 8. Storage, observability and reproducibility

Artifact stores expose immutable references; local/S3 providers and the shared cache own transfer
and verification. `ExperimentRegistry` is a read-only view over result trees, never a second source
of truth. Event/resource/profiler adapters record bounded evidence. Reproducibility profiles,
scoped seed derivation and environment exports make runtime policy explicit.

## 9. Extension policy

Prefer a consumer-project class referenced by `target`. Use entry points only for independently
versioned reusable providers. Keep core dependencies provider-neutral; Plotly, NetworkX, trimesh,
Optuna, BoTorch, cloud stores and trackers stay optional. A new public contract needs constructor
validation, stable re-export, YAML/public-import coverage, focused failure tests, English/Spanish
documentation and an AGENTS entry.

`CudaCompatibilityResolver` probes remote Python/architecture plus NVIDIA driver/compute capability,
verifies an exact official PyTorch wheel, and places its `TorchInstallationPlan` in
`EnvironmentIdentity`. `ManagedEnvironmentProvider` installs and constrains that wheel before other
dependencies, then verifies required CUDA initialization. It never manages host drivers/toolkits.

## 10. Intentional 0.5.3 limits

- No automatic cluster selection or mixed-cluster workflow runtime.
- No server, GUI or resident control-plane daemon.
- No implicit large-result, checkpoint or dataset download.
- No automatic CUDA/driver installation or remote platform wheel synthesis.
- No magical interpretation of arrays as graphs, point clouds or meshes.
- No new HPO mathematics in 0.5.3; CUDA environment resolution does not alter scientific policy.
