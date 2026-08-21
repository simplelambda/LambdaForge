# LambdaForge agent operating guide

This is the low-token entry point for LambdaForge agents. Do not crawl package READMEs. Open only
the relevant [canonical manual](docs/MANUAL.md) section, then the public contract/implementation
under change. Current code and behavioral tests override stale assumptions.

## Product model

LambdaForge is an installable Python >=3.10 PyTorch/Lightning research framework. Consumers own
domain code/data; it runs tasks, datasets, workflows, experiments and HPO across local, SSH/SLURM.
Use public imports only. Never copy it, share its `.venv`, patch `PYTHONPATH`, serialize secrets or
infer CUDA usability from `nvidia-smi` alone.

## Fast routes

| Need | Use |
|---|---|
| Validate without execution | `lf validate CONFIG` |
| Inspect strict materialization | `lf inspect CONFIG --resolved` |
| Read-only execution plan | `lf plan CONFIG [--on CLUSTER]` |
| Run any supported config | `lf run CONFIG [--on CLUSTER]` |
| Discover project configs | `lf configs list`; `lf experiments list`; `lf tasks list` |
| Inspect research history | `lf experiments show/status/history/runs/results NAME` |
| Create a consumer scaffold | `lf init DIRECTORY` |
| Debug preprocessing samples | `lf debug CONFIG --records N` |
| Dataset discovery/convenience alias | `lf datasets plan NAME`; `lf datasets build NAME` |
| Inspect dataset content | `lf datasets show/members/member/diff/stats/verify ...` |
| Place a dataset | `lf datasets materialize SELECTOR --on CLUSTER`; add `--apply` after review |
| Diagnose a cluster | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Diagnose any failed command | read its next action; add `--debug`; use `--json` for tools |
| Reconnect to work | `lf jobs list/status/logs/cancel/retry`; selectors accept ID, name, prefix, `latest` |
| Global runtime view | `lf top --history 120` for humans; `lf overview --json`, `lf jobs list --json`, `lf resources --all --json` for tools |
| Query or compare results | `lf results list/show/compare/export` |
| Plot scientific evidence | `lf plot learning/sweep/seeds/hpo/resources` |
| Inspect/fetch artifacts | `lf artifact inspect/list/fetch/validate/visualize` |
| Preview cache collection | `lf storage gc [--on CLUSTER]`; apply only after review |
| Explain configuration/identity | `lf explain CONFIG`; `lf explain KIND PATH`; `lf explain changes CURRENT --against PREVIOUS` |
| Shell completion | `lf completion bash|zsh|fish` |

`lf` equals `lambdaforge`; grammar is `lf <resource> <action> <object> [--on CONTEXT]`.

## Error and diagnostic contract

Do not parse prose or catch a private implementation exception in agent automation. CLI failures
use the stable categories `configuration`, `validation`, `environment`, `execution`, `resource`,
`connection`, `authentication`, `data`, `storage`, `operation_refused`, `internal`, `warning` and
`cancelled`. Append `--json` anywhere and consume `category`, `exit_code`, `retryable`, `job_id`,
`context` and `commands`; append `--debug` only when traceback internals are needed. `--verbose` is
operational detail, not debug.

Exit codes are 2 for configuration/validation/refusal, 3 for environment/connection/authentication,
4 for execution/data, 5 for resource/storage, 10 for unexpected internal failures and 130 for
cancellation. Warnings remain exit 0. A preflight error explicitly means no job was submitted;
`EXECUTION FAILED` means code started. For failed jobs use `lf jobs logs JOB --tail 300` and
`lf jobs show JOB --json`. Dataset/workflow diagnostics present one root `FAILED` component and
dependent `BLOCKED` components; never treat every blocked node as another root failure.

Boundary failures persist a redacted record below `$XDG_STATE_HOME/lambdaforge/logs/errors`
(normally `~/.local/state/lambdaforge/logs/errors`). Secrets must never be copied from debug output
or injected into commands. When extending LambdaForge, raise `LambdaForgeError(ErrorDiagnostic)`
only where the domain knows WHAT/WHY/IMPACT/FIX; preserve the original with `raise ... from error`.
Ordinary internal `ValueError`/`TypeError` contracts may remain technical because the central CLI
classifier handles the operational boundary. Public diagnostic types live in
`lambdaforge.diagnostics`.

## Consumer installation

```bash
cd /path/to/research-project
python -m venv .venv
source .venv/bin/activate
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

Prefer an immutable wheel for released/offline work. Let the consumer lock and verify PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Install only needed extras. Consumer version bounds are authoritative: update an incompatible
bound deliberately, reinstall, require `pip check`, and never bypass it with `--no-deps`.

## Complete CLI surface

- Author/inspect: `init`, `target`, `validate`, `inspect`, `plan`, `run`, `compose`, `diff`, `explain`, `migrate`, `debug`, `plugins`, `completion`, `project status`.
- Discover: `configs|tasks list|show|validate|plan|run`; `experiments` adds `status|history|runs|results`.
- Control: `doctor`, `overview`, `top`, `resources`, `status`, `logs`, `cancel`, `retry`.
- Clusters: `clusters add|list|show|inspect|set|unset|remove|export|credentials set|delete|test|bootstrap|resources|storage`.
- Jobs: `jobs list|status|show|logs|pause|resume|cancel|retry|delete|reconcile|groups`; `jobs group list|show`.
- Storage: `storage status|gc`; `environments list|show|gc`.
- Data: `data list|locations|inspect|replicate`; `datasets plan|build|list|show|members|member|diff|locations|stats|verify|lineage|add|remove|delete|materialize|replicate`.
- Evidence: `aggregate`, `retain`, `registry`, `dashboard`; `results audit|list|show|compare|export|sync`; `plot learning|sweep|seeds|hpo|resources`; `artifact inspect|export|validate|visualize|list|fetch|plugins`.

## Configuration and execution

Friendly authoring compiles to one strict `MaterializedConfig`. Trusted YAML `target` constructs,
`ref` imports and `params` supplies kwargs. Targets must belong to the installed consumer package.
Use `inspect --resolved` before editing advanced YAML.

The supported document families are:

- `kind: task` for one reproducible non-training operation;
- concise/strict preprocessing, compiled to a Task;
- `kind: dataset` for staged immutable publication;
- `kind: workflow` for a local dependency DAG of tasks/experiments;
- experiment Schema 1.1 for training, seeds, sweeps and HPO.

Supported experiment migration inputs are unversioned and Schema 1.0. Preview with `lf migrate`;
never rewrite source implicitly. Declare every scientific local input at the top level. Physical
paths are resolved before execution but logical identity—not cluster placement—drives scientific
reuse. Default run resumes compatible state; `--rerun`/legacy `--force` repeats terminal science,
`--restart` discards continuation and `--no-resume` only disables continuation. Active identical
experiment revision+cluster is refused unless `--allow-duplicate`; another cluster is allowed.

Authored `resources` is the real scheduler request. CLI resource flags only override explicitly
provided fields. A workflow/dataset top-level request is exact; otherwise LambdaForge derives the
fixed outer allocation from node/stage resources, topological levels and `max_parallel`. Do not ask
users to repeat YAML resources on the command line. `lf run` is canonical for dataset recipes;
`datasets build` is only a compatible selector-oriented alias.

## Task and preprocessing contracts

A project task implements `run(context) -> TaskOutput` (or the supported zero-argument duck-typed
form). Use `context.input(NAME)` and `context.output(NAME, create=True)`. Return small JSON outputs,
numeric metrics and explicit run-relative `ArtifactDeclaration` values. Never escape the run root or
return symlinks.

Preprocessing is `source -> transforms -> sink`. `workers=1` is sequential; `io` uses threads;
`cpu` uses spawn-safe processes while the parent owns the sink/manifest; `gpu` requires one worker.
Sample debugging never finalizes the production sink. Preprocessing does not publish a managed
dataset unless `publish_dataset: true` or compatible `dataset_name` is explicit.

## Dataset lifecycle

Keep these concepts distinct:

```text
DatasetRecipe -> DatasetBuild -> DatasetVersion -> DatasetPlacement
```

Recipe stages compile to the existing Workflow DAG. `required` expresses scientific necessity;
`reuse: auto|never` controls cache policy. `--force-stage X` forces X and all transitive downstream
stages. A successful build validates and atomically publishes the final root plus canonical JSONL
`DatasetIndex`; incomplete stages never become a version.

Remote bundles relocate every explicitly declared local Task input up to 10 MiB, including named
inputs inside embedded recipe-stage tasks; never tell a user to copy data into a hashed bundle or
job directory. Inputs above the bound must fail before submission. Represent them as a DataCatalog
reference with an explicit target-cluster location, a materialized managed DatasetVersion, or an
explicit transfer-provider workflow. LambdaForge must not guess or implicitly move large data.

`DatasetMember` has a stable logical ID, arbitrary partitions/targets/metadata and named
file/directory/record/URI assets with real checksums. Artifact v2 uses path-independent
`content_id == dataset_id`; `build_id` describes recipe provenance. Never mutate a published alias.
`DatasetAsset.sha256` is the file-byte SHA-256; directories use the documented tree fingerprint.
Legacy filename-prefixed file hashes are read-only compatibility, not an authoring contract.

For managed data, `DatasetRegistry` owns exact versions and placements and `DatasetResolver` is
Registry-first. `DataCatalog` remains for external data, aliases, loaders, pins and overrides. Prefer
`dataset:NAME@VERSION/subpath`; an ambiguous unversioned match is an error. Materialization is only
NOOP, verified atomic REPLICATE or durable BUILD. Dataset deletion and storage GC are exact-root,
locked and preview-first.

## Training, post-run and HPO

The standard task expects mapping-shaped batches. Add models, losses and metrics through public
contracts or importable project classes. Use a Lightning callback for batch/epoch validation logic;
`validation_step` exposes detached `model_outputs` and `loss`. Log exact `val_*` names for
checkpointing and HPO.

Use `PostRunAction` for bounded same-allocation analysis after successful training. It receives an
immutable `PostRunContext` and returns `PostRunResult` artifacts. Required failure prevents run
success; optional failure is recorded. Action identity is separate from training identity, receipts
are content-verified, and actions run rank-zero sequentially. Use a Task/Workflow for different
resources, clusters or long dependent work.

Finite sweeps/random/Optuna and adaptive HPO are separate modes; do not combine `sweep` with enabled
adaptive `hpo`. Adaptive space keys are dotted scientific paths with float/int/ordinal/categorical/
bool dimensions and optional conditions. The objective must be an exact dense `metrics.csv` column.
Inspect/dry-run are read-only. START_NEW, checkpoint RESUME and ADD_SEED share cost/memory-aware
ranking; confirmation is separate. Relaunch identical YAML to reconcile/resume controller state.

Memory capacity is explicitly UNKNOWN, UNBOUNDED or KNOWN. A candidate-aware CUDA preflight must
perform representative forward/backward/step in an isolated child; an OOM is censored lower-bound
evidence. Never claim allocator caps are physical isolation or silently shrink batch size. The seed
mean model uses `tau² / n + (v₁ + ... + vₙ) / n²`; with one seed, never invent uncertainty.

Extension policies use the immutable public action/estimate types. Exact searcher, fidelity, seed,
curve, cost, memory, admission and selector signatures are in the HPO section of the manual and
their public docstrings.

## Clusters, environments and jobs

`Transport` and `Scheduler` are real provider boundaries. OpenSSH is preferred and preserves normal
keys, agent, known_hosts and ProxyJump. It opens a client per operation but reuses a private
ControlMaster socket until the configured idle `persist` time. Password mode resolves only hidden
interactive, `keyring:` or `env:` references and must never put values in argv/YAML/bundles/state,
fingerprints or logs.

Keep `PythonRuntime -> PythonEnvironment -> InstalledPackages` separate. New managed profiles use
`python.strategy=auto`: probe the configured/bounded alternatives, reuse a Conda-family manager, or
stage the pinned verified micromamba and create a runtime below `storage.cache_root`. A legacy
`python: python3` string means strict `existing`; migrate it with
`lf clusters set NAME python.strategy auto`. Use `bootstrap --dry-run` before provisioning.
Managed environments are immutable user-space venvs identified by exact framework/consumer/
dependency wheel bytes, resolved runtime, offline policy and Torch plan. Never use `conda activate`,
modify shell startup files, system Python, drivers/CUDA, or silently fall back to CPU.
Bootstrap accepts LambdaForge itself from either an editable PEP 610 source or a regular installed
wheel; never infer a source root from `lambdaforge.__file__` or require `pyproject.toml` inside a
consumer virtual environment. Read `Requires-Python` from release and consumer metadata; do not
duplicate the version floor in code. Runtime/package caches are reconstructible, but GC must retain
runtimes referenced by active jobs, the active pointer or retained environments.
After successful bootstrap, obsolete LambdaForge environment directories are pruned only after the
new pointer is active; the current environment and live-job references remain protected, and a
concurrent build defers cleanup.
Automatic Torch selection uses actual remote Python, driver and compute capability plus official
wheel availability, and the installed environment must pass a CUDA tensor probe when required.
LambdaForge-managed runtimes also record one locally validated host CA bundle and propagate it to
runtime creation, pip/Requests and scientific commands. Never disable verification, download roots
or modify the system trust store. `doctor` exposes `system-python-tls` and `managed-python-tls`.

Remote CLI runs first return a durable `preparing` job and detach bundle/runtime/transfer work;
`--wait-for-submit` explicitly restores synchronous scheduler acknowledgement. `submission_phase`
is machine-readable. Direct jobs then use one detached `ProcessSupervisor`; SLURM remains
authoritative for scheduled work.
`lf top` isolates provider refresh and full-log loading in cancellable processes; never move SSH/
resource/log polling into its keyboard loop. Its default lower list is research work derived from
jobs by identity+target; `v` exposes raw jobs. Crossing the cluster/work boundary changes selection
and `Enter` opens paired history, personal usage and cluster-filtered jobs. Cancellation remains a
rendered, non-blocking confirmation. `--history SECONDS` controls only bounded TUI history.
GUI integrations consume `overview --json` job `timing`/`usage`, `jobs list --json` and
`resources --all --json` `personal` aggregates, not terminal escape sequences. Keep requested
allocations separate from observed use and leave unsupported scheduler accounting unknown.
Job logs have separate lifecycle, submission-worker and scientific streams. Use
`JobService.events` for durable framework/provider transitions and `scientific_logs` for raw
consumer output; scheduler reachability or a supervisor heartbeat is liveness, not proof of
scientific progress. Workflow nodes and preprocessing checkpoints provide generic progress, but a
long opaque consumer operation must emit its own bounded flushed messages.
State, heartbeats, logs and usage are durable. Provider outages report unknown plus last-known state,
not fake scientific failure. Pause retains leases. Signals require matching PID, process group,
creation time and command hash. Direct-host admission is cooperative affinity/visibility, not cgroup
isolation. Mixed-cluster DAG coordination and shared multi-cluster HPO state are intentionally absent.
`jobs retry` is only for failed/cancelled/timed-out attempts; it retains science, increments attempt
and creates a job. Scheduler `jobs resume` only continues a paused allocation.

## Results and publication discipline

Use `ResultService`, Registry and job services rather than filesystem globs. A result is identified
by materialized scientific configuration, logical data, code identity, seed/variant and attempt.
Treat ambiguous successful duplicates as an error before publication. Rebuild summaries with
`aggregate`; use exact metric names and explicit direction for comparisons.

`results sync JOB` transfers only allowlisted small evidence; `artifact fetch JOB NAME` retrieves one
explicit heavy artifact. NPY/NPZ inspection disables pickle, bounds previews and samples large
statistics deterministically. Every plot has a renderer-neutral `PlotSpec` and atomic sidecar. Keep
raw run evidence immutable; retention/compression and deletion are preview-first.

## Extension contracts

- Model: `torch.nn.Module`; mapping output is recommended for named losses/metrics.
- Loss: public `Loss`, consuming `(outputs, batch)` and returning a scalar tensor.
- Metric: public `Metric`; keep state bounded or use documented exact behavior deliberately.
- Task: public `Task` or compatible `run` method returning `TaskOutput`.
- Preprocessing: `PreprocessingSource`, `PreprocessingTransform`, `PreprocessingSink`.
- Dataset profiler and transfer: public `DatasetProfiler`, `DataTransferProvider`.
- Runtime providers: `Transport`, `Scheduler`, `ExecutionBackend`, `ArtifactStore`.
- Training analysis: Lightning `Callback` or `PostRunAction`, according to lifecycle needs.
- Reusable third-party components: `lambdaforge.<kind>` entry points; project-local objects should
  normally use installed `my_project.*` targets.

Public classes require concise docstrings explaining responsibility, invariants and contract. Do not
import or document private file paths.

## Repository modification rules

1. Preserve current YAML, CLI, documented Python APIs, DatasetArtifact v1 reads, supported config
   migrations and public plugin contracts unless the request explicitly authorizes a break.
2. Preserve dataset publication atomicity, content-addressed reuse, result identity, job durability,
   transport/scheduler boundaries, storage safety and HPO semantics.
3. Prefer cohesive modules and simple functions over one-class-per-file ceremony or forwarding-only
   service chains. Do not replace meaningful typed states with ambiguous dictionaries.
4. Check static imports, exports, YAML strings, schemas, examples and entry points before declaring
   code dead. LambdaForge resolves objects dynamically.
5. Keep the base install light; optional providers stay lazy and optional.
6. Update `docs/MANUAL.md` for user-visible behavior, `AGENTS.md` only for agent operations,
   `CHANGELOG.md` for release history and `SECURITY.md` for threat-model changes.
7. Run focused tests after each subsystem, then ruff, mypy, the relevant full suites, package build,
   installed-wheel/CLI smoke and example validation. Do not hide skipped CUDA tests.
8. Change release identity only in `src/lambdaforge/_version.py`; packaging consumes it dynamically.

## Targeted manual routes

Open only the relevant heading in [docs/MANUAL.md](docs/MANUAL.md): configuration (6/20), tasks and preprocessing (7), identity (8), workflows (9), clusters (10), jobs/datasets (11), HPO (12), results
(14), CLI (16), public API (17), architecture (19), migrations (21), process safety (22), outputs
(23), retention (24), components (25), extensions (26), or limitations (28).
