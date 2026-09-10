# LambdaForge agent guide

This is the low-token source of truth for agents using or modifying LambdaForge 0.14.0. Spanish is
in `AGENTS.es.md`. Read the relevant section of `docs/MANUAL.md` only when more detail is needed,
then inspect the public signature or implementation being changed. Current tests and code override
assumptions.

## One execution model

Every YAML executes one or more classes inheriting `lambdaforge.Work`. A class has one lifecycle
entry point: `run(...)`. Reject functions, non-Work classes, required constructor arguments and old
configuration fields locally before submission. Never create a second runner or translate Work YAML
to Task/Experiment/Workflow/DatasetRecipe objects.

Do not reintroduce executable Task/TaskContext/PreprocessingTask, YAML `kind`, `schema_version`,
`target/ref/params`, constructor or method selection, source/transform/sink requirements, arbitrary
DAG edges, recursive object construction, global `lf.current()/metric()/artifact()` APIs, or a
compatibility path unless the project explicitly reverses this architectural decision.

## Fast routes

| Need | Command |
|---|---|
| Validate without execution | `lf validate CONFIG` |
| Explain signature/defaults/resources | `lf explain CONFIG` |
| Read-only expansion | `lf run CONFIG --dry-run` |
| Execute | `lf run CONFIG [--on CLUSTER]` |
| Deliberate new execution | `lf run CONFIG --rerun` |
| Interactive operation | bare `lf` (TTY Research Console) |
| Monitor semantic work | Research Console; `lf overview --json` |
| Work operations | `lf show/logs/cancel/retry/delete SELECTOR`; study Run: `show/logs WORK --run KEY` |
| Low-level jobs | `lf jobs list/show/logs/cancel/retry/delete`; `lf jobs clear [--apply]` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Results | `lf results list/show/compare/analyze/report` |
| Runtime diagnosis | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Cluster profiles | Research Console; `lf clusters add/set/unset/...` for automation |
| Preview safe storage cleanup | `lf clean [--on CLUSTER]`; add `--apply` after review |
| Scaffold | `lf init DIRECTORY` |

Append `--json` for automation and consume stable fields; use `--debug` only for framework
tracebacks. Local and remote run both return after durable asynchronous preparation unless
`--wait-for-submit` is explicit; `--dry-run` is direct and read-only. Never parse prose or secrets.

## Writing Work

```python
from pathlib import Path
import lambdaforge as lf

class Example(lf.Work):
    def run(self, source: Path, limit: int = 100) -> dict[str, int]:
        rows = source.read_text().splitlines()[:limit]
        report = self.outputs.file("report", filename="report.json", role="report")
        report.write_json({"rows": len(rows)})
        self.metrics.log("rows", len(rows))
        self.outputs.value("summary", {"rows": len(rows)})
        return {"rows": len(rows)}
```

```yaml
name: example
run: my_project.Example
with:
  source: {file: data/input.txt}
  limit: 100
resources:
  cpu: 2
  memory: 2GiB
```

Python signatures and docstrings own parameter truth. Use only the typed markers `{file: ...}`,
`{dataset: NAME@VERSION}` and `{from: STEP.OUTPUT}`. Plain strings are never guessed as paths.
Every project class must be importable from its installed consumer package.

Work properties are available only during execution:

- immutable: `name`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir`, `resuming`;
- managed services: `outputs.file/directory/value/dataset`, `metrics.log/log_many`,
  `checkpoints.file/exists/save_json/load_json`, `cache.put/get/file/fetch/rate_limit`,
  `tools.require/run`, `progress.update`,
  `log(message, level=...)`;
- owned paths: `run_dir` is durable per Attempt; `temp_dir` is ephemeral;
- intra-job concurrency: `map(items, function, workers=..., executor=..., retries=...)` is ordered
  and has no persistence; `resume_map(..., key=..., validate=...)` explicitly stores safe
  dependency-aware JSON checkpoints. Legacy `map(..., key=...)` delegates to `resume_map`.

`return` is the primary JSON result. Registered names cannot collide. Never pickle results, mutate
planning inputs, construct hidden paths, create fingerprints, or expose Registry/scheduler services
to scientific code.

Cache is reconstructible; checkpoints are resumable Run state; outputs are durable evidence. Work
code should not manage cache directories, `.part` names, locks, `fsync` or `os.replace`. Prefer
`cache.put/get` for bytes/text/strict JSON, `cache.file/fetch` for path-like content, managed
checkpoint files and managed outputs; `cache.path`, `checkpoints.path`,
`run_dir` and `outputs.artifact` are advanced interoperability escapes, not examples for ordinary
code. A `ManagedFile` is path-like but resumable-map state stores only its logical key/SHA/size.
`resume_map` must validate those dependencies and selectively rerun an invalid item after cache cleanup. Never
serialize machine paths or arbitrary pickle as scientific state.

`outputs.file/directory(..., publish_to=PATH)` publishes a verified result after successful
finalization and refuses different existing content unless `overwrite=True`. After the Execution
record is durable, redundant internal bytes are removed by default; `retain_internal=True` keeps
both. Failed/interrupted managed `artifacts/` are compacted but logs, result, metrics, provenance
and checkpoints remain. Never put disposable bulk bytes directly in `run_dir`. Relative destinations
start at the authored YAML directory. Remotely they require the cluster `project_root` mirror and
map to the same project-relative directory; otherwise use an explicit absolute remote path. Never
describe publication as an automatic transfer back to the controller.

For remote `{file: PATH}` inputs, content up to 10 MiB is bundled automatically. A larger path must
belong to the local `pyproject.toml` project and have an exact counterpart below the cluster's
absolute `project_root`; LambdaForge verifies kind/size/SHA-256 before submission and in the worker,
and never syncs or deletes that researcher-owned mirror. Configure with
`lf clusters set NAME project_root /absolute/remote/project`, then run `lf doctor --on NAME`.
Missing/stale/symlinked content fails closed. Use managed datasets for reusable very large data,
because exact mirror verification reads every byte on each submission.

Project-native tools are declared once in `[tool.lambdaforge.environment]` with `manager="conda"`,
exactly one project-contained `file` or `lockfile`, and bare `required_executables`. Use
`lf clusters bootstrap NAME --project . --dry-run`, review, then apply. Environment files accept
only name/channels/string dependencies; no pip mappings, variables, prefixes, hooks or Torch/CUDA.
Offline native provisioning requires a platform-specific `@EXPLICIT` SHA-256 lock and matching
`package_cache`; pip/Torch also needs the cluster wheelhouse to be fully offline. Managed creation
uses one immutable Conda prefix, exact inventory and verified tools. `tools.require` checks that
prefix and never installs. Pip-only and `environment: existing` behavior is unchanged.

Use `tools.require(..., version_args=...)` and `tools.run(argv, ...)` for external executables.
Commands are argv, never shell strings; child thread variables and environment overrides remain
scoped, output reaches Work logs, and tool provenance belongs only in `environment.json`.

## Clustering contract

Use `lambdaforge.clustering` for non-neural clustering. `KMeans`, `MiniBatchKMeans`, `DBSCAN`,
`HDBSCAN` and `Agglomerative` share `Clusterer.cluster(X) -> ClusteringResult`; sklearn is lazy and
optional through `lambdaforge[clustering]`. Do not add a factory/registry, expose backend estimators,
or create clustering-specific distance types. Reuse `lambdaforge.nn.distances.Distance`, enforce
the algorithm capability table and guard custom precomputed matrices before O(N²) allocation.
KMeans and Ward are Euclidean; reject invalid combinations with the scientific reason. Do not hide
scaling, imputation, PCA, thresholds or stability policy in clusterer defaults.

`print()` and standard Python logging are captured by Job logs. Use `self.log()` for timestamped,
immediately flushed human narration, `self.progress.update()` for completion and `metrics.log()` for
scientific numeric evidence. The Research Console presents bounded Work/study/result read models;
do not put result files, provider calls or scientific logic inside widgets. Automation uses
`lf overview --json` (`work.items[].attempt_history` retains Job IDs), ordinary `lf logs` and
preview-first `lf jobs clear [--apply]` instead of parsing the TUI. Local provider paths belong to
the durable Job and must never be recomputed from the observer's current directory.
Failed Work logs append persisted exception
type/message/phase/result path after any requested tail; request `--verbose` or `--debug` for the
traceback or use `--json` for structured `failure`/`failures`. Immutable
runtime/result models crossing a spawned process boundary must remain pickle-safe; never expose a
raw `mappingproxy` through that boundary. Cache fetch retries incomplete HTTP/
chunked/gzip transfers and transient statuses from clean unpublished temporaries; never add a
consumer-side retry workaround. Human bootstrap progress is stderr-only and machine JSON is clean.
The Research Console calls the same domain services in workers, shares one provider factory,
serializes cluster operations and pauses resource probes during bootstrap. Its activity panel
streams phases/elapsed liveness and is preset/drag resizable; the upper viewport stays scrollable
with a protected minimum height. Mutation confirmations and modal plans render bounded semantic
sections, never raw JSON. Hidden root screens must not load.
Root loading indicators are first-snapshot-only. Later probes retain the last good read model,
display its age and mark it stale on failure; Work rows and bounded Work logs refresh live with at
most one request in flight per view.
Run Work's picker filters to YAML; recents merge existing local Job history with a bounded MRU that
persists only local path/name/time metadata.
Browse/recent selection only populates one selector. The single Submit action must revalidate,
explain and only then asynchronously submit to the captured visible cluster; disable repeated
activation and expose every phase. Preview rendering is diagnostic only and must never block an
already validated submission. Never treat MRU presence as validation or copy authored
configuration into console state. Work display names are not keys: use exact `work_id` identities
for rows and mutation selectors so simultaneous same-name executions remain independent.
Keep YAML syntax failures source-aware and bounded: path, line/column, excerpt, caret and hint once.
Dataset Summary may read the logical index but must not walk assets; Stats/Integrity remain explicit.
Whole-version deletion previews every placement, confirms once and treats
`registered_but_missing` as safe stale-index convergence before forgetting empty records. Its
control remains visible and must immediately expose pending/success/failure state while preventing
duplicate activation.
Parameter studies are ordinary Works declaring `search` or multiple `seeds`, not a
training-specific kind. `work.items[].study_expected` is available before execution telemetry;
`study` becomes non-null
when the current worker publishes its bounded index. Steps, parallel composition and `self.map()`
alone remain ordinary Attempt/log Works and must not create study telemetry.

## YAML composition and studies

`steps` is a sequence. `{parallel: [...]}` is one concurrent level. The following level waits for
all members. Cross-step outputs reference `step.output` and are invalid when the producer has
multiple seeds/trials. `seeds` creates separate Runs and does not inject a `seed` argument.
`search` parameters override normal `with` values and are passed normally to `run()`. `objective`
is legacy `metric`/`mode` or a fixed-range `metrics` utility using `weighted_mean`, `geometric` or
`chebyshev`; composite components must share one integer step. One utility governs every HPO
decision, constraints stay separate hard guardrails and raw-component Pareto status is diagnostic.
Any search with an objective defaults to the full
adaptive policy: scrambled-Sobol startup, optional noise-aware BoTorch mixed-GP qLogNEI with deterministic
mixed-kNN fallback, probabilistic shared-seed racing, curve pruning, convergence and fresh-seed
confirmation. Only proposed candidates become public. `strategy: exhaustive` instead enumerates
exact finite `values`/`when` combinations and every seed; it rejects continuous `range` and
`trials`. Default confirmation seeds are disjoint; `confirmation_seeds: []` deliberately disables
them. `search.fidelity` is explicit cumulative Work-defined budget; `self.fidelity` plus checkpoints
implement continuation and LightningRunner bridges epoch budgets. Controller decisions/state live
in `hpo-control/decisions.jsonl` and `state.json`. Confirmation is immune to performance
pruning/preemption; an incomplete set persists `confirmation_incomplete` and cannot select a
survivor-only mean. `trials` is the executed-candidate budget and
`proposal_pool_size` the larger deterministic pool. Event-driven refill compares `START_NEW`,
`DESIGNED_PROBE`, `ADD_SEED`, `PROMOTE_FIDELITY` and `RESUME_PREEMPTED` after every terminal event.
It automatically balances posterior practical-improvement opportunity O with unresolved-question
entropy K per observed incremental cost; neither is a calibrated information gain or a user-facing
confidence value.
Startup is not a barrier, undispatched queue entries are provisional/replanned with explicit
`CANCEL_QUEUED_ACTION`, and pending `(candidate, seed, fidelity)` identities prevent duplicates. Root `search.reduction_factor` and
`search.confidence` are removed in favour of separately owned fidelity/seed/pruning controls.
Omitted `startup_trials` resolves to `min(trials, max(10, safe parallelism))`; an explicit value is
authoritative. Interleave first seeds across distinct startup candidates before extra seeds.
`ScientificQuestionAnalyzer` is the one shared live/final source for parameter conclusions,
pairwise interactions and the practical optimal region. Scientific `confidence` means stability of
the exact displayed conclusion under deterministic candidate-level/shared-seed resampling—not
coverage, effect size, a p-value or a frequentist interval. Seed noise comes only from
within-candidate repetitions; prioritize shared seeds, incumbent replication and matched valid
counterfactual probes when they reduce an important comparison. Persist purpose, target questions,
O/K, automatic weights, cost and alternatives. Do not add phase/confidence weights or thresholds to
YAML, claim causality, invent a practical margin, expand authored domains, or replace fresh-seed
confirmation.
Controller-critical scientific analysis must stay bounded by live inference precision, never by
raw `proposal_pool_size`: reuse immutable matching geometry and mixed-space distances, score a
rotating representative shortlist, and always retain the optimizer proposal. Never scan the full
pool once per parameter/pair/resample or let explanatory analysis delay Future collection, GPU
cleanup or admission. The full deterministic pool remains authoritative for optimization.
`objective.constraints.METRIC.min/max` are explicit same-checkpoint guardrails. Across seeds,
`seed_aggregation` is legacy `mean`, `worst` or `lcb` with optional `confidence`; missing/insufficient
LCB evidence is infeasible. A utility component may also be constrained. Never
infer guardrails or hidden multi-objective weights from other metrics. `runs_per_gpu` is only a hard
per-device maximum inside the fixed outer reservation; `max_parallel` remains the global maximum.
`resources.gpu_memory` is optional and, when present, is a user safety floor per new Run. Effective
admission uses the maximum of that floor, a candidate-specific conservative future envelope and
known OOM lower bounds, bounded again by live physical free VRAM. Physical VRAM is authoritative.
Never multiply the floor by active Runs or
maintain a second reservation. Cold start admits one unknown Run per GPU; compatible exact/censored
history then enables denser best-fit packing. `RESOURCE_BLOCKED` is reversible and scientifically
neutral; `RESOURCE_INFEASIBLE_ON_DEVICE_TYPE` requires a hard lower bound above device capacity.
Never retry a compatible OOM candidate at the same or lower effective headroom. The resource
planner consumes a bounded ranked scientific frontier, may safely backfill, protects heavy work
from starvation using predicted completion windows, and may stop memory-feasible co-location when
measured aggregate throughput would not improve. If that frontier is entirely blocked, request at
most one bounded extension from the same scientific policy before intentionally idling; never loop
through random resource candidates. Every admitted GPU Run uses a fresh one-worker spawned process that exits on
result/error; never restore a persistent CUDA pool because idle contexts retain VRAM and can
deadlock queued Runs. GPU memory probing must remain a short-lived child process: the controller
must not retain one CUDA context per device or consume a scientific slot. Repeated objective observations need
`step=` for early stopping. `LightningRunner` bridges them automatically; custom loops return at a
safe boundary when `self.stop_requested` is true.

Every adaptive CPU/GPU Run owns a fresh one-worker process. A lost/killed worker and CUDA OOM may
retry as a new checkpoint-compatible Attempt up to `failure_retries` (default 1, max 3); a repeat is
terminal. OOM is censored resource evidence, not an objective value; persist its attempted
allocation/headroom/residency/source and recompute every pending placement before retry. Prefer
best-fit admissible GPUs while preserving a predicted window for still-relevant heavy work. Do not kill healthy siblings or retry forever. Never
retry arbitrary consumer exceptions. One exhausted Run makes the enclosing Work
honestly failed but must not cancel unrelated active/queued Runs. Telemetry records logical GPU
index and exact inherited token; never infer or broaden physical devices from that display field.

Study telemetry is a bounded read model, not another result store. It references per-Run logs and
scalar JSONL, never copies checkpoints/outputs, and exposes exact keys under
`overview --json` → `work.items[].study`. `lf show WORK --run KEY --json` returns parameters,
down-sampled curves, current/best objective and epoch, timing/failure/log; `lf logs WORK --run KEY`
isolates output. Completed HPO uses the mean of each seed's best observed checkpoint; current
same-step curves drive pruning, and pruned Runs are terminal censored evidence—not failures—and
excluded as exact values from completed-objective fitting/statistics. Their parameter-region
pruning rates remain visible. Any performance-pruned seed censors the whole candidate; never average
its earlier completed seeds as survivor-only evidence. Group completed evidence by exact
`(candidate,target,maximum)` rung; never average heterogeneous fidelities or race incompatible rungs.
A candidate-level joint survival model with uncertainty softly
modifies acquisition; multiple seeds do not overcount one candidate and operational failures or
scheduler preemption remain neutral. Default pruning requires two distinct
uncompetitive common steps through `early_stopping.confirmations`. Lightning scalar
callback metrics plus epoch/validation time are automatic. Custom trainers log curves with
`self.metrics.log(name, value, step=epoch)`; use `progress.update` for coarse progress and
`self.log`/print only for human narration.

The controller may cooperatively preempt only a scored non-confirmation fidelity Run with a
verified owned checkpoint, at least 30 seconds of runtime and a competing action exceeding both
the original and continuation priorities by 50%. It never kills for scheduling. Preserve
`PREEMPT` → `PAUSE` → `RESUME_PREEMPTED` evidence; startup without a comparable score is protected,
and a stop arriving after the target still means `completed`.

Adaptive telemetry also includes bounded `hpo_analysis` and the last 25 structured `controller`
actions. The Studies screen consumes it and automation reads
`work.items[].study.hpo_analysis/controller`. Treat per-parameter insights as marginal
associations with explicit coverage/confidence/caveats, never causal claims or a replacement for
the joint multivariate sampler. Enter/right opens the selected parameter's bounded response chart
and pairwise joint-predictive-gain panel; descriptive response/coverage begins at two comparable
observations and predictive gain at three, always with conservative confidence floors. The JSON
includes response points, pruning signal and a bounded matrix.
Retrospective pruner quality lives in `hpo-control/state.json` → `pruner_calibration`; it reports
simulated savings, false prunes, regret and probability/curve calibration without fabricating a
full objective for censored Runs.

The scientific runtime never depends on rendering; the Research Console's base `textual-plot`
widget only visualizes bounded telemetry. Lightning study views label the objective,
proposed/planned candidates, GPU and latest/best epoch. Lightning
projects choose displayed curves with `LightningTrainConfig(epoch_chart_include=[...],
epoch_chart_exclude=[...])`; collection remains complete. `gpu_mem_mb` is peak live allocation,
whereas `gpu_reserved_mb`/`gpu_peak_reserved_mb` are allocator-cache diagnostics. Never use reserved
telemetry as a scheduler request: packing uses explicit `resources.gpu_memory` plus driver free
memory for per-Run dynamic admission, and completed packed Runs empty unused CUDA cache before slot
reuse. Temporary pressure queues and polls; it is not a scientific failure.

Do not confuse `self.map` concurrency inside one Work with a YAML parallel group. A parallel group
uses isolated spawned Work processes inside the enclosing Job's aggregate fixed allocation.
Exhaustive seed/search is serial; adaptive search owns its allocation and schedules child Runs.

## Study Analysis and console

`lambdaforge.analysis.StudyAnalysis` is post-hoc evidence analysis, never a second HPO controller.
Terminal studies persist atomic versioned `analysis.json`; `lf results analyze SELECTOR
[--recompute] [--json]` uses the same core, while `results report` is an optional offline Plotly
renderer. Keep `current_observed_objective`, `best_observed_objective`, `final_objective` and
`selection_objective` distinct. A pruned Run is censored partial evidence and has no fabricated
final score; missing composite components are structured.

For a fixed evidence fingerprint, analysis is deterministic and separates screening from fresh-seed
confirmation. One leading seed is insufficient empirical stability; model-based seed uncertainty is
separate. A central min/max order is mandatory. Surrogate metadata must describe the validation
actually computed. Preserve authored conditional `when` predicates and validate every generated
point. Reliability combines support, CV quality, coverage and extrapolation. Top-region findings
must say observed or predicted; never imply an unpersisted proposal pool was measured. Confirmation
uses the scientific equivalence margin. Keep per-comparable-Run intrinsic resources separate from
total controller spend and scientific Pareto separate from resource Pareto. All effects remain
descriptive/predictive, not causal. Live analysis is provisional; terminal analysis is final.

Adaptive `trials` is the candidate budget and must be consumed by default; sparse coverage or low
analysis confidence is not convergence. Record-streak convergence is opt-in only through a positive
`convergence_patience` (zero/omitted disables it), and every terminal controller snapshot must
persist FINISH plus its exact budget/pool/explicit-convergence reason before final analysis.
Keep the deterministic proposal pool once in planning state. Per-Run specifications contain only
their candidate/seed values and a compact definition; never embed or copy the complete pool into
each specification, because valid large Studies must have linear planner memory.

Bare `lf` opens the Textual Research Console only on a TTY; non-TTY or no-command `--json` prints
help. Overview has separate Cluster, ordinary Work and Study panels, bounded resource history and
no raw JSON preview. Resource and learning-curve plots use the declared native `textual-plot`
widget; do not add a second chart renderer or persist the TUI's bounded samples. Resource history
uses a fixed five-minute window and an external colour key. Refresh must preserve focused panel and
stable entity key. Cluster and Dataset views use semantic cards/bounded sections, not provider JSON.
Root loading indicators are first-snapshot-only; subsequent refreshes retain and age the last good
snapshot. Work rows and bounded Work logs poll live with one request in flight per view.
A terminal `study_expected` Work remains in Studies and uses `study_job_id` to
read the latest telemetry-bearing Attempt. The six primary screens are Overview, Work, Studies,
Clusters, Datasets and Results. Real
navigation is `Study -> Trial -> Seed -> Epoch`; Enter/right pushes, while Esc/left or the Back
button pops one level; ancestor breadcrumbs pop the same real stack. Remote Study/Seed reads must
show loading/error states rather than empty evidence. Seed views preserve
current/best/final/selection semantics, censored prunes,
four-chart pages selectable by mouse or `n`/`p`, a horizontally scrollable all-scalar epoch table,
GPU/resources and isolated logs. Analysis views consume the persisted JSON.
The HPO tab consumes that same analysis: authored/observed/promising parameter domains, reliability,
response uncertainty/support, empirical dispersion, pair interactions and the complete persisted
Action/Trial/Reason controller history. It must not fit another model or imply causality.
Before an Execution result exists, use the bounded live `study.hpo_analysis` through the shared
analysis service; never hide live parameter evidence merely because final Results are unavailable.
Metric page changes reset reused plots to automatic limits; user pan/zoom applies only to the
currently displayed page. Plot categories with their real labels and expose exact point/bar values
on click. Keep terminal uncertainty numeric rather than drawing misleading pseudo-bands; explicit
optional Plotly exports may show genuine shaded intervals, hover, pairwise heatmaps and numeric 3D
surfaces. Coverage uses bounded cards/charts/tables, not raw JSON, and contextual help explains
statistical terms. Decisions append to `study/controller-history.jsonl`; keep
`controller.json.recent` bounded, load the full
history only on explicit drill-down and retain the available tail for legacy Studies without that
file. Internal
metric keys remain stable; human labels may be supplied by
`LightningTrainConfig.epoch_metric_display_names`, and automatic labels must never expose
`__lambdaforge_utility__` instead of “Composite selection score”. Keep trial markers in a separate
column from identifiers.
`Ctrl+P` may list only operations with functioning handlers; inventory names are not parity.
Widgets call Python domain services directly, and slow providers run outside the event loop.
Preserve selection/epoch/log scroll across refresh and last good data as visibly stale after a
transient failure. Destructive actions retain exact-target confirmation and preview/apply safety.
Treat `study: null` as unavailable evidence: open the Study/log workspace and attempt a bounded
service reload instead of converting it blindly or demoting it to ordinary Work.
Study cancellation is a semantic Work operation and must remain available before telemetry exists.
Study deletion is the same preview-first exact semantic Work deletion exposed with visible
progress; active Studies are refused until cancelled and display names never select the target.
Recurring refresh/provider failures render inline stale/error state; they must not emit a toast on
every polling interval.
`lf top`, `lf clusters setup` and `lf clusters modify` are retired public routes; do not restore a
second interactive implementation.

## Neural component route

Before adding a model, inspect `lambdaforge.nn.models` and manual section 14. Existing families
include MLP/CNN, extensive graph and equivariant models, sequence/Transformer/Conformer, sets,
tabular, vision, composition, generative, scientific/implicit and differentiable-tree models.
Do not add a generic `GNN`, redundant alias/factory or domain policy; add only a reusable primitive
with a precise tensor contract and focused conformance tests.

## Identity, attempts and ownership

The nearest `pyproject.toml`, not the active virtual environment, selects `ProjectContext`.
`[tool.lambdaforge].project_id` is an optional stable 1–80 character ID; otherwise derive it from
the resolved root. User cluster profiles/credential references and per-host GPU/process leases are
shared. Controller Jobs/groups/recents, local results/dataset indexes/cache, and remote
state/cache/jobs/datasets/environments are project-scoped. Default remote paths are below
`<workspace>/.lambdaforge/projects/<project-id>`; custom roots append `projects/<project-id>`, while
the lease root remains unscoped. Project catalogs recursively overlay the shared user profile so a
mirror can change without copying credentials/site policy. Never persist derived effective roots
back into an authored profile. Legacy Jobs are visible only when their recorded source belongs to
the current nearest project and must reconnect with recorded provider/work paths; never move old
remote bytes implicitly.

Hierarchy: Work -> Execution -> Run -> Attempt -> Job. Scientific identity includes Work import
path, consumer code identity, normalized parameters, file hashes, dataset content IDs, seed and
trial values; it excludes cluster, paths, IDs and time. Environment/hardware provenance is separate.

Normal execution reuses verified success. Retry means same Run/new Attempt; resume means compatible
checkpoints; rerun means a deliberate new Execution. Published datasets are durable independent
objects. Results/checkpoints are scientific state. Bundle/environment/cache bytes are
reconstructible. Deletion and cleanup must be exact-root, symlink-safe, idempotent and preview-first.

Dataset creation occurs only from `self.outputs.dataset(...)`; it streams members into the existing
DatasetArtifact v2/index/registry format. There is no dataset-build execution protocol. Preserve v1
manifest reads and immutable name/version conflict checks.

## Control-plane invariants

Keep `Transport` and `Scheduler` provider boundaries. OpenSSH multiplexing reuses a private
ControlMaster for its configured idle period. Credentials stay in interactive/keyring/env sources
and never enter argv, YAML, bundles, state or logs. Managed Python environments are immutable,
user-space and wheel-identified; do not modify system Python, CUDA, drivers or shell startup files.
CUDA usability requires an actual tensor probe, not `nvidia-smi` alone. Transient HPO GPU-memory
probe errors pause admission and preserve active Runs; never turn one observation failure into
whole-Study cancellation or launch from stale memory data.

Cluster `gpu_access.mode` is `auto|scheduler|exclusive|shared|command`. Auto selects scheduler for
SLURM and exclusive cooperative leases for direct hosts. Shared admits external occupancy only when
the operator explicitly accepts that risk. Command requires an argv `command_prefix` for a site
claim wrapper; never encode it as shell. Prefer a self-contained `gpu exec`-style wrapper. Optional
`claim_command` and `release_command` are an atomic pair; only `{gpu_count}` expands, the direct
supervisor releases on every terminal path, and persistent claims are invalid with SLURM. In
command/scheduler modes inherited `CUDA_VISIBLE_DEVICES` tokens are opaque grants: never replace or
broaden them. Missing/duplicate grants fail closed; scheduler grants are exact, while an adaptive
Study behind a command launcher may safely scale down to fewer non-empty inherited tokens. The
Research Console edits the
same catalog and credential services as native commands and never shells out to `lf`. Direct versus
SLURM selects process launch; GPU wrappers/claims are the separate `gpu_access` policy, so
`gpu exec` normally uses Direct. Superseded managed
environments are pruned only after a verified replacement is active and live-Job references are protected.
The Add/Edit modal must round-trip the complete durable `ClusterProfile`: guided common identity,
runtime, storage, SSH and GPU policies plus validated Advanced YAML for uncommon OpenSSH/SLURM
mappings. Revalidate with `ClusterProfile.from_mapping` before atomic persistence. Never put a
password value in editor/profile state; edit only its safe reference and leave secret mutation to
the credential service.

Direct/SLURM jobs retain durable state, heartbeat, logs, usage, cancellation and identity checks.
Provider outage is unknown state, not scientific failure. Do not contact a real cluster or run a
real scientific dataset while testing repository changes.
Work-level cancel must attempt every active Job in the semantic Work. Direct cancellation must
terminate and verify the full owned process set, including reparented/new-session workers identified
by the inherited exact Job marker; normal main-process exit enforces the same cleanup. Job/Attempt
cancel stays deliberately narrower. Never let consumer environment overrides replace framework
ownership markers.

## Modification checklist

1. Keep the final path `WorkConfig -> ExecutionPlan -> ControlPlane/Scheduler -> WorkRunner ->
   Work.run()`; infrastructure below Work may be reused.
2. Update `docs/MANUAL.md` and `docs/MANUAL.es.md`, both READMEs, both AGENTS files, schema/examples
   when affected and changelog for user-visible work.
3. Audit dynamic imports before deleting ordinary domain helpers, but delete obsolete execution
   adapters/tests rather than preserving compatibility.
4. Keep base dependencies light and providers lazy.
5. Test the smallest changed subsystem, then ruff, mypy, broader retained suites, wheel build and an
   installed-wheel CLI/scaffold smoke. CUDA tests must be run or explicitly reported as unavailable.
