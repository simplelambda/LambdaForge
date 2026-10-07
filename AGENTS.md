# LambdaForge agent guide

This is the low-token source of truth for agents using or modifying LambdaForge 0.17.0. Spanish is
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
| Resolve all automatic Study policy | `lf config resolve CONFIG [--format json\|yaml]` |
| Inspect project seed streams | `lf seeds [--role confirmation] [--count N]` |
| Read-only expansion | `lf run CONFIG --dry-run` |
| Execute | `lf run CONFIG [--on CLUSTER]` |
| Limited fresh adaptive/fixed Fleet Study | `lf run CONFIG --on-fleet FLEET`; see control-plane limits below |
| Deliberate new execution | `lf run CONFIG --rerun` |
| Interactive operation | bare `lf` (TTY Research Console) |
| Monitor semantic work | Research Console; `lf overview --json` |
| Work operations | `lf show/logs/cancel/retry/delete SELECTOR`; study Run: `show/logs WORK --run KEY` |
| Low-level jobs | `lf jobs list/show/logs/cancel/retry/delete`; `lf jobs clear [--apply]` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Dataset reconstruction/publication | `lf datasets preflight/compare/publish-candidate`; `docs/DATASET_RECONSTRUCTION.md` |
| Results | `lf results list/show/compare/analyze/report/replay`; portable Study: `lf export SELECTOR --output DIR` |
| Runtime diagnosis | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Cluster profiles | Research Console; `lf clusters add/set/unset/...` for automation |
| Inspect/reconcile storage | `lf storage status/reconcile [--on CLUSTER]`; reconcile `--apply` updates only the ledger |
| Preview safe storage cleanup | `lf clean [--on CLUSTER]`; add `--apply` after review |
| Scaffold | `lf init DIRECTORY` |

Append `--json` for automation and consume stable fields; use `--debug` only for framework
tracebacks. Local and remote run both return after durable asynchronous preparation unless
`--wait-for-submit` is explicit; `--dry-run` is direct and read-only. Never parse prose or secrets.

## Writing Work

Study recovery (adaptive, repeated seeds or fixed sweep) is `lf retry STUDY` or **Resume Study…**
in its console workspace. `--dry-run --json` exposes reuse/retry/pending evidence without launching.
Fixed designs restore owned execution/design/Attempt records, never require or fabricate adaptive
`hpo-control/state.json`; adaptive recovery still requires that state. It
reconnects a freshly prepared Job to the exact original owned Execution on the same cluster;
valid completed/pruned evidence, HPO decisions, seeds and spent budgets remain intact. Failed and
interrupted Runs create new Attempts from compatible checkpoints or start fresh when absent.
`--accept-code-change` is an explicit researcher acknowledgement of valid prior metrics/checkpoints,
not a waiver of configuration/input/seed/objective identity checks. Preserve immutable origin
`execution.json`, audit actual revisions in `recovery-history.jsonl`/`current-code.json`, protect
referenced owner Jobs from cleanup and use the existing cross-process lock. Never silently start
over if persisted recovery state is missing/corrupt, automatically retry consumer exceptions,
repeat valid confirmation seeds, or remove physical failed Attempt cost/history. Recovery currently
requires one Study per Execution; it does not migrate state between clusters. Recovery restores a
Study, not necessarily an epoch: checkpoint continuation requires explicit Work support.

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
and never syncs or deletes that researcher-owned mirror. New directory identities use the
versioned canonical tree algorithm, never shell `find | sort`: NFC path components, UTF-8 byte
component ordering, `/` separators, typed length-delimited records and empty directories; ignore
host metadata. Historical markers without an algorithm use the legacy reader. Configure with
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
most one request in flight per view. Overview must use one inventory pass per direct provider (or
active-job status only for schedulers without inventory), local Dataset registry counts and compact
Job/Study projections. Work/Studies root screens must not trigger
resource or Dataset probes. An unverifiable UNKNOWN Work may be forgotten only through
preview/apply local-history deletion; never delete its unverified remote workspace/process or call
it terminal.
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

ActivationCondition is the immutable AND-of-equality/membership authority shared through
ParameterSpace: historical `when: {parent: value}` and explicit `{eq: value}` normalize identically;
`{in: [...]}` requires unique finite parent-domain scalars and has order-independent identity.
Topologically order dependencies, omit inactive keys and never multiply then deduplicate branches.
Root resources alone inherit into steps; reject ignored root with/seeds/replicates/search/sweep/
execution/objective/analysis. Preflight distinguishes Study dispatch budgets from scheduler ceilings
(sum sequential, max parallel). Only live reliable direct GPU capacity may reject impossible
requests; scheduler login/command grants remain runtime-owned and never broadened.
Details: docs/CONDITIONAL_STUDIES.md.

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
confirmation. Only proposed candidates become public. Canonical `sweep.space` instead creates one
required identity for every finite combination times every authored seed; numeric ranges require
`points`, and optional `reference` selects explicit paired comparisons. `strategy: exhaustive`
normalizes to that same fixed `StudyDesign` and common ARI dispatcher. Sweep evidence cannot be
scientifically replanned, although ARI may reorder/place/recover it; Work-internal patience remains
valid. Adaptive-only candidate, replication, pruning and confirmation controls are rejected in a
sweep. Default confirmation seeds are disjoint; `confirmation_seeds: []` deliberately disables
them. `search.fidelity` is explicit cumulative Work-defined budget; `self.fidelity` plus checkpoints
implement continuation and LightningRunner bridges epoch budgets. Controller decisions/state live
in `hpo-control/decisions.jsonl` and `state.json`. Confirmation is immune to performance
pruning/preemption; an incomplete set persists `confirmation_incomplete` and cannot select a
survivor-only mean. Normalize authored YAML once into `StudyDesign`, `EvidencePlan`,
`ExecutionPolicy` and objective policy. Canonical adaptive syntax uses `search.budget`, `.space`,
`.replication` and `.pruning`; legacy flat fields remain aliases. Top-level `execution` owns
`runs_per_gpu`, `max_parallel`, failure retries and Run/time limits; conflicting aliases fail.
`objective.practical_margin` is the single scientific-equivalence authority. Explicit `trials` is
the executed-candidate ceiling; when omitted, automatic scientific convergence owns stopping. A
prefix-stable `scrambled-sobol-prefix-v1` generator materializes and expands bounded windows without
changing old candidates; `proposal_pool_size` is only an advanced window override. Event-driven refill compares `START_NEW`,
`DESIGNED_PROBE`, `ADD_SEED`, `PROMOTE_FIDELITY` and `RESUME_PREEMPTED` after every terminal event.
It automatically balances posterior practical-improvement opportunity O with unresolved-question
entropy K per observed incremental cost; neither is a calibrated information gain or a user-facing
confidence value.
Startup is not a barrier. Geometry-derived initial anchors are protected scientific obligations;
other undispatched entries remain provisional. Use `DEFER_STARTUP_ANCHOR`,
`CANCEL_PLANNED_DISPATCH` and `CANCEL_SCIENTIFIC_ACTION` with their exact meanings, and keep pending
`(candidate, seed, phase, fidelity)` identities unique. Root `search.reduction_factor` and
`search.confidence` are removed in favour of separately owned fidelity/seed/pruning controls.
Omitted `startup_trials` derives a deterministic rank/coverage/D-optimal/maximin
`InitialDesignPlan` from `ParameterSpace`; an explicit value is its authoritative anchor budget.
It must never depend on parallelism. Interleave first seeds across distinct startup candidates
before extra seeds; spare capacity may use replannable `OPPORTUNISTIC_COVERAGE`.
Once an adaptive candidate is proposed, `replication.minimum` distinct seeds are required evidence,
not cancellable hints. Performance-pruned Runs count as attempted/censored but not complete response
evidence; infrastructure failure counts only after retries. Reaching candidate budget stops new
candidates, never existing required debt, continuation, fidelity or confirmation.
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
`ParameterSpace` is the single authored geometry for Sobol, adaptive/Bayesian sampling, scientific
design, resource similarity and Study Analysis. Log scales use log coordinates; integer,
categorical and conditional activity semantics must not be re-inferred from observed candidates.
Candidate mappings and IDs remain unchanged. Coverage state distinguishes attempted/censored
search coverage from completed response coverage and includes matched-context diversity.
Scientific design may request `COVER_PARAMETER_VALUE` or `COVER_INTERACTION_CELL`; their value
must decay as diverse evidence resolves the question. A checkpointed performance-pruned Run may
later use `SCIENTIFIC_CONTINUATION` for a high-value question: preserve Trial/seed identity, create
a new Attempt, bypass competitive pruning, consume Run/time but not candidate budget, and retain
the original prune as valid censored evidence.
`objective.constraints.METRIC.min/max` are explicit same-checkpoint guardrails. Across seeds,
`seed_aggregation` is legacy `mean`, `worst` or `lcb` with optional `confidence`; missing/insufficient
LCB evidence is infeasible. A utility component may also be constrained. Never
infer guardrails or hidden multi-objective weights from other metrics. Integer `runs_per_gpu` is a
hard per-device maximum and integer `max_parallel` a hard global maximum. `auto`/null removes the
artificial cap but still requires a finite host/resource-derived internal dispatch ceiling.
`resources.gpu_memory` is optional and, when present, is a user safety floor per new Run. Effective
admission uses the maximum of that floor, a candidate-specific conservative future envelope and
known OOM lower bounds, bounded again by live physical free VRAM. Physical VRAM is authoritative.
Never multiply the floor by active Runs or
maintain a second reservation. Cold start creates one `BASELINE_ADMISSION` progress lane per idle
granted GPU before co-location; a baseline never consumes an exploration lane or triggers
`UNVALIDATED_LADDER_STEP`. Live right-censored PID/allocator trajectories, phase/progress and
durable checkpoints then evaluate incremental `SAFE_ADMISSION` or `EXPLORATORY_ADMISSION` steps
before terminal Runs exist. Keep one protected compatible-GPU lane at baseline concurrency, one
uncharacterized ladder step per evidence class, and share
provisional knowledge across sibling GPUs. Larger groups may explore distinct resource questions
concurrently, but never duplicate an equivalent experiment or consume the final protected lane.
Tiny allocator maxima are drift when robust physical/allocator evidence says so; persistent signed
growth remains a ramp. `ResourceTrajectoryStatistics`, not the bounded display curve, is the
incremental authority. Hazard advances on progress/allocation evidence, never duplicate polling;
keep phase-specific rare tails and learn relevant phases from compatible Works rather than require a
training checklist. Integrate WaitRegret as exact physical segments, persist it across restart and
reset only when useful idle opportunity disappears or is satisfied. Compare
WAIT/EXPLORE/CHECKPOINT_THEN_EXPLORE to the next evidence event; checkpoint rollback is per resident,
checkpoint cost is learned and a request is not durable acknowledgement. Never invent a one-second
duration or use raw controller score as resource currency: consume `ScientificActionValue` rank,
normalized value, uncertainty and cost. Persist changed evaluations and WHY-WAIT reasons without
poll spam. Next-event estimates retain cadence uncertainty when repeated intervals exist; otherwise
it remains unknown. Scientific pruning is independent of resource completeness.
Capacity-triggered frontier refill and completion-triggered refill share the deferred initial-design/
seed queue and exact deduplication. Generic stepped Work metrics publish bounded
`metric-progress.json` for admission; never scan scalar history per resource poll or equate scalar
progress with a durable checkpoint. Fresh adaptive children use the dispatch-time
`HostResourceLease`; rebalance Torch/BLAS/OpenMP limits and process-tree CPU affinity as residents
change, never native pools in an embedded controller. RAM/storage commitments likewise follow
actual residents while the hard aggregate host ceiling prevents overcommit. Persist the lease and
process-tree CPU/RSS evidence. Exact NVML process attribution is preferred; aggregate
fallback remains censored. OOM constraints belong to the exact packing unless reliable candidate
resident-plus-allocation evidence establishes an intrinsic bound; never restore global concurrency
backoff. Compatible exact/censored
history enables denser best-fit packing. `RESOURCE_BLOCKED` is reversible and scientifically
neutral; `RESOURCE_INFEASIBLE_ON_DEVICE_TYPE` requires a hard lower bound above device capacity.
Never retry a compatible OOM candidate at the same or lower effective headroom. The resource
planner consumes a bounded ranked scientific frontier, may safely backfill, protects heavy work
from starvation using predicted completion windows, and may stop memory-feasible co-location when
measured aggregate throughput would not improve. If that frontier is entirely blocked, request
bounded unseen extensions from the same scientific policy without waiting for terminal Runs. Exact
queued/active action identities and explicit no-new-alternative detection must stop loops; never
iterate random resource candidates. Every admitted GPU Run uses a fresh one-worker spawned process that exits on
result/error; never restore a persistent CUDA pool because idle contexts retain VRAM and can
deadlock queued Runs. GPU memory probing must remain a short-lived child process: the controller
must not retain one CUDA context per device or consume a scientific slot. Repeated objective observations need
`step=` for early stopping. `LightningRunner` bridges them automatically; custom loops return at a
safe boundary when `self.stop_requested` is true.

Every adaptive CPU/GPU Run owns a fresh one-worker process. An exploratory CUDA OOM enters
`RESOURCE_RECOVERY` as a new checkpoint-compatible Attempt of the same logical Run and is bounded
by persisted failed-placement dominance rather than generic retry count. A lost/killed worker and
non-exploratory CUDA OOM may
retry as a new checkpoint-compatible Attempt up to `failure_retries` (default 1, max 3); a repeat is
terminal. OOM is censored resource evidence, not an objective value; persist its attempted
allocation/headroom/residency/source and recompute every pending placement before retry. Prefer
best-fit admissible GPUs while preserving a predicted window for still-relevant heavy work. Do not kill healthy siblings or retry forever. Never
retry arbitrary consumer exceptions. One exhausted Run makes the enclosing Work
honestly failed but must not cancel unrelated active/queued Runs. Telemetry records logical GPU
index and exact inherited token; never infer or broaden physical devices from that display field.

Study telemetry is a bounded read model, not another result store. It references per-Run logs and
scalar JSONL and never copies checkpoints/outputs. Collection reads are hierarchical:
`overview --json` exposes only compact Study counts/objective/leader metadata; opening one Study
loads v3 `study/interactive.json` Trial table cells/aggregate states, never all seeds/parameters.
One Trial loads its own seed/parameter projection; Run curves/logs are selected-Run-only.
HPO/resources use separate host-precomputed files; Analysis uses an aggregate panel without Run
records, while explicit HTML export requests full analysis. Provisional post-hoc analysis is
host-cached on demand outside the scheduling heartbeat; never refit it merely to open HPO.
All Study views use 512 KiB byte pages plus generation/unchanged checks, not an aggregate 8 MiB
ceiling. Hidden parent screens must stop polling. Complete controller history is paged only on
explicit Action-history drill-down. Legacy rich
summaries must be projected on their execution host rather than transferred whole; and
`lf show WORK --run KEY --json` returns parameters,
down-sampled curves, current/best objective and epoch, timing/failure/log plus finalized artifact
paths/checksums/retention; the seed Artifacts tab consumes the same metadata without copying or
opening remote content. `lf logs WORK --run KEY`
isolates output. Completed HPO uses the mean of each seed's best observed checkpoint; current
same-step curves drive pruning, and pruned Runs are terminal censored evidence—not failures—and
excluded as exact values from completed-objective fitting/statistics. Their parameter-region
pruning rates remain visible. Any performance-pruned seed censors the whole candidate; never average
its earlier completed seeds as survivor-only evidence. Group completed evidence by exact
`(candidate,target,maximum)` rung; never average heterogeneous fidelities or race incompatible rungs.
A candidate-level joint survival model with uncertainty softly
modifies acquisition; multiple seeds do not overcount one candidate and operational failures or
scheduler preemption remain neutral. Default pruning requires two distinct
uncompetitive common steps through `early_stopping.confirmations`. `min_step` is eligibility, not
a forecast horizon: use an authored fidelity boundary or one observed local window, and never stop
a candidate still competitive at the exact common step solely from its fitted slope. Lightning scalar
callback metrics plus epoch/validation time are automatic. Custom trainers log curves with
`self.metrics.log(name, value, step=epoch)`; use `progress.update` for coarse progress and
`self.log`/print only for human narration.
Two distinct completed curves identify a retrospective comparison but do not by themselves make
pruning safe. Strong runtime pruning additionally requires finite-sample residual resolution,
retrospective interval coverage and competitive-probability discrimination consistent with the
authored threshold. Persist operational and strong-calibration readiness separately. Protected
startup anchors, required evidence, confirmation and scientific continuations never receive
competitive pruning.

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

Resource scheduling persists a bounded versioned trace. `lf results replay EXECUTION --policy
recorded|ari-v3.1|ari-v3-compat|ari-v2-compat --json` uses the ResultStore API; it is factual only
until `counterfactual_divergence` and labels the remainder trace-conditioned simulation. Legacy
policies exist only as replay approximations, never production schedulers. Missing old trace data
must fail explicitly rather than parse human logs or fabricate metrics.

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

Project HTML tabs use `outputs.html_section(name, title=..., section=...)`, an ordinary managed
file, not a plugin/second runner. Explicit report generation alone collects finalized owned HTML
with exact fingerprints; remote content is paged/cached, never fetched by overview/HPO polling.
Group repeated Run documents by section and render encoded HTML lazily in opaque-origin offline
iframes. Never insert consumer scripts in the parent DOM or relax network/ownership checks.
Documents embed assets/data; limits are 16 MiB/document and 64 MiB/report. Chart axes identify their
member metrics; empirical histogram/CDF distributions are presentation, never new HPO evidence.

Study HTML opens Overview, with exact candidate states and contextual importance. Exclude conditional
parameters from the global chart; retain their branch/support/model score in a separate table, never
label predictive variation causal responsibility. Interpretation/questions use carousels, complete
findings are paged and shared diagnostics live in help. English/Spanish presentation preferences
never translate authored identities or mutate persisted evidence. Ordinary controls use one anchored
searchable dropdown, not a modal; only explicit global search/inspection uses dialogs. Parameters
checkboxes add/remove lines in one chart, preserving independent axes for unknown/different units.
Explore's
primary Analyze/By/Compare controls infer observed views; explicit axes/type/palette/normalization
stay Advanced. Parameters retains removable comparison chips. Configured question cards and category
hierarchy are presentation of persisted semantics. Research owns the finding list; Evidence owns
conclusions/diagnostics/JSON. Arbitrary observed interactions must not relabel persisted HPO models.
Keep legacy saved-view keys, current/final/censored distinctions and the individual Run dashboard.

`study/controller.json.initialization` is durable immutable scientific authority; `recent` is only
a display tail. Preserve objective, practical margin, authored geometry/conditions, policy and seed
streams across refresh/export/reload. Legacy migration reads the durable history once. Live insight
views reuse the controller scientific snapshot. Seed noise v2 distinguishes unresolved/provisional/
calibrated variance and contextual support; one repeated architecture is provisional. Replication
value includes its effect across unresolved questions, not only incumbent comparisons. Endpoint
calibration continuations carry `PRUNER_CALIBRATION_EVIDENCE` and remain protected. Sampled completed
NVML envelopes are usable resource history but never exact lifetime peaks; keep incomplete data
censored and persist `sampled_history_count`. Generic step/progress rates supply throughput; optimize
aggregate evidence/time rather than process count. Human analysis may use higher resampling precision
outside the controller heartbeat; display realization agreement, resolution and noise diagnostics.

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

For sweeps without adaptive INITIALIZE, persisted StudyDesign.space owns the Analysis geometry;
never infer unconditional required keys from sparse branches. Host readers pass sweep geometry
explicitly to old immutable runtimes. Empty valid response contexts are insufficient evidence;
bounded remote errors must preserve the terminal exception, not only the traceback header.

Stable screening stops ordinary optimization, not scientific work: material feasible support and
interaction debt plus confirmation continue until exhausted or hard-budgeted. Compute
`scientific_status` from parameters and material interactions. Scalar-objective
`diagnostic_metrics` are terminal display evidence only and never affect HPO or selection.

`lf export SELECTOR --output PARENT` and the Research Console **Export Study…** action use one
domain service to export the newest Attempt in its current state. A succeeded finalized Execution
uses an atomic, non-overwriting `NAME--EXECUTION_ID` folder; every other state uses a timestamped
snapshot folder and records `status`, `export_kind` and `finalized` in its SHA-256 manifest. Include
only evidence persisted at capture time: exact Execution/Run evidence when available, persisted
Study/controller records, Job lifecycle files, applicable analysis/replay, retained
checkpoints/weights and explicit finalized published artifacts. A pre-Execution snapshot contains
only configuration and control-plane evidence and must never fabricate Runs or conclusions. Never
copy shared datasets, environments, cache or the staged project tree into this package; preserve
their provenance references. Reject symlinks, special files and archive traversal, remove temporary
provider archives on every exit and never publish a partial local export. `--profile default`
keeps complete scientific/decision evidence, deterministic down-sampled metric curves and
summarized high-frequency resource streams; `--profile full` keeps raw streams. Every transformed
or omitted source is listed with reason, byte size and SHA-256. Provider transfer uses a compressed
ZIP64 archive. Put controller transfer/extraction temporaries beside the selected output
parent rather than in system `/tmp`, reuse bytes only from owned extracted evidence, and publish
factual export phases through the optional progress callback. The Research Console keeps exports in
the background and exposes their state outside the initiating workspace without notification spam.

Omitted `seeds` uses the project's versioned collision-free 32-bit `replicate` stream; candidates
share ordinals and minimum replication defaults to one. Explicit `seeds` is authoritative and
finite; `replicates` selects a stream prefix. Fresh automatic confirmation uses the disjoint
`confirmation` stream. Omitted adaptive `trials` uses versioned scientific convergence; sparse
coverage or low confidence alone is not convergence. Record-streak convergence remains a legacy
explicit override. Every terminal snapshot persists FINISH plus the exact convergence/budget/space
reason. Per-Run specifications contain only
their candidate/seed values and a compact definition; never embed or copy the complete pool into
each specification, because valid large Studies must have linear planner memory.
An omitted sweep replicate count means sequential complete shared-seed blocks. Primary stopping
uses the variance-adaptive `paired-pm-eb-cs-v1` predictable plug-in empirical-Bernstein confidence
sequence over paired bounded differences. It controls one authored primary family (treatments
against `sweep.reference`, otherwise the practical top-set pairwise family), supports
`PREFERRED`/`PRACTICAL_TOP_SET`/reference relations, and keeps a one-complete-block lookahead to
avoid a statistical GPU barrier without letting partial evidence affect stopping. No pruning or
per-cell racing is allowed. A missing cell is incomplete, and practical equivalence requires the
authored margin. The family alpha defaults to `0.05`; only automatic sweeps may override it with
advanced `sweep.sequential_alpha`, which fixed replicates reject. Persist point-estimate leader,
exact descriptive conclusion/stability and formal
sequential evidence as three different quantities; `UNRESOLVED` never means converged.
Terminal Studies reconcile every queued identity and expose execution `status`, `design_status`,
`scientific_status` and `finish_reason`. Fixed designs may be complete while scientifically
unresolved; time/Run exhaustion with required debt is incomplete. Keep observed and
evidence-complete candidate counts distinct. Sweep analysis is paired by exact shared seed and
block-resamples seeds; never substitute another seed for a missing cell. Human text and confidence
must derive from the same `ExactScientificConclusion`, while point-estimate leader remains separate.
Without `objective.practical_margin`, unstable winners are `NO_CLEAR_PREFERENCE`/`UNRESOLVED`, not
practical equivalence.

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
surfaces. Seed metric exports are one offline dashboard with a searchable/grouped selector, at most
four objective/validation defaults, raw and per-metric normalized curves, latest/delta bars,
same-step correlations, shared-step X/Y relationships and exact descriptive statistics; categories
collapse and selecting every metric is opt-in. Study HTML navigates persisted candidate comparison,
parameter responses, interactions, coverage, resources and findings without fitting a browser-side
surrogate. Heatmaps offer explicit neutral-centre palettes. Visual preferences use a unique
per-generated-file local-storage key: reopening preserves them and regeneration resets them.
Resizable panels, custom metric categories and saved chart specifications are presentation-only;
they must keep stable metric keys and consume embedded persisted observations without recomputing
scientific evidence.
Study **Explore** supports observed parameter/metric 2D overlays, arbitrary X/Y parameter + Z
metric 3D scatter, heatmaps and numeric surfaces. Keep authored categorical labels and finite
domain gaps; missing values are null, never zero/predicted. Partial/pruned observations are opt-in
display evidence and cannot replace final selection. Exact-value SD is descriptive across Trials,
not seed uncertainty. Package the offline `analysis/assets/study-charts.js` renderer in wheels.
Chart controls preview immediately; Save only persists a view. Additional Y series use searchable
checkboxes. Parameters uses the same observed-metric renderer and a Y selector; keep persisted
adjusted objective/model evidence separate and never relabel it as another metric. Empty plots
must explain absent selection/parameter/filter data without fabricated fallback values.

Study Analysis v8 adds a deterministic `research` read model. `MetricCatalog` owns meaning,
`AnalysisProfile` validates class/YAML declarations, and `ResearchAnalysis`/`ResearchDiagnostics`
own bounded profiling/discovery. Offline assets only render; never run these diagnostics in the
HPO heartbeat or fit scientific models in JavaScript. Freeze declarations and registry defaults
in owned `analysis-semantics.json` before execution; child specs reference it, not a copied catalog.
Keep separate analysis-semantic identity from scientific evidence identity. Test/derived test
metrics cannot govern objective/constraints and cannot enter provisional discovery. Censored
candidates never become exact final evidence. BY-adjusted exploratory permutations are not formal
adaptive-sampling inference or scientific confidence. Show constants/missing metadata explicitly,
preserve family coordinates/lineage, and separate unknown/different-unit Y axes unless the user
opts into visual normalization. Research inbox/search/metric browser/validated saved views use
persisted domain data. Routes and limits: `docs/RESEARCH_ANALYSIS.md` (Spanish sibling).
Coverage uses bounded cards/charts/tables, not raw JSON, and contextual help explains
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

Storage admission shares per-filesystem owned leases: physical free minus active/new commitments
must preserve the larger absolute/percentage safety floor and available inodes. Reservation is not
usage. Never pool volumes or infer foreign owner death from a local PID. Automatic/manual GC shares
`StorageService/StorageOperations`. Environment/runtime marker acquisition is atomic with GC,
including reuse verification. Never replace an invalid complete prefix underneath referenced Jobs.
Publication-copy leases target the destination volume and GC only owned cache roots, never external
publication folders. `lf storage reconcile` measures explicit diagnostic drift; `--apply` only writes
the ledger, never scientific state. It is not a per-write ledger. Protect recovery references and
independent live Work-cache leases.
**Clear storage…** confirms native cleanup; **Clear output** only clears text. Successful unpinned
checkpoints have configurable grace; failed/interrupted recovery remains protected. Named
`checkpoints.pin/unpin`, `outputs.from_checkpoint` and dataset `source_checkpoint` declare independent
publication/retention; never mutable hardlinks. Read `docs/STORAGE.md` for exact scope and unfinished
provisioning/code-layer guarantees rather than claiming those are implemented.

Project-scoped dataset roots affect new publications only. Preserve a verified legacy absolute
placement already recorded for the current project. Content-hash subdirectories are immutable
identity, never random paths: changed bytes require a new logical version. Publication must check
registry compatibility before commit and roll back a newly committed directory if final
registration fails. A conflicting remote registration is safely reconcilable only when its exact
registered directory is proven absent; unreachable or existing bytes remain a hard conflict.

Dataset creation occurs only from `self.outputs.dataset(...)`; it streams members into the existing
DatasetArtifact v2/index/registry format. There is no dataset-build execution protocol. Preserve v1
manifest reads and immutable name/version conflict checks.

Dataset producers call `outputs.dataset_preflight(...)` before expensive work; `datasets preflight`
inspects all indexes without mutation. `intent="rebuild"` seals a pinned unregistered candidate.
`scientific_identity` explicitly declares sources/selection/labels/configuration/algorithm; never
infer operational exclusions. `DatasetComparison.compare`/`datasets compare` verifies all bytes
before trusted project verification with variable-specific policy; identifiers/targets/partitions/
source contracts stay exact. Equivalence keeps both content IDs, never replaces a version.
Publication failures protect artifacts/checkpoints and sealed candidates; `datasets publish-candidate`
previews/applies bytes-only recovery without rerunning science or changing failed history. Inventory
is read-only, merges (name, version, content ID), discovers all configured Console locations and
shows conflicts without ambiguous mutations. See `docs/DATASET_RECONSTRUCTION.md` (Spanish sibling).

## Control-plane invariants

Public `lf run CONFIG --on-fleet NAME` supports fresh adaptive/fixed/repeated/automatic paired Studies,
local coordinator and managed member profiles. Run Work exposes the same targets. Checkpoint/fidelity
continuation, co-location, recovery/adoption, live expansion and distributed export remain gated; see
`docs/COORDINATED_STUDIES.md`. Never describe independent JobGroups as a coordinated Study or
instantiate a second optimizer. Dry-run never acquires grants. Do not claim SSH/command/SLURM
acceptance from loopback tests.

`WorkRunner(dispatcher=...)`, `PreparedShardExecutor` and `CoordinatedDispatcher` retain one native
planning/execution boundary. Persistent owned member Jobs execute finite concrete waves and reuse
native isolated Run/ARI processes. Offers require fresh exact owner heartbeats and all five attested
code/environment/input/numerical/hardware fields; preparation placeholders never enter GlobalRuns.
Inherited GPU tokens remain opaque and cannot broaden. Busy baseline waves offer zero spare slots.
The controller must never compact or read scalar files by resolving execution-host paths locally.
Native worker retention follows durable envelope publication. Live/terminal show/logs use verified
member placement and lazy host-side curves; bounded active scalar summaries are display metadata,
not accepted terminal response evidence. Semantic cancellation stops the parent before enumerating
exact children.
Ordered native scalar streams use bounded complete records, byte cursors and exact lease identities.
Terminal results reach the native planner only after the full stream; interrupted unacknowledged
appends are recovered, never acknowledged corruption. Native pending-aware acquisition and pruning
remain central. Stop commands are immutable/idempotent and do not fabricate terminal evidence;
protected evidence cannot be pruned. Unleased reprioritization audits prior invocations and keeps
science immutable. Verified scalar mirrors, not remote paths, feed historical calibration.
Job-history deletion protects Fleet dependencies and previews the whole family; unsupported retry
must never replay a Fleet as a single-cluster Work, and unsupported export must not omit members.

Fixed/automatic result callbacks replace the unstarted frontier without withdrawing leased or
accepted identities. Preserve coordinator v2 leases/fences, original clock/budgets, quarantine and
unknown ownership. Physical observations and missing heartbeats do not prove granted capacity or
owned-Attempt loss. Internal pause/drain/catalog controls are not public live lifecycle support.
Keep recent scientific backfill, paired-block acceptance and evidence semantics unchanged.

Keep `Transport` and `Scheduler` provider boundaries. OpenSSH multiplexing reuses a private
ControlMaster for its configured idle period. Credentials stay in interactive/keyring/env sources
and never enter argv, YAML, bundles, state or logs. Managed Python environments are immutable,
user-space and wheel-identified; do not modify system Python, CUDA, drivers or shell startup files.
CUDA usability requires an actual tensor probe, not `nvidia-smi` alone. Transient HPO GPU-memory
probe errors pause admission and preserve active Runs; never turn one observation failure into
whole-Study cancellation or launch from stale memory data.

Cluster `gpu_access.mode` is `auto|scheduler|exclusive|shared|command`. Auto selects scheduler for
SLURM and exclusive cooperative leases for direct hosts. Shared admits external occupancy only when
the operator explicitly accepts that risk. Shared Jobs register non-exclusive access, permit other
shared Studies and block new exclusive grants; Run admission uses per-physical-token host locks,
fresh VRAM rechecks and cross-Study start staggering. Prefer observed compute-idle permitted GPUs;
preserve best fit when all are busy/unknown. Never bypass CPU/RAM Job ceilings to exploit spare
VRAM. Fixed/repeated Study parallelism counts required Runs, not just distinct candidates.
Environment build contention waits through provisioning; optional cleanup defers under cache
leases without weakening reference protection. Command requires an argv `command_prefix` for a site
claim wrapper; never encode it as shell. Prefer a self-contained `gpu exec`-style wrapper. Optional
`claim_command` and `release_command` are an atomic pair; only `{gpu_count}` expands, the direct
supervisor releases on every terminal path, and persistent claims are invalid with SLURM. In
command/scheduler modes inherited `CUDA_VISIBLE_DEVICES` tokens are opaque grants: never replace or
broaden them. Missing/duplicate grants fail closed; scheduler grants are exact, while an adaptive
Study behind a command launcher may safely scale down to fewer non-empty inherited tokens. The
optional `visibility_command` reports the current comma-separated opaque grant; `[gpu, exec]`
derives `[gpu, env]`. Intersect it with the original grant, stop only verified workers on revoked
tokens and checkpoint-requeue those logical Runs; keep siblings running and block new admission
when the ownership probe is unavailable. Never accept a newly reported physical token. The
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
