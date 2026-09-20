# Changelog

[Español](CHANGELOG.es.md) · English

All notable LambdaForge changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The repository currently has no Git
tags; the 0.1.0 and 0.2.0 entries below were reconstructed from their version commits and packaged
metadata rather than invented release numbers.

## [Unreleased]

### Added

- Added a deterministic `InitialDesignPlan` whose protected anchors follow canonical
  `ParameterSpace` rank, authored discrete/conditional/numeric coverage, D-optimal information and
  maximin separation—independent of physical parallelism. Bounded search/response coverage state,
  matched-context pair summaries, explicit value/cell probes and hard-infeasible anchor replacement
  are persisted for the Research Console and automation. Controller restart reconstructs completed
  Attempts, protected anchor debt and pending scientific actions without replaying finished Runs.
- Added checkpoint-backed `SCIENTIFIC_CONTINUATION`: a performance-pruned logical Run can resume as
  a new Attempt with the same Trial/seed to answer a later high-value scientific question, without
  fabricating a final objective or consuming another candidate slot. Added `runs_per_gpu: auto` and
  explicit `max_parallel: auto` while retaining finite host/resource-derived dispatch ceilings.
- Refined Adaptive Resource Intelligence to v3.1 with polling-invariant incremental sufficient
  statistics, learned phase-specific peak hazard, exact segmented WaitRegret persistence,
  per-resident checkpoint rollback/cost and tail-aware joint P(fit) composition. Scheduling now
  consumes an explicit `ScientificActionValue` contract and persists compact, versioned traces and
  calibration evidence without turning raw controller scores into resource currency.
- Added `lf results replay EXECUTION --policy ...` and the matching `ResultStore` API for factual
  replay through the first counterfactual divergence and explicitly trace-conditioned simulation
  afterwards. The trace benchmark reports concurrency, utilization, scientific rate, throughput,
  OOM classes, rollback/checkpoint cost, starvation, calibration and prediction error; unknown
  counterfactual quantities remain null.
- Added one canonical authored `ParameterSpace` for Sobol, adaptive/Bayesian sampling, scientific
  design, resource similarity and post-hoc Study Analysis, with shared logarithmic, ordinal,
  categorical and conditional-inactivity semantics while preserving candidate identities.
- Evolved the existing scheduler to Adaptive Resource Intelligence v3. Live trajectories now learn
  material-growth noise, update a weighted phase-aware peak-survival posterior and contract rare
  capacity tails without erasing them. A dimensionally coherent WAIT/EXPLORE comparison normalizes
  scientific ordering, integrates idle-capacity WaitRegret, preserves protected concurrency
  ladders and can request a cooperative Lightning checkpoint before a valuable probe. Persisted
  ExplorationEvaluations expose P(fit), hazard, rollback, information value, wait regret and exact
  rejection reasons without polling spam. Phase-complete pruned Runs remain valid resource evidence.
- Replaced terminal-only GPU cold start with Adaptive Resource Intelligence v2. Active Run PID/NVML
  memory, allocator peaks, phase/progress trajectories and durable checkpoints now feed a
  future-residual model across compatible GPUs. The existing planner distinguishes safe and
  expected-value exploratory admission, advances one unvalidated 1→2→3 step at a time, preserves a
  protected progress lane and persists concise resource transition events. Larger grants may run
  distinct non-redundant resource experiments concurrently; live throughput stops a ladder whose
  aggregate work rate has already regressed.
- Added packing-specific OOM evidence and checkpoint-aware `RESOURCE_RECOVERY`. Unattributed OOMs
  no longer fabricate intrinsic candidate bounds; dominated resident-set experiments are not
  repeated, while a failed heavy+heavy placement does not globally forbid heavy+small packing.
  Resource-space distance now respects authored linear/log/categorical/conditional dimensions and
  widens predictions outside empirical support. Calibration uses parameter-aware leave-one-out
  underprediction, nearby censored bounds widen predictive tails, and atomic active snapshots
  survive interruption only as provisional—not falsely live or exact—knowledge.
- Exposed each Study Run's finalized managed artifacts through `lf show WORK --run KEY` and the
  seed **Artifacts** tab. The shared bounded read model reports the preferred usable path,
  managed/published locations, role, MIME type, byte size, SHA-256, retention and metadata without
  copying or opening remote content.
- Added candidate-specific adaptive resource intelligence for HPO. Bounded physical/allocator
  trajectories spanning the complete active Run, exact and censored peaks, duration/time-to-envelope, OOM lower bounds and
  Work/code/environment/hardware compatibility signatures now feed a persistent mixed-space
  bootstrap demand model that can warm-start compatible Studies.
- Added a resource placement layer after scientific ranking: conservative future commitments,
  hard dominated-retry rejection, explicit `RESOURCE_BLOCKED` versus device infeasibility,
  best-fit multi-candidate packing, predicted-window backfill, heavy-action reservation and learned
  co-location throughput checks. Predictions are conditioned per heterogeneous device; an entirely
  blocked frontier gets one bounded scientific extension, while hard device-type impossibility
  terminates explicitly instead of polling forever. A CPU-only synthetic benchmark reports utilization, useful
  actions/time, blocking, starvation, OOM waste and prediction error against fixed one-Run packing.
- Exposed the backend resource read model in Study Resources: per-device physical free/external/LF
  current/future commitments, predicted headroom, candidate peak interval/support/calibration, P(fit), admission or
  blocking reason and safe-backfill choice. Final Study Analysis records resource-conditioned
  sampling without treating resource scarcity as poor scientific evidence.

### Fixed

- Fixed equal-distance live resource evidence attempting to compare immutable evidence objects
  during adaptive GPU prediction. Resource-neighbour and exploratory-placement ties now have
  explicit deterministic scalar keys, so a valid Study cannot terminate with an
  `ActiveResourceEvidence` ordering `TypeError`. Work failure paths and final Study Analysis are
  also normalized to strict JSON instead of allowing a `PosixPath` to break terminal analysis;
  both conditions are classified as internal framework errors rather than user configuration.
- Replaced monolithic live-Study transfers with a compact `interactive.json` index, lazy selected-
  Run reads and paged controller-history JSONL. Research Console refreshes now preserve table/log
  scroll and explicit plot pan/zoom instead of rebuilding the user's viewport on every poll.
- Fixed an event-driven scheduling hole where controller-deferred actions could remain visible as
  queued while the dispatch queue was empty, leaving a granted GPU idle until another Run ended.
  Resource readiness now requests one bounded scientific frontier immediately and the existing
  memory/throughput planner remains authoritative for admission.
- Fixed false curve pruning caused by treating `early_stopping.min_step` as a forecast horizon.
  It is now only the evidence threshold; absent an authored fidelity boundary, pruning uses one
  observed local window and cannot stop a candidate that is still competitive at the exact common
  checkpoint solely because of a noisy fitted slope.
- Added live command-allocation reconciliation for sites such as CITIUS: `gpu exec` automatically
  pairs with `gpu env` (or an explicit `visibility_command`). A shrinking grant terminates only the
  verified workers on revoked opaque tokens and requeues their logical Runs from checkpoints;
  unaffected devices continue, restored tokens become eligible again, and probe failure blocks new
  admission without inventing or broadening GPU ownership.
- Made Research Console loading hierarchical and bounded: Overview now performs one inventory pass
  per direct provider (or active-job status reads for schedulers without inventory), uses local
  Dataset registry counts and emits compact Job/Study projections;
  Work/Studies avoid unrelated resource and Dataset probes; Study analysis, full action history and
  logs load only on their tabs, while per-epoch evidence remains per-seed. UNKNOWN Work/Study
  history can now be removed through an explicit previewed local-forget operation without deleting
  an unverified remote process or workspace.

- Prevented rejected resource probes from persisting non-finite `-Infinity` sentinels into strict
  Study JSON and crashing the adaptive controller after a healthy Run completed. Cold start now
  establishes one protected exploratory baseline on every otherwise idle allocated GPU even when
  unavoidable driver/context overhead makes physical free VRAM slightly lower than nominal VRAM;
  known lower bounds, failed-placement dominance and evidence-gated co-location remain enforced.
- Made large mirrored `{file: ...}` directory verification independent of shell locale, filesystem
  enumeration, host path ordering and metadata through a versioned canonical tree fingerprint.
  New identities normalize logical Unicode paths, preserve empty directories and delimit every
  typed record; legacy bundle identities remain readable.
- Decoupled startup evidence from GPU/process concurrency, protected pending anchors from queue
  replanning, distinguished dispatch invalidation from scientific cancellation and prevented
  duplicate deferred candidate/seed/fidelity execution. Survival propagation now uses the same
  logarithmic and conditional `ParameterSpace` geometry as every other HPO subsystem.
- Prevented monitor cadence, bounded display downsampling and changing candidate frontiers from
  manufacturing confidence or erasing resource-scheduling liveness. Fixed time-to-next-checkpoint
  semantics and mixed checkpointable/non-checkpointable rollback accounting; rare sub-percent
  capacity tails now remain represented rather than disappearing at a fixed quantile.
- Prevented small allocator/cache maxima, a permanent equiprobable capacity tail, an absolute
  `RAMPING` veto and the former one-second unknown-duration fallback from keeping physically idle
  GPUs at one Run indefinitely. Neighbour candidates no longer impose provisional intrinsic lower
  bounds, joint fit draws no longer correlate unrelated Runs by raw sample index, and real ramps,
  hard OOM bounds, failed placements, run caps and throughput regressions remain safe blockers.
- Made managed dataset publication transactional across the filesystem/registry boundary: a known
  immutable-version conflict is rejected before commit, a late registration failure removes only
  the directory created by that attempt, and stale conflicting remote records become explicitly
  reconcilable only after their registered path is proven absent. Replication diagnostics now use
  the real `--source`/`--destination` CLI options, preflight destination conflicts and explain why
  an unconfigured remote-to-remote relay is refused without changing either side.
- Removed a Study-controller liveness failure caused by recomputing nearest counterfactuals across
  the complete HPO proposal pool for every parameter, pair and evidence resample. Live scientific
  analysis now reuses immutable pool geometry and mixed-space distances, evaluates a deterministic
  bounded reference design and rotates a bounded scientific shortlist while the optimizer's full
  proposal pool remains eligible. Finished/OOM Runs can therefore be collected and slots refilled
  instead of waiting behind descriptive analysis.
- Made `textual-plot` canvases safe under `NO_COLOR`; native charts no longer crash Textual's
  monochrome filter on an unstyled empty dependency segment.
- Added an always-visible **Delete Study History…** action to Study workspaces. It reuses exact
  semantic Work deletion, shows preview/apply progress, requires explicit confirmation and cannot
  delete active Runs or another same-name execution.
- Made adaptive GPU scheduling work-conserving and fair: automatic Sobol startup now fills the safe
  parallel width with distinct candidates, globally free slots prefer the least-loaded/longest-idle
  granted GPU, and an OOM backs off only its device before evidence-based gradual recovery.
  `gpu_memory` remains a per-launch safety floor inside the learned candidate envelope, never a
  multiplied per-active-Run reservation.
- Expanded the Research Console cluster editor from six basic fields to the complete durable
  profile: authentication references, runtime/PyTorch, storage/SSH retention, every GPU-access
  policy field and validated advanced OpenSSH/SLURM mappings now round-trip without silent loss.
  Invalid combinations remain in the modal with a validation error and are never persisted.
- Prevented GPU-requesting Works from silently resolving a CPU-only managed PyTorch environment
  when a site command launcher fails. Adaptive command-launched Studies now scale concurrency down
  to the exact non-empty `CUDA_VISIBLE_DEVICES` grant when it is smaller than the requested ceiling,
  while scheduler grants remain exact and no physical GPU identifier is ever inferred.
- Removed quadratic adaptive-study planning amplification: the deterministic candidate pool is no
  longer copied into every candidate/seed specification, so large valid HPO budgets reach their
  first Run with linear planner memory.
- Kept exact semantic Study cancellation available before the worker publishes telemetry. Periodic
  Study/Seed provider failures now remain as inline stale/error state instead of emitting repeated
  notifications on every refresh.
- Kept every root Research Console snapshot visible during background refresh. The loading
  indicator is now first-load-only, a compact age line reports the last successful update, and a
  failed refresh retains the previous data explicitly instead of replacing it with a loading view.
- Made the Work browser state table and bounded Work log viewer refresh live without overlapping
  probes; unchanged or temporarily unavailable logs retain their last readable snapshot.
- Made validated recent-Work submission transition visibly and atomically into asynchronous
  enqueueing. Rendering a complex validation/explanation preview can no longer strand the launch
  modal before submission: authored brackets/backticks are rendered as literal text rather than
  Textual markup, and validation/submission failures remain actionable in place.
- Prevented same-name local and remote Work executions from crashing Work/Overview tables with a
  duplicate Textual row key. The console now keys rows and cancel/delete operations by exact
  semantic `work_id`, while preserving the human name purely as a label.
- Made DatasetVersion deletion visible from every dataset tab and stateful from the first click:
  preview/apply waiting, cancellation, failure and success are explicit, and duplicate activation
  is blocked while an operation is running.
- Replaced the three competing Run Work actions with one guarded Submit flow. Recent and browsed
  YAMLs now populate the same selector; Submit always validates and explains before launch,
  reports its phase and cannot be repeated while active.
- Removed the hidden per-wave VRAM reservation that double-counted `gpu_memory` for active Runs and
  could cap an 80 GiB GPU at three 20 GiB-threshold trainings despite enough live free memory. GPU
  admission now combines the documented per-launch floor with physical occupancy and the learned
  future envelope. A CUDA OOM remains isolated and retryable, but now lowers the live Study packing ceiling below the
  failing concurrency before requeueing, while unrelated Runs continue unchanged.
- Disabled implicit record-streak convergence by default. Adaptive Studies now consume their
  authored candidate budget unless an explicit Run/time limit or opt-in positive
  `convergence_patience` stops proposals; sparse multidimensional coverage is no longer mistaken
  for convergence. The terminal FINISH event is persisted before the final analysis and records
  the exact candidate/Run/time/pool/convergence reason.
- Preserved categorical HPO labels in terminal plots, removed misleading uncertainty pseudo-lines
  and made plotted points/bars report exact values when clicked.
- Reset every reused learning-curve plot to automatic data limits when changing metric pages, so a
  prior metric's manual pan/zoom cannot hide the next metric's different value range.
- Made HPO parameter evidence available while a Study is still running and has no final Execution
  result. The console now falls back to the bounded live Study analysis, refreshes it periodically,
  shows numeric plus verbal confidence and retains response/support/dispersion/interaction detail.
- Made the root Studies browser refresh live and immediately after submission while preserving its
  selected Study; replaced the unbounded trial/admission JSON preview that collapsed the table with
  a compact objective, Run-state, leader and admission summary.
- Replaced duplicated raw PyYAML scanner output in **Run Work** with one source-aware validation
  panel containing the actual file, line, column, bounded source excerpt, caret and corrective hint.
- Kept cluster actions, tabs and evidence reachable while enlarging the activity console by giving
  the upper workspace a protected, independently scrollable viewport.
- Replaced raw JSON in mutation confirmations, action notifications and modal Work previews with a
  shared bounded semantic renderer.
- Prevented interactive cluster bootstrap from competing with hidden-screen loads or live resource
  probes. The console now shares one provider factory, serializes cluster operations and pauses
  polling until the operation completes, matching the reliable standalone CLI path.
- Made whole-DatasetVersion deletion converge when placement bytes were manually removed: stale
  target/controller registrations are cleaned and no empty version remains in the browser.
- Replaced misleading `unavailable` values for intentionally pruned Run evidence with explicit
  pruned/not-observed wording.
- Fixed failed-Study navigation when the bounded Work read model contains `study: null`; the Study
  workspace now opens its Logs tab after the bounded reload confirms that no telemetry was
  published, exposing preparation/runtime diagnostics instead of trapping the user behind a raw
  `KeyError` message.
- Preserved the focused Overview panel and stable selected entity across asynchronous refreshes,
  eliminating jumps from a selected cluster to the first Study.
- Corrected personal CPU share normalization (process percent divided by host core count) and made
  the GPU comparison like-for-like by plotting total and personal VRAM percentages.
- Prevented one transient CUDA memory-probe failure from terminating an adaptive Study. Admission
  now pauses safely while active Runs continue, retries observation, records the bounded probe
  cause and fails only after a persistent outage with no active Run left to protect.
- Preserved terminal Study identity and its last observed trials/seeds/HPO telemetry across terse
  provider reconciliation, so a failed Study never falls back into the ordinary Work-only view.
- Corrected Study Analysis minimization ordering, truthful leave-one-candidate-out validation
  metadata, exact conditional domains, insufficient one-seed stability, surrogate-aware
  reliability, observed-region coverage wording, equivalence-margin confirmation and the
  distinction between intrinsic per-Run efficiency and total controller spend.
- Fixed Research Console row activation and replaced generic detail modals with a real navigation
  stack. Enter/right now opens Work, Study, cluster, dataset or result workspaces and Esc/left pops
  exactly one level.

### Added

- Added a unified scientific-design layer to adaptive HPO. Live control and final analysis now
  share structured parameter/interaction questions, practical-optimal-region evidence and one
  auditable confidence meaning: stability of the exact conclusion under candidate-level and
  shared-seed-aware resampling.
- Added automatic performance-versus-understanding allocation. The event-driven controller weighs
  posterior practical-improvement opportunity against unresolved-question entropy per observed
  cost, can schedule matched parameter/interaction `DESIGNED_PROBE` actions, and returns to
  optimization when improvement becomes plausible without adding YAML phase knobs.
- Added hierarchical within-candidate seed-noise calibration, deliberate shared-seed comparisons,
  incumbent/challenger replication and persisted action purpose/question/O/K evidence. A
  deterministic same-budget CPU benchmark guards optimization regret while measuring scientific
  conclusions resolved.
- Added automatic consumer-project isolation based on the nearest `pyproject.toml`, an optional
  stable `[tool.lambdaforge].project_id`, `lf project`, project-owned controller/MRU and remote
  state namespaces, recursively layered per-project cluster mirrors, and safe filtered access to
  legacy Jobs. Cluster profiles, credentials and host-wide GPU/process leases remain shared.
- Added contextual statistical help, a visual Coverage dashboard, a pairwise interaction matrix
  and explicit offline Plotly exports for Study, parameter, learning-curve and resource views.
  Interactive reports provide exact hover values, genuine uncertainty shading, heatmaps and
  numeric 3D surfaces without adding Plotly to the base runtime.
- Replaced raw Seed, Pareto and Findings analysis blocks with summary cards, separate scientific
  and resource trade-off tables, and drill-down recommendations.
- Added a YAML-filtered file browser and bounded recent-Work list, seeded from local Job history and
  persistent MRU choices, to **Run Work**; Enter revalidates and directly submits to the visibly
  selected cluster.
- Added a phase-streaming cluster activity console with elapsed feedback, three size presets, a
  draggable separator, clear control and exact apply confirmation.
- Added exact logical split/target summaries, automatic bounded member loading, contextual
  Stats/Integrity actions and confirmed whole-version deletion to Dataset workspaces.
- Added exact numeric Run-state counts below Study Overview bars.
- Added explicit remote-loading and retrieval-error states for Study and Seed workspaces, plus
  clickable ancestor breadcrumbs backed by the real navigation stack.
- Rebuilt the Study Overview with semantic status/trial/leader/compute cards, objective and Run-state
  charts, and a compact leading-parameter table. Trial marks now occupy a dedicated column.
- Rebuilt the HPO workspace around authored versus observed domains, best-supported regions,
  importance/reliability, parameter response and uncertainty plots, observed support/dispersion,
  pairwise diagnostics and the complete persisted Action/Trial/Reason history with decision
  drill-down.
- Added automatic human metric labels and optional
  `LightningTrainConfig.epoch_metric_display_names`; labels never alter raw metric identity or
  objective evaluation, and the internal composite key is presented as “Composite selection
  score”.
- Replaced handcrafted resource sparklines with the native `textual-plot` widget, providing
  human-readable fixed time ticks, Braille-resolution CPU/RAM/GPU lines, an external colour key and
  pan/zoom. The reusable
  dashboard now appears in Overview, the Clusters browser and each cluster workspace.
- Migrated Study learning curves to the same native plotting widget, retaining four-metric pages
  while adding automatic axes, interactive navigation and explicit green-best/red-selected epoch
  markers. Stale editable environments show a reinstall notice instead of retaining a second chart
  renderer.
- Stabilized cluster charts on a fixed five-minute `5m`→`now` axis and moved the total/personal key
  outside the data area. Added mouse-friendly curve page controls, complete horizontally scrollable
  epoch metric tables and a visible contextual Back button to every nested workspace.
- Flattened the Exit control styling so focus no longer draws a competing inner variant block.
- Separated sidebar Action/Browse/Session controls, compacted contextual buttons and introduced
  visual summary cards plus bounded semantic detail for Clusters and Datasets instead of raw JSON.
- Reused the last Overview resource snapshot in the Clusters browser, avoiding a second set of
  remote probes merely to draw the same root-screen data.
- Rebuilt the Research Console Overview as three separate Cluster, Work and Study panels with
  concise progress, formatted selection summaries, live CPU/RAM/GPU history, truthful personal
  usage context and a persistent visible Exit control. Refreshes never overlap, preventing a slow
  cluster from causing duplicate provider connections.
- Added the complete Study → Trial → Seed → Epoch console route with bounded live curves, pages of
  four selected metrics, current/best/final/selection objective semantics, censored-prune markers,
  GPU/resource evidence, isolated logs and exact same-epoch scalar inspection.
- Added navigable Study Analysis views for summary, parameter associations, interactions, observed
  coverage, empirical/model-based seed uncertainty, scientific/resource Pareto and findings. The
  TUI and HTML report consume the same versioned `analysis.json`.
- Added direct service-backed Work, cluster, dataset and result operations with exact preview and
  confirmation boundaries for destructive actions. The command palette now lists only actions
  with a real handler; the parity inventory labels remaining routes explicitly as CLI-only.

### Internal

- Kept the unreachable legacy `LiveJobMonitor` module temporarily as a regression oracle for
  bounded telemetry/rendering while advanced CLI-only operations are migrated. No public `lf top`
  route was restored; deleting it before genuine parity would discard tested behavior.

## [0.14.0] - 2026-09-01

### Added

- Added deterministic, versioned Study Analysis with exact current/best/final/selection objective
  semantics, censored-prune handling, seed bootstrap uncertainty, paired comparisons and separate
  screening/confirmation evidence.
- Added candidate-level surrogate validation, global functional and top-region importance,
  adjusted response curves, interaction surfaces/matrix, marginal/joint coverage, boundary and
  candidate-pool diagnostics, winner stability, constraints, pruning quality and resource Pareto.
- Added atomic automatic `analysis.json`, `lf results analyze`, machine JSON summaries and optional
  self-contained offline Plotly reports through `lambdaforge[analysis-report]`.
- Added structured GPU/CPU/RAM admission evidence so queued Runs explain capacity, per-device VRAM
  and wait reasons without treating temporary resource pressure as scientific failure.
- Added the Textual 8.2 Research Console opened by bare `lf`, with Overview, Work, Studies,
  Clusters, Datasets and Results screens, asynchronous provider loading, stale-data retention,
  contextual help, command palette and confirmation boundaries.

### Changed

- Made Textual the single interactive architecture and kept automation on stable CLI/JSON domain
  services. The former `lf top`, `lf clusters setup` and `lf clusters modify` public routes are
  retired rather than maintaining two UIs.
- `lf` without a subcommand opens the console only on a TTY; redirected or JSON invocations print
  help and exit cleanly.
- Updated all English/Spanish user, manual and agent documentation for the 0.14 workflow and
  clarified that post-hoc effects are predictive/descriptive rather than causal.

## [0.13.3] - 2026-09-01

### Fixed

- Corrected the adaptive controller's multi-fidelity evidence model. Seed results are now grouped
  by exact `(candidate, target, maximum)` rung, one candidate may emit several fidelity
  observations, and neither seed racing nor surrogate fitting can relabel a heterogeneous mean as
  full-budget evidence.
- Removed the remaining dispatch commitment semantics. Every terminal event replans undispatched
  queue entries, records zero-compute `CANCEL_QUEUED_ACTION` evidence for superseded work and fills
  freed slots without an outer round barrier; promotions wait for in-flight comparable rung
  evidence instead of racing a lone partial result prematurely.
- Replaced misleading expected-information telemetry with a centralized, documented
  `controller_value` proxy and observed incremental cost. Seed uncertainty, region coverage and
  fidelity actions now share explicit bounded terms without claiming calibrated information gain.
- Corrected live composite-objective telemetry to evaluate complete same-step vectors while Runs
  are active and preserve an all-time best checkpoint outside bounded metric tails.
- Allowed an objective component to remain a hard constraint and added explicit across-seed
  `mean`, `worst` and confidence-bound (`lcb`) aggregation with fail-closed missing evidence.
- Centralized the survival acquisition adjustment and exposed its coefficients/meaning in decision
  telemetry. Performance-pruned candidates remain censored evidence; operational failures and
  scheduler pauses remain neutral.
- Separated descriptive marginal/pairwise diagnostics from the actual lightweight GP/k-NN
  `surrogate_belief` in `lf top` and overview JSON, retaining the latest model diagnostic beyond the
  bounded controller-event tail without serializing provider models.
- Corrected the optional-provider CI target and added mandatory non-skipping BoTorch tests for
  numerical multi-fidelity, mixed categorical/conditional encoding, pending fidelity, observation
  noise, target-fidelity predictions and survival-adjusted acquisition.

### Changed

- Documented the maintained multi-fidelity architecture honestly as a fidelity-aware surrogate
  plus an external cost-aware action scheduler, including exact rungs, provisional dispatch,
  robust constraints, threshold-dependent metric policy and pruner calibration semantics.
- Expanded controller regressions for real runner-to-observation-to-sampler flow, stale queue
  replacement, controlled provider fallback, live composite utility, retained surrogate belief and
  deliberate late-bloomer false-prune detection.

## [0.13.2] - 2026-09-01

### Fixed

- Unified adaptive HPO around one checkpoint-aligned scientific utility. Objectives may now be
  legacy scalar metrics or fixed-range weighted/geometric/Chebyshev composites; constraints remain
  separate, raw components and Pareto diagnostics stay auditable, and missing same-step components
  never get silently mixed across epochs.
- Reworked pruning from isolated Runs to candidate-level evidence across active and completed seeds,
  with historical incumbents, conservative curve uncertainty, retrospective calibration records,
  separate seed/early-stop thresholds and an explicit no-pruning rule for confirmation Runs.
- Replaced fixed censored-neighbour penalties with a joint candidate-level survival model and
  uncertainty intervals. Performance prunes remain censored negative evidence, while resource,
  infrastructure and scheduler-preemption outcomes remain statistically neutral.
- Removed round/startup barriers from adaptive refill: every terminal Run can choose `START_NEW`,
  `ADD_SEED` or `PROMOTE_FIDELITY` by expected information per observed cost, while pending identities prevent
  duplicate seeds and every decision persists compact alternatives and cost evidence.
- Added conservative checkpoint-safe scheduler preemption for already-scored fidelity Runs. A
  30-second floor, 50% priority hysteresis, confirmation/startup protection and cooperative stop
  preserve work and prevent thrashing; `PREEMPT`, `PAUSE`, `CONTINUE` and `RESUME_PREEMPTED` remain
  auditable and statistically neutral.
- Separated executed candidate budget (`trials`) from `proposal_pool_size`, made fidelity an explicit
  surrogate feature, rejected ambiguous no-op root `reduction_factor`/`confidence` fields, and added
  structured termination, confirmation-completeness and wall/GPU-second accounting.
- Expanded `lf top`/overview HPO evidence with composite components, diagnostic Pareto membership,
  live scheduler slots/actions, candidate-level pruning uncertainty and the exact
  probability/reference behind a prune. Added retrospective pruning reports with simulated
  compute savings, false-prune rate, regret and probability/curve calibration.
- Made early HPO evidence useful instead of blank: marginal response and pair coverage now render
  from two comparable candidates, predictive pair gain starts at three with explicitly low early
  confidence, and a newer `lf top` rebuilds old remote analysis snapshots locally. Pruned Runs now
  contribute visible parameter-region pruning rates and mild censored-neighbourhood avoidance
  without being misrepresented as completed objective values.
- Removed the HPO controller's persistent per-GPU CUDA probe contexts. Free-VRAM observations now
  run in an ephemeral inherited-grant child, so monitoring neither occupies a scientific packing
  slot nor retains memory that can prevent the final `runs_per_gpu` Run from being admitted.
- Made an all-pruned Trial terminal `pruned` instead of `failed`, retained its partial curves as
  censored evidence without contaminating completed-objective statistics, and prevented duplicate
  metric streams from overwriting a terminal Run's distinct current/best objective.
- Kept at least three seed rows visible at common terminal heights by bounding parameter/metric
  previews, with complete detail one drill-down away.
- Separated current and best objective evidence throughout adaptive studies. Completed HPO now
  ranks the mean per-seed best checkpoint while same-step current curves remain the pruning input;
  `lf top` shows best/current values and the winning seed/epoch instead of conflating them.
- Clarified that Direct/SLURM is process launch rather than GPU type, and made the cluster wizard
  exit cleanly from every prompt through `0`, `q`, `quit` or `exit`.
- Made immutable Work result/artifact metadata pickle-safe across adaptive spawned workers. A Run
  that completed with artifacts can no longer fail afterwards with `cannot pickle 'mappingproxy'`;
  that signature is also classified as an internal framework fault rather than user configuration.
- Made terminal studies without a published telemetry index fall back to numbered Attempts and
  ordinary logs in `lf top`, preserving the actual terminal failure instead of an unavailable-data
  placeholder.
- Isolated every adaptive CPU/GPU Run in its own worker so a killed process cannot poison a shared
  pool or abort unrelated trainings. Lost workers and CUDA OOMs receive a bounded checkpoint-aware
  retry; repeated resource failures remain terminal evidence while other Runs continue.
- Replaced the fatal all-GPU `runs_per_gpu × gpu_memory` preflight with dynamic per-Run GPU
  admission. Temporarily occupied devices now wait and are polled, available GPUs continue at
  partial capacity, same-device launches are staggered, and only a bound physically impossible on
  every allocated GPU fails as configuration.
- Prevented adaptive GPU studies from deadlocking after a Run completes: packed Runs now use fresh
  one-worker processes that exit on result or error, releasing their CUDA contexts instead of
  leaving idle pool workers holding VRAM while later Runs wait forever.
- Prevented live study refreshes from racing with worker completion and reverting a terminal Run to
  `running`: controller-owned best-epoch observations now have a separate atomic record, so final
  counters and queued-Run admission cannot be stranded by stale telemetry.
- Made `lf help`, nested command-first help and conventional `--help` exit successfully, including
  when the CLI entry point is embedded and called directly.

### Added

- Added explicit `objective.constraints` outcome guardrails. Bounds are evaluated at the primary
  objective's best epoch and averaged across seeds; missing or violated evidence marks the
  candidate infeasible and excludes it from adaptive fitting/selection without hiding its record.
- Added bounded asynchronous acquisition refill: one or two updated-posterior look-ahead candidates
  can fill capacity before a straggling batch ends, all pending points condition qLogNEI, and every
  decision is audited without speculative mass cancellation.
- Added HPO parameter drill-down in `lf top` with numeric response curves, categorical mean bars and
  a pairwise joint-predictive-gain heat table. The bounded response points and relationship matrix
  are also published in overview JSON with explicit non-causal semantics.
- Added an `i` HPO evidence console to adaptive studies in `lf top`. It explains per-parameter
  numeric direction/possible thresholds or categorical contrasts with coverage, standardized
  effect, conservative confidence, a next-evidence suggestion and the controller's actual latest
  action. The same bounded, explicitly non-causal read model is available in overview JSON.
- Made every objective-bearing `search` use the complete adaptive policy by default: scrambled
  Sobol startup, result-dependent mixed acquisition, probabilistic shared-seed allocation,
  conservative curve pruning, convergence budgets and disjoint fresh-seed confirmation. Optional
  BoTorch noise-aware mixed-GP qLogNEI remains dependency-isolated with deterministic k-NN fallback.
- Added explicit cumulative `search.fidelity` promotion through `self.fidelity` and managed
  checkpoints; Lightning training resumes increasing epoch budgets automatically and final
  confirmation always runs at full fidelity.
- Added compact adaptive replay/audit evidence in `hpo-control/state.json` and append-only
  `decisions.jsonl`, including proposal backend/fallback, `START_NEW`, `ADD_SEED`, `RESUME`, stop and
  confirmation decisions. Result summaries link these records and report seed uncertainty/fidelity.
- Made `strategy: exhaustive` a strict exact-sweep contract for finite `values` and conditional
  `when` branches. Continuous ranges and candidate caps are rejected instead of silently sampling.
- Added expandable failed-Run evidence in `lf top`: `e` toggles persisted traceback details in both
  seed dashboards and full Attempt logs, including failure phase and exact result path.
- Added result-dependent sequential candidate proposal: a space-filling startup set is followed by
  mixed numeric/categorical k-NN acquisition batches, and `lf top` publishes parameters only when a
  Trial is actually proposed instead of displaying the complete deterministic pool as decided.
- Added per-seed GPU index/token provenance to persisted results, JSON study detail and `lf top`.
- Replaced compact resource/study sparklines in `lf top` with framed high-resolution Unicode
  time-series charts inspired by nvtop's terminal layout, without a new plotting dependency.
- Added horizontal panning for long live logs and per-dashboard Lightning curve selection through
  `epoch_chart_include`/`epoch_chart_exclude`.
- Added semantic terminal colour with `NO_COLOR` support, compact cluster-history traces, complete
  selected-Trial parameter grids, paged four-curve study dashboards, selectable epoch tables,
  curve selection markers, full per-epoch scalar detail and an explicit raw-output toggle.
- Simplified study selection tables by removing redundant truncated summaries, adding a complete
  selected-seed metric panel and using layout-independent `n`/`p` curve-page controls.
- Added objective-aware study navigation: Trial tables name metric and direction, seed rows separate
  latest and best epoch, and Run charts/tables retain a green best-epoch marker alongside the red
  selected epoch even after bounded downsampling.
- Added `lf clusters setup` and `lf clusters modify` as explained, dependency-light terminal flows
  over the existing native profile commands; they cover credentials, paths, runtimes, storage,
  schedulers and GPU policy without a second configuration backend.
- Added arrow/Enter navigation and focused consequence/risk help to cluster setup choices, retaining
  the equally documented numbered fallback and safe exit at every prompt.
- Added atomic per-cluster GPU claim/release argv around command-wrapped Jobs. Self-contained
  wrappers such as CITIUS `gpu exec` are preferred; persistent claims release after success,
  failure, cancellation or failed submission and are rejected with SLURM.

### Changed

- Adaptive Bayesian proposals now use joint noisy expected improvement with seed standard errors
  and pending-batch awareness. HPO insight confidence has conservative sample floors, explicitly
  identifies marginal explanation versus joint decision-making, and reports pruned evidence as
  censored. Curve pruning requires two distinct unfavorable common steps by default.
- Training telemetry now names peak live CUDA allocation separately from current/peak PyTorch
  allocator cache, prioritizes epoch/validation timing, and releases unused CUDA cache between
  packed study Runs. HPO admission remains based on explicit per-Run GPU memory and driver free
  memory, not allocator-reserved telemetry.
- Active study Runs now show live duration and measured timing as it arrives, with a labelled
  elapsed-per-epoch estimate only while explicit timing telemetry is unavailable.
- Command/scheduler GPU Runs now treat inherited `CUDA_VISIBLE_DEVICES` entries as opaque site
  grants (including UUID/MIG tokens), narrow them per child and fail closed instead of inventing or
  broadening an absent/insufficient allocation. CUDA probes use the same site command wrapper.

## [0.13.0] - 2026-08-30

### Fixed

- Fixed `lf top` crashing while unpickling nested immutable study lists from its background
  snapshot process. Parameter-study intent is now persisted before submission, so Works declaring
  `search` or multiple `seeds` open the study dashboard during preparation instead of falling
  through to the ordinary Attempt log.
- Stopped treating every Work with a telemetry file or several workflow Runs as a training study.
  Only declared search/repeated-seed studies publish the compact study index; preprocessing,
  composition and mapped Works consistently open their ordinary Attempt log from the main view.
- Made Work cancellation hierarchical instead of cancelling only its primary Job. Every active Job
  is attempted, direct supervisors terminate and verify process-group plus reparented/new-session
  workers by exact inherited Job identity, and normal main-process exit reaps surviving children.
- Prevented terminal Jobs and repeated failures from retaining unbounded partial managed outputs:
  failed/interrupted Attempt artifacts are compacted while logs, metrics, results, provenance and
  checkpoints remain, and verified `publish_to` duplicates are removed after durable publication;
  `lf clean` previews/applies the same policy to retained historical Jobs.
- Prevented managed environments from accumulating one multi-gigabyte immutable prefix per
  historical dependency identity by pruning superseded prefixes after both bootstrap and automatic
  run preparation, while protecting the active prefix and every live-Job reference.

### Changed

- Made objective searches with multiple seeds adaptive by default. Successive halving allocates
  seeds progressively using conservative mean/standard-error ranking, supports cooperative
  step-based early stopping and retains explicit `strategy: exhaustive` for complete studies.
- Made adaptive study concurrency an explicit outer-allocation policy: `runs_per_gpu` packs
  independent spawned trainings per reserved GPU, requires a per-Run GPU-memory bound above one,
  checks current free memory and divides CPU/RAM/storage among active children.

### Added

- Added bounded live study observability: `lf top` drills from a Work into parameter Trials and
  individual seed Runs with isolated auto-refreshing logs, learning curves, scalar/timing summaries
  and terminal failures. `overview --json`, `show WORK --run KEY --json` and
  `logs WORK --run KEY` expose the same read model without duplicating checkpoints or large outputs;
  `LightningRunner` emits scalar callback curves plus epoch/validation timing automatically.
- Added per-cluster GPU admission modes for scheduler allocation, conservative direct exclusive
  leases, deliberately shared direct hosts and argv-only site claim wrappers.
- Added `retain_internal=True` as the explicit opt-in to preserve a second physical copy of a
  published managed output, terminal `retention.json` receipts and `Work.stop_requested` for custom
  adaptive training loops; `LightningRunner` handles the same stop contract automatically.

## [0.12.1] - 2026-08-25

### Fixed

- Prevented relative remote output publication from disappearing into hashed Job workspaces:
  cluster profiles can now declare a persistent researcher-owned `project_root`, and relative
  `publish_to` paths preserve the authored YAML layout locally and remotely.
- Allowed typed project directories above the inline bundle limit to use an explicitly prepared
  remote mirror. Exact kind, byte count and SHA-256 are verified before scheduler submission;
  missing, stale, partial or symlinked content fails without launching scientific code.
- Retried truncated chunked/gzip HTTP transfers, including `IncompleteRead`, from a clean private
  temporary on every rate-limited attempt; permanent HTTP errors fail promptly and exhausted
  downloads cannot publish partial cache bytes or records.
- Preserved structured Work failures across remote Job logging: tailed human logs and `lf top` now
  show the terminal exception and persisted result path, while JSON/debug views expose bounded
  structured evidence without duplicating an existing traceback.

- Reconciled installed native-package subdirectories from regular `conda-meta` records when
  micromamba 2.8 omits `subdir` from `list --json`, while retaining exact version/build/channel/
  subdir verification before publication, after pip and on reuse. Inventory failures now show a
  bounded field-level diff instead of dumping the complete prefix.
- Prevented package names such as `libssh2` in native inventory diagnostics from being mistaken for
  SSH transport failures; typed native-preparation errors now take precedence and classify as
  environment failures.
- Prevented direct SSH supervisors from copying an already staged Job workspace onto itself, which
  failed before scientific launch with recursive `SameFileError` data.
- Made pre-launch stdout/stderr streams stable and taught Job logs to distinguish absent legacy
  streams from the supervisor's durable failure reason.
- Fixed local asynchronous Jobs becoming `unknown` after scheduler acknowledgement because the
  detached submitter and later observers resolved a relative provider root from different working
  directories. New Jobs persist absolute project-anchored roots and old records recover from their
  saved source path.

### Changed

- Added phase and periodic liveness output to human cluster bootstrap and automatic, scroll-safe
  refresh to the `lf top` Attempt log viewer; machine JSON output remains uncontaminated.

- Made normal local `lf run` use the same durable, non-blocking Job handoff as remote execution;
  supervised children still execute inline and `--dry-run` remains read-only.
- Added forward/back arrow navigation across Work, Job, cluster detail and full logs in `lf top`.
- Replaced redundant `v`/`l` monitor shortcuts with confirmed `d` selected-history deletion and `D`
  terminal-history cleanup; destructive provider work remains asynchronous and active Jobs survive.
- Made Work cache a real reconstructible storage category in preview-first `lf clean`, coordinated
  with an active-Work shared lease rather than merely describing raw paths as collectible.
- Consolidated managed path containment and file/directory fingerprinting and made the common
  `CrossProcessFileLock` the direct dataset-cache and Work-cache coordination primitive.
- Made plain `Work.map` an ordered concurrency helper without hidden persistence and introduced the
  explicit `resume_map` spelling for stable-key per-item checkpoints; legacy keyed calls remain
  compatible.
- Changed `lf top` drill-down from raw Job rows to human-numbered Work Attempts, including cluster
  detail and log titles, while retaining machine Job IDs in `overview --json` and `lf jobs`.
- Documented the complete existing neural model-family catalog instead of adding redundant generic
  MLP/GNN aliases or another model factory.
- Propagated each target's configured `storage.cache_root` into direct and scheduled Work processes
  so the simple and file cache APIs share the intended reusable/collectible location.

### Added

- Added `clusters add --project-root`, mutable `clusters set NAME project_root ...` configuration,
  `doctor` mirror checks and bilingual guidance distinguishing small bundle snapshots, shared
  project mirrors and immutable managed datasets.
- Added `Work.log(message, level=...)` for timestamped, immediately flushed messages while retaining
  normal `print()` and Python logging capture.
- Added path-like `ManagedFile`, integrity-checked `Work.cache.file/fetch/rate_limit`, validated
  atomic checkpoint files and automatically finalized `outputs.file/directory` helpers.
- Extended `Work.map` with field keys, restored-result validation, per-item retry/backoff and safe
  logical managed-cache dependencies so cache cleanup recomputes only affected items.
- Added `Work.tools.require/run` with argv-only execution, streamed bounded logs, child-scoped thread
  controls, explicit version probes and single-source environment provenance.
- Added optional `lambdaforge.clustering` adapters for KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN and
  agglomerative clustering, one immutable result contract, explicit Distance compatibility,
  guarded precomputed metrics and generic ARI/silhouette/stability evidence.
- Added a complete Spanish manual and security policy alongside synchronized English/Spanish
  README and agent guidance.
- Added preview-first `lf jobs clear [--apply]` and definitive individual Job deletion that removes
  the exact owned workspace, events and detached-submission record.
- Added the small `Work.cache.put/get` API for atomically stored bytes, text and strict JSON without
  exposing cache paths or unsafe pickle serialization.
- Added optional `publish_to`/`overwrite` managed-output publication: a verified Attempt artifact
  remains authoritative while an explicit local or remote-host copy is published atomically;
  symlink traversal, type changes and self/owner directory replacement are refused.
- Added `attempt_history` to the semantic Work read model so TUIs and external wrappers share the
  same numbered-attempt information.
- Added optional project-owned native dependency declarations for managed clusters. Exact Conda
  solves use LambdaForge's verified micromamba, platform/inventory-aware environment identity, one
  immutable prefix, required-executable provenance, `bootstrap --project` dry-run explanations and
  explicit SHA-256 offline locks/package caches without changing pip-only projects.

### Removed

- Removed the forwarding-only cache-specific file-lock subclass; dataset and Work caches now use
  the shared runtime lock directly.

## [0.12.0] - 2026-08-23

### Changed

- Made `lambdaforge.Work` the only YAML-executable contract and `Work.run()` the only scientific
  lifecycle entry point.
- Replaced multi-family authoring with a strict Work schema supporting normal parameters, typed
  file/dataset values, sequence, parallel groups, seeds, finite/random search and metric objectives.
- Bound immutable configuration, inputs, resources, seeds and trial context plus managed outputs,
  metrics, progress, checkpoints, cache and paths directly to each Work instance.
- Replaced Task-based execution/materialization in the control plane with direct `WorkConfig` plans,
  Work bundles and `WorkRunner` execution. Parallel Runs use spawn isolation.
- Unified scientific execution under `lf run`; dataset creation now uses
  `self.outputs.dataset(...)`, while dataset inspection, verification and placement commands remain.
- Simplified the public package root and CLI around Work/Execution operations, retaining Jobs as the
  advanced scheduler/process view.
- Renamed the reusable Lightning helper and logging policy so no non-executable public class is
  presented as a Task.

### Added

- Durable Work Execution/Run/Attempt result envelopes with scientific fingerprints, environment
  manifests, captured logs, metrics, outputs, artifacts, datasets, failures and resume evidence.
- Immutable submitted-configuration snapshots ensure retry keeps the same Execution definition
  even when the authored YAML changes later.
- Seed-aware objective summaries rank parameter variants by their mean metric and retain every
  contributing Run and seed.
- Safe JSON checkpoint helpers, identity-scoped cache paths, progress snapshots and resumable
  bounded `Work.map()`.
- Per-Job mutable remote workspaces, embedded controller code identity and supervisor-consumed Work
  progress; immutable bundle caches are never used as execution directories.
- Preview-first, idempotent deletion for direct local Executions as well as scheduled Work roots.
- Current Work JSON Schema, Work-only scaffolds/examples and focused Work runtime tests.

### Removed

- Function execution, global current-runtime helpers and legacy YAML execution fields/kinds.
- Separate dataset-build command/protocol and the old authoring schemas/examples.
- Compatibility/migration routes for superseded pre-1.0 execution configuration.
- The unused `ExecutionBackend` hierarchy and parallel resource/retry policy layer; scheduler
  providers are the single operational execution boundary.
- The unused catalog-oriented `DataService`/transfer-provider path; `DatasetService` is the single
  dataset lifecycle service.

## [0.11.0] - 2026-08-22

### Added

- Function-first Work authoring with `run`, `with` and `resources`; ordinary project functions now
  execute through the established strict Task runner without a framework base class or context
  parameter.
- Explicit nested `{file: PATH}` and `{dataset: NAME@VERSION}` parameters, preserving content
  identity, managed resolution and bounded remote staging while leaving ordinary strings alone.
- Public `current`, `metric`, `artifact` and streaming `publish_dataset` runtime APIs. Metric history,
  final values, artifact checksums and immutable DatasetArtifact v2 publication are merged into the
  existing result and registry contracts.
- Simple seed/search/objective expansion, sequential steps, explicit parallel groups and an
  advanced class/method escape hatch, all compiled to the existing Workflow DAG and HPO sampler.
- Name-first `lf show`, preview-first `lf delete` for exact terminal Work roots and `lf clean` as a
  conservative cache-GC alias. Tiny completion receipts make local/remote deletion retries
  idempotent. The same active Work identity is refused only on the same cluster.
- Local integration tests for callable execution, explicit files, metrics, artifacts, streaming
  publication, steps/parallel, seeds/search, duplicate prevention and exact-root cleanup.

### Changed

- `lf init` now generates normal Python functions and minimal YAML instead of requiring Task,
  Lightning or object-graph authoring for starter projects.
- The README, manual, examples and agent guide lead with Python-defined science and YAML-defined
  execution. Strict Task, Workflow, Experiment and DatasetRecipe forms are documented as advanced
  compatibility surfaces.
- Remote tasks receive the selected cluster's dataset registry/root environment when available, so
  callable publication uses the same permanent placement boundary as recipe publication.
- Release identity advances to 0.11.0 from the single canonical `_version.py` source.

### Security

- Runtime artifacts and dataset assets must exist below the active run, reject path escape and
  symlinks, and are hashed before success/publication. Dataset publication remains staged,
  verified, immutable and atomic.
- Work deletion accepts only terminal tracked job IDs and exact children of the configured job
  root; it is preview-only without `--apply` and cannot select datasets, shared caches or
  environments.

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

[Unreleased]: https://github.com/simplelambda/LambdaForge/compare/v0.13.3...HEAD
[0.13.3]: https://github.com/simplelambda/LambdaForge/compare/v0.13.2...v0.13.3
[0.13.2]: https://github.com/simplelambda/LambdaForge/compare/v0.13.0...v0.13.2
[0.13.0]: https://github.com/simplelambda/LambdaForge/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/simplelambda/LambdaForge/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/simplelambda/LambdaForge/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/simplelambda/LambdaForge/compare/v0.10.1...v0.11.0
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
