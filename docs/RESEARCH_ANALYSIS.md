# Study research workspace

[Español](RESEARCH_ANALYSIS.es.md) · [Manual](MANUAL.md#16-study-analysis)

Current application release: **0.17.0**. Analysis v8 and saved-view v1 remain compatible; the
implementation reports below retain their historical release context.

## Contents

1. [Open a workspace](#1-open-a-workspace)
2. [Declare meaning, not execution policy](#2-declare-meaning-not-execution-policy)
3. [Families and questions](#3-families-and-questions)
4. [How discovery works](#4-how-discovery-works)
5. [Explore and save](#5-explore-and-save)
6. [Architecture, safety and limitations](#6-architecture-safety-and-limitations)
7. [Implementation report](#7-implementation-report)
8. [Second-round simplification report](#8-second-round-simplification-report)

## 1. Open a workspace

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

## 2. Declare meaning, not execution policy

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

## 3. Families and questions

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

## 4. How discovery works

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

## 5. Explore and save

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

### Project HTML tabs

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

## 6. Architecture, safety and limitations

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

## 7. Implementation report

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

## 8. Second-round simplification report

This section records the focused follow-up on 2026-10-03; section 7 describes the earlier foundation.
The [conditional Study guide](CONDITIONAL_STUDIES.md) includes grammar and planning examples.

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
