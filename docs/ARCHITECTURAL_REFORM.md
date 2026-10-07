# Architectural reform: audit and implementation ledger

[Español](ARCHITECTURAL_REFORM.es.md)

This is a continuation ledger, **not a declaration that the requested reform is complete**.
Baseline: `d2f526d`, LambdaForge 0.17.0. No WISDOM files, productive cluster Jobs or persisted
scientific records are modified by this reform.

## Audit findings

- `work/runner.py` combines execution, controller orchestration, retry classification, termination,
  scientific summaries and resource dispatch in roughly 10,000 lines. Responsibility-based
  extraction requires characterization tests; moving blocks alone is not the goal.
- Run results describe physical Attempts. Recovery already preserves valid logical outcomes for
  adaptive, repeated and fixed designs. Adaptive recovery requires its real persisted controller;
  fixed designs must not fabricate one.
- Retry and termination used different classifiers. Generic `CUDA error` was interpreted as a
  resource failure, although illegal access and dtype errors are not allocation failures.
- Telemetry changed failed state to retrying before archiving it, losing its displayed failure.
  Retry queues were also counted as occupied scientific slots. Eight-entry display history could
  not explain the complete physical cost of a long recovery sequence.
- At baseline, product contracts, a durable ProductRegistry, native product dependencies and
  portable import were absent. The contract/catalog/product-transport layer now exists below;
  native decisions, explicit typed inputs and declared single-Study publication now exist;
  dependency orchestration and complete distributed Fleet export/import remain pending.
- Dataset preflight, sealed rebuild candidates, project-owned comparison and publication recovery
  exist. At baseline durable certificates and contract-aware consumer resolution were absent;
  certificates now reuse the product catalog, while Work consumer resolution remains pending.
- Fleet supports fresh centrally planned native Studies with managed members, leases and fencing.
  Public distributed recovery, live membership, member co-location, checkpoint transfer and full
  distributed export remain gated; internal coordinator controls do not satisfy public acceptance.
- Storage leases and preview-first cleanup exist. Provisioning has no separate complete admission
  budget, and environment identity still includes project code. Dependency/code layer separation
  must preserve immutable prefixes and numerical provenance.

## Implemented foundation

- `diagnostics/failure.py` is the pure operational classifier used by Run termination, OOM retry
  recognition, lost-worker retry eligibility and persisted Work diagnostics. Its
  `FailureDisposition` uses the existing `ErrorCategory` and `RetryDisposition`; it is not a second
  diagnostics framework. Eligibility does **not** authorize a retry: ARI, budgets, checkpoint
  compatibility and placement dominance still govern admission. No generic consumer exception is
  retried merely because its text mentions a lost worker or memory.
- `work/state.py` derives `StudyState` from existing obligations/current logical states, with
  `operational`, `evidence` and `health` dimensions. `required_evidence()` is shared by live and
  terminal telemetry. Required unproposed adaptive candidates do not become fictitious debt.
  Performance pruning retains its existing censored-evidence semantics, not a fabricated final
  objective or an operational failure.
- `work/attempt_history.py` archives terminal Attempts before logical transitions. The visible tail
  remains bounded; cumulative Attempt counts/failures/durations survive truncation. Old truncated
  histories are explicitly lower bounds (`history_complete=false`), not invented exact history.
  Authoritative result files and budget accounting remain separate and unchanged.
- Compact Study/overview projections carry `lifecycle`; Console Study Overview displays it without
  requesting Attempt files. Waiting retries do not occupy live process slots.
- Latest recovery outcomes prefer the greatest Attempt number within the same logical evidence
  cell; delayed older delivery cannot replace a recovered success. Fidelity rungs stay distinct.

Existing `status`, result versions, HPO decision criteria, ARI placement, exact hashes and recovery
locks remain compatible. This first foundation does **not** yet replace every final execution,
analysis, Fleet or dependency consumer with the aggregate lifecycle contract.

## Further integration and hygiene corrections

Native finalization and output references use latest logical outcomes, not `all(run.ok)` over
failed physical history. Summary/objective aggregation does not duplicate older Attempts of the
same evidence cell. Native evidence aggregation shares the telemetry authority, distinguishes
phases/fidelity rungs, handles unparameterized repeated designs and retains every composed Work's
debt. Observed physical records remain intact; history omitted by recovery remains incomplete.

ResultStore projects legacy lifecycle in memory; Analysis metadata and export manifests carry it
without changing scientific fingerprints. This is not a migration of every HTML, cached analysis,
recovery or Fleet reader. Full physical-history migration, dynamic obligations and overhead
accounting remain unfinished.

Storage previews previously created cache roots and GC/controller locks; probing cache ownership
could also create a lease. Inspection, GC/environment previews and Job compaction previews now
create none of these. Existing lease probes open without writing. Apply still acquires ownership
locks and recomputes targets: an advisory preview is not deletion authority. Uninspectable ownership
protects its cache. Windows/NFS behavior of read-only lease mode has not been tested there.

Native ResultStore deletion verifies terminal state and persisted ownership, serializes with Study
import and holds the controller lock through removal. Read-only preview creates no lock. Confirmed
deletion prunes the single owned Work parent only if empty, using root-descriptor operations without
following links. Sibling/foreign content, configured roots and independently published products
remain. Imported running snapshots are evidence-only and may be deleted. This is not yet the
project-wide audit of every empty-owned-parent cleanup path.

## Product model, catalog and transport layer

`products/models.py` separates explicit versioned meaning, exact content and original producer
provenance in deeply immutable, pickle-safe models. `products/registry.py` publishes independent
artifact copies with exact raw-byte SHA-256 verification, immutable names/contracts, native writer
locking and bounded metadata-only reads. Reads/previews create nothing. Producer configuration is
provenance, not a compatibility requirement; additional origins do not rewrite the original.

Native `lf products list/show/provenance/consumers/verify/publish/export/import/select/decide/status/finalize` call the same API. Product
bundles verify manifests, artifact bytes and the complete provenance set before idempotent import.
Export → delete original → concurrent import preserves promoted bytes and historical identity.
`products/selection.py` deterministically selects completed full-fidelity latest-Attempt model
snapshots with explicit artifact-bound metrics, grouping, constraints, top-k and declared ties.
Only selected model bytes are read and independently promoted. Generic Run best/last metrics
never establish arbitrary checkpoint identity. `outputs.from_checkpoint(..., metadata=...)` records
the project-declared snapshot evaluation. A tiny native CPU Study validates this path end-to-end.
Native `StudyDecision` publication preserves the persisted selection and exact Analysis identity,
with unresolved conclusions explicit and no refit. Typed `with` product inputs resolve contracts
and scientific expectations; bundles pin content IDs, workers receive the project-owned root and
actual Attempts append immutable consumer audit records. Remote preflight verifies materialization
without downloading model bytes. A single Study may declare `products`: native finalization
persists science first and promotes products before compaction. Separate bounded status and
publication history retain failures; `products finalize` retries only publication under native
Execution ownership locks, not training. See [Products](PRODUCTS.md). This is **not** composed/Fleet
promotion, dependency waiting/replanning or complete Fleet export.

Native `lf import PACKAGE [--apply]` now verifies/registers single-host version-2 Study exports,
including exact sealed product bundles. Original Execution/Run/Attempt and producer bytes remain
intact in a portable archive; import records operational placement separately and never execute or
become native recovery state. Local/provider export includes published receipt-bound products.
The Console Products browser pages metadata/audits and exposes explicit verification/export and
Study import with worker feedback, revalidation and confirmation. This is not distributed Fleet
import/export, unsigned authenticity proof or automatic dependency orchestration.

`DatasetEquivalenceCertificate` now seals the existing whole-dataset comparison in the same
ProductRegistry, as a versioned ScientificReport. Exact content IDs, full science contract,
verifier/policy/result/evidence are bound; timestamps and machine roots are operational provenance.
Unresolved, corrupt or truly different comparisons fail closed. The native API does not merge
identities, infer transitivity or modify ordinary Work input resolution. Reports exceeding bounded
metadata fail explicitly rather than truncate approval evidence. See
[Dataset reconstruction](DATASET_RECONSTRUCTION.md). Contract-aware YAML resolution remains pending.

Managed-file/checkpoint constructors and cache reads without an active worker lease now create no
empty data/record/lock trees. First writes retain native atomic publication and per-key locking.
An explicit active-Work cache lease remains eager for GC safety; fully lazy lease acquisition is
still pending. Root checkpoint publication continues to reject invalid intent before missing-file
checks, without requiring an empty checkpoint directory to exist.

Public package imports now reuse the existing `LazyExports` authority for Work, products, HPO,
metrics, clustering and Analysis. Merely reading a contract/catalog or CLI help does not import
Torch, the execution engine or scientific Analysis. CLI execution/results/export and configuration
planning load those services only on their relevant routes. Public names and class module identities
remain unchanged; no second runner or import compatibility layer is introduced. This is a concrete
startup boundary correction, not the unfinished executor/ARI responsibility extraction.

## Remaining work, in dependency order

| Request sections | Boundary | Status / acceptance still required |
| --- | --- | --- |
| 1–3, 63–64 | Full audit and architectural extraction | Initial boundary audit only; extract executor, dispatch/retry ownership and persistence after characterization. |
| 4–9, 46–47 | State, failures, recovery and budgets | State/failure/finalization foundation implemented; pending remaining adapters, taxonomy, dynamic obligations, full physical/overhead accounting and self-healing. |
| 10–18, 31–32, 50, 54–56 | Products, contracts, selection and dependencies | Contract/catalog, promotion, local ModelSet/native StudyDecision, typed Work inputs, roots/consumer audits, single-Study YAML publication and publication-only retry implemented; pending composed/remote/Fleet selection and dependency waiting/replanning/DAG. |
| 19–20 | Portable import and complete export | Native single-host Study/product import/export, verification and idempotence implemented; pending complete Fleet artifact inventory/transfer and distributed acceptance. |
| 21–24 | Dataset contracts and equivalence | Existing reconstruction preserved; durable certificates via the native comparison/product catalog implemented. Pending explicit additional representation policy, scalable report artifacts and downstream contract-aware Work resolution. |
| 25–30, 60 | Filesystem, storage, environments and cache | Storage previews/lazy roots and locked terminal ResultStore deletion/empty Work-parent pruning corrected; pending remaining read-only/lease/cleanup-path audit, provisioning admission, dependency/code layers and cleanup economics. |
| 33–42 | Production Fleet lifecycle | Pending pause/resume/reconcile/adoption, membership/drain/capabilities, spare capacity, co-location and resumable verified transfers. |
| 43–45 | CLI, Console and HTML | Lifecycle Overview, product metadata/audit browser and confirmed Study import integrated; remaining recovery/selection/dependency dashboards and complete explanations. |
| 48–49, 51–53, 59 | Compatibility, bounded reads and safety | Preserve current contracts; audit new readers, migrations, dry-run purity, idempotency, locks and bounded stores for every new feature. |
| 57–58, 65–70 | Acceptance, documentation and delivery | Final acceptance A–D and full validation remain required after the remaining implementation; no broad completion claim. |
| 61–62 | Documentation and migration | Initial English/Spanish ledger and lifecycle guide; final public routes/examples and migration must follow working implementations. |

No pending feature should be exposed as working through a stub command, guessed product identity,
silent restart, certificate-free dataset equivalence or coordinator-only Fleet export.

## Validation checkpoint

Before extraction: 69 characterization tests passed (Study observability, evidence plans and
diagnostics). The same 69 passed after the initial integration. Added focused lifecycle tests cover
OOM versus consumer/kernel failures, worker-boundary retry, disk-full policy, late delivery,
bounded history with cumulative cost, recovery and missing/censored required evidence. Record final
test outcomes in the handoff; this ledger does not pre-claim final acceptance success.

First foundation validation: Ruff and mypy passed; full suite **1454 passed**, with four existing
Lightning warnings, before the product layer. Installed-wheel smoke and one tiny native CPU Work
passed outside the checkout using a fresh installation prefix and reused dependency packages.
Product layer checkpoint: full suite **1508 passed**, before native decisions/typed inputs.
Products/lazy-storage focused checkpoint: **84 passed**; latest integration **237 passed**, including
publication, decisions, typed inputs, recovery, evidence and providers. Subsequent full suite:
**1554 passed**, four existing Lightning warnings, before Study import/Console additions.
Installed wheel outside checkout passed native CPU publication → consumption → producer deletion →
product import, using a fresh package prefix with existing dependency packages reused. Native import
and provider-transfer regressions and Console headless tests are added; record their final outcome
in the handoff. Loopback remote preflight/export is not production SSH/Fleet acceptance. This
checkpoint does not claim the reform or acceptance A–D is complete.

Continuation checkpoint: the full run completed with **1588 passed / 3 failed** (four existing
Lightning warnings); the three failures were test doubles still assuming eager WorkRunner globals
or the pre-import evidence path. They were corrected and all three passed on rerun; a focused
48-test integration pass also covered those modules. Additional certificate regressions:
**20 passed**, including all-member rejection, corruption, explicit tolerances, immutable/pickle
transport, relocation, bounded-report failure and historical approval versus current integrity.
Ruff and mypy passed (**580 source files**). Current installed-wheel smoke and native two-seed CPU
Study export → original deletion → import passed outside the checkout, preserving Analysis,
producer provenance and model bytes. Headless Products/import screens were rendered and inspected.
This checkpoint is not a fresh single-run full-suite pass or final reform acceptance.
