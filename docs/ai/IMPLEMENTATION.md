# Agent-native integration acceptance ledger

[Resumen técnico en español](IMPLEMENTATION.es.md).

This is the implementation ledger, not a claim that the complete integration has shipped.
Baseline: LambdaForge 0.18.0, HEAD `56c48957b448a6f631e606188687e4b57b6bf4d3`,
including the existing uncommitted operational-reliability changes. No live cluster, WISDOM
source, persisted scientific record, GPU grant or scientific fingerprint is modified by this work.

Release scope: LambdaForge 0.19.0 packages the implemented foundations and operational fixes.
The original 0.18.0 baseline and verification history below remain historical; this version bump
does not change the pending acceptance items or enable unsupported scientific mutations.

## Architecture and audit

Five separate layers: repository/consumer instructions; packaged procedural Skills; project-pinned
Python facade and MCP adapter; Markdown knowledge; existing scientific/operational services.
The adapter must never instantiate another runner, refit HPO, accept shell commands, or infer
permission from natural language. Imported evidence is read-only. Fleet limitations remain native.

The former AGENTS.md was a detailed contract/reference rather than a discovery entry point.
Its content has been retained in FRAMEWORK_CONTRACTS.md; sections classify as follows:

| Existing section | Classification / destination |
|---|---|
| Execution model | Essential global rules, root guide |
| Fast routes | Procedures and CLI reference, packaged discovery / Skills |
| Writing Work | Public API, recovery/security contracts, Work/results references |
| Clustering | Component capabilities and scientific restrictions, model reference |
| Composition/Studies | Scientific/ARI implementation contracts, specialized reference |
| Analysis/Console | Scientific semantics and presentation procedures, analysis reference |
| Neural components | Discovery route, model Skill and actual API catalog |
| Identity/ownership | Essential security rules plus storage/identity reference |
| Control plane | Provider/Fleet/security contracts and explicit unsupported operations |
| Modification checklist | Essential contributor workflow and specialized acceptance ledgers |

AGENTS.md remains canonical (126 lines). Spanish material is explanatory, not a second independently
maintained policy. Consumer instructions must describe public APIs, not modifications to internals.
No root-guide migration is complete until all reference links and packaging have been verified.

## Requirement matrix

Number ranges refer to the attached 0–60 implementation request. `pending` is deliberate and
must not be presented as available by the CLI, capability catalog, or documentation.

| Sections | Deliverable / affected component | Acceptance | State |
|---|---|---|---|
| 0, 1 | HEAD/standards audit; domain-service map | Official sources + installed signatures | foundation audited |
| 2–4 | Compact canonical guide; consumer templates | Rules retained; no broken links/dual policy | implemented |
| 5–7, 49 | 24 packaged specialized Skills + executable examples | skills-ref, native snippets, wheel | implemented; full snippet inventory still a final gate |
| 8–10, 18–24 | Typed facade, catalog, MCP/project isolation | Real handshake, schemas, limits, permissions | in progress |
| 11–17 | Native discovery/Work/Study/results/data/product/provider tools | Domain parity, bounded lazy reads | in progress |
| 23, 42–43 | Trusted approval plans / audit / replay protection | Expiry, changed args, concurrent apply, injection | approved knowledge actions verified; scientific actions pending |
| 25–26 | Codex and second client | Real stdio clients; preserve client configuration | real Codex discovery/calls verified; second vendor/conversation pending |
| 27–31 | Scaffold/init/sync/dependencies | Legacy scaffold, personalization, offline failures | implemented; real offline installer failure verified |
| 32–36, 40–43 | Markdown memory + derived index + evidence references | External edits, CAS, corruption, backlinks | in progress |
| 37 | Opt-in local semantic retrieval | Incremental/hybrid/versioned model, textual fallback | pending |
| 38–39 | Optional Obsidian adapter | Headless Markdown, existing/external vaults | headless and explicit local desktop helpers tested; external binding/real desktop pending |
| 44–47 | Python GUI API; Textual configuration | Same services, feedback, no shell CLI adapters | facade/panel/example implemented; complete orchestration pending |
| 48, 50–53 | Package and security/protocol/scaffold tests | Installed wheel outside checkout, no LLM key | installed wheel, note approvals and loopback HTTP verified; full scientific gates pending |
| 54–55 | CPU scientific and GUI end-to-end scenarios | Across client sessions; exact evidence preserved | native tiny Work/evidence/memory and headless panel verified; actual vendor-agent conversation pending |
| 56–60 | Bilingual quickstart, technical report, final gates | Ruff/mypy/tests/CI results honestly recorded | pending |

## Standards consulted (2026-10-11)

- [AGENTS.md](https://agents.md/): plain Markdown, scope by nearest guide.
- [Agent Skills](https://agentskills.io/specification): named directories, YAML frontmatter,
  progressive references and `skills-ref` validation.
- [MCP specification](https://modelcontextprotocol.io/specification/latest).
- [Official Python SDK](https://py.sdk.modelcontextprotocol.io/): v2 `MCPServer`, typed tools.
  Package index and isolated Python 3.10 installation verified **mcp 2.3.0**.
- [Codex MCP](https://developers.openai.com/codex/mcp/) and
  [Skills](https://developers.openai.com/codex/skills/): configuration must use the current client.
- [Obsidian CLI](https://obsidian.md/help/cli): optional desktop integration, not memory storage.

## Security boundaries to preserve

Native Work validation imports project Python. A READ session must not execute arbitrary imported
classes under the name of validation; PLAN access to trusted project code needs explicit operator
configuration. Metadata queries must not import consumer code. Knowledge text is untrusted data,
not permission, policy, scientific verification, or a runnable command.

Agent tools cannot provide their own approvals. Local human approval is a separate trusted channel;
the integration cannot sandbox a client which independently has unrestricted same-user shell access.
Paired local HTTP is READ/PLAN only, loopback/exact Host/Origin guarded, with token verification on
every request. It is not an OAuth server: public/network serving remains unsupported. Scientific
SUBMIT/OPERATE/DESTRUCTIVE permissions do not enable unimplemented actions.
No optional SDK installation in the development sandbox changes a cluster or the user's Codex config.

## Verified implementation checkpoint — 2026-10-11

- Public `lambdaforge.agents.AgentServices` uses native domain owners with typed response contracts.
  Parameter types are preserved by the decorator; catalog/schema construction needs no MCP import.
  Compact operations cover project/version/Work/config, Jobs/Works/Study panels, native results,
  products, local dataset placements, registered clusters and Markdown. Model export discovery is
  static and does not import Torch merely to list names. This is not the complete requested tool set.
- SDK 2.3.0 real in-memory and stdio handshake with schemas/resources/prompts. A client in a different
  cwd remains project-pinned. Explicit native validation is isolated: consumer prints cannot corrupt
  stdio, cancellation reaps the owned child, timeout cannot become silent Work execution.
- Markdown source, incremental FTS, 11 note kinds, exact evidence refs, backlinks, revision-checked
  update/link, duplicate/corruption warnings and explicit corrupt-index rebuild with old copy retained.
  No embedding model, LLM key, desktop app or plugin is needed for these operations.
- 24 Skills pass the reference `skills-ref` validator from official commit
  `69ef37e9424c0a7ea9dd2293b559e43ec8176379` (validator requires Python >=3.11; verified in isolated
  Python 3.14). A separate native fixture test validates Work/repeated/adaptive YAML and executes
  the tiny Work, product promotion, ResultInput/ProductInput consumers and linked knowledge.
- Init flags, TTY questions, preview/apply and managed-hash sync preserve custom guides/Skills,
  notes and client-global configuration. Codex configuration is a reviewable example, not an
  installed/connected client. Legacy science-only init stays unchanged without opt-in flags.
- **Agents / knowledge…** Console panel: status, installed Skills, search and confirmed asset setup.
  `examples/agent_gui.py` illustrates the shared nonblocking facade, not a chatbot.
- Focused suite: 26 tests passed (22 agent/protocol/scope/assets/panel/cancellation tests + 4
  structure checks). Existing Work/Console regression run passed 74 tests, with two structure
  failures subsequently corrected and all four structure checks rerun green.
- `ruff check src tests` passed; full mypy passed over 615 source files. All 73 files checked in
  the touched formatting scope passed. Whole-tree format check still reports 24 pre-existing
  unrelated files; their user changes were preserved rather than bulk reformatted.
- Installed wheel outside checkout verified resources/Skills, optional-import laziness, native
  observer and legacy authoring smoke; final rebuild/reinstall is rerun after the last edits.
  CPU/protocol CI gate added; no assertion that remote CI has run.

## Continued implementation checkpoint — 2026-10-11

- `ApprovalService`: closed knowledge.create/update/link/export action set, pure previews, exact
  project/arguments/revision hashes, signed requests, expiry, distinct local human approval,
  cross-process single-use claim and metadata audit. MCP has no approve tool. CLI requires TTY
  plus typed full ID; Console renders exact arguments and requires the same human confirmation.
  Approval never applies a mutation. Stale/forged/expired/concurrent/crash/replay cases are tested.
  Request listings now validate only the selected page, not every historical request body.
- Operator permissions are distinct from available capabilities. READ/PLAN remains the default;
  WRITE_PROJECT enables separately approved note edits/ZIP snapshots. Scientific mutations have no
  registered implementation and cannot be enabled by an agent-supplied permission or boolean.
- Optional `agents` dependencies are inspected without import or network. Fixed-extra preview/apply
  uses the invoking virtualenv/Conda, refuses system Python/arbitrary packages and supports an
  explicit project-owned offline wheelhouse. Tests include real pip failure in a disposable venv,
  timeout cleanup, stale plans and CLI nonzero diagnostics without deleting generated project files.
- Official SDK streamable HTTP is opt-in paired loopback, with private token preparation that never
  returns the secret, per-request token/project checks, exact Host/Origin and loopback-peer checks,
  bounded request/session/concurrency settings and untrusted forwarded headers. Actual socket
  client handshake/calls/rotation and ASGI unauthorized/oversize tests passed. Not public OAuth.
- Actual installed Codex **0.162.0-alpha.17.2** app-server connected to the native stdio server,
  discovered tools and called project_info. Temporary CODEX_HOME configuration, ephemeral thread,
  foreign cwd, no inference/model credentials and no global client settings modified. This is
  deliberately **not** proof of an LLM conversation or a second vendor client.
- Installed public models now resolve constructor/forward declarations and actual docstrings
  statically through bounded export chains; every listed model was checked without importing Torch.
  Recipes, selected Run curves/logs, artifact text preview, result comparison, product contract/
  members/provenance/consumers, exact Job state/incidents and Fleet config use native owners.
  Explicit resource probes use a project-owned observation cache, not another project's view.
- Related-note queries explain exact link/tag/evidence relationships. Local evidence-reference
  verification and hypothesis/decision/open-question views do not verify the interpretation.
- Dataset discovery separates exact content metadata, cached known placements and explicitly
  selected logical member pages, without scanning assets or every provider implicitly.
- Canonical note export preserves exact BOM/CRLF bytes in a compressed ZIP with checksum manifest;
  pure preview binds every note and refuses stale plans, corrupt/duplicate notes, symlinks and
  overwrite. CLI uses native preview/apply; MCP uses the same separate human approval authority.
  Native create/update/link and export record metadata-only intent/completion events with paged
  byte cursors. Interrupted intent is not success; external editor authorship is not fabricated.
- Thirteen short SDK prompts reference actual installed Skills/tools. Resources add installed Study
  signatures, source-path/recipe/note indexes and exact native results/products under toolset gates.
- Optional Obsidian helpers separate passive inspection from explicit local version/open requests.
  PATH is not proof of compatible/running Desktop; official URI targets only the owned README.
  Headless/missing/disabled/timeout paths are tested without changing desktop preferences. No live
  Obsidian instance was used; existing/external vault binding remains pending.
- Latest focused gate: **67 tests passed**, including real Codex opt-in, HTTP, approvals,
  offline installer, native CPU/evidence, Console, structure and release consistency. External
  edits occurring after native validation are checked again before committing the note.
  `ruff check src tests` passed;
  mypy passed over **622 source files**. These are not a claim of full backend pytest coverage.
  Normal imports remain SDK/Torch-lazy. Wheel installed/smoked outside checkout at 0.18.0;
  final documentation-only rebuild is performed after this ledger update.
- Whole-suite run (started before the final desktop/journal regressions were added): **1930 passed,
  1 failed, 4 warnings in 1195.10 seconds**. The sole failure was the Spanish AGENTS navigation
  page omitting the current version, not a scientific/backend failure. That page was corrected;
  all four release-consistency tests and the final 67-test focused gate then passed. The entire
  suite has **not** been rerun after that correction; do not label this run wholly green.

## Next implementation boundary (do not redo verified work)

1. Extend the same approval authority beyond exact Job cancellation to native SUBMIT/OPERATE/DESTRUCTIVE actions, binding actual
   scientific/provider plans, identities, budgets, ownership and revalidation. Do not add a second
   backend or enable scientific mutations before their security/native integration tests.
2. Complete remaining bounded dataset/product/provider planning and resources/prompts. Public
   model declarations are available; missing scientific contracts must remain explicit, not inferred.
3. Second real vendor client and actual conversational CPU-science/memory/GUI acceptance. Codex
   transport calls and SDK tests do not substitute for these scenarios.
4. Opt-in local semantic/hybrid retrieval, existing/authorized external Obsidian vault binding,
   actual desktop acceptance, GUI orchestration and journal recovery UX for interrupted writes.
5. Full requested scientific/security/protocol/scaffold/packaging gates and final bilingual technical
   report. **The attached request's final-completion requirements are not met yet.**

## Refinement checkpoint — 2026-10-12, baseline `8df2a1c`

This continues the new 1–65 architectural refinement request. It is not a completed reform or
a new release; historical verification above remains historical.

- Native v1 envelopes now distinguish typed diagnostic/approval failures and preserve redacted
  native diagnostic references. Unexpected internal errors do not leak private tracebacks.
  Real MCP failures set `isError`; declared successful payloads have validated specific schemas.
  SDK 2.3 output models publish those schemas through public Pydantic/SDK contracts, not private
  manager mutation. Installed/session-enabled/unsupported and provider freshness stay distinct.
- Deterministic technical EN/ES intent discovery supplies native owner, permissions, costs,
  preconditions, error codes and existing next tools. This is not semantic memory or confidence.
  Result queries and exact Job actions are extracted behind the unchanged public facade.
- First operational action: exact Job cancellation with OPERATE and shared signed human approval.
  Pure metadata preview; expiring single-use request; separate CLI/Console review; fresh native
  state/binding revalidation before `JobService.cancel`; saved outcome and no replay after claim.
  Siblings/evidence remain untouched. Unknown owners, changed paths/providers and Fleet fail closed.
  Agent submission, recovery, deletion and coordinated recovery remain **pending**.
- All 24 Skills now contain task-specific procedures/decision cases rather than repeated generic
  advice; command examples are checked with the native parser and tool names against the facade.
  Approval plan scroll is separate from pinned ID/confirmation controls.
- CI creates its agent venv explicitly. Mock installation tests isolate interpreter facts;
  the real offline failure fixture is independent of an ambient SDK. Retired CI test paths are
  removed. Existing formatting-only changes preserve ASTs; no scientific algorithm was altered.

Verification of this checkpoint: **97 targeted tests passed** (all agent tests with official
SDK and the opt-in real isolated Codex client, plus structure/version consistency). Ruff checks
pass; **869 Python files** pass format check; mypy passes **628 source files**. All 24 Skills
pass both official `skills-ref` and creator validation; native parser/facade tests verify their
command/tool examples without executing them. Wheel 0.19.0 is built/installed with no dependencies
in an owned temporary environment and smoke-tested outside the checkout. No full backend suite,
real cluster scientific workload, second vendor conversation or live desktop acceptance is claimed.

Next: expand native approved operations with their own evidence/security tests, extend
executable task examples, then contextual/optional semantic memory, safe vault/import/edit recovery,
GUI/MCP Apps/tasks, monitor ergonomics and explicit checkpoint/Fleet boundaries. None is made
available by a boolean, notes or this ledger. Follow acceptance tests and installed-wheel checks
before claiming each feature; do not repeat the previous Work/ownership/modal diagnosis.

## Managed report assets and reboot ownership — 2026-10-13

- Managed `outputs.html_bundle` uses the existing artifact lifecycle: bounded UTF-8 entry,
  explicit versioned/attested manifest, owned opaque files and exact native/raw checksums.
  No scientific selection, renderer domain logic, point clipping or scheduling changes.
- Public scoped browser loader, parent-owned verified buffer transfer and native read-only
  capability loopback viewer preserve opaque origins/network restrictions. Selected remote
  reads use existing compressed Transport, bounded threads/loads, deadlines and cancellation.
  Tab/document disposal unloads iframe/readers; project renderers own their own buffer disposal.
- Native Work/provider reports and portable export/import regenerate deployment references
  against retained package bytes. Regeneration from ResultInput is an ordinary retained-only
  Work, not training/forward inference. Legacy `html_section` is unchanged. EN/ES guide, schema
  and consumer example ship with the package.
- ctgpgpu16 read-only native probes confirmed a newer OS boot than the recorded WISDOM owners.
  Post-boot inaccessible access sessions were falsely blocking recovery. Boot/lifetime evidence
  now scopes them correctly; live markers, missing birth/boot evidence and newer uncertain owners
  still block. Native previews preserve v6b/c's 47 completed Runs each and v6f's completed Run.
  v6d was succeeded; v6e/g had no scientific Execution yet. No remote recovery was launched,
  scientific evidence modified or WISDOM code changed. Zinc is an ordinary Work: retained
  checkpoints are present, but Study recovery does not certify its single-Work continuation.

Verified: 132 targeted regressions across reports, Work results, export/import, recovery and
Console; 29 final managed/legacy HTML tests including a real Chromium 70 MiB asset, complete
206-record consumer catalogue, exact float32/NaN/mask bytes, isolation, corruption, cancellation,
deadline, portable moves/import and local/remote-profile native readers over disposable loopback.
The four local/remote Work export tests also passed separately after expanding their parametrization.
86 focused ownership/provider/Console tests passed. Ruff passes; 686 checked files are formatted;
mypy passes 631 source files. Wheel 0.19.0 built/installed into a disposable venv and smoked outside
checkout, reusing already-installed dependencies without downloading/installing new dependencies.
No whole-suite run, live remote bundle producer or scientific cluster recovery is claimed. The
earlier architectural/agent-refinement boundaries above remain unfinished; this is not a release.

## Resource robustness continuation — 2026-10-13, no release bump

Native GPU observation now shares bounded UID/host/boot caches, optional official NVML,
isolated/reaped deadline-bound helpers, measured/unknown quality and per-UUID circuits. Numeric
CUDA visibility is pinned once through isolated driver identity enumeration without a compute
context; ordinary memory observations never import Torch or initialize CUDA. UUID/MIG grants
need no driver identity probe. Shared/command admission uses the same placement planner with
temporary pre-spawn commitments, resident reconciliation, bounded cooperative waiters and
validated operational profiles. Failed observation pauses starts, not healthy siblings.

Host/cgroup capacity, PSS/USS/RSS attribution, affinity/quota and child thread ceilings protect
known RAM/CPU budgets. Outer direct CPU/GPU leases are host/boot scoped and GPU/MIG UUID keyed;
release/child visibility use retained identity, not a new ordinal mapping. Legacy unscoped
owners remain protected and queue feedback explains the ambiguity. GPU discovery runs outside
the outer lock. Operational history rotates compressed windows, retains sampled lifetime peaks,
and recovers from archive temporary leftovers without touching scientific evidence. Saved bounded
sanitized reports use native CLI/Console/READ facade services; unsafe generic claim/release pairs
remain explicitly unsupported, never an unscoped provider mutation.

Verification: **293 passed** in the broad targeted gate; **211 passed** in the final overlapping
resource/provider/ARI gate; **58 passed** in Work regressions; **59 passed** in final outer lease/
provider/policy checks; **9 passed** in selected Console checks. Ruff and diff checks pass;
mypy checks **641 source files**. CPU synthetic packing/cache benchmarks and installed wheel
0.19.0 smoke pass without downloading dependencies. The official MCP protocol test is skipped
(SDK absent), not accepted. No whole suite or real GPU/MIG/gpuctl/SLURM acceptance is claimed.

The full attached resource reform is **not finished**. Continue from the explicit pending list in
`docs/ai/IMPLEMENTATION.md#history-resource-robustness-audit`: online RAM/host-global unknown-demand startup and process/PSI
backpressure, phase-aware peer commitments/crash reconciliation, native legacy ownership
reconciliation, complete historical hardware-event reports and authorized provider acceptance.
Do not repeat the verified observer/cache/UUID/fairness work or change WISDOM to pass these gates.

## Incremental fixed Study extensions — 2026-10-14, unfinished

The attached incremental extension request is **not completed**. The implementation checkpoint
is [STUDY_EXTENSIONS.md](../MANUAL.md#guide-study-extensions); no release version is bumped.

Implemented and tested:

- Native fixed-design comparison: one changed rectangular sweep dimension or appended explicit
  seeds, order-independent candidate/cell identity, preservation of origin Trial ordinals,
  exact numeric types, native delta validation and no-op detection. Mixed/deleted/conditional/
  auto-replication/adaptive/composed designs are rejected, not routed through retry.
- `StudyExtensionService.preview` verifies owned successful native Execution/design/Attempt
  evidence, exact input identities, per-Run scientific identity, declared objective evidence,
  origin/current code compatibility and recorded CPU runtime. Global CodeIdentity, Run IDs,
  original configuration, controller/recovery logic and spent cost are unchanged.
- Extension-specific clean Git HEAD / selected-YAML-only proof, native source digest evidence
  for non-Git sources and fail-closed changes, symlinks, other projects, UNKNOWN ownership,
  missing/corrupt result evidence and altered package metadata. Recorded native CPU metadata
  is not a complete arbitrary dependency/hardware/runtime attestation; wider apply remains gated.
- Internal `StudySeriesStore` first-reservation journal: atomic envelope, immutable revision
  hashes, chained append-only history, duplicate/concurrent request idempotency, bounded owned
  metadata, no invented Job/Execution and native controller-lock retention. Local native result
  deletion refuses retained evidence and corrupt/lost journal authority. Job-backed origin
  reservations are explicitly unsupported until their cleanup/retention integration exists.
- CLI `studies extend` defaults to read-only preview; `--apply` refuses before target/Work load.
  CLI show/history reads the domain service's exact owned metadata. No public operation creates
  a series yet. Python preview returns `apply_supported=false` and `series_persisted=false`.

Verification: **89 passed** in the final design/preview/reservation/fixed-recovery/deletion gate;
**13 passed** in a separately selected, overlapping deletion/runtime gate. Ruff and diff checks
pass; mypy checks **8 touched source files**. The 20-cell scalar CPU origin executes through
native Work/Run/Attempt contracts, with only its CPU process boundary inlined in disposable
fixtures; no real scheduler/cluster workload is claimed. The actual 15-cell delta is **not yet
executed**, so 20+15=35 execution acceptance has not passed. Installed-wheel smoke is recorded
below after its completion; no dependency download, live WISDOM change or cluster mutation.

Resume from these remaining gates, without repeating the verified foundation:

1. Source-context-preserving native delta materialization and stronger compatible runtime/source
   provenance; reject unprovable dependencies rather than treating metadata equality as a waiver.
2. SubmissionService stable request/Job handoff, precommit/postcommit crash reconciliation and
   apply revalidation. Never call a second runner or blindly resubmit an uncertain reservation.
3. Immutable operation transitions and repeated extensions; reference guards for Job/storage
   deletion and compaction, exports/imports and products before enabling those live references.
4. Accumulated native read model and StudyAnalysis adapter with original IDs/provenance, paired
   seeds, physical cost, exploratory interpretation and fresh evidence/analysis fingerprints.
5. CLI apply/analyze and nonblocking Console selection/confirmation/history and scientific catalog
   grouping, keeping real Jobs visible. Packaged Study Skill and safe agent READ/PLAN integration.
6. Actual CPU 20→35/20→32 native submission execution, failed delta recovery, concurrency and crash
   fault injection, repeat extension and authorized single-cluster/runtime/provider acceptance.

Fleet, arbitrary migration and silent compatible-code acknowledgement remain unsupported.

Installed-wheel gate passed outside the source checkout: wheel 0.19.0 was built without build
isolation/index/dependency resolution and installed only into an owned temporary target. Smoke
imports the installed service/journal, validates the native 20→35 comparison and CLI grammar,
and verifies both packaged extension guides. Existing dependencies were reused; no new dependency
was downloaded or installed. This packaging check does not certify delta execution or remote support.

## Developer experience continuation — 2026-10-15, unfinished

Incremental UX implementation over HEAD `8df2a1c`, preserving dirty resource/report/Study-extension
work. Version stays **0.19.0**; the attached broad UX reform is **not finished**. See
[DEVELOPER_EXPERIENCE.md](../MANUAL.md#guide-developer-experience) for commands and precise boundaries.
SHA-256 checks confirm all seven protected extension service/design/journal/CLI/test files are
byte-identical to the starting checkpoint; config diff reuses only their pure JSON helpers.

| Area | Implemented | Still pending |
|---|---|---|
| Init/authoring | Safe derived names, real instructions/tests; installed schema, static field query, native config diff | Editor merge, native recipe copying; diff does not authorize recovery/extension |
| Discovery | Actual-parser bilingual task help, typo suggestions, pure Bash/Zsh/Fish completion | Real Zsh/Fish sessions and dynamic entities; Bash syntax tested |
| Operations | Compact exact-ID inspect/diagnose, native history/diagnostics, bounded change-only NDJSON watch | Consolidated native retry eligibility and per-Run resource/log streams |
| Console | Local Work/Study filters, consistent exact opening, retained scroll/snapshots, late-reply disposal | Separate debounced/paged global entity search, sorting/paging, Workspace extraction |
| Results | Native `ResultStore.table` JSON/CSV pages, exact selectors, mandatory provenance, latest/all Attempts, censored snapshots | Cross-Execution compatible/paired comparisons via StudyAnalysis/AnalysisProfile; remote table integration |
| Agents | Compact native context/doctor; installed official Codex grammar, reviewed registration and TTY launch | Registry is not handshake; SDK absent here; real registration/conversation and unrelated-config preservation acceptance |
| Skills | Work authoring and Study designs/recovery/extension boundaries with executable toy examples | Other priority procedures and richer workflows |

Verification: **116 passed** in CLI/client/assets/project-isolation/operational-health/extension
regressions; **50 passed** in semantic and native Work/results regressions; **16 passed** in final
overlapping table/inspection/Skill-example checks. Earlier overlapping gates: **72 passed,
1 deselected** for selected agent/Console/diagnostic tests, **9 passed** for focused navigation.
Ruff and diff checks pass; mypy checks **19 touched source files**. Both changed Skills pass the
Skill creator validator; the new example test validates native fixed/repeated/adaptive WorkConfig
and all documented CLI lines without running science. No full backend suite or GitHub CI claimed.

Wheel 0.19.0 built without build isolation/index/dependency resolution, installed only into an
owned temporary target; generated consumer package built/installed separately without dependencies.
Outside the checkout, installed help/schema/completion stay free of Torch/consumer imports;
disposable native scalar CPU Work execution and exact ResultStore/table catalogue pass. No live
WISDOM evidence, GPU/SSH/SLURM workload, cluster mutation, Codex registration/launch, global client
settings or release number changed. Registry tests use isolated simulated client state.

Continue from the pending column, without repeating verified blocks. Keep diagnosis/recovery
authority in native services and compare explicit metrics/splits/designs, never survivor-only
confirmation. Add editor previews/merges, offline report navigation/accessibility, remaining Skills,
shared action/capability metadata and typed MCP responses incrementally. Preserve default READ/PLAN.
Complete real SDK/client/config-preservation, isolated shell/editor acceptance and repeatable light
command/Console metadata benchmarks before declaring the full reform accepted.

## Historical checkpoint: first executable fixed Study delta — 2026-10-16

The acceptance section below supersedes the pending scope of this historical checkpoint.

The request to finish intuitive **remote WISDOM** extension is **not completed**. This checkpoint
implements a first explicitly reviewed **local CPU** delta and the shared Console dialog; remote/
GPU attestation, successive accumulated revisions/analysis, full handoff-crash reconciliation and
series-aware portable exports remain pending. The earlier read-only checkpoint is superseded only
for this stated local capability. No release bump, WISDOM edit or live cluster operation.

Implemented: revalidation of the reviewed plan, checksummed immutable revision and handoff events,
preallocated native Job identity, same-directory private YAML snapshot preserving source-relative
inputs/name, pinned bytes and scientific identity at native preparation, duplicate/concurrent
request suppression, and explicit `studies refresh` reconciliation of exact completed delta cells.
Job/Execution/Series identities stay distinct. Native result deletion and Job history/storage GC
retain series references; completed delta identity must match its pinned submission. Missing or
uncertain handoff authority blocks replay. No synthetic Job/Execution or second scientific runner.

Console **Extend Study…** preselects the authored YAML, offers the existing picker, shows conserved/
new/total counts and changed values/seeds, invalidates review on selection changes and submits once.
Loading, errors (including consumer import SystemExit) and acceptance remain visible. Source YAML
remains untouched; recents route delta Jobs to the authored YAML rather than the private snapshot.
Study actions use a two-row grid. Imported evidence is read-only; remote/GPU is explicitly rejected.
Already dispatched series/deltas cannot become new origins and silently replay prior cells.
The packaged Study procedure was updated with skill-creator and its validator passes.

Verification: **46 passed** in the focused design/preview/apply/Console gate; **69 passed** in an
overlapping apply/storage/deletion gate; **82 passed** in storage/submission/recovery/deletion;
**22 passed** in Console recovery/history/products; **31 passed** in authoring/packaged-asset checks.
The final **9-test** apply/dialog gate additionally covers preparation code drift, relative inputs,
implicit names, duplicate/concurrent handoff, visible SystemExit and completion. CPU evidence really
executes 20 original + 15 disjoint new native Runs with unchanged original bytes; only the provider/
process preparation boundary is inlined in that fixture. This is not real detached/GPU acceptance.
An initial broad gate had 137 pass / 1 asynchronous history-view failure; separate reruns passed.
The new dialog test was made condition-waiting instead of assuming one UI pause completes a thread.
Ruff/diff checks pass and mypy checks 14 touched sources. Installed-wheel results are appended below.

Continue with native same-placement managed/GPU provenance and verified delta execution before
claiming the user can extend WISDOM on its cluster. Then implement cumulative cell authority and
budgets across revisions, series grouping/analysis and exact portable retention/export. Keep first
successful evidence immutable and never bypass these gates with Resume, a full rerun or manual
journal edits. Do not repeat the verified first local delta/UI foundation.

Installed-wheel import/API smoke passed outside the checkout, including the reviewed apply/refresh
CLI, native service/journal, Console dialog and both packaged guides. Build/install used only an
owned temporary target, no index, dependency resolution or new dependencies. This is packaging
acceptance, not a claim of real detached provider execution. Final completion validation also
reuses native recovery input/Run checks and compares the delta's recorded runtime fingerprint.

## Incremental fixed Study series acceptance — 2026-10-16

Implemented the reviewed extension workflow through the native CLI and Research Console. Edit
the authored YAML, open **Extend Study…**, review retained/new Runs and confirm its exact plan.
Only disjoint new cells are submitted through SubmissionService. Successive revisions compare
the accumulated design, not just the origin: native CPU fixtures execute **20→35→56** and
**20→32**, preserving original evidence bytes and exact Execution/Run/Attempt identities.

Single-cluster managed execution now inspects authoritative evidence on its original host and
reuses its retained immutable environment through the native preparation/scheduler/launcher.
Runtime and granted GPU-type compatibility are checked before consumer execution; GPU ordinals
and grant count are not hardware identity, and provider visibility is never broadened. Remote
metadata is paged and checksummed without downloading model/checkpoint/artifact histories.
Code, input/dataset, objective, seed, ownership and cumulative budget checks remain mandatory.

The Console groups opted-in revisions as one Study series with cumulative Runs/metrics, native
Job navigation, revisions and history. Analysis uses the existing StudyAnalysis engine with exact
provenance and explicit post-hoc exploratory interpretation, not survivor-only confirmation.
Series exports preserve independent native evidence packages plus cumulative JSON/HTML/journal;
referenced Jobs, Executions and environments remain protected from deletion/GC.

Handoff crash reconciliation reuses the preallocated Job only when no request/provider execution
exists. Missing or uncertain authority fails closed. Duplicate applies return the accepted Job.
Failed deltas use native explicit retry of their original Execution, retain successful cells and
Attempt costs, and remain attached through exact retry lineage. Live unsealed metrics are a
display overlay, not silently accepted scientific evidence.

Verification: **128 passed** in the focused extension/design/Console/retry/export/agent-assets
gate; **63 passed** in storage/ownership/deletion; **70 passed** in remote paths, coordinated
submission, TLS and GPU-access regressions. Ruff passes and mypy checks 19 changed source files.
The Study Skill passes its frontmatter/reference validator. A locally built **0.19.0** wheel was
installed into an owned temporary target without dependency resolution or index access; outside
the checkout its CLI, service, Console, retained-runtime hooks and bilingual packaged guides
pass smoke checks. No release bump, WISDOM edit or live cluster operation was performed.

Limits: the origin must be an owned successful rectangular fixed sweep with explicit seeds;
each revision adds values in one existing dimension OR appends seeds. Adaptive/conditional/
composed/Fleet expansion, migration, unmanaged remote environments, local GPU extension and
executable Series import remain unsupported. Checkpoint training continuation still requires
consumer support. Remote acceptance uses an isolated loopback provider and native CPU processes;
GPU contracts use synthetic metadata, not a claim of real SSH/SLURM/CUDA or WISDOM acceptance.
See [Study extensions](../MANUAL.md#guide-study-extensions) for the supported workflow and exact limitations.

## Conditional/selective fixed extension refinement — 2026-10-16

The previous rectangular-only restriction was real: conditional origins failed before comparison.
Native configuration now supports additive complete scalar points through `sweep.include` and
exact sparse selections through mutually exclusive `sweep.candidates`. Both retain authored
ParameterSpace activation; inactive keys stay absent. Include unions base evidence, not new
Cartesian domains. Explicit candidate mode does not enumerate a Cartesian pool. Existing YAML
without these keys keeps its native variants/design identity.

Fixed extension compares exact active candidate/seed sets, rejects removed historical candidates
or changed activation contracts and constructs only their disjoint native delta. Shared controls
are not repeated. A new joint candidate still executes when all its marginal values were already
observed. Explicit seed growth remains a separate revision. Native submission, revision authority,
cumulative analysis, costs, code/input checks and historical records are unchanged.

Verification: the focused regression gate passed **148 tests**, including native conditional
delta execution, successive revisions, Console, recovery and loopback retained-prefix dispatch.
The expanded sparse-specific suite passed **15 tests** (overlaps that gate), including schema and
no-Cartesian-enumeration checks. Ruff and mypy pass for six changed source interfaces; schema and
diff checks pass. Four synthetic count-shape fixtures cover 60+32, 48+28, 48+32 and 96+28 Runs;
these are not inspection or acceptance of actual WISDOM V6 records. No WISDOM edits/submissions,
live GPU/SSH/SLURM checks or release bump were performed.

Operational limit: controller installation alone cannot add syntax to an older retained immutable
remote environment. Preview now checks native sparse-parser capability before handoff, and dispatch
rechecks the reviewed capability. Missing support fails closed with an explicit diagnostic; this
change does not implement an acknowledged compatible-runtime transition or patch that environment.
Do not promise existing remote V6 extension until its real reviewed preview passes. Adaptive,
Fleet, migration, composed and structured-valued extension exclusions remain in force.

Final gate after consolidating point validation through ParameterSpace: **57 passed** across
sparse/conditional comparison, native cumulative series and retained-prefix dispatch (overlapping
the earlier gate). The final 0.19.0 wheel built without index/dependency resolution and passed
installed validation outside the checkout: include → exact sparse delta, schema and both packaged
guides. Installation was only into an owned temporary target; the smoke executes no science.

## Fixed extension authoring-order repair — 2026-10-16

Read-only inspection of WISDOM V6a `job-20261011132247-5c2dec2e` confirmed that sorted frozen JSON
keys regenerated 12 of 15 Trial positions differently. The Execution-owned authoring snapshot has
exactly the same JSON values and reconstructs the original fingerprint with zero Trial mismatches.
The extension reader now uses that owned, bounded, regular snapshot only after exact semantic
equality with frozen configuration; original native fingerprint and per-Attempt checks stay intact.
Explicit ordered JSON survives canonical wire paging. Local preview/completion and the source-
shipped remote reader share this behavior; no live edited YAML supplies historical authority.

Regression checks cover multidimensional conditional ordering, live-source edits, remote loopback
paging, exact integer/float distinction and missing/corrupt/changed/symlinked snapshots. The focused
preview/series/remote gate passed **42 tests**; the expanded order/apply/dialog gate passed **16**
(overlapping), with Ruff and mypy passing on three changed sources. No release version changed.

The actual V6a evidence reader now verifies **60 successful Runs / 64 spent Attempts** without
rewriting evidence or submitting computation. The subsequent real read-only preview remains
blocked by existing code compatibility: its original Git provenance was dirty, and its changes
digest differs from today's dirty tree despite the same HEAD. No equivalence waiver, code/runtime
migration, WISDOM edits or extension apply was performed. This repairs the screenshot's false
configuration mismatch, not a claim that current V6a is ready to submit.

Final combined order/preview/apply/series/remote/design/sparse/Console regression gate:
**81 passed**. No additional scientific state changes were performed.

### Revisioned fixed Study expansion (2026-10-16)

The subsequent code-compatibility refusal was real but too restrictive for deliberate expansion.
Default `code_policy="revisioned"` now retains the immutable origin and permits a reviewed
current-project/native-runtime revision. It is not a code-equivalence waiver. Exact compatible
extensions retain their existing behavior; `--code-policy strict` preserves equivalence-only
review. Fixed parameters/content/datasets/objectives/seeds/design and successful Attempt authority
remain checked. Only the exact new cells execute, without refunding failed Attempts or time.

Managed remote revisions use ordinary bundle/environment/scheduler preparation, not a patched
historical prefix. Current source/dependency declaration is pinned through detached preparation.
A host-side native preflight compares the actual staged semantic delta and records its own
scientific identity separately from the controller pin. It does not mutate seed coordinates;
host namespace differences are not permission to alter literal seeds/design. Worker config/code
checksums and completion/retry use that attestation. Metadata readers select the actual recorded
native interpreter for both extension preflights, not the login Python by accident.

The series keeps all exact native provenance and raw metrics. Unproven code/runtime cohorts never
contribute to a pooled seed mean or cross-revision effect analysis. StudyAnalysis runs per revision;
the default is the latest cohort, JSON includes `code_cohort_analyses`, and HTML has sandboxed
**Code · revision-N** native-analysis tabs. Original metric semantics stay with original evidence.
The Console shows the warning before **Extend · current code** confirmation and records code
policy in revision navigation. This is one history, not a fake relabelled or restarted experiment.

Read-only WISDOM V6a preview for `job-20261011132247-5c2dec2e` now permits **60 reused + 32 new
= 92 cells**, retaining **64 spent Attempts**, with `revisioned-current` and ordinary managed
deployment proposed. No WISDOM source/evidence was edited and no real extension was submitted.
The disposable loopback provider exercises the actual detached worker/preflight/native CPU Runs
and collection, with fresh host namespace/current code, plus rejection of changed staged seeds.
Provider/environment preparation is intercepted; this is not real SSH/GPU/managed-install acceptance.

Verification: **149 passed** in the combined extension/order/recovery/Console/report regression
gate; **1 additional successive-code-revision test passed**. Ruff and mypy pass on the changed
sources. The installed wheel is checked outside the checkout, without installing dependencies.
No release version bump. Unsupported adaptive/Fleet/migration/local-GPU/unmanaged-code-deployment
boundaries remain explicit. Per-revision HTML still obeys the existing 16/64 MiB document/report
limits; many revisions may require separate reports rather than unbounded embedded documents.

### Large extension preflight / pre-Execution retry repair (2026-10-16)

Read-only diagnosis of WISDOM V6a extension `job-20261016125643-ce4bcdd6` found a controller-side
preparation failure in `prepare_native_delta`, not a consumer Work exception. Its reviewed contract
was 539,412 compact JSON bytes (530,026 metric-semantics bytes) and was passed verbatim in SSH argv.
Password SSH failed with connection reset at this call. There was no scheduler ID, remote provider
request or new Execution result; the original successful revision was not rerun or altered.

Preflight now sends canonical per-field SHA-256 commitments and recomputes complete contracts on
the host. No field/content identity is truncated or ignored, numeric types remain exact, mismatches
name their fields, and native scientific fingerprint/config/code checks are unchanged. A real
**read-only** preflight of that already-prepared configuration passed with a **5,747-byte** SSH
command, without staging a guard, dispatching a provider or calling Work.run.

Native retry now retains extension ownership and the original submission pin even if preparation
failed before an Execution existed. It retries that delta's preparation only, keeps its native
retry lineage and prior evidence, and does not invent checkpoint continuation. Local fallback
discovery filters Execution manifests by the exact Job, avoiding an unrelated original Study.
Successful existing-Execution recovery still retains its original preparation attestation.

Verification: **106 targeted tests passed**, including a >600 KB native AnalysisProfile preflight,
changed descriptions/parameters/seeds rejected, canonical type integrity, read-only validation,
preparation retry through the real native WorkRunner/series collection, duplicate retry rejection,
existing code-cohort/retained-prefix dispatch and adaptive/fixed recovery. Ruff and mypy pass on
changed sources. No version bump, WISDOM edit, persisted scientific rewrite or real retry was applied.

Remaining live recovery choice: WISDOM's current source review pin now differs from the submitted
one. This is separate from the diagnosed SSH failure. No source approval was silently refreshed;
retry still protects the reviewed source. Re-approving different code for this pending extension
or resubmitting its retained prepared code is not implemented by this focused repair and requires
an explicit reviewed workflow, not manual edits to its journal or a full-experiment restart.

## Manual consolidation, clustering modules and generic FGW — 2026-10-16

The 32 former topic/audit documents are consolidated without dropping their substantive content:
public material is indexed in `docs/MANUAL.md` / `docs/MANUAL.es.md`; historical audits are archived
below, not presented as new acceptance. Only the two manuals and four essential contributor/ledger
references remain under `docs/`. Packaging, repository links, agent documentation metadata and all
24 packaged Skills point to retained chapters. Skill-creator validation and local link/anchor
regressions verify that migration; specialized procedures, examples and authority boundaries remain.

Clustering strategies and shared contracts/results now have one implementation class per module.
Existing public clustering imports, backend policies and numerical behavior are retained. Generic
`lambdaforge.distances` provides `StructuredMeasure`, `FusedGromovWassersteinDistance` and `FGWResult`:
optional POT conditional gradient, balanced q=2 costs, arbitrary feature metrics/precomputed cross
distances, explicit units/normalized weights, feasible warm starts, marginal and local-convergence
diagnostics and dense-memory guards. No four-index loss tensor, domain-specific parsing, clustering
or unearned global-optimality/metric claim is introduced. The optional POT CI job gates real solves.

Local acceptance: **66 targeted tests passed**, covering actual POT numerical objectives, marginals,
identity/permutations, uneven/zero weights, custom attributes, failure/iteration limits, optional
imports, memory guards, existing clustering, manuals/examples, packaged Skills and neural exports.
Ruff check/format passed for touched implementation/tests; mypy passed for 21 source files. All
24 Skills passed the skill-creator validator. A clean temporary wheel built and its installed
observer/CLI/resource/asset smoke passed outside the checkout; the installed FGW example also ran
with POT. POT was installed only into the explicitly authorized temporary test target.

The additional global structure/public-API run had **5 passes and 1 failure**: four existing classes
lack docstrings (`GPUObservationPolicy`, `ResourceSafetyPolicy`, `ExtensionReply`, `SeriesReply`).
Those unrelated implementations were not changed or reported as passing. No WISDOM source, real
cluster, persisted evidence, scientific workload or normal project environment was modified.
Version remains 0.19.0; no release bump was requested.

## Idle-GPU dispatch diagnosis and correction — 2026-10-17

Read-only inspection of WISDOM's actual managed runtimes/metadata identified two independent
failures (not inferred from a utilization percentage alone):

- `wisdom-v6g`, `job-20261016165002-1ab71d38`, cluster 12: 66 completed, 54 pending, zero active,
  about 79 GiB physically free on the granted H100. The native planner approved baseline work but
  `RESOURCE_LAUNCH_WAIT` recorded `study_ram_allocation_headroom_or_unknown`. A controller-only RAM
  share plus a whole-tree launch check charged helpers against an already fully distributed
  allocation, leaving the first replacement Run permanently blocked.
- `wisdom-v6h`, `job-20261016132142-9e6d1052`, cluster 16: 110 completed, 104 pending and two active
  H200 Runs, one per granted token (`1`, `3`), about 121 GiB physically free each. Persisted live
  statistics retained startup positive streaks after 39/47 real progress cycles without a material
  peak. Both residents stayed RAMPING and the exploratory lane could not promote, while a generic
  rare full-device tail became each resident's deterministic full-device commitment.

Native corrections reconcile derived elastic RAM shares with the same fresh process-tree
measurement used for launch, without shrinking independently authored requirements or exceeding
the Job/physical/cgroup ceilings. Real flat progress closes an old growth streak; duplicate polls
do not. Valid compatible terminal envelopes can bound stable resident commitments, retaining live
rare-tail samples/weights for joint-fit risk; cold-start uncertainty and OOM bounds remain intact.
Fresh launch checks reuse the approved native planner's sibling context and saved exploration
WAIT comparison, rather than restarting a single-device planner after consuming wait regret.
`launch_waits` and idle reasons distinguish a planner proposal from a worker actually starting;
operational history remains deduplicated. Candidate/seed/objective/design and GPU grants are unchanged.

Verification: 127 targeted resource/runtime/recovery/documentation tests passed; a broader native
dispatcher, grant recovery, resource benchmark and runtime set passed 189 tests (sets overlap).
Ruff check/format and mypy on the three changed implementation modules passed. Checks used local
disposable fixtures and read-only remote metadata/driver queries, not scientific workloads.

Operational boundary: source changes do not hot-patch immutable environments or already running
controllers. The operator explicitly approved cancellation/recovery of cluster 12 only, keeping
cluster 16 running. Native cancellation of `job-20261016165002-1ab71d38` occurred with zero active
Runs. Native recovery preview verified unchanged YAML/science/consumer code: reuse 66 successful
Runs, retry 3, retain 51 pending, preserve 72 spent Attempts and physical budgets. No code-change
acknowledgement override was required. Native retry created `job-20261017102915-3bb4b922`, using
new immutable `env-72dd487020659c94443a910f`; read-only runtime inspection verified the corrected
launch/lease implementation. The same `execution-27fc9197382c151f` then reported two active Runs
on the original granted H100 token and all 66 completed Runs retained. No WISDOM source, scientific
record, checkpoint or cluster configuration was manually edited. Cluster 16's original Job was
not cancelled, restarted or patched; its already-running controller still requires a separately
authorized future native recovery to adopt the fix. Existing scientific identity and recovery
gates remain authoritative; checkpoint-level continuation is never implied without Work support.

## Historical audit archive

Preserved reports from previous implementation passes; their dates and remaining items are historical, not new verification claims.

<a id="history-architectural-reform"></a>

## Architectural reform: audit and implementation ledger

[Español](IMPLEMENTATION.es.md#history-architectural-reform)

This is a continuation ledger, **not a declaration that the requested reform is complete**.
Baseline: `d2f526d`, LambdaForge 0.17.0. No WISDOM files, productive cluster Jobs or persisted
scientific records are modified by this reform.

Follow-up at `c8f2d2e`: [scientific correctness audit](#history-scientific-audit) closes ordered paired
evidence, persisted lookahead recovery and conclusion/stability inconsistencies. It does not close
the larger lifecycle, learned host allocation, product dependency or distributed Fleet scope below.

<a id="history-architectural-reform-audit-findings"></a>

### Audit findings

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

<a id="history-architectural-reform-implemented-foundation"></a>

### Implemented foundation

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

<a id="history-architectural-reform-further-integration-and-hygiene-corrections"></a>

### Further integration and hygiene corrections

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

<a id="history-architectural-reform-product-model-catalog-and-transport-layer"></a>

### Product model, catalog and transport layer

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
Execution ownership locks, not training. See [Products](../MANUAL.md#guide-products). This is **not** composed/Fleet
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
[Dataset reconstruction](../MANUAL.md#guide-dataset-reconstruction). Contract-aware YAML resolution remains pending.

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

<a id="history-architectural-reform-remaining-work-in-dependency-order"></a>

### Remaining work, in dependency order

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

<a id="history-architectural-reform-validation-checkpoint"></a>

### Validation checkpoint

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

<a id="history-scientific-audit"></a>

## Scientific correctness audit

[Español](IMPLEMENTATION.es.md#history-scientific-audit)

Baseline: `c8f2d2e`, LambdaForge 0.17.0. This closes the ordered-evidence and
conclusion-consistency corrections below, **not the entire requested engineering audit**.
No WISDOM code, real cluster, historical scientific manifest or GitHub CI is changed.

<a id="history-scientific-audit-confirmed-defects-and-corrections"></a>

### Confirmed defects and corrections

| Classification | Authority / previous behavior | Correction and evidence |
| --- | --- | --- |
| CONFIRMED_BUG | `SequentialSweep.evaluate` sorted the intersection of numeric seed IDs; newly completed blocks could reorder previous weighted observations. | Requires explicit persisted `seed_order`, consumes only its complete prefix, records exact inference seeds. Tests use the real affine ProjectSeedStream, reversed worker arrivals and missing early cells. |
| CONFIRMED_BUG | `runner._execute_fixed_evidence_group` initialized `committed_lookahead=None` even during recovery. | Restores the existing `sweep-blocks.json` commitment through shared read-only validation; a partially completed speculative block prevents opening another. An integration test preserves original Run records, manifest and physical spend across interruption. |
| CONFIRMED_BUG | Block telemetry reported min/max completed ordinals even across holes. | Reports the complete contiguous prefix only. Initial block inventory is durable before dispatch, including interruption before the first terminal result. |
| CONFIRMED_BUG | `WorkRunner._summary` counted only initial authored obligations, potentially calling a sweep succeeded with a committed lookahead cell missing. | Recovery and native finalization share `recovery.fixed_requirements`: created blocks remain required without rewriting the immutable design. Integration preserves three valid Runs and exposes the fourth as pending, rather than restarting them or hiding debt. |
| CONFIRMED_BUG | ScientificDesign changed unsupported predictions to `UNRESOLVED` while retaining another hypothesis's stability. | Unsupported descriptive conclusion has zero realization support; predictive conclusion/stability and missing-support reason remain separate. No coverage is relabelled confidence. |
| CONFIRMED_BUG | `_exact_conclusion` and sweep post-hoc analysis assigned `1-modal_probability` to an unresolved event; empty distributions claimed stability one. | Uses the actual unresolved token mass, or zero without realizations. Weak numerical leaders no longer rewrite observed equivalence realizations into an unrelated winner probability product. |
| CONFIRMED_BUG | Study Analysis treated any sequential `stop=true`, including `INCOMPLETE`, as formal resolution. | Shared formal-state reader requires a scientific decision; operational stopping remains unresolved. Absence of a sequential procedure is `None`, not formal approval. |

The remote retry reader embeds the same pure block validator from its authoritative source, like
the existing recovery/seed readers. A regression executes it with isolated Python (`-I -S`),
without an installed LambdaForge runtime; it does not claim real SSH acceptance.

`ScientificQuestionState` now exposes descriptive and formal resolution separately. Its legacy
`resolved` accessor retains exploratory descriptive applicability when no formal procedure exists;
it must not be read as a formal confidence guarantee.

<a id="history-scientific-audit-statistical-contract"></a>

### Statistical contract

The estimator formula remains the predictable plug-in empirical-Bernstein sequence from
[Waudby-Smith and Ramdas, Theorem 2 and equation 15](https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf).
Its weights depend only on the preceding observations. Paired differences are bounded and the
theorem assumes a common conditional mean; deterministic seed generation alone does not prove
that scientific assumption. Primary-family Bonferroni and authored practical margins are unchanged.
Monte Carlo tests check implementation regressions, not the theorem or real-model calibration.

<a id="history-scientific-audit-compatibility-and-ownership"></a>

### Compatibility and ownership

- Runtime inference is `paired-pm-eb-cs-v2`, with `committed-acquisition-prefix-v1` and exact
  `evidence_seeds`. Historical v1 records remain inspectable, never rewritten or relabelled v2.
- Immutable historical `StudyDesign.replication_policy` declarations remain unchanged; the actual
  evaluation record owns its runtime policy version. No Run/Attempt/scientific IDs change.
- Block inventory remains version 1: it already contains ordinals and commitment. Recovery
  rejects duplicate/discontinuous ordinals, conflicting commitments and inconsistent seed metadata.
  Initial coordinates come from `seed_source.resolved`, not guessed numeric order. Missing
  authoritative coordinates fail before recovery writes. Read previews remain read-only.
- Scientific-question snapshots are version 4; parameter conclusion semantics are version 3.
  Existing `confidence` aliases mean descriptive stability of the exact displayed event, never
  probability of truth or formal coverage. Old persisted analyses are not silently recomputed.
- Explicit finite seeds keep authored ordinal order; fixed sweeps do not gain adaptive pruning.
  Retry/continuation remain new Attempts of the same logical Run. Imported evidence remains inert.

<a id="history-scientific-audit-reproducible-verification"></a>

### Reproducible verification

```bash
pytest -q tests/hpo tests/work tests/analysis tests/products tests/tui
python benchmarks/sequential_sweep_audit.py
python benchmarks/hpo_scientific_design.py
python benchmarks/resource_scheduling.py
ruff check .
mypy src/lambdaforge
```

The audit benchmark peeks at every prefix of 200 paths, up to 96 blocks, with three reference
comparisons in degenerate, low-variance and high-variance null scenarios. The resource benchmark
is an existing synthetic regression, not real-GPU validation or a new scheduler improvement.
The existing scientific-design benchmark covers only three small landscapes; it is insufficient
to claim full eight-scenario scientific acceptance.

Local results for this change:

- Broad retained HPO/Work/Analysis/products/Console suites: **718 passed, 628.84 s**, before the
  final committed-debt and isolated-reader additions; then the final affected backend suites:
  **144 passed, 43.70 s**. This is not a claim that full repository pytest was rerun.
- Ruff, mypy (**581 source files**) and whitespace checks passed.
- Installed wheel outside the checkout: packaging/import/assets smoke, CLI help/scaffold/validation,
  native spawned CPU automatic sweep with four Runs and read-only recovery preview passed.
  Dependencies were reused in a temporary prefix; no user project environment was replaced.
- Calibration: **0/200** family violations in each of three null scenarios, 96 possible looks.
  The three-landscape benchmark retained mean regret approximately **3.70e-17** for both policies.
  Existing synthetic resource benchmark: adaptive makespan **58 s**, fixed **109 s**, both zero
  OOMs. These are regression results of existing scheduling, not a new scheduling improvement.
- CUDA hardware, real SSH/SLURM and new trace-derived resource experiments were **not run**.

<a id="history-scientific-audit-explicit-unfinished-boundaries"></a>

### Explicit unfinished boundaries

| Classification | Remaining work | Acceptance needed |
| --- | --- | --- |
| CORRECTNESS_RISK | Full lifecycle/continuation fault injection and physical overhead accounting across all readers. | Interrupted checkpoint/publication, delayed terminal delivery, heterogeneous rungs and no-information continuation tests through all native paths. |
| DEFERRED_FEATURE | Demand-learned unequal CPU leases and additional ARI calibration. | Real owned-process demand/progress experiments and trace-derived replay superiority, preserving grant/affinity safety. Existing residency-based leases and ARI are not replaced speculatively. |
| DEFERRED_FEATURE | Revised scientific action valuation across correlated questions and costs. | Expanded eight-landscape comparisons with support, calibration, cost and regret; existing question/optimization authorities remain intact. |
| DEFERRED_FEATURE | Broader WorkRunner extraction, contract-aware dataset substitution and product dependency waiting. | Characterization and end-to-end ownership/identity/recovery tests; explicit certificate policy, no inferred transitivity. |
| DEFERRED_FEATURE | Public Fleet recovery/adoption, resumable evidence transfer and distributed export. | Complete lease/fencing/dispatch/acknowledgment/recovery/export acceptance chain. Existing distributed gates stay explicit. |

Existing products, certificates, import/export, resource replay and failure classification are
covered by retained suites; that is regression coverage, not completion of their pending features.

<a id="history-resource-robustness-audit"></a>

## Resource robustness audit — 0.19.0, 2026-10-13

Baseline HEAD: `8df2a1c504110b068eb728f97b4ab860d42b2632`. The working tree already contains
unreleased agents/recovery/report work; preserve it. This audit does **not** establish the cause
of CiTIUS server restarts. No remote connection, reservation operation or stress workload is authorized.

<a id="history-resource-robustness-audit-actual-ownership-and-process-hierarchy"></a>

### Actual ownership and process hierarchy

```text
controller CLI/Console → ControlPlane/SubmissionService (preparation; native provider)
  → ProcessScheduler → detached ProcessSupervisor (one per admitted direct Job)
    → site gpu exec wrapper, when configured → python WorkRunner / Study controller
      → fresh one-worker spawned ProcessPoolExecutor per admitted GPU Run
        → consumer Work.run → DataLoader/tool/map descendants and native library threads
```

The CLI need not remain alive. SLURM supplies its allocation rather than direct whole-device
leases. Fleet uses the existing StudyDispatcher/member boundary; recovery remains explicitly
unsupported where documented. Planned Runs, active Runs, OS processes and library threads differ.
ResourceRequest is a declaration; HostResourceLease is an elastic share inside a Study, not a
physical quota or a host-global allocation. ProcessGuard narrows affinity/native thread pools
and watches parent death; it does not silently edit a consumer's DataLoader worker count.

<a id="history-resource-robustness-audit-hypotheses-existing-protection-and-required-changes"></a>

### Hypotheses, existing protection and required changes

| Hypothesis | Actual owner / protection | Necessary condition and possible effect | Change / verification |
|---|---|---|---|
| A: simultaneous memory admission | runner `_launch_gpu_run`; SharedGPUAdmission serializes check/start in shared mode | multiple controllers see free bytes before a new process ramps | stable UUID lock plus reconciled startup commitment; deterministic competing controllers |
| B: command bypass | supervisor sets admission root only for shared, not command | same external grant reused by multiple Jobs | enable cooperative coordination without claiming/releasing provider resources |
| C: CUDA observation contexts | `_gpu_memory_inventory` and `_gpu_hardware_labels` spawn Torch/CUDA helpers | periodic controllers or model-label read | ordinary NVML observation, no Torch/context; fake backend/import checks |
| D: duplicate driver queries | runner attribution/activity, supervisor usage, ResourceService and DirectProbe each query | concurrent polling of same host/device | one bounded shared observation cache; test concurrent cache misses/TTL |
| E: blocked heartbeat | `_record_usage` subprocess has **no timeout** | stuck driver/tool query | isolate/time-bound observation; heartbeat survives, no scientific cancellation |
| F: declared RAM versus actual | supervisor capacity leases sum declared RAM; zero is allowed; HostResourceLease divides current resident budget | undeclared/underdeclared usage or cgroup smaller than machine | physical/cgroup capacity, startup commitments and qualified PSS/USS; do not call RSS sum physical truth |
| G: descendant amplification | ProcessGuard / supervisor inspect descendants | many DataLoaders, map/tool workers or native threads | report process/thread counts and inherited affinity; test PID reuse/access denial |
| H: native pools | ProcessGuard limits Torch/threadpoolctl and runner rebalances Run leases | libraries initialized before worker budget or consumer explicitly spawns workers | preserve current child-only limits; do not rewrite scientific worker parameters |
| I: filesystem bursts | checkpoint atomic APIs, storage admission, changed-write signatures in ResourceHistoryStore | concurrent large snapshots/publications/checkpoints | retain integrity and required checkpoints; measure bounded operational write/probe overhead |
| J: occupancy versus throughput | GPUPlacementPlanner already learns compatible slowdown and protects heavy work | sparse/heterogeneous performance evidence | retain planner, test coordinated overhead/throughput; no permanent single-Run cap |
| K: revoked grant | runner periodically calls visibility_command, terminates only verified affected worker | provider revokes initial token | parse declared contract, preserve siblings; unknown probe never means revocation |
| L: driver degraded | runner pauses admission on probe failure, existing OOM placement dominance | repeated inaccessible/reset/identity failures | scoped expiring observation circuit/backoff; never infer ECC/Xid from generic exceptions |
| M: lease versus external grant | GpuAccessPolicy command wraps the scientific child; native leases are cooperative | internal lease interpreted as provider ownership | grants remain provider-owned; never broaden inherited visibility or release a manual reservation |

<a id="history-resource-robustness-audit-safety-boundaries-verified-by-source-inspection"></a>

### Safety boundaries verified by source inspection

- Native GPU Run workers are fresh spawned one-worker processes, collected/closed on completion;
  no idle persistent CUDA pool is required. Scientific/HPO decisions stay at the existing controller.
- GPUPlacementPlanner/ResourceDemandModel retain learned peak, censored live trajectories,
  exploration lanes, exact packing OOM evidence, starvation/throughput checks and authored ceilings.
- ResourceHistoryStore persists exact/censored operational evidence separately from objectives.
- SharedGPUAdmission currently has per-token short locks and launch staggering but no startup
  byte commitment; raw CUDA ordinals are not safe shared physical identities.
- Supervisor capacity leases currently use command-sensitive `matches()` as death evidence;
  inaccessible or exec-changed owners must instead remain protected by native lifetime checks.
- StorageAdmission already separates physical free bytes from declared future commitments;
  failed publication/checkpoints remain protected. This task does not delete any scientific bytes.
- OperationalHealth distinguishes stale/unreachable/unknown/lost and scientific outcomes.
- MonitorService has one persisted monitor owner and bounded provider concurrency, not CUDA
  compute workers. UI/provider snapshots are informative observations, not allocation grants.
- CrossProcessFileLock checks real regular paths and has bounded acquisition, but additional
  shared-root UID/permissions/ancestor checks are necessary for new cross-project coordination.

<a id="history-resource-robustness-audit-source-and-provider-evidence"></a>

### Source and provider evidence

Official NVIDIA binding distribution is `nvidia-ml-py`, importing the module `pynvml`; the
deprecated third-party distribution named `pynvml` is not the dependency to add.
[NVIDIA package](https://pypi.org/project/nvidia-ml-py/),
[CUDA visibility/order contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
CUDA ordinals can be reordered. Do not equate them with NVML index; resolve actual visible UUIDs,
including MIG, or fail closed. Reading host GPU metadata never grants the right to use it.

The requested Spanish wiki URL initially returned an old cached version without gpuctl;
the current English official page documents exec/manual reservations, environment preparation,
MIG UUIDs and preemption. [CiTIUS official manual](https://wiki.citius.gal/en:centro:servizos:servidores_de_computacion_gpgpu).
Reuse of an active reservation and its manual release are distinct. Its exact production
ownership/cleanup behavior with simultaneous wrappers still requires authorized real-site proof.
Never run `gpu release` merely because a configuration contains a release argv.

<a id="history-resource-robustness-audit-implementationacceptance-tracking"></a>

### Implementation/acceptance tracking

The attached 32-section request is in progress. Implement measurement/timeouts/cache first,
then stable shared admission/commitments, then RAM/CPU/backpressure/diagnostics and synthetic gates.
Update this ledger with actual checks. No production gpuctl validation or solved hardware cause
is claimed by this audit.

<a id="history-resource-robustness-audit-implemented-and-locally-verified-blocks"></a>

#### Implemented and locally verified blocks

- Optional official NVML observer, one shared UID/host/boot cache, distinct admission/informative
  freshness, single-flight refresh, isolated helper deadlines, backoff and no unowned respawns.
  Runner, supervisor, direct resource view and process view reuse it. CPU-only heartbeats skip it.
- Numeric CUDA grants are resolved once through bounded driver identity (no compute context),
  not guessed from NVML indices. UUID/MIG and shell visibility parsing preserve original grants.
  After baseline, qualified per-device failure pauses only that device's new admission.
- `command` joins `shared` startup coordination by UUID. Before-spawn persisted commitments
  reconcile observed bytes without double counting and protect peer demand from baseline or
  exploratory bypass. Missing worker metadata is not death; native Job absence can reconcile
  an orphan. Compatible time-to-envelope extends a finite startup window.
- Per-device age/priority/served-launch fairness yields only to a feasible waiting Study. It
  preserves light backfill and independent GPUs; no new scientific queue/design authority.
- Physical/cgroup RAM, PSS/USS/RSS, CPU quotas/cpuset and native thread environment/affinity
  boundaries. Job admission subtracts only unmaterialized qualified demand. Controller overhead
  and aggregate Study RAM are checked before dispatch. Unknown demand remains explicitly unknown.
- Expiring NVML/device-identity circuits; native candidate/placement OOM dominance remains
  separate. Device-read failure is not a claim of ECC/Xid or permission for scientific retry.
- Cluster-operational `resource_safety` profiles, CLI/Console/agent saved-evidence report and
  compressed bounded operational usage windows with cumulative sampled peaks. No training
  metrics/logs/checkpoints are compressed/deleted by this feature.
- Unsafe generic claim/release pairs are readable but refused before external mutation; cleanup
  cannot execute unscoped `gpu release`. Provider-owned wrappers/manual grants remain separate.

Verification uses disposable local metadata, independent CPU processes, fake NVML/visibility
providers and real local helper timeout/reaping. Synthetic packing/throughput/replay gates and a
shared-cache microbenchmark retain existing resource intelligence; no production speedup claim.
Exact test totals are recorded in the implementation ledger after final gates. Version stays
0.19.0. The official MCP SDK is absent in the available environments; protocol acceptance is
explicitly skipped, not simulated into a claim of real integration.

<a id="history-resource-robustness-audit-still-required-do-not-claim-whole-request-completion"></a>

#### Still required; do not claim whole-request completion

1. Online RAM prediction/temporary host-global startup demand for undeclared budgets and richer
   process/PSI backpressure. Reading PSI and enforcing known capacity alone do not complete this.
2. Complete phase-aware peer reconciliation after the finite startup window, stronger long-lived
   coordinator-crash/pending-marker recovery and a full historical interval report with confirmed
   provider driver events. Latest cached metadata and sampled lifetime peaks are not a timeline.
3. Native reconciliation of ambiguous legacy leases without host/boot ownership evidence.
   New outer CPU/GPU leases are host-scoped and GPU/MIG UUID keyed, with retained-identity
   release and synthetic shared-home/order/MIG tests. Old unscoped records remain protected,
   not reinterpreted against local PIDs; the queue explains that ownership ambiguity.
4. More Fleet/SLURM/older-runtime observation acceptance, authorized real NVML/MIG/gpuctl
   preemption/reservation-lifetime validation and installed official MCP protocol acceptance.

Next safe block: host-global RAM startup/backpressure with bounded native operational evidence;
then explicit legacy lease ownership reconciliation. Never mutate WISDOM, active studies or site reservations
to complete these acceptance gates.

<a id="history-resource-robustness-audit-local-verification-checkpoint--2026-10-13"></a>

#### Local verification checkpoint — 2026-10-13

- Resource/provider/ARI/recovery/Fleet/agent-read gate: **293 passed**. Final focused native
  runtime/resource/provider/ARI gate after outer UUID lease changes: **211 passed**. These
  overlap; they are not 504 distinct tests or a whole-suite run.
- Native adaptive Work regressions: **58 passed**. Final outer-admission/policy/provider gate:
  **59 passed**. Console workspace/editor/resource-report gate: **9 passed**, 45 deselected.
- Ruff over source/tests/benchmarks and `git diff --check` passed; mypy checks **641 source
  files**. Wheel 0.19.0 installed into a disposable environment with no dependency downloads
  and smoke-tested outside checkout, including optional-observer imports, cached CPU fixture,
  native report facade and ordinary installed inventory.
- Official MCP protocol tests are **skipped** because the optional SDK is absent; the typed
  READ facade tests are not a substitute for protocol acceptance. No site probe, reservation,
  hardware fault, WISDOM change or scientific cluster workload was performed for this reform.

Synthetic CPU benchmark: the six-action fixture completes in 109 simulated seconds with a
one-Run policy versus 58 with the existing adaptive planner, with 40.24% versus 75.63% memory-time
utilization and no simulated OOMs. This is an exact-prediction fixture, not evidence of H200
throughput or a realistic zero-OOM guarantee. The shared-cache microbenchmark issued 100 reads,
one fake backend query, zero helpers and about 0.053 seconds local total coordination time.
Its uncoordinated reference assumes one query per request; real NVML/driver latency was not measured.

<a id="history-operational-reliability-progress"></a>

## Operational reliability implementation ledger

Baseline: `56c48957b448a6f631e606188687e4b57b6bf4d3`, LambdaForge 0.18.0.
This is an acceptance ledger, not a declaration that unfinished capabilities exist.
The supplied specification has 34 sections. Keep each requirement mapped to native authorities;
never add a second JobStore, runner or scientific coordinator.

| Sections | Native authority / demonstrated gap | Implementation and acceptance evidence | Status |
|---|---|---|---|
| 0 | JobService, JobStore, JobObservation, OverviewService, ResearchWork | Audit real HEAD and contracts; retain this ledger | Audited |
| 1–2 | get/list/reconcile swallowed provider errors or refreshed stored state files | OperationalHealth and locked ordered JobStore observations; failed/empty inventories; provider fault tests | Implemented core |
| 3 | Work/Study/Overview presentation used state alone | Shared health/freshness cells/details and OperationalWorkspace incident table; imported read-only distinction | Implemented core |
| 4 | Supervisor heartbeat and scientific progress are independent signals | Configurable grace/freshness/progress policy; hysteresis, clock-skew and no metric-free death inference tests | Implemented core |
| 5 | Detached per-Job supervisors existed, no independent catalog observer | MonitorService start/stop/status/config/diagnose; real local detach/restart/duplicate tests; Linux installation preview | Partial: OS logout/reboot and non-Linux service validation pending |
| 6 | CredentialService and diagnostics redaction exist | Durable deduplicated incidents, TLS SMTP with references, bounded retries/resolution events; Console settings/test | Partial: real SMTP/TLS server and full duration/escalation matrix pending |
| 7 | SLURM discarded errors/reasons | Structured queue/accounting evidence; NODE_FAIL/OOM/TIMEOUT/PREEMPTED/delay/query-error fixtures | Implemented core; real site pending |
| 8 | ProcessScheduler read state.json as liveness | Read-only host lifetime/marker observation; legacy immutable runtime bridge uses same implementation; live-child ambiguity refuses recovery | Implemented core |
| 9 | diagnostics/failure is the classification authority | Shared native operational diagnostics for scheduler/source facts, generic CUDA versus OOM and unknown causes; incidents carry recommendations | Implemented core; real site/complete source matrix pending |
| 10–12 | Native recovery preview/apply and lock already exist | Native request IDs before apply/explicit inspection, durable rejected requests, derived admission/runtime/progress phases and exact locked revalidation; CLI dry-run remains read-only | Implemented single-host core; complete operational reconciliation matrix pending |
| 13–15 | Retry recorded original source path, no public alternate YAML | --config CLI/API/Console picker; semantic origin/seed checks; preview/apply pin and preparation guard; moved-YAML reuse 9/retry 1 test | Implemented single-host core |
| 16–17 | Managed checkpoints and recovery preserve Runs | Opt-in configure/should_save and terminal retention; exact JSON/managed-file hashes, owner/step/write time, paged host verification, metadata-only recovery table | Partial: live rotation, application restoration acknowledgement and cross-host transfer pending |
| 18 | Fleet coordinator v2 fencing exists; public recovery gated | Native coordinated recovery and verified compatible placement; never replay as single-host Work | Pending |
| 19–20 | Native Job events existed, no incident workbench | OperationalWorkspace, Resume Study picker and paged offline Recovery history; persistent preview/config/apply failure dialogs with exact request/phase/log/history/plan controls; no toast polling | Implemented single-host core; full checkpoint/compatibility workbench pending |
| 21 | Existing jobs/overview/retry parser and domain services | Implemented health/monitor/incidents, jobs recovery audit and retry --config; shared CLI/Console domain services and JSON | Implemented core |
| 22 | Work Results 0.18.0 already implemented | Retained Work Results/import/export/product suites; internal checkpoint metadata excluded by the shared payload authority | Verified retained local core; complete compatibility/fault matrix pending |
| 23 | Last-good UI snapshots and bounded workers already exist | Nonblocking cross-process project/provider probe gates, separate SLURM lanes, receipt ordering and abrupt observer-death tests; cached clocks unchanged | Implemented probe core; complete UI race/platform matrix pending |
| 24 | runner owns scientific runtime | Incremental control-plane observability extraction, no fingerprint/runner changes for display | In progress |
| 25 | Job/Attempt/lease identity, locks, safety already exist | Corruption/late response/unknown-owner tests; never duplicate ambiguous computation | Pending verification |
| 26–27 | Existing real local process and fixed/adaptive retry tests | Two native CPU Studies through a faulty simulated provider and Textual YAML picker; confirmed loss, reuse 9/retry 1 each, durable incidents/audits and Console reopen | Partial: real provider preparation and full A–BW/Fleet chaos matrix pending |
| 28 | CI lint/types/tests/wheel exists | Explicit operational/fixed/adaptive reliability gate; packaged-guide and installed-wheel import/parser smoke | Partial: full chaos/platform acceptance pending |
| 29 | Manuals, READMEs, AGENTS and changelogs bilingual | Operational reliability guide, emergency steps, monitor example and public references; packaged EN/ES resources | Implemented for delivered core |
| 30 | Existing schemas and immutable evidence | Additive metadata, legacy reads, no imported active Jobs; retained suites | Pending verification |
| 31–33 | Acceptance and final report | Report actual tests/platform boundaries; do not claim complete before end-to-end acceptance | Pending |

<a id="history-operational-reliability-progress-demonstrated-defects-not-hypothetical-claims-about-the-real-incident"></a>

### Demonstrated defects (not hypothetical claims about the real incident)

- `JobService.list` and `OverviewService._jobs` returned prior records after exceptions without
  recording that a provider observation failed.
- `JobService.reconcile` turned an exception into an empty inventory and refreshed state-file facts
  without checking the recorded process lifetime.
- `ProcessScheduler.state` read a durable state file, not a current process check.
- SLURM queue/accounting failures collapsed to `UNKNOWN` without preserving the query failure.
- Native retry had no alternate configuration parameter and accepted a source path before checking
  whether it still exists. Recovery does correctly protect scientific identity and original Runs.
- A lost-confirmed UNKNOWN origin counted as active and blocked its own retry. The exact origin is
  now excluded only on the native eligible recovery path, with provider revalidation under lock;
  unrelated active Jobs still block duplicate execution.
- Checkpoint registration introduced internal files that Work Results wrongly listed as outputs.
  Runtime and result views now share the payload filter, including the standalone host reader.
- SMTP originally held the incident lock through network delivery. Delivery now reserves an attempt
  then releases the lock; an old acknowledgement cannot commit a superseding transition.
- Recovery rejected before enqueue lacked a request ID/history. Apply and explicit Console
  inspection now persist an operational request before validation; native dry-run is still read-only.
- Loading nine reused Runs could be mistaken for new recovery progress and resolve an incident.
  A baseline and subsequent positive runtime advancement now distinguish new computation from reuse,
  obligation totals, counter resets and observed process loss.
- An `squeue` transport timeout skipped accounting, and cached query details could survive failure.
  A bounded accounting fallback now preserves actual evidence and clears stale query details.

No productive cluster or WISDOM evidence has been inspected or modified during this audit.

<a id="history-operational-reliability-progress-verification-and-remaining-acceptance"></a>

### Verification and remaining acceptance

<a id="history-operational-reliability-progress-specification-fault-matrix-abw"></a>

#### Specification fault matrix (A–BW)

This maps requirements to evidence, not to a blanket HPC acceptance claim. **Local** means native
local implementation with the test's stated faults; **partial** means only the listed boundary is
covered. Transport fixtures do not certify real SSH, SMTP, SLURM, logout or reboot behavior.
All file references below are beneath `tests/`.

| Cases | Local evidence / remaining gap | Boundary |
|---|---|---|
| A, H, X | `controlplane/test_operational_health.py`: failed probe retains confirmation/science then recovers; `test_operational_providers.py`: failed inventory is not empty success | Local provider faults |
| B, C, E | `test_operational_providers.py`: timeout inventory and queue/accounting transport failures | Local faults; actual SSH host outage pending |
| D | `test_operational_health.py`: typed authentication failure separate from science and unknown query errors | Local typed failure; real SSH auth pending |
| F | `test_operational_providers.py`: successful empty inventory becomes unknown, not process-loss proof | Local |
| G | `test_operational_diagnostics.py`: malformed response is not unreachable; corrupt direct records fail closed | Partial: full malformed inventory/source matrix pending |
| I | `test_operational_health.py`: late observation cannot erase a newer fact or cancellation | Local locked observations |
| J, K, M | `test_operational_health.py`: real short-lived process terminates, retained state stays running but ownership reports lost; ambiguous live-child markers prevent loss inference | Local process + marker fixture; supervisor/child outage combinations partial |
| L, AF, AG | `test_operational_health.py`: heartbeat freshness distinct from optional progress expectations | Local clocks/policy; no inference of death from metric silence |
| N | `controlplane/test_process_scheduler.py`: reused/mismatched PID identity rejected | Local identity checks |
| O, P | `test_operational_providers.py`: live queue and positive terminal accounting, same-read details | Local SLURM fixtures |
| Q, W | `test_operational_providers.py`: delayed/empty accounting unknown; query failure remains error and clears old details | Local SLURM fixtures |
| R, S, T, U, V | `test_operational_providers.py` and `test_operational_diagnostics.py`: exact scheduler reasons, OOM not inferred CUDA OOM, preemption distinct from user cancellation | Local SLURM fixtures; real site pending |
| Y, Z | `test_operational_monitor.py`: actual disposable detached monitor, duplicate start, stop and restart | Local process; OS logout/reboot acceptance pending |
| AA, AB, AC, AD, AE | `test_operational_monitor.py`: durable dedup/escalation/resolution, bounded delivery, rejection/acceptance and superseding transition | Local SMTP transport doubles; real TLS/mail server pending |
| AH | `controlplane/test_study_retry.py`, `work/test_study_recovery.py`, `work/test_fixed_study_recovery.py` | Local native fixed/adaptive recovery |
| AI | `RecoveryObservation.eligible` and shared terminal-state recovery path | Partial: explicit native TIMEOUT preview/apply integration pending |
| AJ, AK | `test_operational_providers.py`: live/unresolved ownership rejection; `tui/test_operational_recovery_e2e.py`: confirmed loss + exact retry | Local; provider admission simulated in E2E |
| AL | `work/test_fixed_study_recovery.py`, `work/test_study_recovery.py`: existing controller lock prevents duplicate execution | Partial: full concurrent public request/admission race pending |
| AM | `test_operational_providers.py`: apply-time ownership recheck; `tui/test_recovery_history_console.py`: rejected apply remains visible | Partial: complete mid-operation connectivity matrix pending |
| AN | Request IDs/rejected enqueue auditing use native error boundary | Pending ambiguous transport/admission acknowledgement acceptance |
| AO | `controlplane/test_submission_tls_monitor.py`: worker persists pre-scheduler failure | Local worker boundary |
| AP | `test_recovery_observation.py`: accepted/preparing/scheduler phases are not progress | Partial: sustained never-admitted preparation integration pending |
| AQ | `test_recovery_incidents.py`, `tui/test_operational_recovery_e2e.py`: baseline reuse is not new progress; exact audited progress resolves origin | Native CPU execution; provider/admission simulated |
| AR, AS | `test_recovery_audit.py`, `tui/test_recovery_history_console.py`: durable rejected requests, reopened offline history, persistent preview/config/apply errors | Local native audit + Textual |
| AT, AU, BA | `test_relocated_recovery_config.py`, `tui/test_operational_recovery_e2e.py`: original snapshot, moved YAML/comments, source missing, original seed namespace | Local |
| AV | Existing input semantic identity checks and worker binding | Partial: dedicated relocated large-input equivalent-path integration pending |
| AW | `test_study_retry.py`, `work/test_fixed_study_recovery.py`, `work/test_study_recovery.py`: explicit code acknowledgement preserves revisions, never overrides science | Local |
| AX, AY, AZ | Native fixed/adaptive changed-configuration/identity guards and relocated seed-policy tests | Partial: individual Dataset/architecture source matrix pending |
| BB | `test_relocated_recovery_config.py`: corrupt original identity; native owned JSON validation | Local |
| BC, BD, BE | `work/test_checkpoint_evidence.py`, `work/test_fixed_study_recovery.py`: host hashes, corrupt JSON refusal, missing checkpoint restarts only failed Run | Local; no application restoration certificate |
| BF | `test_study_retry.py`: foreign execution preview refusal; `work/test_checkpoint_evidence.py`: foreign Execution/Run owner rejects load/replace/reference/inventory without byte mutation | Local managed identity; cross-host/application compatibility pending |
| BG, BH, BI, BJ | Fixed/adaptive suites: nine successes reused, partial/pending exact cells, missing adaptive state rejection and spent budgets preserved | Local native recovery, including spawned CPU workers |
| BK, BL, BM, BN, BO, BP | Fleet retry still refuses independent single-host replay; coordinator fencing is not distributed recovery/transfer acceptance | Pending native Fleet recovery/compatible materialization |
| BQ | `tui/test_recovery_history_console.py`: slow worker + close/unmount is responsive; stale callbacks do not mutate removed screen | Partial: abrupt Console process-death matrix pending |
| BR | `test_observation_gate.py`: concurrent service probes, receipt ordering, distinct SLURM lanes and abrupt observer death | Native locks/processes + provider doubles |
| BS, BT, BU, BV | Retained full suites plus `tui/test_work_evidence_console.py`, `work/test_work_results.py`, `work/test_study_import.py`, products suites and outside-checkout wheel smoke | Full-suite outcome recorded separately; no imported live-host inference |
| BW | Legacy record/checkpoint reads and retained compatibility suites | Partial: complete historical-version snapshot matrix pending |

Executed during this implementation:

- Native fixed/adaptive recovery suites: 60 passed before later operational presentation changes.
- Combined provider/monitor/direct-process/recovery/relocated-YAML/Console regression batch:
  58 passed; targeted health/provider/request/Console batch: 22 passed.
- Relocated YAML preview/apply plus native recovery: 5 passed, including reuse of nine successes,
  new Attempt only for seed 54 and immutable original Execution/Attempt results.
- Broad control-plane/products/recovery/Console batch: 585 passed and one legacy host-reader failure,
  subsequently fixed and covered by 47 passing targeted tests.
- Current operational/fixed/adaptive/checkpoint/Console batch: 159 passed. Further source/preemption,
  incident-resolution and unsafe-path regressions were added afterward and run separately.
- Complete pytest invocation: 1,816 passed, one Work Results checkpoint-listing regression failed
  (1,218.35 seconds; four Lightning CPU warnings). The regression was fixed; the complete Work
  Results/Console evidence/checkpoint batch then passed 38 tests. This is **not** a claim of a second
  all-green full-suite invocation on the final revision.
- Provider/native recovery/probe/Work Results/checkpoint batch: 62 passed.
- Recovery audit/native retry/configuration/Console batch: 33 passed. A broader health/incident/
  audit/Console batch passed 40 tests but exposed an unmount callback warning; message-based
  delivery then passed the three history tests with thread-exception warnings treated as errors.
- SLURM fallback and native recovery/audit regressions: 38 passed. Subsequent positive-progress/
  loss, incident, audit and history regressions: 32 passed; per-test-clock incident batch: 12 passed.
  The final UI batch/full-suite results are recorded after completion, not inferred from these.
- Latest audit/monitor/Console/Work Results/checkpoint batch: 80 passed; checkpoint/native fixed/
  adaptive recovery/retention batch: 87 passed. Shared diagnosis/provider/incident batch: 51 passed.
- Positive-runtime baseline/native preview/audit/Console/structure batch: 43 passed. The full suite
  exposed the pytest module-alias clock fixture defect: patching a namespace import did not update
  the collected module's clock. The fixture now patches its own globals, and an injected six-hour
  collection delay is verified independently, without sleeping or weakening freshness policy.
- After the fixture alias correction, the combined incident/health/audit/Console batch passed
  36 tests and the six-hour-delay injection passed all 16 incident tests.
- Complete follow-up pytest invocation: 1,850 passed, three failures (two collected-clock failures
  and missing Message class docstrings), four Lightning CPU warnings, 1,212.09 seconds. Those defects
  are corrected. A subsequent invocation was interrupted after 1,221 passes when the remaining
  clock-alias defect was reproduced; neither run is presented as final all-green acceptance.
- **Final delivered-core revision:** complete pytest passed **1,866 tests**, four existing Lightning
  CPU smoke warnings, **1,125.33 seconds**. The JUnit report was generated in the temporary
  validation directory. This invocation includes the corrected clock fixture, managed checkpoint
  owner/registration guards, native recovery baseline, saved audit and persistent Console errors.
  A passing local suite is not acceptance of the unimplemented Fleet or real-site boundaries.
- Two-Study Textual recovery plus incident/core tests: seven passed. The provider/submit
  acknowledgement is simulated; actual CPU WorkRunner execution, YAML validation, locks, Attempts,
  original success bytes, incident resolution and Console reopen are native local implementations.
- Ruff and mypy passed (603 source files). No remote CI run was triggered.
- Source/wheel distributions built. Installed-wheel smoke passed outside the checkout with the
  existing dependency environment explicitly shared; package code/metadata were verified to come
  from the temporary wheel installation. CI retains a fresh dependency installation gate.
  The final wheel is rebuilt after recording this outcome; no version/release publication occurred.

CUDA is detected on the local machine but CUDA fault/site tests have not been executed in this
phase. SMTP delivery tests inject transport acceptance/failure; no real mail was sent. systemd
tests preview the unit; no service was installed on this workstation. Existing monitor lifecycle
tests launch/stop only disposable project observers.

Do not mark the complete supplied specification accepted. Remaining work includes native Fleet
fencing/recovery and compatible transfer, live checkpoint rotation/application restoration receipts,
the complete checkpoint/compatibility workbench, real OS service/mail/site validation and
the full A–BW fault matrix. The local two-Study recovery fixture does not certify real SSH, scheduler
admission or a complete native detached background preparation cycle.
Existing single-host recovery checks, products and Work Results remain the native authority.

<a id="history-operational-reliability-progress-inventory-follow-up--2026-10-11"></a>

### Inventory follow-up — 2026-10-11

Saved WISDOM controller observations (read locally, without contacting execution hosts) exposed
two distinct causes of host-wide UNKNOWN/degraded displays: a corrupt historical direct Job
aborted the inventory, and a long-lived observer shipped the wrong helper definitions after
editable source line offsets changed. Neither observation establishes scientific failure.

Native direct inventory now returns bounded, validated per-Job error rows without adopting them,
mutating host evidence or authorizing recovery. Healthy siblings receive fresh facts and clear
prior query errors. Terminal outcomes remain preserved; root/transport failures still fail closed.
Observer construction selects named definitions and their helper closure from current source
snapshots rather than stale loaded function offsets. The Study overview helper follows the same
rule. Console/monitor sessions open before this correction must be restarted to load it, not Jobs.

Validation: **480 tests passed** in **414.79 seconds**, covering all control-plane tests, operational
Console/recovery E2E and repository structure; the new inventory file contains 12 regressions.
An earlier broad run reproduced three missing-helper failures after source edits (475 other tests
passed), prompting the source-snapshot fix; it is not reported as green. Ruff check and mypy
(603 source files) passed. A newly built wheel's observer/CLI smoke passed outside the checkout,
with existing dependencies explicitly shared. No real cluster, WISDOM code, dataset, scientific
record or training process was modified. Whole-repository formatting still reports findings
outside this focused fix; changed implementation/test files are formatted separately.
