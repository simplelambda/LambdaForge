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

1. Extend the same approval authority to native SUBMIT/OPERATE/DESTRUCTIVE actions, binding actual
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
