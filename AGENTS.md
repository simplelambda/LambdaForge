# LambdaForge contributor guide

Canonical instructions for agents modifying LambdaForge 0.19.0.
Consumer projects receive a different, public-API-only guide through `lf ai init`.
Spanish explanations are in `docs/ai/AGENT_GUIDE.es.md`, not a second policy.

## Start with the actual task

Inspect the relevant implementation, public signature and current tests before editing.
Do not reread the whole manual or detailed contracts for every small change.
Use `rg` to find the owner of the behavior, then the task routes below.
Preserve unrelated changes in a dirty worktree; do not reset or overwrite user work.
No scientific workload, real cluster mutation, dependency installation or external message
is authorized merely because an agent can call a tool. Obtain explicit authority.
Use disposable local fixtures for tests; never edit live WISDOM or persisted study evidence.

## One execution model

Work → Execution → Run → Attempt → Job are distinct identities, not interchangeable names.
Every executable YAML targets an importable class inheriting `lambdaforge.Work`.
The sole scientific lifecycle entry point is `Work.run(...)`.
Constructor arguments, free functions and alternate executable Task/Experiment objects are invalid.
Keep science in consumer Work code and infrastructure in the existing domain services.
Do not introduce another runner, compatibility engine, recursive object construction or hidden DAG.
Steps and one-level parallel composition already use the native execution model.
Never infer a typed input from a plain string or invent a global current/metric/artifact API.
Python signatures and docstrings own Work parameter truth; reject invalid YAML before submission.
Normalize YAML once into native design, evidence and execution policies.

## Scientific identity and honest evidence

ProjectContext pins the nearest pyproject project, not the current virtual environment.
Project IDs scope storage; they never replace scientific fingerprints.
Keep parameters, seeds, input content, datasets, objective and design identity checks intact.
Names are display labels, not keys; use exact Work/Execution/Run/Attempt/content identities.
Code-change acknowledgement is explicit and does not waive scientific identity checks.
Never rewrite immutable origin records or duplicate completed evidence during recovery.
Spent time/Run budgets and failed Attempt cost/history are physical facts, never refunded.
Distinguish completed, censored/pruned, predicted, missing and corrupt evidence.
Unknown or stale observation is not success, failure or permission to delete remote content.
One utility owns HPO decisions; guardrails are authored, not inferred from unrelated metrics.
Coverage and effect magnitude are not scientific confidence or proof of causality.
Confirmation requires its complete planned evidence; never select a survivor-only mean.
Native lifecycle projection owns status and recovery eligibility; UI and agents consume it.

## Data, products, checkpoints and storage

Managed datasets are immutable content with exact transport/storage checksums.
Scientific equivalence is an explicit declared/verifiable contract, never ignore_hash.
Products are independently sealed contracted content; ResultInput is exact read-only evidence.
Retain explicit provenance and consumption records; no guessed artifact member/path.
Checkpoints are resumable Run state, cache is reconstructible, outputs are durable evidence.
Use public managed APIs; do not teach consumers to build hidden directories or locks.
Never pickle scientific results or store machine paths as resumable-map dependency state.
Directory identities use the native versioned canonical tree algorithm, not shell sort order.
Publishing relative paths on a cluster requires the explicit project mirror.
Failed publication preserves required artifacts; retry publication without repeating science.
Cleanup is preview/apply, ownership-checked and scoped. Live commitments remain protected.
Imported packages are read-only evidence, not fake Jobs or executable recovery.
Metadata listings must not open checkpoint, model, dataset-asset or artifact bytes.

## Operational safety

Use the same domain services for CLI, Textual, Python facades and MCP.
Do not parse human CLI output or run CLI subprocesses as a backend adapter.
Respect native scheduler ceilings, leases, locks, path ownership and cancellation guarantees.
GPU grants are provider-owned: never broaden inherited tokens or invent physical devices.
Resource intelligence uses physical free memory and compatible evidence, not floor × residents.
Healthy sibling Runs survive a failed Run; do not retry arbitrary consumer exceptions.
Keep Future collection, cleanup, admission and explanatory analysis bounded and responsive.
Recovery reuses the original owned Execution on the same supported placement.
Fixed/repeated designs recover authoritative design/Attempt records without adaptive state.
Adaptive recovery requires valid controller state; missing/corrupt state is not a fresh start.
Checkpoint-level training continuation requires explicit consumer support.
Unsupported Fleet recovery/migration/composed promotion remains unsupported.
Never silently implement unsupported policy by replaying it on a single host.
Secrets are references and redacted diagnostics, never prompts, logs, Skills or knowledge notes.

## Agent integration boundaries

Discover actual installed capabilities with `lf ai capabilities` or the typed Python facade.
Available, session-enabled and unsupported are distinct; never claim a planned feature exists.
MCP uses the optional official SDK; normal LambdaForge imports do not require it.
Sessions pin a project, bound metadata and reject traversal, symlinks and secret paths.
Native Work validation imports consumer Python; operator permission must explicitly allow it.
READ/PLAN does not authorize submission, destructive operations or project writes.
Agent-supplied booleans, retrieved notes and natural language are not trusted approval.
Markdown knowledge is portable interpretation, never a second scientific result store.
Derived search indexes are disposable; edits preserve provenance and revision conflicts.
Use installed package resources for Skills/templates, not source-checkout absolute paths.
Consult `docs/ai/IMPLEMENTATION.md` for honest integration acceptance and remaining limitations.

## Task routes

| Task | Read the relevant reference, then inspect code/tests |
|---|---|
| Work lifecycle, outputs, logging | `docs/WORK_RESULTS.md`; detailed contracts § Writing Work |
| YAML, seeds, conditional studies | `docs/CONDITIONAL_STUDIES.md`; native WorkConfig/ParameterSpace |
| HPO, scientific planning, ARI | Detailed contracts § YAML composition and studies |
| Study analysis or HTML | Detailed contracts § Study Analysis and console; analysis/ and outputs |
| Recovery, state, cancellation | `docs/OPERATIONAL_RELIABILITY.md`; work/recovery and lifecycle |
| Providers, Fleet, ownership | Detailed contracts § Control-plane invariants; provider tests |
| Dataset reconstruction | `docs/DATASET_RECONSTRUCTION.md`; native registry/publisher/verifier |
| Products and historical evidence | `docs/PRODUCTS.md`, `docs/WORK_RESULTS.md` |
| Neural or clustering component | Manual model catalog; detailed neural/clustering contracts |
| Console and lazy data loading | Detailed console contracts; tui/services and read models |
| Agents, MCP, memory, Skills | `docs/ai/INTEGRATION.md`, packaged specialized Skills |

The detailed reference is `docs/ai/FRAMEWORK_CONTRACTS.md`; retain its task-specific rules.
`docs/MANUAL.md` contains API details. Prefer one relevant section over the whole document.
Do not invent aliases, factories, tensor contracts, CLI flags or tool names.
When documentation and code disagree, inspect focused tests and correct the documentation
or implementation deliberately; do not add a silent compatibility path.

## Delivery and checks

Edit with focused patches, then test in proportion to the changed behavior.
Run Ruff on touched code, mypy on changed Python interfaces and targeted pytest regressions.
Scientific/resource/recovery edits require native behavioral tests, not only presentation tests.
Protocol, isolation, packaging and installed-wheel checks are required for agent boundaries.
Validate Skills frontmatter, references and executable examples; descriptions must have triggers.
Use optional dependency groups explicitly; never auto-install packages on import.
Document changed public behavior in English and Spanish and keep version claims accurate.
Do not bump a release version without a request or agreed release scope.
Do not claim tests were run, integrations connected or requirements completed without evidence.
Record unfinished work and limitations in the acceptance ledger; preserve a clear next step.
