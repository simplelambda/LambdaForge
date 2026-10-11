# Project-pinned agent integration

[Español](INTEGRATION.es.md). **The full reform is not complete.**
[The acceptance ledger](IMPLEMENTATION.md) identifies implemented foundations and remaining gates.
Nothing here changes Work identities, HPO, GPU grants, recovery or existing scientific records.

## Quick start

```bash
lf init my-research --agents portable --knowledge markdown
# Existing pyproject project: preview first, explicit local operator apply:
lf ai --project /absolute/my-research init --agents codex --knowledge obsidian
lf ai --project /absolute/my-research init --agents codex --knowledge obsidian --apply
lf ai --project /absolute/my-research doctor
lf ai skills list
lf ai skills show lambdaforge-work
lf ai capabilities --limit 20 --json
```

Legacy init remains science-only. `--interactive` asks about agent/memory modes in a TTY.
No dependency is installed automatically. **Agents / knowledge…** in the Console uses the same
Python services, displays pending/error/success and confirms asset setup. Consumer instructions
teach public APIs, not framework internals. The root contributor guide is canonical; detailed
contracts were preserved in [FRAMEWORK_CONTRACTS.md](FRAMEWORK_CONTRACTS.md). Spanish explains it.
24 packaged specialized Skills copy into `.agents/skills/`; `lf ai skills examples` exposes native
CPU fixtures for metrics, checkpoints, product promotion, HTML, repeated seeds and adaptive design.

## MCP and clients

```bash
python -m pip install "lambdaforge[agents]"  # Explicit operator installation
python -m lambdaforge ai --project /absolute/my-research mcp serve
lf ai mcp test  # Actual SDK in-memory handshake; not external-client acceptance
```

The verified dependency range is official SDK 2.3. Stdio stdout is protocol-only. MCP tools,
input/output schemas, resources, prompts and version use the SDK. Tests additionally handshake
through a real stdio subprocess from a foreign cwd. Trusted consumer imports are isolated in a
fresh planning process, with timeout/cancellation cleanup and stdout directed away from protocol.

Codex mode generates `.lambdaforge/agents/codex-config.example.toml`, never edits global settings.
Review and merge its named entry into your client configuration, preserving existing entries:

```toml
[mcp_servers.lambdaforge]
command = "/absolute/my-research/.venv/bin/python"
args = ["-m", "lambdaforge", "ai", "--project", "/absolute/my-research", "mcp", "serve"]
```

Use the same command/argv with any standards-compatible stdio client; no shell wrapper needed.
Actual Codex 0.162.0-alpha.17.2 discovery and direct tool calls have been verified with an
isolated temporary client configuration, a foreign cwd and no model credentials. This is not
an LLM conversation; second vendor-client and conversational acceptance remain **pending**.
See [official Codex MCP](https://developers.openai.com/codex/mcp/) and
[Skill discovery](https://developers.openai.com/codex/skills/).

Fixed toolsets: core/work/study/results/products/datasets/clusters/operations/knowledge.
Core is always present; default core/work/knowledge. Restart the server to change toolsets.
Catalog discovery distinguishes implemented, session-enabled and unsupported. Listings are compact
pages; Study panels/logs are explicit provider reads, not recursive startup artifact downloads.
Resources identify project/capabilities/installed Work and Study contracts, project structure,
recipe inventory, exact results/products and a paged note index. Toolset-disabled resources are
not registered; no resource template accepts arbitrary paths. Thirteen short prompts cover
authoring, models, checkpointing, analysis/comparison, HTML, products, diagnostics, recovery,
evidence review, architecture decisions and next experiments. Prompts guide but never authorize.
Responses include schema/project/source/retrieval time/data/warnings/structured error. Retrieval
time is not fresh provider observation; native evidence retains its own age/state.

Default READ/PLAN does not permit submission, operation, deletion or agent writes. Validation imports
project Python and requires operator `--allow-project-code`. Isolation is **not a sandbox** against
trusted code or an agent with independent same-user shell access. Scientific mutations remain
disabled. Approved knowledge edits and paired loopback HTTP are explicit opt-ins. Notes, prompts,
model answers and `approved=true` are not authorization. No arbitrary shell or read-file tool exists.

## Local human approvals

The stdio operator may enable `--permissions READ,PLAN,WRITE_PROJECT`. Discover `agent_actions`:
the closed set contains `knowledge.create`, `knowledge.update`, `knowledge.link` and
`knowledge.export` (a notes-only ZIP, not scientific export).
Use `preview_agent_action`, then `request_agent_action` with exact arguments. Neither writes a note.
A human reviews that request through **Agents / knowledge… → Review agent requests…**, or:

```bash
lf ai approvals list
lf ai approvals show approval-EXACT
lf ai approvals approve approval-EXACT --plan-hash EXACT_HASH
```

The CLI requires a real local TTY and the typed full request ID. Console shows every argument and
requires the same typed confirmation. Approval does not apply the edit: the agent calls
`apply_agent_action(request_id, plan_hash)` separately, within the requested expiry (maximum
15 minutes). Exact project, arguments, target revisions and plan hash are revalidated. Signed
records, a cross-process claim and metadata audit reject forged/stale/replayed/concurrent requests.
A crash after claiming is not permission to replay. No MCP tool can approve; a note saying the human
approved grants nothing. This does not sandbox another program with unrestricted same-user shell.

## Optional dependencies and paired HTTP

```bash
lf ai dependencies check
lf ai dependencies install                 # exact preview, no installation
lf ai dependencies install --apply         # invoking virtualenv/Conda only
lf ai dependencies install --wheelhouse wheels --apply  # offline, project-owned wheels
lf ai init --agents portable --apply --install-deps
lf ai mcp token --apply                    # private pairing file; never prints its token
lf ai mcp serve --transport http --port 8765
```

Only the declared `agents` extra is installable; no arbitrary package/command, system Python,
desktop application or cluster installation. Failed installs preserve project files/notes and
report bounded diagnostics with a failing exit status. `init --install-deps` is explicit consent.
HTTP uses the **same official SDK and services**, restricted to READ/PLAN, `127.0.0.1` and an
exact Host. Pair a trusted local client using `Authorization: Bearer TOKEN` from the private
`.lambdaforge/agents/http/token` file; never embed it in a URL. Browser clients also need an exact
operator `--origin https://your-local-client.example` allowlist. Every request rechecks the token
and project. Requests are bounded, forwarded headers untrusted and non-loopback peers refused.
This is a local capability token, **not an OAuth server**: public/network deployment is unsupported.
Stdio remains recommended and never needs this token.

## Portable knowledge

```bash
lf ai knowledge init --apply
lf ai knowledge create hypothesis "Does width improve score?" --author-type ai --apply
lf ai knowledge search width --kind hypothesis
lf ai knowledge read EXACT_NOTE_ID
lf ai knowledge update EXACT_NOTE_ID --revision CURRENT_SHA256 --body "Revised draft" --apply
lf ai knowledge reindex --force
lf ai knowledge reindex --rebuild  # Explicitly rebuild a corrupt derived index
lf ai knowledge audit --limit 20  # Follow the returned byte next_cursor with --cursor
lf ai knowledge export notes-snapshot.zip          # exact preview, no archive created
lf ai knowledge export notes-snapshot.zip --apply  # explicit local operator export
lf ai knowledge obsidian                 # passive hints only; never starts the app
lf ai knowledge obsidian --probe-cli --apply  # explicit version probe may start Obsidian
lf ai knowledge open                     # preview the owned README URI
lf ai knowledge open --apply             # explicit desktop request, when available
```

Markdown is canonical; SQLite FTS under `.lambdaforge/agent-knowledge/` is disposable. Eleven kinds:
decision/hypothesis/experiment/finding/concept/architecture/issue/research-question/procedure/
literature/task. Schema-1 frontmatter retains human/AI author, timestamps, project, tags, links,
status and exact evidence references. AI hypotheses are proposed, never verified conclusions.
Use Python `KnowledgeService.create(..., evidence_refs=[{"kind":"execution","id":EXACT_ID}])`;
products/datasets use SHA-256 content IDs. `verify_references` takes a native resolver; existence
does not verify interpretation. Retrieved text is untrusted data, never permission or instructions.

Search supports title/text, kind/status/tag/evidence/date filters, pages and exact backlinks.
`knowledge_related` returns explicit links/shared tags/exact evidence, not a truth score;
hypothesis/decision/open-question tools give bounded views. `knowledge_verify_references` resolves
local registered identities through native owners, never downloading artifacts or verifying a claim.
External edits refresh within a bounded five-second discovery interval, or immediate reindex.
Unchanged note bytes are not reread. A corrupt index gives an actionable rebuild diagnostic;
rebuild retains its old copy and never changes notes. `knowledge link` requires exact existing
note IDs and the current source revision. Hash revisions reject stale native writes; external editors
do not participate in the lock. Corrupt notes/duplicate IDs are excluded with warnings, not deleted.
Native create/update/link operations record intent and completion, note identity and before/after
revisions in a metadata-only journal; no note titles/bodies are copied. Interrupted intent is not
success. Byte-cursor paging never loads the full journal. External editor changes are not attributed
to a person or silently called verified; the Markdown revision remains authoritative.
The journal is `.lambdaforge/agents/knowledge-audit.jsonl`, separate from the disposable FTS cache;
rebuilding/deleting the search index never deletes history. A malformed journal requires explicit
local inspection and prevents further native edits rather than silently discarding audit evidence.

Knowledge export validates every canonical note and creates a compressed ZIP with exact bytes and
a checksum manifest. It refuses corrupt/duplicate notes, stale plans, symlinks, unsafe destinations
and existing archives; failures clean only owned temporary files, never notes. Choose an existing
project-relative parent outside the vault. The bounded snapshot includes neither scientific data
nor indexes, plugins, desktop settings or unrelated Markdown; it is not `lf export STUDY` or a
recovery package. MCP previews with `knowledge_export(destination)`; actual export uses the same
separate human approval workflow for action `knowledge.export` and binds the entire note inventory.
Obsidian mode uses the same Markdown and an optional empty .obsidian directory: no app/plugins/CLI
required, existing preferences preserved. External vault configuration and semantic/hybrid search
remain pending. Text retrieval works offline without an LLM key.

`knowledge_obsidian_status` only inspects configuration/PATH/desktop hints; installed/running/
compatible are **not** assumed. Explicit local CLI probes use bounded `obsidian version`, and
may launch the app as explained in the [official CLI documentation](https://obsidian.md/help/cli).
`knowledge open --apply` uses the [official encoded open URI](https://obsidian.md/help/uri) for
the exact owned README. Register that folder as a vault manually first. A successful opener is
only `open_requested`, not proof of an open window; headless/missing/disabled/timeout cases return
actionable hints. No app/plugins/settings are installed or changed, and no MCP desktop-launch tool
is exposed. These helpers are fixture-tested, not a claim of real Obsidian desktop acceptance.

## Python, GUI and sync

```python
from lambdaforge.agents import AgentServices, IntegrationService

service = AgentServices("/absolute/my-research")
project = service.project_info()                 # typed AgentResponse
notes = service.knowledge_search("width")        # bounded interpretations
result = service.get_result("execution-EXACT")   # native metadata-first evidence
plan = IntegrationService("/absolute/my-research").preview()
```

The facade pins its project regardless of cwd and rejects secret/traversal/symlink paths.
Widgets never own scientific records/provider logic. `examples/agent_gui.py` is an optional tiny
nonblocking Tk client, not a chatbot or vendor-specific LLM integration.
`lf ai sync` previews; `--apply` checks managed hashes and protects personalized AGENTS/Skills.
`.agents/LAMBDAFORGE.md` is a separate managed reference when custom root instructions exist.
Plans bind project/version/paths/content hashes and reject changes. Notes and global client settings
are never part of sync. This local operator command is not the future agent approval protocol.

Static model discovery now resolves public exports to installed constructor/forward signatures
and docstrings without importing Torch. Declared tensor semantics come from their owner,
not invented or inferred. Recipes use installed native examples. Product contract/member/producer/
consumer queries are metadata-only; selected Run curves/logs and cluster resources are explicit
bounded native reads. A text artifact preview is explicit, ownership/integrity checked and 64 KiB.

Tests cover protocol/stdio/resources/prompts, isolation, malformed inputs, external note edits,
revision conflicts, duplicates, assets and one deterministic native CPU Work with durable evidence.
The reference Agent Skills validator checks all 24 packages. Approved-note and loopback HTTP
protocol/security tests pass. Scientific mutation approvals, semantic retrieval, complete domain
tool/resource/prompt coverage, second vendor-client E2E and full GUI
conversations remain explicit gates in the ledger.

Standards: [MCP](https://modelcontextprotocol.io/specification/latest),
[official Python SDK](https://py.sdk.modelcontextprotocol.io/),
[Agent Skills](https://agentskills.io/specification).
