# Agent-native integration acceptance ledger

[Resumen técnico en español](IMPLEMENTATION.es.md).

This is the implementation ledger, not a claim that the complete integration has shipped.
Baseline: LambdaForge 0.18.0, HEAD `56c48957b448a6f631e606188687e4b57b6bf4d3`,
including the existing uncommitted operational-reliability changes. No live cluster, WISDOM
source, persisted scientific record, GPU grant or scientific fingerprint is modified by this work.

Release scope: LambdaForge 0.19.0 packages the implemented foundations and operational fixes.
The original 0.18.0 baseline and verification history below remain historical; this version bump
does not change the pending acceptance items or enable unsupported scientific mutations.

## Selected native library resolution — 2026-11-14

Managed Conda launch now binds `native-prefix-libraries-v1` to its selected dependency prefix,
including code venvs. A stdlib-only `-I -S` bootstrap execs the target with scoped loader selection
before imports/site hooks; no global export, dependency downgrade, import-order workaround or
prefix/binary edit. Absolute inherited driver paths remain after the selected directory; empty
or relative entries are omitted. GPU visibility/grants are copied unchanged. Direct supervisors
also start under the native envelope, rather than importing Torch before it. SLURM receives that
same scientific argv, and normal descendants inherit the environment. Named isolated runtimes
detach consumer-injected paths; active-native `tools.run(env=...)` cannot replace loader variables.

Readiness/reuse checks both import orders, small CPU Torch/KDTree operations and actual loaded
C++/GCC origins. Bootstrap, Job, new receipts and Run provenance expose policy/loaded paths.
Cached immutable prefixes/old evidence are never rewritten; compatible bases reuse a fresh
attestation while current exact execution wheels carry the new framework code.

Native regressions compile two local ELF runtimes with incompatible CXXABI versions and reproduce
the first-load failure, then verify selection in either order. Installed Torch/SciPy are exercised
in CPU-only subprocesses and a detached disposable supervisor/child; `/proc` confirms loaded paths.
No dependency downloads/installations or WISDOM source/data/remote Job mutations were performed.
Real ctgpgpu16 bootstrap/retry remains an operator action, not claimed cluster acceptance.
An initial broad run hit an unrelated GPU fixture admission timeout; final verification restricts
CUDA visibility to empty in the test invocation. Three manual supervisor fixtures now include the
storage metadata already required by the native request contract; no queue/backend behavior was
changed to mask their missing metadata. APBS exit-13 and stable-prefix PDB2PQR regressions remain.
Final CPU-scoped verification: 236 pytest cases passed, 2 CUDA-only cases skipped; Ruff and
focused mypy (12 files) passed. No claim is made about live GPU/ctgpgpu16 interoperability.

## Stable-prefix managed publication — 2026-11-14

Confirmed locally that renaming an installed prefix breaks both a pip-created absolute console
script and shell/Python launchers (exit 127). Managed dependency and execution layers now install
at their lifetime paths; only readiness is atomically published after final-path verification.
No binary/text prefix substitutions or consumer-specific launcher patches. The native build/GC
lease covers the entire transition, and micromamba receives an empty prefix. Readiness receipts
attest the publication contract/path; failures retain redacted, owned diagnostics and are not active.

Environment identity v4 and dependency contract v2 select distinct prefixes without mutating old
Jobs' installations. A validated previous frozen resolution is read-only migration evidence and
supplies exact pins; missing/bad authoritative receipts are not fabricated. Known failed unsealed
builds are collectable only through native GC without active/recovery references. Unresolved
interrupted builds remain protected. No release version is changed. Manual §10 documents native
bootstrap/retry and optional preview/apply cleanup, not manual remote launcher edits.

Acceptance uses real local fixture wheels, console/shell processes, native metadata/inventory
checks, concurrent preparation and live build/GC locks; the heavyweight Conda download/solve and
Torch import are fixture substitutes. APBS's strict version exception remains tested. The lease
fresh-heartbeat regression now samples time at test execution instead of collection, preventing a
long combined suite from accidentally turning its fresh heartbeat stale. Real micromamba/PDB2PQR
bootstrap and the reported WISDOM retry are deliberately not run: they need operator authorization.
No WISDOM source/data, remote prefixes, Jobs or scientific records were modified. Verification:
202 targeted pytest cases passed across publication, probing, native/layered environments,
bootstrap/runtime, storage and ownership suites; Ruff and focused mypy passed. Documented
bootstrap/retry/cleanup commands were parser-checked without invoking any backend action.

## Managed native version probing — 2026-11-14

The common managed-prefix probe now verifies APBS's owning package and version banner, allowing
only APBS 3.4.1 `--version` to return 13. Other nonzero exits remain failures. Bootstrap, submission
and cache reuse use this same check; executable containment/ownership and exact Conda inventory
verification remain in force. Probe receipts/failures include argv, status and bounded diagnostics.
Native regressions execute disposable local binaries and check successful/failing probes, ordinary
fallback, provenance, prefix escape and package build drift on reuse. No packages were installed,
no WISDOM Job was launched, and no managed remote prefix was changed. Remote deployment and real
APBS preparation remain operator actions after updating the controller's LambdaForge build.

## Chained bounded Work recovery — 2026-11-12

Corrected the current working-tree inspector: it previously checked every later Attempt's input
path against the origin Job before applying any revision handling. Attempt 1 now exclusively
anchors historical role bindings. Every later Attempt validates its exact Run/Attempt envelope,
own Job/bundle/configuration, role set, raw bytes, compatible historical fingerprints/sizes and
non-bounded scientific parameters. Different Job owners additionally require an owned request in
the same placement, targeting the original Execution, and an accepted native recovery audit.
No search for a convenient passing Attempt and no skip of contradictory/corrupt evidence.
Diagnostics retain the cause and identify Attempt, actual/origin owners and input role where
applicable; repair of infrastructure provenance is not a request to change scientific YAML.

Legacy v3/v4 bundle/Attempt receipts and either recorded compatible historical/own-basename input
fingerprints remain read-only. Missing origin evidence or unprovable later lineage is refused.
The same helper serves CLI, Console and native recovery. A current metadata preview can run on
an old host interpreter but does not upgrade/attest it: prepared-runtime capability
`bounded-role-chain-v2` is required before scientific handoff. Managed preparation selects current
exact LF wheels; old existing interpreters fail with an update/bootstrap diagnostic, not a late
foreign-path exception. No historical prefix is patched.

Offline native tests execute first failure, accepted-code recovery/failure, second recovery/failure
and third recovery/success across distinct staged Job roots, including renamed files. Six role
identities, original manifests/results and expensive producer checkpoints survive unchanged.
Tests reject changed equal-length bytes, swapped/duplicate roles, forged owner/Execution/placement
requests, missing/corrupt bundles/audit, symlinks, traversal and changed scientific parameters.
Console tests use the actual native preview for both success and actionable failure, without a
real submission. Runtime-capability tests reject old/missing helpers. Ruff and mypy check touched
interfaces; targeted recovery/Console regressions are recorded with final verification below.

Both real read-only commands, for origin `job-20261111182119-e5e63cda` and recovery
`job-20261112091733-4cb30fdc`, pass using the current inspector: same Execution
`execution-95fe53f00a13002a`, Run `run-3bed97f306afd66f7dbe`, next Attempt 3. This is a valid plan,
not evidence that the next WISDOM scientific computation will complete. No Job launch, recovery
apply, historical record/checkpoint rewrite, deletion or WISDOM edit performed. Researcher applies
against the latest failed recovery Job after deploying current LF through native preparation.
Current read-only inventory confirms 450 geometry NPZs, 450 Zn annotations and 449 surface-view
NPZs; these are existence counts, not substitutes for native hashes or WISDOM validators.
Verification: 82 targeted recovery/Console tests pass, plus 62 adaptive/fixed/Study-retry regressions
(overlapping cases). Ruff and mypy pass on the three touched production interfaces. Built-wheel
smoke imports `bounded-role-chain-v2` and inspection helpers without Torch.

## Historical bounded-file recovery identity — 2026-11-12

Implemented `bounded-role-v1`, a scoped compatibility contract from original bundle raw transport
receipts, exact configuration roles and owned Attempt input evidence. Current local/staged bytes
must match; only then reconstruct original filename-bound digests. WorkConfig carries immutable
operational bindings (not authored scientific overrides), passed through preview, dependency
inspection, fixed/single validation and actual native Run specifications. Origin manifests, Run
IDs and checkpoint ownership remain untouched; code acknowledgement stays independent. No global
legacy identity redefinition, prefix guessing, symlink/copy workaround or directory-name relaxation.
Diagnostics identify role, historical basename and expected/observed raw digests.

Read-only inspection of Job `job-20261111182119-e5e63cda` confirms all six TXT raw hashes match
their exact original bounded bundle receipts and current local roles. Recovery preview verifies
scientific identity `sha256:95fe53f00a13002a70b7af4bc6cc9b16a27e57ecc1a55781752d2dd3f687c81d`,
retains Run `run-3bed97f306afd66f7dbe` and proposes Attempt 2 with explicit code-change acknowledgement.
Native metadata inventory: 323 frozen sources (156,005,809 bytes), 450 geometry NPZs
(5,945,229,037 bytes), 450 annotations (32,801,030 bytes), 4,301 structural files
(17,253,625,402 bytes). Collection preview sees 5,398 registered files, no missing/size-invalid
registered payloads; registration is not full payload hash/application validation. Dependency
preview reports 323 present/unverified references, zero unavailable and 1,223 source-unbound
records; these are not universal proofs of scientific compatibility. WISDOM validators must run.

Disposable tests cover duplicate authored basenames across roles, relocated YAML, a second native
staging prefix, exact unchanged origin manifests, accepted/rejected code changes, equal-size
content changes, swapped roles, missing/corrupt bundle receipts, preserved producer bytes,
native public retry/dependency preview and failed checkpoint GC protection. No scientific Job,
recovery apply, environment deployment, checkpoint relocation or WISDOM edit performed.
The exact public `lf jobs retry job-20261111182119-e5e63cda --accept-code-change --dry-run`
completes with `state: planned` and the original Execution/Run. Targeted recovery/dependency suite:
95 passed (including 10 staged-identity cases); Ruff and mypy pass on the five changed interfaces.
Built-wheel smoke verifies the installed `bounded-role-v1` helper and exact renamed-file identity
without importing Torch or installing dependencies.
Pending: researcher-approved real retry through updated managed LF; no expensive payload downloads
or promise that all application validators will pass. Existing scientific/runtime safeguards remain.

## Retained acquisition dependencies and HTTP cooldown — 2026-11-11

Implemented durable per-Attempt cache namespace provenance, retained-Run reachability (including
accepted revisions and local Job result references), GC deletion-time rechecks and protected
bytes/chains/pressure reasons. Failed mapped Works retain potentially recoverable artifacts.
Retention is conservatively namespace-level, not exact per-file reclamation. Live leases still
protect writers. Deleting exact owning history releases references through existing native guards.

Managed map references carry namespace/content evidence; recovery resolves only owned historical
namespaces with exact fingerprints. Observed fetch receipts pin first-request contract/digest
in the original Run; missing observations fail instead of querying a changed URL. Map inputs
are explicit dependencies also under spawn, with optional scientific_policy; thread collectors
also capture cache access. Hidden subprocess I/O and scientific validity cannot be inferred from
PDB IDs, plain strings or a code acknowledgement. Consumers must expose ManagedFile inputs/results,
seal external acquisitions with checkpoints and supply validators/policy. No WISDOM source edits.

Native `jobs recovery-dependencies JOB` and typed services inventory metadata and bounded output.
Explicit apply holds the Execution controller lock and cache/key/GC leases, verifies candidates
against original historical logical-basename fingerprints and atomically copies matching bytes
from owned artifacts, without moving checkpoints, creating symlinks, replaying URLs or science.
Size-only candidates are unverified; missing/unbound records do not become successful evidence.
Fixed the existing temporary-basename validation mismatch in ManagedFileStore builders without
changing persisted fingerprint defaults or published identities.

HTTP service cooldowns use shared-filesystem locks/deadlines across threads/processes. Retry-After
seconds/date and explicit HttpCooldownPolicy JSON hints apply even after local retry exhaustion;
cancellation and permanent errors remain bounded. This is not cross-host coordination without
shared cache storage and does not replace instance-local rate pacing.

Acceptance uses disposable fixtures: terminal quota pressure, accepted-code recovery with zero
new HTTP calls, URL replay refusal, exact legacy artifact repair, live controller refusal,
deleted-history release, local/composed Runs, explicit spawn inputs, concurrent 429/503 cooldown,
spawned HTTP clients, numeric/date/JSON hints, cancellation and permanent 400. Ruff and mypy cover
the changed public sources; a built-wheel smoke imports the new lazy APIs without Torch.
Verification: 114 targeted recovery/storage/HTTP/runtime tests pass, plus the 84-test
single/sequential/adaptive/fixed recovery suite (overlapping tests, not 198 distinct cases).
Ruff passes and mypy checks 13 touched interfaces. The native dependency CLI help and a
metadata-only JobService probe are exercised without preparing a runtime or launching science.

Real read-only preview of origin `job-20261110225702-865a3c13` found 28,727 unavailable cache
references, 28,213 source-unbound records, zero present and zero artifact candidates in the owned
inventory inspected. Both revision namespaces are absent. Prior counts/session preservation do
NOT prove these lost inputs recoverable. No real repair, download, Job launch, registry edit or
scientific mutation was performed. WISDOM's minimum-negative-sequence blocker is separate science.

Pending operational acceptance: deploy current LF before new work/repair; existing immutable
supervisors are not hot-upgraded and retain their old terminal-GC behavior. Full same-snapshot
recovery of the reported lost bytes cannot be promised. Require original digest-matching bytes
from a valid source, otherwise a researcher-authored new acquisition. Do not reuse source-unbound
legacy inspections against new coordinates. No automatic third-party archive/digest reconstruction,
cross-host throttle without shared storage, or detection of hidden subprocess dependencies is claimed.

## Sequential Work recovery — 2026-11-11

Implemented native `retry`/Console recovery for sequential unseeded single-Run steps. Persisted
configuration and child Attempts establish topology and ownership without HPO state. Completed
predecessors restore typed outputs; the worker verifies artifacts/inputs before writing recovery
state. Failed/interrupted children reuse original Run checkpoints in a new Attempt; downstream
steps wait. Root/step resources, scientific identity, spent budgets, controller/provider ownership,
code acknowledgement and Job cleanup dependencies remain enforced. Parallel/composed Studies
are explicitly unsupported. Old interrupted requests without step ownership remain blocked.

Verified on disposable fixtures: first/later failure, resumable item/session reuse, typed files,
repeated interruptions, corrupt/missing evidence, code acknowledgement, budget exhaustion,
read-only native preview, live locks/provider, duplicate/concurrent retry, GC dependencies and
Console confirmation. Targeted recovery/Console suite: 38 passed; existing adaptive/fixed,
relocated-config and Study Console regressions: 80 passed. Ruff and mypy on four changed Python
interfaces pass. No optional integration deployment is implied by those local tests.

Read-only real inspection of WISDOM Job `job-20261110225702-865a3c13` confirms original
Execution `execution-7ba21fa34368be0b`, discover Run `run-bb1f5dd43712f5987e08`, next Attempt 2,
four pending steps and 234 registered checkpoint files with no missing/size-invalid payloads.
Native canonical fingerprint/size of `discovery-session.json` verified; map record files:
28,167 coordinate downloads, 28,167 discovery inspections, 13 functional metadata, 5 negative
mapping. These counts are not proof that every dependency/application validator will pass.
Preview verifies authored configuration identity and requires code acknowledgement.
No real retry, environment deployment, checkpoint relocation or WISDOM source/data edits performed.
Pending: researcher-reviewed live recovery and validation of all restored map dependencies.

## Explicit direct-transfer passwords — 2026-11-09

CLI --ask-password and masked Console input explicitly authorize source→destination password
delivery over the existing encrypted control transport. Automatic key/Kerberos behavior remains;
no saved controller secret or private key is forwarded implicitly. Native JobService/Scheduler/
Supervisor accept bounded ephemeral stdin through anonymous pipes, never requests/argv/env/logs.
Native PasswordSshTransport on the source supplies the same binary/archive/atomic Dataset engine;
the acknowledged source-host Job does not depend on the laptop. Preview sends no asset bytes.
Unknown/changed keys remain blocked, failed authentication never silently relays/reverses/retries,
and generic retry refuses input-bearing Jobs because their credential is intentionally absent.

Verification: real password-only loopback SSH, tiny temporary datasets and temporary trust keys:
six tests passed using the existing optional-dependency test interpreter (no installation), including
complete source-host replication, no secret persisted, failed/changed authentication, missing trust
and CLI prompt behavior. Three native ephemeral-input tests passed; Console masking/confirmation/
secret clearing passed. Existing transfer suites passed 86 tests / one optional skip; queue/provider/
Console/CLI suite passed 47 / four skips; final native retry/input/Console suite passed 27 / three
optional skips. Ruff, mypy on twelve interfaces, CLI help and built-wheel API/source checks passed.

Read-only inspection of ctgpgpu16 confirmed Paramiko is available but the active immutable runtime
lacks ephemeral-input capability. **Pending:** explicit authorization to bootstrap/deploy a new
source runtime and a separate user-approved real transfer. No prefix was patched, dependency
installed, live Job launched or dataset copied. No MFA/keyboard-interactive UI, password storage,
automatic secret recovery or password-based reverse-pull policy is claimed. Controller connectivity
is still required until the secret and Job acknowledgement reach the execution host.

## Shared dataset validation receipts — 2026-11-09

Implemented public Work.validate_dataset over DatasetValidationCache, separate from seed-scoped
Work caches. Native runner binds its already computed consumer code identity without changing
scientific fingerprints. Receipts pin exact dataset/placement, code, contract/version/settings
and runtime. First/stale use verifies native bytes and runs the explicit consumer callback;
reuse inventories bounded POSIX metadata only. Concurrent callers share a native lock and GC
lease. Failures, changed content, unsafe paths/permissions and corrupt receipts cannot imply
success; force revalidates rather than ignoring hashes. Local checksummed v2 assets only.

Verification: 17 validation regressions passed, including native two-seed Work lifecycle reuse,
concurrency, changed/restored-mtime corruption, contract/code changes, relocation, callback failure,
GC leases, metadata bounds, unsafe paths and rejected URI/legacy content. Queue/fixed-recovery/
validation suite passed 78 tests before final boundary additions; 26 Work runtime/lazy-storage/
HTTP-cache regressions also passed. Ruff, mypy on five interfaces, diff checks and built-wheel
public API imports (without Torch import) passed. Only disposable fixtures were used.

Pending consumer acceptance: WISDOM must move deep structural validation into this callback and
use a light binding path after receipt acceptance; LF cannot intercept its arbitrary Python
calls. No consumer source/data, current runtime or live Job was modified. This does not accelerate
the existing Study in place or prevent reservation expiry during a first long check. Receipts
are trusted local change-detection cache, not hostile-filesystem cryptographic proof.

## Sequential execution queues — 2026-11-08

Implemented opt-in native `lf run --queue`, `lf queue` metadata inspection and Console submit
selection/queue table. FIFO tickets persist on the project/placement host; ordinary detached
preparation feeds the same ProcessScheduler/ProcessSupervisor, not a separate runner. Preparation
order cannot overtake registration order. Admission precedes training-resource commitments, waits
for released native ownership, advances on any terminal outcome and survives closing the Console
after provider acknowledgement. Duplicate dispatch, unsupported old runtimes, unsafe/corrupt/missing
metadata and unverified predecessors fail closed. Cleanup receipts and predecessor retention
preserve ordering after native GC. Recovery does not replay original tickets. Queue release failure
is operational evidence, not permission to rewrite completed science.

Verified with disposable metadata, concurrent registrations, native detached local CPU commands,
preparation failure/cancellation, reverse preparation order, nonzero exit followed by successful
successor, no waiting CPU/GPU/storage commitments, CLI routing, Console selection/read-only view,
runtime capability guards and retention tests. No real cluster, WISDOM or dataset was changed.
Limits: POSIX direct hosts only, not SLURM/Fleet; controller needed during preparation; no automated
recovery/priority changes/cross-project queue or site-reservation renewal. Root manifests are
bounded; administrative history compaction is not implemented. See Manual §10 for precise use.

Acceptance completed 2026-11-09: broad suite 161 passed / two real-CUDA cases skipped, then 90
focused native queue/provider/recovery/storage/Console tests passed / two CUDA cases skipped after
final hardening. Ruff, mypy on 13 changed interfaces, CLI help, wheel packaging/imports and shipped
metadata RPC execution passed. Shared-storage fixtures prove separate cluster FIFO namespaces;
release/GC tests do not conflate queued preparation with missing scientific recovery state.

Queue-scope clarification 2026-11-09: Console labels and both manuals explicitly distinguish
whole-Study FIFO ordering from concurrent internal Trials/seeds. A native detached CPU regression
requires two seeded Runs to overlap before allowing the next queued Job to start; unchanged
resource envelopes are asserted. The queue/Console/evidence-plan suite passed 42 tests. Read-only
inspection of V14d confirmed no queue ticket, three active Runs and two still validating consumer
inputs before CUDA allocation; no live Job, grant or consumer source was changed. Startup
validation cost is not addressed by changing queue policy or bypassing unmeasured-Run admission.

## Isolated-tool preflight import isolation — 2026-11-08

The ctgpgpu16 SLAE bootstrap failure exposed a framework-induced pip/setuptools import-order bug,
reproduced locally without CUDA or consumer code. The tool probe now uses only stdlib metadata
before importing the declared scientific modules; LF parses requirements and validates their
reported versions with full PEP 440 semantics outside that interpreter. Bootstrap and Work share
the same checks. VCS revisions, sealed inventory, ABI imports, CUDA probes/grants and environment
identities remain unchanged; no distutils environment override, dependency downgrade or cache
rewrite is introduced. Added regressions cover real setuptools import without pip, root/local
build pin checks, corrupt metadata and failed Work preflight not becoming verified. No remote
bootstrap, SLAE reinstallation or scientific execution was performed for this correction.

## Isolated scientific tool runtimes — 2026-11-07

Native declarations, bootstrap/prepared submission and Work tools now support independent
immutable Python tool prefixes without LF/consumer dependencies, shell hooks or another Work
runner. Exact root/source/ABI policies, import preflight, explicit staged entrypoints and native
resolution receipts are implemented. Tool execution inherits provider GPU tokens unchanged,
streams logs, bounds thread requests and cancels only owned descendants. Native GC protects
current aliases, Job snapshots, recovery owners, preparation and active environment/code leases.
Main-prefix replacement pruning does not collect isolated tools. Dry-run reports unresolved tool
package plans honestly; it does not solve by installing. POSIX provisioning is supported; Windows
provisioning and hermetic/offline artifact availability are not acceptance claims.

Authorized disposable Linux CPU validation: SLAE commit
`078b94e436b8eceb02251a254c3c224358ec6ee2` imported and loaded the 882,525,854-byte base checkpoint
after exact SHA-256 verification (`ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99`).
The reviewed recipe and observed versions are in Manual §4.5 / `examples/isolated_tools.toml`.
This used no WISDOM environment install, source edit, dataset mutation or cluster submission.
**Pending:** WISDOM's standalone extraction adapter, target CUDA recipe/driver acceptance and a
coordinated small protein extraction with scientific output/checkpoint validation. LF provisioning
alone does not repair the consumer's current in-process SLAE import. No such acceptance is claimed.

Verification: 160 targeted tests passed, with two real-CUDA cases explicitly skipped; 111 focused
tests passed again after source/TLS hardening. Ruff and mypy (12 changed interfaces) passed, as did
wheel construction, packaged recipe/module byte checks and isolated wheel imports. The opt-in
`test_slae_tool_acceptance.py` runs the real public tool API against an explicitly supplied disposable
binding and checkpoint; normal CI skips it rather than downloading packages or weights implicitly.
The local ignored acceptance receipt records all 115 resolved package versions. No extraction,
cluster execution, WISDOM source/environment change or dataset/checkpoint mutation was performed.

## Managed dependency layers and recoverable cache collection — 2026-11-06

Implemented in the existing environment/provider/storage services, without another runner or
consumer edits. New v3 execution prefixes install exact code wheels independently over a frozen
dependency base; initial resolution/package inventory and immutable bootstrap wheels are retained.
Dependency/runtime/native/Torch/offline contracts, not code/YAML/seed changes, select the heavy base.
Sealed resolution receipts live outside GC, supply exact version constraints on reconstruction and
reject drift before publication; prepared environments/Jobs record the actual resolution identity.
Legacy prefixes are not rewritten. Native preparation-reference leases bridge staging to durable
provider hand-off; active/recovery/series/process references retain code and its base.

Explicit `clean --environments-only` selects obsolete complete prefixes without requiring an authored
quota. `storage repair` audits exact abandoned operational references, checks controller death on its
own host plus installer descendants on the build host, and never changes scientific state. Missing
state with a readable request is scoped; unresolvable ownership stays protected and diagnosed.
Preview can explain hypothetical post-repair collection but cannot authorize apply. Console adds
exact selection/preview/confirmation and meaningful no-change feedback. Routine storage reads use
cached measurements; deep collection measures unique file blocks and excludes surviving hard links.
The retained quota remains apparent-file-byte based; observed filesystem free deltas are separate.

Read-only case validation on citius-ctgpgpu16: 91 prefixes (87 complete), 793,224,586,015 apparent
file bytes versus 695,754,940,416 unique allocated file bytes. Their declared policies/dependency
metadata form 12 families (not a certificate of historical transitive/binary equality); 49 prefixes
contain LF 0.19.0/WISDOM 0.13.0. Three missing-state/request Job references and three laptop-owned
stale builds pass explicit repair preview with controller-host death and build-host inspection.
Hypothetical environment-only plan: **82 candidates**, **631,804,428,288 releasable file-block bytes**,
nine protected prefixes. The unreadable b0b58dcb build marker/temp remain protected. Original failed
enrichment Job `job-20261106115040-ce45a8b3`, environment `env-31506119e900d58a32cd6e97`, bundle,
all scientific/checkpoint bytes and published datasets are preserved. Full exact-path/byte JSON and
public commands are delivered locally under `.lambdaforge/control/plans/` (ignored, not a new registry).
**No real repair, deletion, cancellation, dependency installation or scientific launch was applied.**

Regression owners: `test_environment_layers_and_reference_repair.py`, existing storage/build,
native environment, CUDA/runtime/bootstrap/provider/staging suites and `test_storage_clear_console.py`.
They use disposable tiny CPU wheel fixtures (no network installation) and loopback transports;
native code layers/imports/entry points, immutable reuse, orphan/corrupt/foreign/PID-reused ownership,
live staging hand-off, recovery references, hard links, read-only previews and Console confirmations
are exercised. Ruff and changed-interface mypy are required in delivery. Remaining boundaries:
unverifiable legacy owners are not force-reclaimed; no retrospective byte-level deduplication or
separate complete provisioning byte-reservation ledger is claimed.

Verification: the latest 156-case run passed 154 and timed out in the two real local CUDA provider
cases waiting on `peer-startup-future-envelope-headroom` (about 3 GiB physically free on the 4 GiB
desktop GPU). They do not exercise dependency installation and were not made to pass by weakening
admission. The scoped CPU/provider/storage suite and frozen-resolution rebuild regression are
verified separately: **154 passed in 53.31 seconds**, including the native CPU provider cases,
eviction/rebuild with retained resolution pins and corruption/drift rejection. Ruff/format passed
on 20 touched Python files; mypy passed on 15 source interfaces; `git diff --check` passed.
Live cluster checks were read-only. No broad CUDA acceptance claim is made.

## Single Work file-checkpoint recovery — 2026-11-06

The reported remote failed Job was inspected read-only. Its owned single Execution has no
Study design/controller and contains one failed Run; the preventive application voxel guard
was not OS OOM evidence. Native preview now recognizes its authoritative resolved `single`
design instead of requiring adaptive state. Verified public CLI `lf retry ... --config ...
--dry-run --json` returned the original Execution/Run, next Attempt 2, 1176 managed receipts
(1175 NPZ payloads plus the failure report; 1,433,721,912 registered bytes), zero missing/size
invalid entries, explicit `payload_hashes: not_checked`, exact original scientific identity,
code-change acknowledgement and effective RAM 96 GiB rather than the origin's 16 GiB.
No environment solve, launch, remote installation, checkpoint payload download, consumer edit,
scientific state rewrite or dataset publication was performed for that validation.

This extends the existing recovery inventory/dispatcher and Job retry path, not another engine:
single Works preserve the same logical Run/checkpoint collection in a new Attempt, aggregate the
latest outcome and keep old failure/cost history. Native provider and read-only existing controller
lock probes reject live/unknown ownership; apply retains atomic controller/recovery locks and GC
dependency references. Authored/staged display labels are normalized only in this single-design
recovery context. Science, input/dataset identity, objectives, execution/publication contracts
remain checked; compatible code needs explicit acknowledgement. Only reviewed CPU/RAM/wall-time
resource changes are supported and propagate to both Job and public Work resources/audit. Selected
YAML bytes are pinned; native Work Console now inspects/confirms with feedback before apply.
Exhausted persisted execution Run/time budgets block recovery rather than refunding old Attempts.

Regression owners: `tests/work/test_single_work_recovery.py`,
`tests/controlplane/test_single_work_retry.py`, `tests/tui/test_single_work_recovery_console.py`,
plus existing adaptive/fixed/grant/recovery suites. Disposable native Work fixtures prove intact
file producers are not rebuilt, failed/interrupted producers retry, corrupted bytes are not reused
as valid evidence and publication happens only after complete validation through source_checkpoint.
No mocked Work lifecycle replaces those tests; asynchronous provider acknowledgement alone is
simulated in control-plane tests to prevent workload launch.

Remaining boundaries: metadata inspection does not hash/deserialize payloads or infer missing
producer counts from opaque keys; public managed access and consumer validators retain authority.
It does not promise training epoch resume, Fleet recovery or cross-cluster migration. The actual
managed environment still needs normal preparation of the corrected LF/compatible consumer build
when the researcher deliberately applies recovery. No real apply or full producer replay is claimed.

## Entmax15 sparse backward — 2026-11-04

Read-only inspection of V11d's latest native Attempt results confirmed 20 succeeded Runs and
four failures of the entmax15 cell (seeds 211, 223, 227, 229), rejected downstream as nonfinite
SchNet distance weights. Two earlier OOM Attempts were retried successfully and are distinct
history, not the four terminal numerical failures. A local CPU reproduction demonstrated
correct `[0.5, 0.5, 0, 0]` probabilities with NaN gradients from unused zero-root thresholds.

The existing Entmax15 now retains the identical closed-form forward and computes its analytic
simplex VJP from the saved probabilities, with zero sensitivity outside support. Safe root
evaluation in backward also permits higher derivatives on fixed-support regions; no C2 claim
at actual support transitions. Constructor, names, dtype promotion, state dict and exact zeros
remain unchanged. No optional package, softmax fallback, epsilon smoothing, generic gradient
repair or downstream finiteness-check relaxation. Tests compare the old forward byte-for-byte,
analytic Jacobian, CPU/CUDA float32/64/16/bfloat16, arbitrary/noncontiguous axes, valid -inf masks,
singleton/support boundary gradients, finite differences and a tiny synthetic optimizer step.
Catalog documentation is regenerated from the native class.

Consumer code/data, persisted evidence and managed environments were not modified; no real
Study was resumed or launched. A managed deployment of the reviewed LF build is still needed
before deliberate recovery; an existing nonfinite checkpoint is not healed by a numerical fix.

## Live resource evidence authority — 2026-10-30

Read-only inspection of the reported running Study found long consumer-side managed-split
initialization (approximately 44 minutes per observed Run before CUDA), not a stopped coordinator.
The dispatcher had admitted additional Runs; a zero-CUDA initializing probe held the unvalidated
ladder lane while a compatible sibling remained protected. Free VRAM alone does not attest the
future demand of that initializing worker. Consumer source/data, live runtimes, Jobs, reservations
and scientific records were not changed; no workloads were launched or stopped.

The actual LF defect was endogenous evidence: the live trajectory update used the preceding
placement forecast (including its own posterior and unmaterialized peers) as historical residuals.
Repeated polls could change its predictive distribution without new physical/progress evidence.
The existing ResourceDemandModel is now used with recorded outcomes only for that historical
prior; an empty-history cold prior is not passed off as history. Placement still receives fresh
censored live trajectories. Zero-prefix CPU initialization cannot train GPU demand; compatible
measured prefixes supply an honest weighted-median point forecast, not an intrinsic neighbour
bound. Upper tails, physical admission, failed-packing dominance and protected lanes are retained.
Diagnostics add materialized/unmaterialized resident counts, elapsed time and actual planner wait
reason without a second scheduling policy or increased notification cadence.

Regression owner: `tests/hpo/test_live_resource_evidence_authority.py`, plus native ARI/dispatcher,
grant-recovery, shared-commitment and scheduling/replay regressions. Tests use disposable CPU-only
fixtures. **Remaining boundary:** consumer initialization can still leave GPU capacity unused while
future demand is uncharacterized; LF does not disable scientific validation or promise saturation
before allocation evidence exists. Existing immutable remote runtimes need explicit managed
deployment/recovery to adopt this correction; no live rollout is claimed.

## Connectors, tables and bounded preview contracts — 2026-10-29

Task source: the current local dirty/untracked working tree and installed native policies,
not a remote checkout or a stale client payload. Changes remain entirely in LF. No consumers,
researcher data, clusters, GPU reservations or live databases/Jobs were modified/contacted.
No new scheduler, graph interpreter, Work runner, HPO or Dataset registry was introduced.

| Requirement | Implemented and checked | Explicit limit / remaining work |
| --- | --- | --- |
| CSV/TSV | Additive native options, explicit headers/names/encoding/BOM/quote/escape/comments/skip/decimal/error policy; incremental projection/row ceiling, gzip decoded guards; existing materialized/internal-path route retained | Wrong-width malformed records now fail unless explicit skip is selected. Native CSV field limit retained, no global override. JSON is not falsely incremental. |
| Tables | Versioned Column/TableSchema and actual JSON schema/example; cast/null constants, typed declarative row predicates, selection/drop/rename, stable sort, dedup, exact concat, explicitly derived aggregate IDs/lineage, FromRows keys | Sort/dedup/aggregate/concat are bounded materializations, not disk-backed external algorithms. Learned imputer/tokenizer expansion and Parquet/Arrow remain absent/unadvertised. Existing train/fold-only standardizer tests retained. |
| SQL readers | Existing-file SQLite read-only/authorizer/read transaction/progress timeout; optional Psycopg PostgreSQL read-only repeatable-read/server cursor, separate bound parameters, validated identifiers, explicit TLS/host credential refs, cursor cleanup | SQLite real fixtures pass. PostgreSQL driver boundary/redaction tested with a mock; real ephemeral TLS PostgreSQL requires explicit opt-in fixture and is skipped when absent. No real server certification claimed. Other dialects need mature backend/contract tests, not an advertised empty plugin. |
| SQL writes | Explicit native operator SQLWritePolicy; insert/upsert/row replace/DDL authority separated, schema/keys/idempotency, bounded atomic rollback on failed/cancelled writes | Operator owns authorization; unrestricted Python is not sandboxed. No raw DDL, auto-create DB file, arbitrary destructive query, JSON flattening or infinite retry. |
| Preview | Shared row/input-byte/source/deadline/decoded-retention budgets; allowed roots, cancellation, owned temporary cleanup; honest prefix/full/unknown identity, row/source provenance; NPY mmap/selected C-order NPZ incremental prefix | Cooperative allocation guards, not OS RSS hard quotas; native I/O/driver calls can allocate/block before Python checks. NPY/NPZ object/structured and Fortran multi-axis NPZ previews refused. Native typed Work sealing may still fingerprint full input bytes; no scientific identity waiver. |
| Effects/catalog | operation-effects-v1 plus schema resources/secret refs, native bounded-preview methods, plugins conservatively disabled for native preview; selected signature/default/abstract validation and metadata-only completeness audit | Native PreviewSession is not a sandbox or generic executor. Not every native model has a complete symbolic tensor equation: audit reports 93 models, 1 abstract / 15 declared structures / 77 signature-only. No invented compatibility guarantee. |
| Model reuse | Structured module injection/state/variadic/multiple-output contracts for ensemble/multitask/mixture/NeuralODE/calibration; actual CPU calls and all native signature descriptors audited without instantiation | Existing GVP/PaiNN/DimeNet/NNConv/sklearn implementations reused. No new scientific architecture or false differentiable classical estimator. Calibration fits held-out data, not train/test. |
| Providers/config | Versioned installed ClusterProfile schema generated from native defaults + constraints, actual effective resolution and CLI work/cluster schema; synthetic gpu executable / gpu exec / custom explicit visibility fixtures | Existing local/SLURM only; no site protocol inferred from command name and no unsupported bare claim/release ownership relaxation. Existing grant-shrink and cancellation/provider tests retained. |
| Distribution | Working-tree catalog and cluster-schema freshness guards, exact wheel/content receipt, two equal offline builds and installed non-editable CSV/SQLite/native Work/dataset/sweep/HPO/CLI checks | No client application bundle update or release bump claimed; consumer must install/pin reviewed wheel. Optional dependencies never installed automatically. |

Public EN/ES manual chapter `guide-tabular-connectors`, packaged JSON schemas and actual examples
define additive migration and limitations. Test owners: `test_tabular_preview`, `test_sql_etl`,
`test_authoring_effect_contracts`, `work/test_tabular_authoring`, installed wheel acceptance plus
existing catalog/ETL/sample/Work/authoring/grant tests. Optional real PostgreSQL and skops tests
remain explicit skips, not successful integrations. Final verification counts are recorded in
the delivery response; no claims are derived from a test's mere existence.

## Preparation orphan and binary dataset streams — 2026-10-28

Read-only inspection of V10e's failed preparation found an environment-build marker owned by a
now-absent process on the same controller host. No matching remote installer or inaccessible
process inventory entries remained. CacheBuildLease acquisition now checks that controller's
process lifetime, then CAS-reclaims unchanged orphan metadata under the existing GC lock only
after remote exact-prefix child inspection. Heartbeat age is not sufficient; foreign, live,
unreadable, changed, symlinked and unexpected-entry cases remain protected. Environments, Jobs,
scientific bytes and grants are unchanged; no real recovery/submission or unlock was performed.

Direct SSH probes on the real 16↔12 endpoints now fail at authentication, not host-key trust.
Read-only effective SSH configuration shows neither default site identity files nor an agent.
No authorized_keys, private credentials or host trust were altered. The separate latest relay
failure revealed Paramiko's text-mode readline vs binary read contract: PasswordSshTransport now
opens public binary ChannelFile readers and preserves original handles until cleanup. Diagnostics
explain that fingerprint approval does not provide noninteractive login authorization.

Verification: **177 passed, 1 skipped** in focused storage/runtime/native-environment, binary,
orphan recovery, transfer/reliability/durable replication and SSH trust regressions. The skipped
real-Paramiko test in the development environment was run with already-installed Paramiko using
the existing consumer interpreter against LF checkout/tests: **10 passed** in binary transport.
Only disposable local fixtures were created; no dependencies installed or real bytes transferred.
Ruff/mypy and the final diff whitespace check pass.

## Scientific authoring operations for visual clients — 2026-10-28

The source of truth for this work is the actual dirty working tree, including untracked neural
operators, not the older Studio payload or HEAD alone. The consumer authoring guide/adapters and
current public model exports/tests were inspected before adding operations. Existing graph
families were reused; no WISDOM/AIStudio source/data, real Jobs, clusters or dataset records were
changed. No release version bump or external dependency installation was performed.

| Area | Implemented / evidence | Remaining boundary |
| --- | --- | --- |
| Catalog | Versioned packaged JSON + schema; SDK/CLI; native signatures, aliases, plugin metadata sidecars, structured errors; metadata-only and selective-resolution subprocess tests | Not every native operator has complete machine-readable shape/range/symmetry equations. Unresolved ports explicitly require native documentation; no fake validation guarantee. No consumer editor integration was performed. |
| Factories | Portable constructor/kwargs provenance; independent state, receiver normalization widths; native class/kwargs injection preserved | Runtime-only alias registrations are not portable plugins; fixed-activation models do not gain unsupported injection. |
| Samples | Existing native DatasetArtifact/index only; NPZ/NPY/JSON/CSV/sidecars/targets/meta; optional/default/presence, strict dtypes/shapes; bounded preview; stack/ragged/pad/disjoint graphs | Index scan is bounded at 100k members/64 MiB, not a paged billion-sample engine. Read deadlines are cooperative, not OS preemption. URI assets require native materialization. |
| ETL | Bounded direct Python operations, strict joins/ids/splits, shared field validation, managed staging, native atomic publication; train/fold-only standardizer | General derive expressions/control flow belong to client-generated Python. Many-to-many joins, arbitrary codecs/tokenizers and proof that authored split labels are honest are not claimed. |
| HTTP | Existing Work.cache.fetch extended with bounded decoded bytes, cancellation/retry checks, HTTPS bearer references and same-origin redirects; native checksum/atomic cache retained | Socket/filesystem calls can block until their timeout; rate-limit waiting retains its existing pacing behavior. No implicit forwarding of controller credentials. |
| Estimators | Mature optional sklearn trees/forest/boosting/SVC/SVR; seeded/thread-scoped capabilities and provenance; no torch-module fiction | Optional skops round-trip test is skipped when extra absent. Missing-extra diagnostic is tested; no pickle fallback or unreviewed type trust. |
| Lifecycle | Synthetic ordinary Work: CSV+JSON→checkpointed member publication→two named inputs→field/component sweep **and** native HPO; interruption preserves files and no partial Dataset is registered | No second runner, scheduler/HPO/registry or GUI graph interpreter. Native identity/recovery rules remain authoritative. |
| Distribution | Offline reproducible wheel builder, freshness guard, dirty/HEAD/version/content receipt; two equal builds; installed non-editable SDK/CLI + prepare/sweep/HPO CPU flow tested outside checkout | No Studio bundle deployment is claimed. Consumer must install the exact reviewed wheel and update its own payload. Backend/environment reproducibility still requires pinned dependencies. |

Checks live in `test_operation_catalog`, `datasets/test_sample_contract`, `test_etl_operations`,
`work/test_visual_research`, `work/test_visual_download`, `packaging/test_visual_authoring_wheel`
and the existing native HTTP/managed-infrastructure, plugin, neural/graph/import regressions.
Ruff and mypy cover changed interfaces; optional skops absence is reported, not hidden as a pass.
The EN/ES manual's scientific-authoring chapter is the public integration contract, with actual
JSON/YAML/consumer-Python examples. Future complete symbolic neural contracts and client
integration are explicit follow-ups, not metadata capability claims.

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
- Storage leases and preview-first cleanup existed at this baseline. Provisioning still has no
  separate complete admission budget. The 2026-11-06 update above implements dependency/code
  separation while preserving legacy immutable prefixes and numerical provenance.

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
| 25–30, 60 | Filesystem, storage, environments and cache | Previews/lazy roots, locked terminal deletion, immutable dependency/code layers and audited reference repair/physical collection implemented; remaining broad cleanup-path audit and separate provisioning byte admission are pending. |
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

<a id="scientific-algorithms-expansion"></a>

## Generic scientific algorithm expansion — 2026-10-18, staged continuation

**The complete supplied specification is NOT accepted.** Initial delivery implemented P0–P4;
the [P5/P6 follow-up](#scientific-algorithms-fgw-follow-up) below updates the current state to
**P0–P6 implemented and tested, P7–P12 pending**. Do not repeat completed stages on continuation.
No WISDOM code, persisted evidence, Job, cluster, dependency environment or release version
was changed. Baseline was clean `main`, `0eb0310` (the cited FGW/scheduling commit).

### Audit and architecture

| Initial category | Actual baseline |
|---|---|
| Complete existing components | StructuredMeasure, balanced q=2 FGW/POT adapter and FGWResult; KMeans, MiniBatchKMeans, DBSCAN, sklearn HDBSCAN, Agglomerative and shared results/capabilities; regression ConformalPredictionInterval. |
| Partial for this request | `supports_precomputed=True` existed, but native string adaptation rejected public precomputed inputs; silhouette treated them like features without matrix-specific checks. ARI/stability only compare already aligned sample arrays. Work.resume_map/checkpoints/cache already exist and must own future block persistence. |
| Absent | H0 graph topology, stable spatial regions, independent-control null calibration, FGW block/resume API and barycenters, medoids, prototype assignment/rejection, mixture adapters, classical MDS and extended stability integration. |

Actual environment: Python 3.10, NumPy 2.2.6, SciPy 1.15.3, sklearn 1.7.2, Torch 2.3.1,
pytest 9.1.1. POT was **not installed**. Supported extra remains `POT>=0.9.6,<0.10`;
POT installation was requested for a temporary test target, not performed. Public POT docs were
consulted; their latest signature is not represented as an installed-backend verification.

Existing native APIs were inspected before editing, including Work.resume_map and lifecycle
contracts. New namespaces are lazy `lambdaforge.topology` and `lambdaforge.stats`; dissimilarity
validation belongs to existing `distances`, not a second point-distance hierarchy. No factories,
runner, cache, normalization engine or consumer-specific data model was introduced.

### Implemented stages and scientific decisions

| Stage | Implemented evidence / limitations |
|---|---|
| P0 | Audit above, existing contracts/tests/dependencies inspected. Full request remains staged. |
| P1 | `distances.validate_dissimilarity`, memory check before conversion/CPU transfer; DBSCAN/HDBSCAN/compatible Agglomerative precomputed inputs and silhouette. Ward/KMeans reject. Explicit averaging only; default finite nonnegative/zero-diagonal/symmetry checks. HDBSCAN uses backend `copy=True` after a regression exposed in-place mutation. Matrix guard does not bound all backend workspace. |
| P2 | Union-find plateau-batched SuperlevelMergeTree, Branch/MergeEvent/MergeTree; elder ties by minimum birth vertex, explicit essential None, event/subtree index, regions/components/mass curves/persistence. Exhaustive tiny-graph oracle, tie/edge/permutation/monotone/deep-chain checks and benchmark. Tied branch names may change with vertex reindexing; partitions and birth/death multiset do not. |
| P3 | StableRegionSelector/RegionSelection/StableRegion; widest feasible growth-event interval, exact uniform and explicit discrete threshold memberships, core/support/mass and rejection reasons. Death boundary is left-open; no crossing. Default greedy disjoint support policy is not a global packing optimum. Dense retained memberships have a byte guard. weighted_iou/weighted_dice support binary/fuzzy membership, not probabilistic confidence. |
| P4 | EmpiricalNullCalibrator/CalibrationResult: inclusive add-one upper/lower ranks, grouped max/min, same-statistic evaluation, ties and resolution warning. BootstrapResult + bootstrap_statistic/bootstrap_quantile: independent observations or complete groups, optional fixed strata, reproducible percentile CIs. No posterior/FDR/conditional guarantee or regression-conformal reuse. Unequal-group pooled statistics are explicitly documented. |

Detailed public examples/contracts/math and limits are in
[manual chapter 34](../MANUAL.md#guide-graph-science) and its Spanish counterpart.
`examples/graph_regions_calibration.py` is a runnable synthetic topology/control example,
**not** the still-required full topology→FGW→prototypes example.
New public result classes have individual files. Existing clustering constructor signatures
remain intact; silhouette adds only the keyword-only matrix guard. Root exports and existing
FGW/StructuredMeasure semantics are unchanged. No new dependencies/extras were added.

### Actual verification

- Baseline: `pytest tests/test_clustering.py tests/test_fused_gromov_wasserstein.py -q`:
  **30 passed, 16 skipped** (POT unavailable).
- Final focused command:
  `pytest tests/test_clustering.py tests/test_precomputed_clustering.py tests/test_superlevel_merge_tree.py tests/test_stable_regions.py tests/test_empirical_null.py tests/test_scientific_algorithm_imports.py tests/test_fused_gromov_wasserstein.py tests/test_manual_links.py -q`:
  **128 passed, 16 skipped**, 2.47 seconds. All skips require POT. This is not numerical
  validation of a new FGW solver or barycenter (none was added).
- Earlier broader command also included `tests/test_structure.py`: **130 passed, 16 skipped,
  1 failed** before adding the last discrete-roundoff regression. The failure identifies four
  baseline classes missing docstrings: GPUObservationPolicy, ResourceSafetyPolicy, ExtensionReply,
  SeriesReply. `git show HEAD:<path>` + AST inspection confirmed all four were already missing
  their docstrings; they were left untouched. New public classes have docstrings.
- Ruff passes on touched scientific source/tests/examples; mypy passes for **21 source files**.
- Both standalone synthetic examples/benchmark were executed. Chain benchmark with tracemalloc:

  | Vertices / edges | Seconds | Peak Python-tracked MiB |
  |---|---:|---:|
  | 1,000 / 999 | 0.0497 | 0.75 |
  | 10,000 / 9,999 | 0.4905 | 7.89 |
  | 50,000 / 49,999 | 2.4800 | 39.39 |

  These are one-machine synthetic timings, not a universal bound. Python-tracked allocations
  are not total process RSS or all native allocations. Tests include a 3,000-deep event chain
  without recursive traversal. Discrete membership uses correctly rounded subset sums rather
  than 100 normalized increments that can spuriously exceed 1. No clipping is introduced.
- A wheel was built locally with `python -m build --wheel --no-isolation` in a disposable
  target. A smoke process with cwd `/tmp` imported directly from the wheel (module paths
  asserted to contain `.whl`), ran topology, region selection, calibration, bootstrap,
  dissimilarity validation and precomputed DBSCAN successfully. Existing numerical dependencies
  were shared read-only; no package was installed. Lazy-import assertions passed before requesting
  clustering. This verifies the artifact, not a fresh dependency installation or POT integration.
- Repeated final focused run after documentation: **128 passed, 16 skipped**, 3.37 seconds.
  The other three structural tests separately passed (the known baseline docstring test was
  explicitly deselected). `git diff --check` passes. No full backend/resource/cluster suite was
  needed or run for these standalone algorithm changes.

### Next steps recorded at initial delivery (superseded by the follow-up below)

1. **P5**: extend existing FGW pairwise API with deterministic square/rectangular blocks, bounded
   diagnostics and incomplete NaN/error state. Inside Work, use native resume_map/checkpoints/cache
   with exact object + solver identities and corrupt/dependency validation. Do not store machine
   paths or introduce another persistence/parallel engine. Read `work/runtime.py` (resume_map),
   managed-file/dependency tests and public Work services before designing it.
2. **P6 mandatory**: verify installed supported `ot.gromov.fgw_barycenters`, implement real
   barycenters, explicit nonzero-diagonal relational representation without weakening default
   StructuredMeasure, numeric shared features, masses, init/seed/multistart, genuine convergence,
   backend vs postcheck objective/coupling diagnostics and actual POT numerical tests.
3. **P7/P8**: weighted honest PAM/medoids, real observation representatives, cross-distance
   PrototypeAssigner and explicit rejection/calibration. No fake probabilities/prediction API.
4. **P9/P10**: sklearn finite/truncated-variational Gaussian mixtures with distinct [N,K]
   responsibilities and genuine estimator prediction; Classical MDS with explicit negative-eigenvalue
   policy and distortion. Do not feed an arbitrary FGW matrix to GMM as though it were features.
5. **P11**: observation-aligned clustering/stability/representative/neighbour metrics and grouped
   bootstrap integration. The new spatial overlap functions cover only part of this stage.
6. **P12**: full synthetic integration example, remaining bilingual docs, real optional-backend
   regression/installed-wheel matrix and final whole-spec acceptance report. No H1/H2, unbalanced OT,
   domain logic, custom GPU solver or distributed graph engine is part of the requested scope.

Keep exact numerical/scientific evidence distinctions; no completed claim based on placeholders.

<a id="scientific-algorithms-fgw-follow-up"></a>

### P5/P6 follow-up — real POT matrices and barycenters, 2026-10-18

**Current verified scope: P0–P6. Next implementation stage: P7. The whole request remains open.**
The operator explicitly authorized POT installation only in disposable test directories. POT
**0.9.7.post1** and the declared minimum **0.9.6** were installed with `--no-deps --target`
in separate `/tmp/lf-pot-*` directories. Ordinary `.venv` still reports `find_spec('ot') is None`.
No dependency installation in WISDOM/project/cluster environments, real scientific workload, cluster
mutation, release bump or push was performed.

P5 adds `distances.FGWMatrixBuilder`, `FGWMatrixBlock`, `FGWMatrixResult` and the internal
versioned `matrix_identity` codec. Square and rectangular comparisons use the existing actual FGW
solver, deterministic row-major blocks, explicit triangular reflection and bounded numerical thread
pools. Dense/streaming budgets include workspace and metadata estimates, not a hard RSS/callback
memory guarantee. No dense coupling history is accumulated for a matrix. Explicit recorded errors
remain NaN in memory / null + error in JSON, with completeness separate from convergence.

An executing Work supplies `work=self`; every block uses **native Work.resume_map**, sequentially,
without a new runner/cache/checkpoint/lock protocol or hidden retries. Keys include exact numerical
object contents, dtype/shape/endian, total masses, relational semantics, ordered collections,
scientific solver settings/class, threads and library versions. Opaque feature encodings and callable
metric revisions are explicit authored contracts; closure equivalence is not guessed. Completed
JSON records have a checksum and are validated before reuse. Missing/malformed/incomplete/corrupt
records and recorded failed blocks are recalculated. Interruption loses at most the current block;
successful comparisons inside an incomplete block may be repeated, an explicit block-granularity
tradeoff rather than refunded physical compute. Public ordinary FGW pairwise behavior is untouched.

P6 adds the **real** `distances.FGWBarycenter` and `FGWBarycenterResult` using installed
`ot.gromov.fgw_barycenters`. Supports shared numeric features, explicit masses/support cardinality,
initial C/X, bounded iterations, seed/restart seeds and sequential multistart. Endpoints alpha=0/1
are explicitly unsupported by this barycenter API; ordinary FGW still supports them. Candidate
validity means finite feasible postchecked evidence, not necessarily convergence/global optimality.
Multistart selects the lowest **weighted final-prototype postcheck objective** among valid starts,
and retains failed-start diagnostics. Both supplied initializations are not silently perturbed.

`StructuredMeasure(..., relational=True)` is a new **keyword-only opt-in** for symmetric nonzero
diagonal relations. Defaults/previous positional constructor behavior remain intact. Barycenter
structures are preserved, never zeroed or projected onto metric matrices. Default/source geometry
and masses now explicitly reject complex inputs rather than dropping their imaginary part.
The same ordinary FGW solver can compare a relational prototype afterwards.

Installed POT source inspection confirmed its barycenter OR stopping test and lagging last-update
transports. The result therefore separates raw feature/structure change histories from freshly
solved final-prototype couplings/costs, with `backend_objective=None` when the chosen logging mode
does not supply one. Convergence requires **both** changes small plus converged final comparisons;
otherwise it warns honestly. No synthetic physical realization, uniqueness or global optimum is
claimed. The memory preflight accounts for POT's per-iteration transport logging.

#### Verification actually performed

- Latest targeted full scientific command, with POT 0.9.7.post1 on process-only PYTHONPATH:
  `pytest tests/test_clustering.py tests/test_precomputed_clustering.py tests/test_superlevel_merge_tree.py tests/test_stable_regions.py tests/test_empirical_null.py tests/test_scientific_algorithm_imports.py tests/test_fused_gromov_wasserstein.py tests/test_fgw_matrix.py tests/test_fgw_barycenter.py tests/test_fgw_optional_backend.py tests/work/test_fgw_block_resume.py tests/test_manual_links.py -q`:
  **171 passed, no skips**, 4.19 seconds.
- Independent minimum-backend command with POT 0.9.6:
  `pytest tests/test_fgw_matrix.py tests/test_fgw_barycenter.py tests/test_fgw_optional_backend.py tests/test_fused_gromov_wasserstein.py tests/work/test_fgw_block_resume.py -q`:
  **66 passed**, 2.89 seconds. Earlier intermediate runs of 53/61/63 cases were also green;
  they are superseded by the expanded verification, not counted again.
- `tests/work/test_fgw_block_resume.py` contains **six native lifecycle regressions**: interrupt
  a real Work, preserve two good blocks, then recover only missing/malformed/checksum-invalid/
  incomplete blocks; a recorded failed block is not reused. These use disposable local fixtures,
  actual WorkRunner/checkpoints and real POT, not an alternate mock persistence engine.
- Numeric tests cover scalar-vs-block matrices, both orientations/explicit symmetry/rectangular
  collection, identities, memory guard, opaque contracts, NaN failure, one/two/unequal objects,
  weights, exact/permuted inputs, actual POT comparison, nonzero relational diagonal, reproducible
  multistart, invalid starts, max-iteration/nonconvergence, OR-vs-BOTH stopping and final marginals.
  FGW matrices also feed the native precomputed clustering API in an integration regression.
- With **no POT target on PYTHONPATH**, optional-backend diagnostics/lazy imports:
  **3 passed**. Importing new classes does not load POT/Torch/sklearn.
- Ruff passes on scientific source/new tests/examples; mypy passes for **41 source files**.
  `git diff --check` passes. The earlier unrelated four baseline docstring omissions remain
  untouched; no green whole-repository structure claim is made.
- `examples/fgw_barycenter.py` ran with actual POT: weighted 2/3-node objects, a complete matrix,
  reproducible two-start barycenter and rectangular prototype comparisons. Its observed relational
  diagonal `[0.16666667, 0.33333333]` was retained, with converged postchecked objective
  `0.11291666666666667`. This is a synthetic example, not a universal numerical benchmark.
- The optional POT CI job now requests its clustering extra explicitly and includes the new
  numerical/native-resume tests and standalone example. GitHub CI was **not** launched/observed.

Public contracts/examples are in [the FGW matrix section](../MANUAL.md#fgw-blocks) and
[barycenter section](../MANUAL.md#fgw-barycenters), with Spanish counterparts in the same single
manual structure. Changelog remains Unreleased; release version is unchanged. Final wheel-artifact
smoke: the rebuilt 0.19.0 wheel was imported directly with cwd `/tmp`, asserting its module paths
come from `.whl`, sharing only existing numerical dependencies and the temporary POT target.
Real barycenter optimization retained its nonzero diagonal, a complete native block matrix was
computed, and imports remained lazy until solving. This is artifact validation, not a fresh
dependency installation. Repeated final regression runs after the normalized-support underflow
guard: **171 passed in 4.53 s** (POT 0.9.7.post1), **66 passed in 3.41 s** (POT 0.9.6).

#### Continuation without repeating P0–P6

Start **P7**: inspect Clusterer/ClusteringResult/capabilities and add honest weighted PAM/medoids,
real observation indices and arbitrary-label representatives under the existing memory/distance
contracts. Then **P8** cross-distance prototype assignment with explicit rejection (not invented
probabilities/HDBSCAN prediction), **P9** finite/truncated-variational GMM adapters, **P10** classical
MDS with negative-spectrum policy, **P11** aligned clustering/representative/neighbour stability,
and **P12** full topology→FGW→clustering→medoid/barycenter→assignment example and final acceptance.
No new algorithm placeholders/factories were added for these stages. Native Work lifecycle and
HPO/resource/recovery policy were not modified by this standalone algorithm work.

## 2026-10-19 — deleted-Job startup commitments and idle V8 GPUs

Read-only native inspection of WISDOM found `wisdom-v8a` Job
`job-20261019114516-0054cc25` on cluster 12 with zero active Runs and 16 queued,
approximately 79 GiB physical free VRAM; `wisdom-v8b` Job
`job-20261019114504-779e02a6` on cluster 16 had two Runs on GPU token `0` while
its granted token `1` was idle with approximately 140 GiB free. Both now use auto caps.
Their fresh launch veto was `peer-startup-future-envelope-headroom`, not a YAML cap.

The host-wide startup ledgers still referenced deleted morning Jobs
`job-20261019084926-ba7a6173` and `job-20261019084951-8927477a`. Their controller
lifetimes were dead, worker identity files and entire Job directories absent, and the native
process-marker scan found no surviving marked processes or relevant unresolved ownership.
Missing request/state files prevented the old reconciliation from retiring those commitments.

The shared admission service now uses `ProcessSupervisor.detached_owner_absent` only when
the entire owned Job directory is absent. It reuses native lifetime/marker-scope semantics;
no timestamp-only expiry, physical-idleness inference, fabricated Job state or relaxed grant
checks. Existing incomplete/corrupt directories stay fail-closed. Repeated commitments from
one owner/path share their absence check within one reconciliation. This changes operational
startup accounting only, not scientific status, recovery eligibility or GPU access.

Verified locally: **118 passed** across `tests/hpo/test_adaptive_resources.py`,
`tests/runtime/test_shared_gpu_commitments.py`, `tests/controlplane/test_recovery_process_scope.py`
and `tests/controlplane/test_process_scheduler.py`. New tests cover deleted Jobs with dead/reused
controllers admitting a new Study, surviving orphans, inaccessible/alive owners, corrupt records
and unsafe paths. Ruff and mypy cover the touched operational modules. No WISDOM source,
remote Job, grant or persisted evidence was modified. Adoption by already running immutable
runtimes still requires explicitly authorized native cancellation/recovery; not performed.

## 2026-10-20 — reusable graph technologies from the prospective V9 design

Read the report's `atomic_encoder_design.typ`, the included V9 family specifications and
`docs/auditoria_v9.md` under the user's TFM/Memoria in read-only mode. The report explicitly
calls this a prospective architecture comparison, not trained results. Existing GCN, GraphSAGE,
GIN, R-GCN, PNA, GATv2, sparse GraphTransformer and EGNN are reused, not duplicated.

Implemented independent public modules under `nn/models/graph`, exported through the existing
graph/models/nn namespaces:

- GINE layer/stack with explicit GINEConv per-message ReLU; full and basis NNConv layer/stack,
  sum/mean/root controls, centered basis coefficients and node-side basis matrix application.
  Existing ECMP already covers receiver/sender/edge MPNN; relation embeddings do not falsely
  claim the report's independent-per-relation MLP semantics.
- Gaussian/sinusoidal radial features and cosine cutoff; SchNet-style scalar node interactions
  with explicit distances and optional independent support. SiLU is documented, not silently
  identified with the full original shifted-softplus application recipe.
- GVP primitive, residual scalar/vector GVP-GNN stack and whole-vector-channel dropout;
  PaiNN sinusoidal interaction/local mixing and stack. Persistent polar vectors, invariant
  node readouts, explicit geometric inputs and zero initial vectors, no chemical featurizer.
- Bounded explicit GraphTriplets, canonical polynomial envelope, normalized spherical Bessel
  and real m=0 harmonic bases, distinct bilinear DimeNet and factorized DimeNet++ interaction
  modules. The generic DimeNet node encoder selects the interaction equation explicitly;
  standard Torch initialization/arbitrary features/node readouts are documented adaptations.
  Optional `graph-directional` extra requires SciPy only to construct root constants; no
  dependency installation occurred. Forward is differentiable Torch. Spherical small-argument
  series/recurrence match SciPy through all eight supported orders; all 64 roots are distinct
  and ordered in each of those orders. Outside-support powers/trigonometry are bounded to
  avoid nonfinite backward even for very distant edges.
- Graphormer-inspired global structural stack/layer and explicit directed BFS encoding.
  Batch graphs cannot attend to each other. Unreachable/self/path codes and degree codebooks
  are explicit; no arbitrary index-selected path attributes. Supplied edge biases must retain
  permutation equivariance. Degree/path conditioning per block is a declared generic variant,
  not original graph-token/pretraining parity. Dense score/shortest-path byte limits fail before
  corresponding matrix allocation; they are not claimed as full training-memory forecasts.

The framework does not implement the consumer's semantic gates, relation branches, baseline
context, capacity matching, signed data/code selection or campaign protocol. In particular no
WISDOM file, dataset, study, cluster or scientific workload was changed/launched. Full-paper
benchmark parity, CUDA conformance/performance and predictive improvement remain unverified;
these are component APIs and numerical contracts, not evidence of a completed V9 experiment.

Verification (existing Python 3.10 / Torch 2.3.1): **168 passed in 3.22 s**, new graph tests plus
existing graph/primitives/attention/equivariant/catalog regressions; **6 passed** in the existing
general model tests. New cases include scalar equations, explicit edge-matrix/triplet sums,
permutation/multiplicity, isolated/empty nodes, O(3) transformations, state_dict, finite backward,
bf16 CPU autocast, malformed input, memory bounds and missing optional dependency diagnostics.
Ruff passes for graph changes, exports, tests and example; mypy passes for **60 source files**
including graph interfaces and the example. `examples/graph_encoders.py --directional` executed
all eight variants with finite forward/backward.

Built the 0.19.0 wheel using `build --wheel --no-isolation` (no dependency install), then imported
it directly with cwd `/tmp`, verified `.whl` provenance and `graph-directional` metadata, and
executed the packaged example with both directional equations successfully. Artifact:
`/tmp/lf-graph-v9-wheel.HWcYl8/lambdaforge-0.19.0-py3-none-any.whl`. This shares existing Torch/SciPy
dependencies in read-only mode; it is not a clean-environment install claim. Manual/example are
packaged and documented EN/ES; release version remains unchanged. Prior unrelated scientific
algorithm P7–P12 acceptance work remains pending as recorded above, not folded into this task.

## 2026-10-20 — public graph composition/source-control follow-up

Implemented only the missing public interfaces from the follow-up request; no WISDOM/thesis
edits, cluster contact, scientific Jobs, dependency installation, automatic commit or release bump.
The [public capability matrix and exact signatures](../MANUAL.md#graph-operator-controls) distinguish
existing equations from new composition boundaries, with the same Spanish section.

- GCN/GraphSAGE optional signed aligned message weights preserve original topological degrees
  and Mean counts. Canonical GCN loops replace supplied loops with modulation one; root/bias
  policy remains explicit. Both model stacks forward the same modulation without new state.
- GIN/GINE and PNA forward now reuse public messages/aggregate/update stages. ECMP exposes
  projection and those stages per existing layer index. One update may combine independent
  relation messages without duplicated roots/MLPs. PNA's scaler/aggregator order and authored
  statistics remain unchanged; degree_dtype preserves original AMP arithmetic. Independent
  branch objects retain unused stage parameters; execution/capacity accounting must be honest.
- SchNet/PaiNN source weights jointly control radial features and cutoff, keeping unit directions
  separate. Public radial_inputs and forward_from_radial layer boundaries allow native bases and
  complete supports to be combined per relation before one node update, with no hidden cutoff.
- DimeNet distance-source control retains an angle-only constant-one radial baseline, not invalid
  physical zero lengths. forward_from_basis accepts explicit relation-specific native supports;
  initial states, incoming/outgoing triplet products, post-interaction states and readout messages
  honor differentiable weights. Mixed typed paths/multiplicity survive without topology changes.
  Fractional factors compound over stages; legacy zero-input node-readout constants are explicitly
  distinguished from messages when all relations close. No original state keys were replaced.
- Existing structural attention remains global/dense; its score guard now uses promoted dtype
  under autocast. This is a memory-accounting correction, not a new Graphormer path-edge recipe
  or permission to override a consumer's prohibition on dense N×N matrices.

Compatibility: constructors/default equations/state keys preserved, checked against hand references
and pre-refactor equations (including PNA/ECMP bf16 AMP); malformed ECMP topology/relation IDs
are now rejected rather than silently truncated. Precomputed interfaces validate tensor/topology
contracts, not hidden physical provenance. Radial controls cannot infer arbitrary consumer edge
attribute semantics; duplicate scalar distance columns remain the caller's responsibility.
Source control does not claim to disable directions/angles or the node's own update constants.
No new generic GNN factory, scientific gate registry or application architecture was introduced.

Verification: **274 passed in 3.67 s**, across graph_operator_controls, v9_graph_components,
graph_models, graph_primitives, graph_transformer, equivariant_graph_models, models, pna,
relational_gcn and nn_component_catalog. New synthetic controls cover signed/zero/fractional
weights, canonical/explicit loops, common roots, relation/attribute separation, mixed triplets,
distance closure/cutoffs, long relation support, permutations, edge reversal, disjoint batching,
O(3), finite gradients/bf16, strict state_dict, invalid geometry and controlled memory guards.
Ruff lint/format pass; mypy passes for **59 source files** (graph interfaces, ECMP and example).
`examples/graph_operator_controls.py --directional` ran with finite forward/backward, using
public methods only. Wheel built with existing dependencies and --no-isolation; importing the
wheel directly from `/tmp` verified module provenance and ran that packaged example successfully:
`/tmp/lf-graph-controls-wheel.aVa31E/lambdaforge-0.19.0-py3-none-any.whl`.
No clean dependency installation, real-protein fit, CUDA conformance, GPU performance or predictive
improvement is claimed. Earlier unrelated scientific algorithm acceptance remains pending.

### 2026-10-20 — Bound live ARI posterior feedback and joint-fit work

Read-only inspection of `wisdom-v8d`, Job `job-20261020101604-883baff7`, cluster16:
the grant contained physical tokens `1,3`. Token 1 had about 139 GiB free and one 842 MiB
idle CUDA worker; token 3 had three active training workers. The first Run on token 1 had
already persisted `succeeded`, but the controller had not collected it. Study/admission
projections were stale despite the supervisor heartbeat and sibling training continuing.
Persisted live supports had 3,446/3,447 points and candidate predictions had 6,739; resource
projection was about 30 MiB and Study summary about 106 MiB. A read-only native debugger stack
confirmed `_normalized_weights → _weighted_quantile → _joint_fit_probability →
_resource_experiment_signature → _exploration_value → place → _execute_gpu_admitted_runs`.
The debugger detached; no Job was cancelled, restarted, retried or remotely patched.

Native correction: bound posterior mixtures and convolution inputs to 128 weighted points;
retain endpoint masses and conservatively round interior bins upward. Shared-latent fit sweeps
CDF events once instead of sorting every distribution at every quantile. Live trajectory updates
retain historical residual weights instead of making a rare tail equiprobable with each body
support point. Exact small supports, physical checks, grants, OOM dominance and scientific
planning policies are preserved. No second scheduler or concurrency override was added.

Verification: **175 passed in 3.90 s** across resource-distribution liveness, adaptive resources,
v3.1 invariants, scheduling benchmark/replay, native live refill/recovery grants and runtime
commitment/observation/device-health/attribution regressions. Eight new cases cover conservative
CDFs, exact endpoint masses, exhaustive small-support ensemble agreement, bounded legacy inputs,
70 live feedback cycles without terminal evidence and residual-tail weighting. Ruff lint/format
passes for three touched Python files; mypy passes for both changed source interfaces. A local
synthetic 6,740-point/four-distribution comparison timed out the previous calculation after 3 s;
the corrected calculation took about 0.042 s. This is not a real-GPU throughput claim.

Operational limitation: the already-running immutable managed runtime still contains the old
implementation. Adopting the correction in that Job requires a separately authorized lifecycle
operation; none was performed. No live WISDOM evidence, application code or dependencies changed.

### 2026-10-27 — Durable operational Dataset replication

CLI `datasets replicate --apply` and Console Replicate now enqueue the same native
`dataset-replication` Job. They reuse JobService/JobStore and provider schedulers/supervisors,
not a scientific Work, alternate lifecycle or second copy engine. Direct SSH runs on the source
host independently of the PC. A compressed relay runs under a detached local Job and survives
UI/launcher exit but still needs its PC/network. Authentication snapshots retain references only;
interactive-password relay is refused. Site SSH remains host-key verified without forwarding.
No live transfer was launched, resumed or cancelled, and no remote installation was performed.

Dataset Transfers and `datasets transfers [DATASET]` / `transfer-status JOB` reconnect to bounded
metadata and native logs. Exact active requests are locked/deduplicated; unknown ownership and
unacknowledged submission never justify duplicate dispatch. Native CPU/storage/grant policies
remain intact. Checksum-verified destination acknowledgements reconcile the controller placement
on reconnect without changing identity; failures/corruption are not complete replicas. Active
source/destination copies are protected from Dataset deletion even when hosted on the controller.
Derived profiles are not project-scoped twice. Terminal observations are cached via native metadata
merge for cheap history; explicit status still reads host metadata. UI refresh retains old rows on
observation failure and does not repeatedly notify. Completed placements update Locations.

Verification: **142 passed in 73.72 s** across all Dataset tests, native ProcessScheduler and
Dataset Console tests. Includes real detached disposable local Jobs, parent exit before copy starts,
concurrent deduplication, direct/relayed compression, scoped-project routing, corruption refusal,
bounded metadata-only reads, cached terminal history, ownership/path safety, endpoint deletion
protection and Console pending-versus-verified feedback. Ruff lint/format pass for 12 Python files;
mypy passes for 9 changed source interfaces. Wheel built without dependency installation; new
modules and CLI parsers imported directly from that wheel in `/tmp` with provenance asserted.
No clean installation, real site SSH/SLURM connectivity or real large-dataset throughput is claimed.

Limits: old in-session transfers are not retroactively detached; relays cannot survive PC shutdown;
interrupted copies restart the byte stream, not byte-offset resume. Already-exact destination
verification/reconciliation remains the existing synchronous no-new-transfer path. A new transfer
is durable after native Job acknowledgement, not merely after clicking Submit. No version bump.

### 2026-10-27 — Relative relay request paths and early-failure diagnostics

Read-only inspection of the supplied local `stderr.log`, `request.json` and `work/replication.json`
for `job-20261027090952-65861abc` confirmed that the request existed but its command argument was
relative. The native supervisor correctly used its absolute work directory as cwd; the worker
therefore looked for the project-relative request below that directory and exited before copying.
No dataset-integrity/scientific failure was inferred. No WISDOM file, dataset, registry or original
Job was modified, and no actual transfer or remote command was launched for this correction.

Replication now reuses native local submission anchoring *before* constructing paths, staging or
serializing the worker command. Local profile workspace/state/cache/run/dataset/lease roots and
explicit interpreter paths are pinned to ProjectContext; the registry was already absolute.
Snapshots retain those absolute local paths. SSH profiles remain host-owned and unchanged;
no local resolving of remote paths, symlink workaround, auxiliary copy or worker cwd change.

Status reads a bounded 8 KiB owned stderr tail for failed Jobs, redacts before exposing/caching,
and returns a nonempty cause plus `logs_command` when no result/progress was ever written. Missing
logs produce an explicit diagnostic/link, not an empty error. Failed caches from before this fix
refresh once. CLI renders the cause and Console log details include it; outcome/identity/publication,
active-transfer deduplication and uncertain-ownership policies are unchanged.

Verification: **146 passed in 81.62 s** across all Dataset tests, native ProcessScheduler and Dataset
Console regressions. Added real worker execution from work_dir with relative/absolute local roots
(controller invoked from a project subdirectory), complete local snapshot anchoring, unchanged SSH
profiles, early missing-request failure with no destination/result/progress, redaction, legacy empty
failure cache refresh and public CLI JSON/human diagnostics. Existing active deduplication and atomic
checksum publication regressions pass. Focused early-failure checks were rerun after adding bounded
tail handling. Ruff lint/format pass for four touched Python files; mypy passes for three changed
source interfaces. No dependency installation or release bump. Reattempt uses the ordinary explicit
`datasets replicate` preview then apply; the original failed Job/logs remain historical evidence.

### 2026-10-27 — Immutable nested resource telemetry and Study recovery review

The supplied V10a generator/items traceback is reproduced locally without CUDA or dataset access.
`dataclasses.asdict` reconstructs dict subclasses with a pair generator; FrozenJsonMapping formerly
accepted only Mapping. The constructor now accepts the normal dict pair-iterable protocol and
continues recursively freezing nested values. Resource dataclasses now serialize through one
field/portable-container boundary instead of asdict/deepcopy reconstruction. Active/terminal
history, admission, exploration, commitment ledgers and trace scientific-value contracts use it;
nested frozen lists/mappings and mapping proxies become detached built-in JSON containers.
Already-portable wire values remain equal and parameter resource identity hashes remain unchanged.
Unsupported non-JSON values still fail explicitly, preserving the prior atomic snapshot.
No generic catch, GPU admission/monitor disablement or alternate recovery engine was introduced.

Verification: **213 passed in 72.93 s** across immutable JSON/resource serialization (12 new cases),
ARI/v3.1/distribution liveness/replay/benchmark, monitor immutable IPC, shared GPU commitments,
adaptive/fixed Study recovery and GPU-grant recovery. Coverage includes the actual persist_active
round-trip at synthetic step 89/final-evaluation phase, exact original values/immutability, JSON
validity, all corresponding resource write paths, stable wire representation and atomic rejection.
Recovery regressions preserve valid result paths/bytes and checkpoint compatibility, reuse original
Execution/Run identities, retain physical costs/history, requeue interrupted seeds and do not infer
completion from telemetry. No recovery code needed changing. An additional native-environment
identity/cache regression passed. Ruff lint/format pass for four touched Python files; mypy passes
for both changed source interfaces. Wheel built --no-isolation without installing dependencies;
from `/tmp`, exact wheel module provenance, nested asdict and native persist_active reload verified.

Deployment remains an explicit operator action: use the corrected controller framework package,
bootstrap the consumer's managed cluster environment from the consumer project, inspect native
retry --dry-run, then deliberate retry. Exact wheel bytes participate in EnvironmentIdentity even
when the release stays 0.19.0. Do not patch the old prefix or rewrite original evidence. The V10a
YAML was read only to confirm its display name/seed policy; no real Run, transfer, remote probe,
environment install, dataset/index mutation or recovery was performed. The supplied remote error
file was not fetched; no live per-Run completion counts or checkpoint compatibility are claimed.
The unrelated simultaneous Dataset transfer is not attributed as a cause. Version unchanged.

### 2026-10-27 — Direct push/pull replication, explicit relay and honest observation

Read-only inspection of the supplied transfer Job found receiver-write EOF followed by another
Paramiko EOF during cleanup. This proves a broken receiver stream, not whether the network failed,
the receiver exited or the controller suspended. Native events recorded the failed process at
11:21:33 UTC, about five seconds after its 11:21:28 exit; 11:39 was a later confirmation, not the
first failure detection. The exact prior UI desynchronization is not proven. Dataset listings now
refresh native transfer observations without requiring transfer-status or an open detail panel.

One native transfer service now probes effective bounded readonly SSH push then inverse pull,
classifying authentication, host key, timeout, network, tools, permissions and indeterminate causes.
Push Jobs live on source, pull Jobs on destination; both run the existing exact archive/publication
engine with no controller data socket. Auto only proposes controller relay. CLI interactive consent,
explicit --route relay for automation, and Console confirmation show path, original size and PC
dependency. Approved direction/diagnostics persist; a changed route before dispatch is rejected.
No trust weakening, key copying, agent forwarding, second runner or automatic unbounded retry.

Status uses native OperationalHealth/Job evidence, separating process outcome, observation age,
byte advancement/age and observed verification/publication. Unknown observation never fabricates
failure or confirmed running. No compressed percentage is inferred from original size. Binary
stderr is drained independently and bounded/redacted; peer exit is unknown when unavailable.
Cleanup cannot replace the original exception and owned subprocesses are still reaped. Bounded
receiver phase responses distinguish actual verification/publication during direct push as well
as pull. Exact checksums, atomic publication, source ownership and active-transfer exclusion remain.

Verification uses only disposable local fixtures and simulated site transports: actual detached
workers for both directions, no controller stream, route/consent changes, distinct diagnostics,
unobservable/terminal process views, separate byte/verification clocks, early sender/receiver failure,
primary/cleanup EOF preservation, incomplete-publication rejection and existing duplicate guards.
No real site SSH probe, transfer, Job cancellation/restart, dependency install or WISDOM/data/index
mutation was performed. Site authentication/trust and filesystem/runtime readiness remain operator/
administrator responsibilities. Probes are snapshots, not future-connectivity guarantees. Partial
streams restart on deliberate retry, not byte-offset resume; verified destinations are reused.
The release stays 0.19.0. No claim is made that real cluster-to-cluster connectivity is configured.

Checks: the broad Dataset/ProcessScheduler/SSH-policy/binary/Console suite passed **179 tests** in
108.07 s. After adding receiver-reported publication phases, the affected replication/binary/Console
subset passed **77 tests** in 72.17 s (including two new direction/phase regressions). Ruff lint and
format checks pass for 16 touched files; mypy passes for 12 source interfaces. The wheel builds
without dependency installation; extracted-wheel import provenance, helper presence and route
classification were smoke-tested from `/tmp`. These are fixture/packaging checks, not site acceptance.

### 2026-10-27 — Explicit no-laptop replication policy in the Console

Read-only local Job inventory inspection of `job-20261027120734-c733ed69` confirms that `host=local`
is accurate: it was a controller-stream relay. Its persisted direct-push probe reported missing
trusted ED25519 host key for the destination; inverse pull reported authentication denied on source.
This is evidence of those particular failed probes, not a current live connectivity guarantee.
A fresh preview from this agent could not authenticate because the session's keyring backend was
unavailable; no keys, credentials, profiles, real Jobs or datasets were changed and no transfer launched.

The Console previously exposed only endpoint selection, then converted a relay proposal into relay
policy at confirmation. It now explicitly offers Direct only, Auto and Relay, defaults remote
sources to Direct only and forwards the chosen policy unchanged to the native preview. A direct
failure opens readable scrollable diagnostics; no implicit relay confirmation is possible. Local
endpoints explain their PC dependency. Confirmation also rejects a relay plan under direct-only
policy defensively. Transfers label relay as `via this PC`; existing Jobs retain their pinned route.
English/Spanish manuals explain that execution host is route-derived, not a freely selectable label.

Checks: replication Console plus native route regressions **33 passed** in 43.99 s; final expanded
Console suite **10 passed** in 17.97 s, including route selection/default, fixed action controls at
80×24, readable literal diagnostics, no direct-to-relay confirmation and explicit relay labeling.
Ruff lint/format pass for three touched files; mypy passes for two source interfaces. The correction
cannot create site-authorized inter-host SSH access: an operator/administrator must verify host keys
and authorize noninteractive access in at least one direction. No trust bypass or credential forwarding.

### 2026-10-28 — Explicit first-use peer fingerprint approval

The supplied screenshot identifies unknown ED25519 trust from 16 to the exact destination IP;
inverse authentication denied is a separate failure. Accepting keys on the controller does not
configure the initiating cluster, and hostname/IP aliases can use different trust records. A new
read-only WISDOM preview was attempted but this agent session cannot access its keyring backend;
current live SSH/authentication state is therefore not asserted. No real trust, credential, Job,
dataset, registry or WISDOM file was changed, and no real transfer was launched.

Structured native direct-route errors now expose diagnostics without parsing a human traceback.
SiteSshTrustService/its shipped host helper implement bounded read-only inspection of effective
SSH configuration, existing user/global trust and offered keys. DatasetService, `datasets trust-ssh`
and Console consume that same operation. A separate first-use modal names initiator, peer/IP/port,
remote file and SHA256 keys and warns that keyscan does not authenticate them. Explicit approval
pins/re-scans every fingerprint, rechecks under a native lock, preserves existing bytes/permissions
and atomically adds hashed entries to the initiating user's safe effective known_hosts. Concurrent
changes, mismatched fingerprints, existing changed/marked keys, symlinks and unsafe paths fail closed.
HostKeyAlias, proxy routes and special path expansions require site-managed trust rather than
approximate support. No SSH strict-check disablement, credential forwarding or automatic key replace.

After trust approval the Console repeats the normal native preview. Dataset copy still requires
its own confirmation; canceling trust changes nothing. Recording server identity neither grants
noninteractive login nor fixes a password-only peer account. Existing native direct push/pull Jobs,
relay consent, identity/checksum verification and atomic dataset publication remain unchanged.

Verification: **194 passed in 100.32 s** across Dataset tests, first-use trust, binary/SSH policy and
Dataset Console. Covers native fingerprints vs ssh-keygen, no-write preview, explicit approval,
hash lookup, original bytes/permissions, idempotence/concurrency, rotation/revocation and unsafe
paths, port-specific labels, CLI delegation, real helper subprocess execution with fixture-only
SSH binaries/disposable HOME, modal cancel/approve/re-preview and separate copy confirmation.
Ruff lint/format pass for 11 touched files; mypy passes for nine source interfaces. CLI help works;
wheel builds --no-isolation without dependency installation, and extracted-wheel imports verify
module provenance, helper presence and public trust APIs. OpenSSH's primary manual was checked
for keyscan's unauthenticated nature. No live site acceptance or release-version bump is claimed.

### 2026-10-28 — V10d CPU startup diagnosis and GPU zero-attribution correction

Read-only inspection of `job-20261028082934-ca20dad0` on ctgpgpu16 found the two granted tokens
2/3 intact, three live scientific workers and a fourth seed queued. Workers consumed about one
CPU core each, read tens of GB and remained in the logged managed-split loading stage without
CUDA process entries or epochs. The immutable consumer runtime's source confirms that structural
registration calls full sidecar validation and AtomicSASA.validate recomputes SASA during that
initialization. This is consumer CPU preparation, not proof of a GPU-launch deadlock. Without
intervention, the first workers reached epochs 3–4 and LF admitted seed 229 at 09:17:10 UTC.
No Work/YAML, datasets, persisted scientific records, GPU grants or processes were changed.
Intrusive stack attachment was not used: it would pause a live scientific worker and requires
separate operator approval. Code, logs, open-file metadata and advancing CPU/I/O were sufficient.

A separate LF bug was reproduced with fixtures: CPU-only live workers absent from the CUDA PID
table fell through to aggregate attribution, inheriting device-wide driver/other usage (about
617 MiB here). GPUObservation now explicitly attests process-inventory completeness, including
NVML/SMI process-inspection failures and truncation. The native runner recognizes absent owned
processes as measured zero only with a complete measured inventory, validated live lifetime,
observation after worker birth and no conflicting positive allocator evidence. Unknown per-PID
bytes, partial/legacy observations and reused lifetimes remain unknown/conservative; uncertain
multi-device usage cannot be mislabeled exact. No physical free memory, future envelope,
protected progress lane, exploration policy, device grant or scientific result was relaxed.

Changes are limited to GPUObservation, native runner attribution and a focused regression module,
plus EN/ES documentation/changelog. **153 tests passed in 2.59 s** across native ARI/v3.1, resource
distribution/replay/benchmark, attribution/observation and shared GPU admission. Tests include the
actual live-update path, reserved/foreign VRAM, unknown vs zero, birth time, reused PID, unknown
per-process bytes, allocator conflict and backend query/inspection failures. Ruff lint/format
passes for all three touched files; mypy passes for both source interfaces. No dependency install
or release bump. Existing immutable running Jobs are not hot-patched: the correction is for the
next explicitly deployed runtime. Reducing the long SASA initialization requires consumer-side
work; no WISDOM optimization or current-runtime rollout is claimed here.
