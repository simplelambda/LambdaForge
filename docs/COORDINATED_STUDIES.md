# Coordinated Studies: implementation status and contracts

[Español](COORDINATED_STUDIES.es.md)

## Status: fresh adaptive and fixed Fleet launch; recovery/transfer integration pending

`lf run CONFIG --on-fleet FLEET` now queues one durable local coordinator for repeated Runs,
fixed sweeps, automatic paired sweeps and fresh adaptive HPO without fidelity continuation.
The Research Console's Run Work target selector exposes
the same `fleet:NAME` route. Scientific Work YAML remains unchanged. This is a **limited public
capability**, not completion of the full coordinated adaptive-HPO request.

The existing single-cluster route and independent `MultiClusterSubmissionService` remain unchanged.
Independent submissions are not coordinated HPO.

### Implemented

- One native `WorkRunner` and the original adaptive/fixed/`PairedSweepSequentialAnalyzer` planners. The
  dispatcher executes exact proposals and invokes the existing result/frontier callbacks; it
  never creates candidates, seeds or an optimizer on a worker. One Execution and final analysis.
- One persistent provider Job per member, prepared by ordinary `ControlPlane` and `JobService`.
  Its owned member runtime receives finite leased waves, reuses native isolated Run processes
  and ARI, and drains accepted work before releasing its allocation. Waves do not create new
  GPU claims or another scheduler Job.
- Fresh member heartbeats under the exact owned Job attest immutable code/environment/file-input/
  numerical identities plus actual homogeneous hardware. All five identities must match across
  members before any scientific Run is leased. Preparation placeholders are never placement evidence.
- Baseline GPU admission uses only inherited opaque tokens, native short-lived hardware/memory
  probes, per-member caps and physical headroom. No broadening of device grants. A busy wave offers
  zero spare slots: **GPU co-location and incremental ARI refill inside a Fleet member are pending**.
  Offers expire; physical `lf fleets offers` observations alone never grant dispatch capacity.
- The common prepared-provider path retains direct/site-command/SLURM policy, TLS, bundles and
  immutable managed environments. Loopback CPU and tiny real CUDA acceptance exist; this is not
  yet end-to-end SSH, site-command or SLURM acceptance.
- Typed file markers retain authored scientific parameters and verify canonical bytes/size on
  preparation and worker binding. Large inputs still require the existing project mirror.
  Distributed dataset attestation remains rejected; matching NAME@VERSION is not enough.
- Existing atomic coordinator v2 leases, submission fences, unknown ownership, quarantine,
  caps, original clock/budgets and reconciliation remain authoritative. The global placement wave
  additionally respects the native global parallelism ceiling. No lease is retried from a lost
  connection alone.
- Member Jobs remain in `lf jobs` but do not become independent semantic Works. The parent is
  shown as `fleet:NAME`; active and terminal per-Run metadata includes exact
  cluster/Job/shard/lease/Attempt. Native worker Run records project bounded latest/best scalars
  into the coordinator's existing Study view. `lf show STUDY --run KEY --json` and
  `lf logs STUDY --run KEY` read live or terminal curves/logs on the verified member owner, lazily.
  The coordinator never opens a remote path locally. Display summaries do not substitute for
  the durable scientific scalar stream below.
- Native `metrics.jsonl` and `training-metrics.jsonl` are transported incrementally in complete,
  SHA-256-verified records, at most 32 KiB per channel/Run/read, with optional lossless compression.
  Exact Study/Run/Attempt/lease/member/shard identities fence every read; byte offsets preserve
  order and deduplicate retransmission. Durable receiver cursors recover unacknowledged appends
  after interruption; missing/corrupt acknowledged data, gaps and contradictory replay fail closed.
  Workers retain their native evidence through disconnection. Logs and bulk artifacts are not
  part of this stream. A terminal result reaches the planner only after its full scalar stream.
- The same central adaptive planner consumes completed evidence and all leased/queued/running
  proposals across members. Its existing pending-aware acquisition (BoTorch qLogNEI when available,
  deterministic mixed-kNN fallback otherwise) is retained. Spare admissible capacity invokes the
  native bounded scientific frontier, not worker-side/random proposal generation. Unleased
  reprioritization preserves scientific identity and records the prior invocation/priority.
- Central pruning reuses native utility/history/calibration rules and sends durable idempotent
  exact-lease stop requests. Required/startup/confirmation evidence stays protected. Requests
  stop cooperatively; an unavailable connection is not pruning acknowledgement or terminal evidence.
  Verified local scalar mirrors feed historical calibration without opening execution-host paths.
  Checkpoint continuation/recovery is still gated, not inferred from similarly named local paths.
- `lf cancel STUDY` stops the coordinator before enumerating and cancelling exact member owners.
  Semantic deletion previews the whole owned Job family and refuses active members. Individual
  cleanup cannot remove evidence referenced by the family. Remote bulk compaction stays on its
  execution host; coordinator paths are independently owned.

### Commands available now

```bash
lf run study.yaml --on-fleet research --dry-run --json  # No grant, upload or Job.
lf run study.yaml --on-fleet research                  # Durable asynchronous hand-off.
lf show STUDY --json
lf show STUDY --run trial-00001-seed-4 --json
lf logs STUDY --run trial-00001-seed-4
lf cancel STUDY --dry-run
lf cancel STUDY --apply
lf delete STUDY                                       # Preview terminal family.
lf delete STUDY --apply
```

The coordinator is currently `local`; members must be managed cluster profiles, not the built-in
local target. Dry-run validates source/signature/design, member roles and credentials, reports caps
and deferred environment/equivalence checks, and does not probe or acquire GPUs. The durable hand-off
captures operational profiles and credential references, never credential values. Ordinary
`--on` and `--on-fleet` are mutually exclusive. `--rerun`, `--restart`, hidden recovery options and
`--wait-for-submit` are rejected for this route.

### Not implemented yet

Checkpoint stream/continuation and distributed recovery; predictive/lookahead
policy; member co-location; public pause/resume/reconcile and adoption; coordinator restart recovery;
live membership expansion/drain; dataset-placement equivalence; checkpoint/artifact transfer,
compression/resumption and distributed export; full placement/utilization TUI/HTML; StudyDecision
and dependency gating. Internal coordinator lifecycle/catalog controls are not substitutes for those
public operations. Fleet retry refuses single-cluster replay; export refuses a coordinator-only
package that would silently omit member evidence.

The request is therefore **not complete**. Do not describe this limited route as production-ready
full distributed production readiness or claim provider acceptance from loopback/fake tests.

### Incremental validation (2026-10-05)

Ordered stream regressions cover bounded multi-chunk reads, compression, replay, incomplete records,
receiver interruption, foreign leases and symlinks. Member regressions cover directed pruning,
idempotence, immutable commands and protected evidence. Public loopback CPU integration runs the
native adaptive planner across two prepared allocations and retains one HPO state/final analysis.
Historical pruning is also exercised using verified mirrors with deliberately absent owner paths.
These tests do not establish real remote/GPU adaptive acceptance or coordinator restart recovery.

### Validation of this implementation pass (2026-10-04)

- `ruff check .`: passed.
- `mypy src/lambdaforge`: passed, 558 source files.
- Complete `pytest`: 1,246 passed; four Lightning warnings in CPU tests.
- A freshly built, installed wheel passed packaging smoke checks, installed-import verification,
  `run --help`, scaffolding, validation and read-only dry-run.
- Public Fleet integration exercised two loopback CPU members, including live Run metrics/logs;
  separate prepared-provider tests exercised tiny real local CUDA workloads. These are not
  real SSH/site-command/SLURM acceptance tests. WISDOM and real clusters were not modified or run.
- GitHub CI was not executed. Passing local checks does not establish the pending capabilities
  listed above.

## Fleet catalog

Fleet configuration belongs in `lambdaforge.clusters.yaml`, never the scientific Work YAML:

```yaml
clusters:
  a: {transport: ssh, host: a-login, user: USER, scheduler: local}
  b: {transport: ssh, host: b-login, user: USER, scheduler: local}
  queue: {transport: ssh, host: queue-login, user: USER, scheduler: slurm}
fleets:
  research:
    coordinator: local
    members:
      - {cluster: a, max_gpus: 2, max_runs: 4}
      - {cluster: b, max_gpus: 4}
      - {cluster: queue, max_inflight_jobs: 2, required: false}
```

Cluster profiles keep their own GPU launch/claim policy, project mirror and data environment. All
references must resolve locally before any probe. User < project < explicit precedence is shared
with the existing catalog. A fleet override replaces its whole membership list; it does not merge
lists accidentally. Cluster definitions, credential references and execution profiles are retained.
Caps accept positive integers or null; online/draining/disabled are operator controls, not probes.

```bash
lf fleets list --json
lf fleets show research --json
lf fleets offers research --json         # Explicit read-only probes; not an allocation.
lf fleets drain research b               # Preview catalog change.
lf fleets drain research b --apply
lf fleets disable research b --apply     # Never signals active Jobs.
lf fleets enable research b --apply
```

Use `lf fleets --catalog PATH ...` to select an explicit catalog. Otherwise mutations go to the
project catalog. Saving a fleet does not save secrets or change Work scientific identity.

## Ownership and persistence

The current local driver owns a project-scoped coordinator below the durable JobStore's
`fleets/PARENT_JOB/coordinator`, with independently owned native Execution metadata beside it. `coordinator.json`
is schema version 2 (with a v1 reader) and is atomically fsynced with the existing JSON publisher under the existing
cross-process file lock. `leadership()` fences the planning loop. Short state transactions never
hold a lock over network I/O. The state contains immutable initialization/budgets, Run definitions,
all leases/Attempt history, placement reasons, shard identities, member controls and quarantine
references. Separate immutable digest-named `results/` files hold accepted envelopes; `quarantine/`
holds contradictions. Missing/corrupt/version-incompatible state fails closed, never silently resets.
Restart cannot silently change coordinator authority. Readiness attestations must be explicit
booleans and proposed scientific payloads must be strict finite JSON before they are persisted.

The global Run key hashes `(study_identity, candidate, seed, phase, fidelity)` using the existing
scientific identity primitive. A lease adds Attempt number, opaque lease ID and assigned cluster.
Two drivers cannot reserve the same logical Attempt. Offers exclude locally acknowledged resident
leases; the coordinator subtracts additional unacknowledged leases to prevent repeated stale
observations from overbooking during the prepare/submit gap.

A submission-intent fence is persisted **before** the provider call. An ambiguous acknowledgement
therefore requires observation by immutable shard ID, not another submit or a new Attempt. The
executor protocol is independently idempotent by shard ID. No credentials belong in a shard.
Late submission acknowledgements attach to their exact Attempt: they cannot overwrite a completed
result, move a running Run back to queued, or attach an old Job ID to a replacement Attempt.

Fresh/healthy executor offers can permit new work, but expired offers cannot. Transport failure
marks active leases `unknown_remote`, with exponential reconnection backoff capped at five minutes.
It does not cancel workers, create failures or release scientific identity. Missing observations
also remain unknown. Lost-attempt retry requires a local scheduler/owned-process/shard authority
proof naming the exact lease and confirming no live owner, and remains bounded. Arbitrary consumer
exceptions are not retry triggers. Old shard records do not consume the replacement Attempt's Job cap.

Result envelopes must match Study, Run, Attempt, lease, target, parameters and execution equivalence.
Exact duplicated bytes are harmless; contradictory or late released-Attempt results are quarantined
with a compact diagnostic, never overwrite valid evidence. A crash between immutable result write
and atomic index write is repaired by repeated delivery. Per-Run result reads verify the digest.
Root/evidence symlinks are rejected. Overview snapshots expose compact counts, not every result.

Reservation currently conservatively consumes the `max_runs` dispatch allowance; an operator
withdrawal before the submission-intent fence refunds its unspent reservation, preserving the
Attempt audit. Wall-time is
measured from immutable coordinator creation across restart. Physical resource accounting and
reversible unstarted reservation accounting still need the production dispatcher integration.

## Reproducible foundation acceptance tests

No real cluster or WISDOM execution is needed:

```bash
python -m pytest -q tests/controlplane/test_coordinated_study.py tests/controlplane/test_fleets.py
python -m pytest -q tests/work/test_concrete_shard.py
python -m pytest -q tests/work/test_study_dispatch_boundary.py tests/controlplane/test_coordinator_pause.py
python -m pytest -q tests/work/test_coordinated_cpu_dispatch.py tests/controlplane/test_prepared_cpu_executor.py
python -m pytest -q tests/controlplane/test_shard_preparation.py tests/controlplane/test_prepared_provider_dispatch.py
python -m pytest -q tests/controlplane/test_fleet_study_service.py tests/controlplane/test_member_allocation.py
```

The fixed design test supplies 56 candidates × 4 seeds = 224 unique required Runs and fake targets
with capacities A=2, B=3, C=1. All are centrally ingested once; the exact synthetic final objectives
match a serial reference. The partition test disconnects B, loads a fresh coordinator instance,
redirects new work to A/C and reconciles one resident, one completed and one positively lost B
Attempt; only the lost Run becomes Attempt 2. Other tests cover competing drivers, ambiguous
acceptance, stale offers, caps, checkpoint locality, environment/hardware mismatch, provenance,
budgets, operator drain and immutable global identities. These validate the operational contract,
not yet end-to-end distributed Work execution or predictive optimizer quality.
The CPU shard test invokes four real spawned Work processes at concurrency two, keeps successful
and failed scientific evidence and native paths, and proves repeated delivery does not create
another Attempt. It does not submit a provider Job, grant a GPU or start an optimizer.

Prepared-provider acceptance also exercises actual bundle staging, JobService, a detached direct
supervisor, CPU children and a minimal CUDA tensor child with native ARI. Installation/runtime
resolution are fixtures; the transport is loopback, not SSH. The GPU test uses explicitly shared
local access and inherits the supervisor's grant; it skips when CUDA is unavailable. This does not
establish SLURM/site-command acceptance or managed package installation. Persistent-allocation
cases verify fresh baseline offers, exact Job reuse and drain without releasing the grant between
waves. Input identity/mutation and opaque-grant regressions are separate.

The prepared-provider tests run four fixed identities or three complete two-candidate paired
blocks over two detached direct CPU targets, ingest native outcomes centrally and produce one
final analysis. This is real
local ProcessScheduler execution, not a fake provider, but it is still an internal prepared-CPU
test. `tests/controlplane/test_fleet_study_service.py` additionally tests the product service's
real two-member allocations, single final analysis, terminal Run-owner reads, semantic cancellation,
captured asynchronous request and read-only CLI grammar/preflight. Installation remains a fixture;
the test does not pretend to be real SSH or complete adaptive Fleet acceptance. The live-read
fixture holds scientific Runs open until the coordinator has actually read their isolated log
and step-7 metric through the exact member owner. Foreign seeds/Trials/Attempts/paths are rejected;
remote display scalars cannot become completed response evidence or trigger local path reads.
Dispatcher-injection tests also exercise the
existing native adaptive planner without creating a worker-side optimizer. Pause tests cover
network partitions, completed-result ingestion, pre-submit races, restart and immutable budgets.

Fixed authored sweeps will keep performance pruning disabled. Automatic sweeps will retain complete
paired blocks; adaptive searches may prune under their existing scientific contract. Future
predictive work belongs to the current Study and its ordinary budgets: spare capacity is not
permission to launch scientifically dependent downstream work or uninformative random proposals.
