# Coordinated Studies: implementation status and contracts

[Español](COORDINATED_STUDIES.es.md)

## Status: operational foundation, not yet an execution feature

This is the implementation record for the requested coordinated multi-cluster Study. **There is
not yet a supported `lf run --on-fleet` route.** Do not use independent submissions as a workaround
and call them coordinated HPO. No production runner/planner has been redirected to this foundation.
The existing single-cluster route and independent `MultiClusterSubmissionService` remain unchanged.

Implemented and tested:

- `Fleet`/`FleetMember`: operational roles, membership and per-target caps in `ClusterCatalog`.
- `FleetResourceService`: reuse `ResourceService` observations, explicitly separate physical facts
  from executor-attested admission, and report optional/required reachability.
- `ExecutionEquivalence`: exact verified code, environment, inputs, numerics and hardware stratum.
- `GlobalRun`: immutable candidate/seed/phase/fidelity identity without placement coordinates.
- `GlobalPlacementBroker`: hard readiness, freshness, stratum, memory, cap and checkpoint-locality
  filters; rank by known estimated completion/load, with deterministic name tie-breaking. Unknown
  times stay unknown in the audit and rank behind fully known estimates; no invented probabilities.
- `StudyCoordinator`: atomic per-Attempt leases, unique shard assignments, submission-intent fencing,
  unknown remote states, explicit bounded lost-Attempt retry, restart reconciliation, result-digest
  validation and contradiction quarantine. Scientific proposals are inputs, not invented here.
- `ShardExecutor`: a protocol for idempotent submission and factual owned-executor observation.
  This is a boundary, **not yet a Process/SLURM/command-backed implementation**.
- `work.shard.execute_concrete_shard`: internal fresh-CPU worker proof using the existing isolated
  dispatcher, not a second planner. It validates a finite leased queue, stores outcomes immediately,
  isolates consumer failures and makes completed re-delivery idempotent. It deliberately rejects
  GPU execution, recovered Attempts and checkpoint continuation until their provider/identity
  bindings are ready. Verified preflight equivalence is a required caller input, not independently
  established by this internal worker. It is not connected to public submission.
- `lf fleets list/show/offers/drain/disable/enable`: catalog discovery and preview-first operator
  controls. Catalog controls do not yet manage running production Studies. `offers` reports
  observation-only readiness, never claims dispatch is ready.

Still required before exposing coordinated execution:

1. Connect the existing `WorkRunner` dispatch boundary to the coordinator without copying the
   adaptive or fixed scientific planner. Keep `PairedSweepSequentialAnalyzer` and complete block
   lookahead as the automatic-sweep authority.
2. Implement owned shard workers through existing `ControlPlane`, bundles/environments,
   `JobService`, `ProcessScheduler`/`SlurmScheduler`, GPU access policy and local ARI. Shards must
   receive concrete Run invocations only and never own a second optimizer.
3. Preflight verified data/environment/numerics/hardware equivalence. Acquire fresh local offers
   inside exact grants; apply per-member GPU caps locally. An observation from `nvidia-smi` cannot
   authorize placement.
4. Integrate remote metric/checkpoint/artifact locality and exact transfer manifests with existing
   Study telemetry, results, recovery and export. Checkpoints currently block non-local placement;
   automatic replication is deliberately not implemented or pretended to work.
5. Add versioned primary/predictive search policy to the **existing** scientific planner, with
   pending-aware acquisition, budget accounting, stale queued withdrawal and explicit idle reasons.
   The foundation accepts/audits a predictive Run only with planner/evidence/model revisions and
   a reason/policy, but it does not generate predictive proposals.
6. Add `StudyDecision`, explicit upstream decision gating (no arbitrary workflow language), native
   launch/reconcile CLI, TUI and single scientific HTML with placement provenance/utilization.
7. Finish provider-adapter integration and the full predictive acceptance cases before calling the
   feature complete. The fresh CPU-worker subprocess test below proves only concrete native Work
   execution; fake-executor tests are not proof of Process/SLURM/command integration.

LambdaForge 0.17.0 releases the tested Fleet catalog and coordination foundations described here.
It does not yet provide complete multi-cluster Study execution; the integration steps above remain
pending.

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

The future driver owns a project-scoped coordinator directory below its Execution. `coordinator.json`
is schema version 1 and is atomically fsynced with the existing JSON publisher under the existing
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

Fixed authored sweeps will keep performance pruning disabled. Automatic sweeps will retain complete
paired blocks; adaptive searches may prune under their existing scientific contract. Future
predictive work belongs to the current Study and its ordinary budgets: spare capacity is not
permission to launch scientifically dependent downstream work or uninformative random proposals.
