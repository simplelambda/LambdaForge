# Storage admission and safe cleanup

[Español](STORAGE.es.md) · English

## Inspect and clear

```bash
lf clusters storage CLUSTER --json
lf storage status --on CLUSTER --json
lf storage reconcile --on CLUSTER --json         # measure; no metadata mutation
lf storage reconcile --on CLUSTER --apply       # update diagnostic ledger only
lf clean --on CLUSTER              # preview; no deletion
lf clean --on CLUSTER --apply      # recheck and collect safe candidates
```

In the Research Console, open **Clusters → your cluster → Clear storage…**. The operation
previews reclaimable categories and protected references, requires confirmation, runs outside
the UI loop and reports pending/success/failure in the activity panel. **Clear output** only
clears the visible terminal text. Neither operation forgets scientific history. Use native
`lf delete` or `lf jobs clear` for deliberate history deletion.

Explicit storage inspection inventories owned directories. Normal resource probes use filesystem
capacity and small lease records, not recursive scans, checkpoints or dataset assets. Reports
separate category sizes from physical filesystems; apparent bytes are not physical free space.
`reconcile` compares deep measurements with `state/storage-ledger.json`, reports size/file-count
drift and can atomically update that versioned ledger. It never removes files or changes scientific
records. Overlapping categories are not additive. The ledger does not replace physical free-space
checks or existing references; a corrupt ledger is reported rather than silently overwritten.

## Admission

```yaml
resources:
  storage: 100GiB
```

This declares a future storage commitment for the outer Job allocation, not a filesystem quota
and not its current measured usage. It does not prevent arbitrary consumer code from writing
more bytes. Cooperating LambdaForge Jobs on the same host use a shared, filesystem-specific
lease authority. Submission never combines free capacity from different volumes.

Admission requires `physical free - other active commitments - new commitment >= safety` and
available inodes when the filesystem reports them. Usage does not shrink the declared commitment;
it is released at supervisor completion or after positively verified owner death. Unreadable,
corrupt or ambiguous ownership blocks new commitments rather than granting fictitious capacity.
PID reuse alone does not prove death. An existing owned allocation is not reserved again by its
workers or Fleet offer.

When admission cannot proceed, direct Jobs wait before acquiring CPU/GPU leases. The state
includes requested/free/reserved/safety bytes and the reason. The same authority is available
inside scheduler-owned native workers. Safe automatic GC is attempted before continued waiting;
healthy science is not killed to reclaim disk. No Run is restarted by cleanup.

Configure the shared authored cluster profile (example):

```yaml
storage:
  state_root: /durable/lambdaforge/state
  cache_root: /scratch/lambdaforge/cache
  run_root: /scratch/lambdaforge/jobs
  dataset_root: /durable/lambdaforge/datasets
  lease_root: /durable/lambdaforge/host-leases
  cache_max_size: 500GiB
  cache_max_age: 30d
  safety:
    min_free: 20GiB
    min_free_percent: 5
  terminal_jobs:
    grace_period: 2d
```

The larger absolute/percentage floor wins. Default free-space safety is 5%; default successful
checkpoint grace is two days. Project scoping changes operational roots, not the host-wide lease
root. The console profile editor exposes these fields as well as native `clusters set/unset`.

Pressure is reported as NORMAL, SOFT_PRESSURE, HARD_PRESSURE or CRITICAL using physical free
space, safety, leases and inode exhaustion. Reservations are cooperative, not protection against
another user's writes or an external disk quota.

For SLURM, no portable scratch directive is assumed. Positive storage requests produce an
omission warning unless `resource_mapping.storage` is configured for the site, for example
`{option: tmp, value: "{storage_mib}"}`. Worker-side root admission does not imply SLURM has
reserved that space. Fleet placement requires fresh known storage fit for a positive requirement;
unknown capacity is not permission to dispatch.

## What collection can remove

Collection respects active Job, exact environment, runtime, bundle and Work-cache references.
Environment builds protect their own prefix and mutating package categories, not unrelated
Work caches. Unknown ownership stays protected. Only positively identified dead, stale same-host
build owners permit orphan temporary cleanup; foreign-host/legacy markers are retained.
A dead local controller alone is insufficient if pip/Conda still references the exact temporary
prefix. Unreadable process evidence also keeps it protected. Bootstrap uses an existing host
interpreter for ownership metadata; that interpreter need not satisfy the scientific Work version.

Automatic cache collection observes authored quota/age, pressure and proven orphan status; without
an authored quota its ceiling is 10% of the cache filesystem capacity. Reconstructible lower-cost
categories are selected before expensive environments, then by last use. Protected entries can
prevent reaching a quota; the plan reports unresolved excess instead of deleting them.
`lf clean` additionally includes inactive reconstructible Work caches. Categories include pip,
Conda/native/runtime packages, managers, environments, runtimes, bundles and temporary content.

Cache deletion records intent, renames an exact entry into owned trash, and resumes interrupted
trash deletion idempotently. It never replays an uncommitted intent against newly referenced
original content. `state/storage-gc.jsonl` records applied operational collection. Results, logs,
metrics, provenance, published datasets, researcher-owned mirrors, successful unpublished
artifacts and active/unknown Jobs are not cache candidates.

## Checkpoints and publication

Failed/interrupted Executions retain recovery checkpoints. Terminal artifact compaction continues
to discard partial failed artifacts and verified redundant published output bytes. A **successful**
Execution may release unpinned checkpoints after its grace period. Keep durable scientific models
as registered outputs; explicitly pin resumable state that must outlive ordinary retention:

```python
self.checkpoints.pin("best/model.ckpt", reason="retain-for-follow-up")
self.checkpoints.unpin("best/model.ckpt")  # metadata only; does not immediately delete
model = self.outputs.from_checkpoint("model", "best/model.ckpt", release=True)
```

`from_checkpoint` seals an independent registered artifact and checks source/snapshot identity.
Dataset publication can read an owned checkpoint directory directly, avoiding an intermediate
Attempt copy:

```python
self.outputs.dataset(
    name="example", version="1", members=members,
    source_checkpoint="prepared-records",
    release_checkpoints=["prepared-records"],
)
```

Member asset paths are relative to that named checkpoint directory. Explicit release becomes
eligible only after successful Execution finalization and a verified independent artifact or
dataset publication; changed/missing destination bytes invalidate the proof. Copies use
copy-on-write where supported, otherwise independent regular copies—never writable hard links.
This is not a promise of zero-copy on every filesystem.
Managed snapshot copies reserve their additional bytes on the **destination** filesystem, including
external `publish_to` and dataset assets. An insufficient volume triggers one safe, same-volume cache
collection attempt before a compact space diagnostic; GC never scans or deletes that external
publication directory. Uncommitted copies retain their original checkpoints. These incremental
copy commitments are released on success/error, do not silently lower the Job's declared commitment,
and do not multiply physical payload bytes for a successful reflink. A clone checks safety/inodes;
its full-copy fallback must first reserve the payload bytes.

Cleanup takes the Execution writer lock, rechecks pins/identity and journals checkpoint deletion;
it cannot race an active Study recovery writer. `checkpoint-retention.json` records removals while
Attempt results, logs, metrics and history remain. Corrupt/missing retention evidence fails closed.

## Current boundaries

Run-root admission and managed publication-copy reservations are implemented. Bundle staging and
dependency provisioning do not yet have separate byte reservations on their destination volumes.
Dataset index/manifest generation is not a disk quota. Managed environments
still use identity v2 with exact wheel hashes: dependency environments and independent code layers
have **not** yet been separated. Build markers are conservatively protected, but cross-host orphan
proof remains conservative. Both Python-runtime and managed-environment build markers now use
the same atomic GC handshake and bounded heartbeat, including reuse verification. Invalid complete
environment prefixes are preserved rather than replaced underneath referenced Jobs. Explicit
`lf storage reconcile` provides a diagnostic measurement ledger, not a complete per-write allocation
ledger. Do not treat cleanup as protection against every ENOSPC during preparation.
