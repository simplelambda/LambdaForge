# Artifact retention

[Experiment guide](../README.md) · [Repository guide](../../../../README.md) ·
[Español](README.es.md)

This package safely reduces completed experiment suites. It can retain selected checkpoint roles,
archive large intermediate regular files, and prune explicit disposable files. Retention is
preview-first, task-agnostic and disabled unless configured or explicitly requested.

## Contents

- [YAML contract](#yaml-contract)
- [Eligibility](#eligibility)
- [Protected artifacts](#protected-artifacts)
- [Checkpoint roles](#checkpoint-roles)
- [Rules and archives](#rules-and-archives)
- [Object API and CLI](#object-api-and-cli)
- [Transaction and recovery](#transaction-and-recovery)
- [Produced artifacts](#produced-artifacts)
- [Limits](#limits)

## YAML contract

Retention was added by experiment Schema 1.1:

```yaml
schema_version: "1.1"

retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect:
    - reports/**
    - predictions/final.json
  rules:
    - action: compress
      include: [predictions/**/*.json, embeddings/*.npy]
      exclude: [predictions/final.json]
      min_size_bytes: 1048576
      compression:
        level: 9                # optional override of archive.compression_level
        only_if_smaller: true
    - action: prune
      include: [scratch/**]
      exclude: []
      min_size_bytes: 0
  archive:
    name: artifacts.zip
    compression_level: 6
  lock_timeout_seconds: 60
```

Omitting the block is equivalent to `mode: disabled`, `keep: all`,
`prune_unselected: false`, no rules, `artifacts.zip`, compression level 6 and a 60-second lock
timeout.

- `disabled` preserves historical behaviour. Explicit `Experiment.apply_retention()` or CLI
  `--apply` is still an intentional manual request.
- `preview` enables planning but never deletes, archives, creates locks or writes reports by itself.
- `apply` asks the final successful aggregation to apply retention automatically.

Every mapping is strict and rejects unknown keys. Validation rejects empty include lists, negative
sizes, unsupported actions, ZIP levels outside 0–9 and compression options on prune rules. Runtime
validation also rejects absolute, drive/UNC, parent-traversing, backslash and NUL patterns.

## Eligibility

Application requires a current `aggregate/aggregation_receipt.json`. The receipt is written
atomically only by a final aggregation and is complete only when:

1. the expanded suite contains at least one run;
2. every expected variant and seed exists;
3. every run has `status: ok`;
4. every variant is complete and terminal with exact expected/completed counts;
5. each run still has safe `config.yaml`, `environment.json`, `hparams.json` and `result.json`;
   `metrics.csv` is also committed when `trainer.write_epoch_metrics_csv` is enabled;
6. every `experiment.required_artifacts` path exists safely inside its run;
7. core aggregate CSV/JSON outputs were published;
8. configuration and committed input/output fingerprints still match.

Failed, interrupted, ignored, pending and dry-run results never qualify. Incremental per-variant
aggregation invalidates an old receipt and passes `final=False`, so it cannot trigger retention. A
new training launch invalidates the receipt before workers start.

`Experiment.preview_retention()` returns `not_ready` without writing when the receipt is absent,
incomplete or stale. Sources are fingerprinted again immediately before archive and quarantine
operations; any mismatch rolls the transaction back.

## Protected artifacts

Generic rules never touch:

- `config.yaml`, `environment.json`, `hparams.json`, `train.log`, `metrics.csv` or `result.json`;
- `checkpoints/**`, which has its own role-aware policy;
- every exact `experiment.required_artifacts` path and its subtree;
- paths matched by `retention.protect`;
- retention ZIPs, locks, journals, manifests and quarantine metadata;
- suite-level `aggregate/**` and summary artifacts;
- symlinks, junctions/reparse points, special files or anything outside the run;
- files not matched by exactly one rule.

If two rules select the same regular file, planning fails before mutation. Include/exclude patterns
are relative POSIX globs evaluated from each run directory.

## Checkpoint roles

`trainer.checkpoint_policy` controls what Lightning creates while training.
`retention.checkpoints` independently controls what survives after successful final aggregation.
`prune_unselected: false` keeps every checkpoint regardless of `keep`.

When pruning is enabled, `best` and `last` resolve only inside the current run's `checkpoints/`
directory. The resolver supports recorded paths, moved-run rebasing and LambdaForge/Lightning
`best-*`, `last.ckpt` and `epoch-*` conventions. `last` for an `all` training policy is the
unambiguous greatest generated epoch. If a requested role is absent or ambiguous, checkpoint
pruning for that whole run is skipped. There is deliberately no `none` retention choice.

Pruned recorded paths are removed atomically from `result.json`. Completion and resume inspect safe
local files, so `checkpoint_policy: all` remains reusable even when Lightning provides no best/last
callback path. `CheckpointChoice.AUTO` loads best, then last, then the latest safe local checkpoint;
exact `BEST` and `LAST` choices do not silently cross roles.

## Rules and archives

A `compress` rule streams selected files into one ZIP per run and effective compression level.
Published names derive from the configured base, level and plan fingerprint, for example
`.lambdaforge/retention/artifacts-l9-0123456789ab.zip`. Keeping policy-owned files below the
per-run internal directory prevents a later rule or archive-name change from selecting an older
archive. Archives are immutable and never overwrite an existing path. Zip64 is enabled. Before
originals move, LambdaForge reopens the ZIP and checks CRCs, member names, sizes and SHA-256 values.
An internal manifest records the plan and member hashes.

With `only_if_smaller: true`, members whose compressed payload is not smaller remain in place. If
every member is optional and the complete ZIP including metadata is not smaller than the originals,
no archive is published and all originals remain.

A `prune` rule archives nothing. Its files still move to reversible quarantine before the commit
marker. Generic compression and pruning never select checkpoints.

## Object API and CLI

```python
from lambdaforge import Experiment, LambdaForge

experiment = Experiment.from_yaml("experiment.yaml")
plan = experiment.preview_retention()       # strictly read-only
print(plan.status, plan.operations)

result = experiment.apply_retention()       # explicit mutation request
print(result.status, result.reclaimed_bytes, result.archives)

same_plan = LambdaForge.preview_retention("experiment.yaml")
```

Typed `ArtifactRetentionPlan` and `ArtifactRetentionResult` objects retain mapping/JSON
compatibility, reject mutation and expose stable enum statuses.

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
lambdaforge retain experiment.yaml --apply --json
```

The command is preview-only unless `--apply` is present. Preview returns 0 only for a ready plan.
Apply returns 0 for `applied` or `already_applied`; not-ready, conflict and partial outcomes return
1. Syntax errors return 2.

## Transaction and recovery

Training owns an exclusive suite activity lock while workers are active. Final aggregation owns a
shared activity lease and an exclusive aggregation lock. Retention acquires exclusive activity,
aggregation and transaction locks in that order, so two LambdaForge processes cannot train,
publish aggregates or prune the same suite concurrently. OS-owned file locks are released after
normal or abrupt process termination and live handles are never spawned.

Application uses a durable journal:

1. write a synced `prepared` journal;
2. stream, close, sync and verify immutable ZIPs;
3. revalidate every source fingerprint;
4. atomically rename selected sources into suite-local quarantine;
5. publish the `committing` marker;
6. update checkpoint metadata, purge quarantine and publish immutable/latest results;
7. refresh the aggregation receipt and remove the journal.

A restart before `committing` restores quarantined sources and removes transaction ZIPs. A restart
after `committing` finishes forward. Conflicting copies, unreadable journals, replaced files and
unsafe paths are preserved and reported; LambdaForge never guesses which copy to delete. A
rolled-back plan may be retried. Reapplying a committed plan returns `already_applied` without a
second ZIP.

## Produced artifacts

```text
<suite>/
├── .lambdaforge/
│   ├── activity.lock
│   ├── aggregation.lock
│   └── retention.lock
├── <variant>/seed=<seed>/
│   └── .lambdaforge/retention/
│       └── artifacts-l<level>-<plan-prefix>.zip
└── aggregate/
    ├── aggregation_receipt.json
    ├── retention/
    │   ├── <plan-id>.json
    │   └── latest.json
    └── summary.json
```

`summary.json` starts with `status: not_applied` and a null manifest. Only a committed transaction
updates it atomically with the real latest manifest, status and plan identifier. Results list every
planned operation, its state, selected/reclaimed bytes, archive hashes, warnings and errors.
Rollback history uses a status-suffixed immutable result and cannot block a later commit.

## Limits

- Retention currently targets local filesystems and regular files. Remote/object-store backends
  need their own atomicity and lease contracts.
- ZIP/Deflate is the only archive codec. Checkpoints are pruned or retained, never compressed.
- Open/read-only files can make an operation roll back where the platform prevents rename/removal.
- Preview is deliberately non-locking and can become stale; apply rebuilds and revalidates its plan
  under locks.
- YAML remains trusted configuration. Patterns cannot escape a run, but configured Python targets
  and plugins retain the broader trusted-code boundary.
