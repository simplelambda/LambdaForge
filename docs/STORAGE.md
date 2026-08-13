# Internal storage and conservative garbage collection

[Español](STORAGE.es.md) | English | [Root guide](../README.md)

## 1. Categories

`storage status` reports exact roots, bytes and file counts for state, bundles, environments,
shared package cache, job workspaces, temporary files and the configured dataset root. These
categories have different lifecycle rules:

| Category | Meaning | Default GC |
|---|---|---|
| state | Small registries and pointers | never |
| bundles | Immutable, reconstructible execution input | stale/incomplete if unreferenced |
| environments | Content-addressed verified venv | stale/incomplete if unreferenced |
| package cache | Reusable pip downloads | reported; provider policy |
| job workspaces | State, logs and mutable scientific work | never automatically |
| temporary | Incomplete reconstructible cache work | eligible |
| datasets | Scientific data placements | never |

Results and retained checkpoints are scientific evidence, not cache.

```bash
lambdaforge storage status
lambdaforge storage status --on atlas
lambdaforge storage status --all --json
lambdaforge storage gc --on atlas          # preview
lambdaforge storage gc --on atlas --apply
lambdaforge environments list --on atlas
lambdaforge environments show ENV --on atlas
```

## 2. GC safety

GC starts with a plan. It checks active/queued job bundle and environment references, requires an
exact descendant of a configured internal root, refuses symlinks and never accepts dataset/result
roots. Apply deletes only the displayed reconstructible candidates. Cache age/budget settings are
optional; absence means no aggressive age deletion.

Collectors use one cross-process lock. GC also fails closed while an environment-build marker is
present, so it cannot race `pip install` or atomic publication. Investigate a leftover marker only
after proving that no bootstrap is active.

An environment build uses `.env-ID.tmp-...`, installs exact wheels with a shared pip cache,
verifies LambdaForge/Torch/CUDA and only then atomically renames to `env-ID`. Failures remove the
exact temporary path when safe. A complete marker makes an environment reusable. The framework
keeps framework/consumer/dependencies in one exact identity: 0.6 deliberately does not split a
heavy runtime layer when doing so could make project dependencies ambiguous.

## 3. Quota planning

Put small state on reliable home storage and reconstructible heavy cache/work on scratch. Put
datasets on project/shared storage and register their placements. Both bytes and file counts matter
on HPC systems. `cache_max_size` and `cache_max_age` express policy, but GC remains preview-first;
they do not authorize deletion of science.
