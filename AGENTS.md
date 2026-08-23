# LambdaForge agent guide

This is the low-token source of truth for agents using or modifying LambdaForge 0.12.0. Read the
relevant section of `docs/MANUAL.md` only when more detail is needed, then inspect the public
signature or implementation being changed. Current tests and code override assumptions.

## One execution model

Every YAML executes one or more classes inheriting `lambdaforge.Work`. A class has one lifecycle
entry point: `run(...)`. Reject functions, non-Work classes, required constructor arguments and old
configuration fields locally before submission. Never create a second runner or translate Work YAML
to Task/Experiment/Workflow/DatasetRecipe objects.

Do not reintroduce executable Task/TaskContext/PreprocessingTask, YAML `kind`, `schema_version`,
`target/ref/params`, constructor or method selection, source/transform/sink requirements, arbitrary
DAG edges, recursive object construction, global `lf.current()/metric()/artifact()` APIs, or a
compatibility path unless the project explicitly reverses this architectural decision.

## Fast routes

| Need | Command |
|---|---|
| Validate without execution | `lf validate CONFIG` |
| Explain signature/defaults/resources | `lf explain CONFIG` |
| Read-only expansion | `lf run CONFIG --dry-run` |
| Execute | `lf run CONFIG [--on CLUSTER]` |
| Deliberate new execution | `lf run CONFIG --rerun` |
| Monitor semantic work | `lf top`; `lf overview --json` |
| Work operations | `lf show/logs/cancel/retry/delete SELECTOR` |
| Low-level jobs | `lf jobs list/show/logs/cancel/retry` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Results | `lf results list/show/compare` |
| Runtime diagnosis | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Preview cache cleanup | `lf clean [--on CLUSTER]`; add `--apply` after review |
| Scaffold | `lf init DIRECTORY` |

Append `--json` for automation and consume stable fields; use `--debug` only for framework
tracebacks. Remote run returns after durable asynchronous preparation unless `--wait-for-submit` is
explicit. Never parse prose or secrets.

## Writing Work

```python
from pathlib import Path
import lambdaforge as lf

class Example(lf.Work):
    def run(self, source: Path, limit: int = 100) -> dict[str, int]:
        rows = source.read_text().splitlines()[:limit]
        self.metrics.log("rows", len(rows))
        self.outputs.value("summary", {"rows": len(rows)})
        return {"rows": len(rows)}
```

```yaml
name: example
run: my_project.Example
with:
  source: {file: data/input.txt}
  limit: 100
resources:
  cpu: 2
  memory: 2GiB
```

Python signatures and docstrings own parameter truth. Use only the typed markers `{file: ...}`,
`{dataset: NAME@VERSION}` and `{from: STEP.OUTPUT}`. Plain strings are never guessed as paths.
Every project class must be importable from its installed consumer package.

Work properties are available only during execution:

- immutable: `name`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir`, `resuming`;
- managed services: `outputs.value/artifact/dataset`, `metrics.log/log_many`,
  `checkpoints.path/exists/save_json/load_json`, `cache.path`, `progress.update`;
- owned paths: `run_dir` is durable per Attempt; `temp_dir` is ephemeral;
- intra-job concurrency: `map(items, function, key=..., workers=..., executor=...)` preserves input
  order, requires unique stable keys and stores JSON checkpoints for retry/resume.

`return` is the primary JSON result. Registered names cannot collide. Never pickle results, mutate
planning inputs, construct hidden paths, create fingerprints, or expose Registry/scheduler services
to scientific code.

## YAML composition and studies

`steps` is a sequence. `{parallel: [...]}` is one concurrent level. The following level waits for
all members. Cross-step outputs reference `step.output` and are invalid when the producer has
multiple seeds/trials. `seeds` creates separate Runs and does not inject a `seed` argument.
`search` parameters override normal `with` values and are passed normally to `run()`. The objective
must name a scalar logged through `self.metrics`; variants with several seeds are ranked by their
mean, never by the best individual seed.

Do not confuse `self.map` concurrency inside one Work with a YAML parallel group. A parallel group
uses isolated spawned Work processes inside the enclosing Job's aggregate fixed allocation;
seed/search members of each Work definition remain serial.

## Identity, attempts and ownership

Hierarchy: Work -> Execution -> Run -> Attempt -> Job. Scientific identity includes Work import
path, consumer code identity, normalized parameters, file hashes, dataset content IDs, seed and
trial values; it excludes cluster, paths, IDs and time. Environment/hardware provenance is separate.

Normal execution reuses verified success. Retry means same Run/new Attempt; resume means compatible
checkpoints; rerun means a deliberate new Execution. Published datasets are durable independent
objects. Results/checkpoints are scientific state. Bundle/environment/cache bytes are
reconstructible. Deletion and cleanup must be exact-root, symlink-safe, idempotent and preview-first.

Dataset creation occurs only from `self.outputs.dataset(...)`; it streams members into the existing
DatasetArtifact v2/index/registry format. There is no dataset-build execution protocol. Preserve v1
manifest reads and immutable name/version conflict checks.

## Control-plane invariants

Keep `Transport` and `Scheduler` provider boundaries. OpenSSH multiplexing reuses a private
ControlMaster for its configured idle period. Credentials stay in interactive/keyring/env sources
and never enter argv, YAML, bundles, state or logs. Managed Python environments are immutable,
user-space and wheel-identified; do not modify system Python, CUDA, drivers or shell startup files.
CUDA usability requires an actual tensor probe, not `nvidia-smi` alone.

Direct/SLURM jobs retain durable state, heartbeat, logs, usage, cancellation and identity checks.
Provider outage is unknown state, not scientific failure. Do not contact a real cluster or run a
real scientific dataset while testing repository changes.

## Modification checklist

1. Keep the final path `WorkConfig -> ExecutionPlan -> ControlPlane/Scheduler -> WorkRunner ->
   Work.run()`; infrastructure below Work may be reused.
2. Update `docs/MANUAL.md`, README examples, this file, schema and changelog for user-visible work.
3. Audit dynamic imports before deleting ordinary domain helpers, but delete obsolete execution
   adapters/tests rather than preserving compatibility.
4. Keep base dependencies light and providers lazy.
5. Test the smallest changed subsystem, then ruff, mypy, broader retained suites, wheel build and an
   installed-wheel CLI/scaffold smoke. CUDA tests must be run or explicitly reported as unavailable.
