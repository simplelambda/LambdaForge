# LambdaForge agent guide

This is the low-token source of truth for agents using or modifying LambdaForge 0.12.0. Spanish is
in `AGENTS.es.md`. Read the relevant section of `docs/MANUAL.md` only when more detail is needed,
then inspect the public signature or implementation being changed. Current tests and code override
assumptions.

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
| Low-level jobs | `lf jobs list/show/logs/cancel/retry/delete`; `lf jobs clear [--apply]` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Results | `lf results list/show/compare` |
| Runtime diagnosis | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Preview cache cleanup | `lf clean [--on CLUSTER]`; add `--apply` after review |
| Scaffold | `lf init DIRECTORY` |

Append `--json` for automation and consume stable fields; use `--debug` only for framework
tracebacks. Local and remote run both return after durable asynchronous preparation unless
`--wait-for-submit` is explicit; `--dry-run` is direct and read-only. Never parse prose or secrets.

## Writing Work

```python
from pathlib import Path
import lambdaforge as lf

class Example(lf.Work):
    def run(self, source: Path, limit: int = 100) -> dict[str, int]:
        rows = source.read_text().splitlines()[:limit]
        report = self.outputs.file("report", filename="report.json", role="report")
        report.write_json({"rows": len(rows)})
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
- managed services: `outputs.file/directory/value/dataset`, `metrics.log/log_many`,
  `checkpoints.file/exists/save_json/load_json`, `cache.put/get/file/fetch/rate_limit`,
  `tools.require/run`, `progress.update`,
  `log(message, level=...)`;
- owned paths: `run_dir` is durable per Attempt; `temp_dir` is ephemeral;
- intra-job concurrency: `map(items, function, workers=..., executor=..., retries=...)` is ordered
  and has no persistence; `resume_map(..., key=..., validate=...)` explicitly stores safe
  dependency-aware JSON checkpoints. Legacy `map(..., key=...)` delegates to `resume_map`.

`return` is the primary JSON result. Registered names cannot collide. Never pickle results, mutate
planning inputs, construct hidden paths, create fingerprints, or expose Registry/scheduler services
to scientific code.

Cache is reconstructible; checkpoints are resumable Run state; outputs are durable evidence. Work
code should not manage cache directories, `.part` names, locks, `fsync` or `os.replace`. Prefer
`cache.put/get` for bytes/text/strict JSON, `cache.file/fetch` for path-like content, managed
checkpoint files and managed outputs; `cache.path`, `checkpoints.path`,
`run_dir` and `outputs.artifact` are advanced interoperability escapes, not examples for ordinary
code. A `ManagedFile` is path-like but resumable-map state stores only its logical key/SHA/size.
`resume_map` must validate those dependencies and selectively rerun an invalid item after cache cleanup. Never
serialize machine paths or arbitrary pickle as scientific state.

`outputs.file/directory(..., publish_to=PATH)` optionally publishes a verified copy after successful
finalization and refuses different existing content unless `overwrite=True`. Relative destinations
start at the authored YAML directory. Remotely they require the cluster `project_root` mirror and
map to the same project-relative directory; otherwise use an explicit absolute remote path. Never
describe publication as an automatic transfer back to the controller.

For remote `{file: PATH}` inputs, content up to 10 MiB is bundled automatically. A larger path must
belong to the local `pyproject.toml` project and have an exact counterpart below the cluster's
absolute `project_root`; LambdaForge verifies kind/size/SHA-256 before submission and in the worker,
and never syncs or deletes that researcher-owned mirror. Configure with
`lf clusters set NAME project_root /absolute/remote/project`, then run `lf doctor --on NAME`.
Missing/stale/symlinked content fails closed. Use managed datasets for reusable very large data,
because exact mirror verification reads every byte on each submission.

Project-native tools are declared once in `[tool.lambdaforge.environment]` with `manager="conda"`,
exactly one project-contained `file` or `lockfile`, and bare `required_executables`. Use
`lf clusters bootstrap NAME --project . --dry-run`, review, then apply. Environment files accept
only name/channels/string dependencies; no pip mappings, variables, prefixes, hooks or Torch/CUDA.
Offline native provisioning requires a platform-specific `@EXPLICIT` SHA-256 lock and matching
`package_cache`; pip/Torch also needs the cluster wheelhouse to be fully offline. Managed creation
uses one immutable Conda prefix, exact inventory and verified tools. `tools.require` checks that
prefix and never installs. Pip-only and `environment: existing` behavior is unchanged.

Use `tools.require(..., version_args=...)` and `tools.run(argv, ...)` for external executables.
Commands are argv, never shell strings; child thread variables and environment overrides remain
scoped, output reaches Work logs, and tool provenance belongs only in `environment.json`.

## Clustering contract

Use `lambdaforge.clustering` for non-neural clustering. `KMeans`, `MiniBatchKMeans`, `DBSCAN`,
`HDBSCAN` and `Agglomerative` share `Clusterer.cluster(X) -> ClusteringResult`; sklearn is lazy and
optional through `lambdaforge[clustering]`. Do not add a factory/registry, expose backend estimators,
or create clustering-specific distance types. Reuse `lambdaforge.nn.distances.Distance`, enforce
the algorithm capability table and guard custom precomputed matrices before O(N²) allocation.
KMeans and Ward are Euclidean; reject invalid combinations with the scientific reason. Do not hide
scaling, imputation, PCA, thresholds or stability policy in clusterer defaults.

`print()` and standard Python logging are captured by Job logs. Use `self.log()` for timestamped,
immediately flushed human narration, `self.progress.update()` for completion and `metrics.log()` for
scientific numeric evidence. In `lf top`, Enter/right drills from Work to numbered Attempt to logs
and left backs out; the primary TUI hides long Job IDs. `d` deletes one confirmed terminal
selection and `D` clears confirmed terminal history while preserving active Jobs. Automation uses
`lf overview --json` (`work.items[].attempt_history` retains Job IDs), ordinary `lf logs` and
preview-first `lf jobs clear [--apply]` instead of parsing the TUI. Local provider paths belong to
the durable Job and must never be recomputed from the observer's current directory.
Attempt logs in `lf top` refresh automatically. Failed Work logs append persisted exception
type/message/phase/result path after any requested tail; request `--verbose` or `--debug` for the
traceback, or `--json` for structured `failure`/`failures`. Cache fetch retries incomplete HTTP/
chunked/gzip transfers and transient statuses from clean unpublished temporaries; never add a
consumer-side retry workaround. Human bootstrap progress is stderr-only and machine JSON is clean.

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

## Neural component route

Before adding a model, inspect `lambdaforge.nn.models` and manual section 14. Existing families
include MLP/CNN, extensive graph and equivariant models, sequence/Transformer/Conformer, sets,
tabular, vision, composition, generative, scientific/implicit and differentiable-tree models.
Do not add a generic `GNN`, redundant alias/factory or domain policy; add only a reusable primitive
with a precise tensor contract and focused conformance tests.

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
2. Update `docs/MANUAL.md` and `docs/MANUAL.es.md`, both READMEs, both AGENTS files, schema/examples
   when affected and changelog for user-visible work.
3. Audit dynamic imports before deleting ordinary domain helpers, but delete obsolete execution
   adapters/tests rather than preserving compatibility.
4. Keep base dependencies light and providers lazy.
5. Test the smallest changed subsystem, then ruff, mypy, broader retained suites, wheel build and an
   installed-wheel CLI/scaffold smoke. CUDA tests must be run or explicitly reported as unavailable.
