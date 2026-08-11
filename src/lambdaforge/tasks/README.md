# LambdaForge generic tasks

[Repository guide](../../../README.md) · [Español](README.es.md)

Generic tasks are LambdaForge's unit for reproducible work that is not necessarily model training:
preprocessing, downloads, feature extraction, inference, evaluation, export, figures or any other
batch operation. They use an independent strict Schema and reuse the framework's object factory,
environment/plugin provenance, atomic results, attempt history and ResultCatalog.

## Start here

A task has four visible parts: declared **inputs**, the configured Python **task object**, returned
**outputs/metrics**, and verified **artifacts** on disk. Input contents and configuration determine
the task fingerprint. Outputs are small JSON-compatible values for downstream logic; artifacts are
files/directories with hashes. `required_artifacts` is an additional success condition, not a list
of files LambdaForge creates automatically.

Use a task for one bounded operation. Use an experiment when Lightning training semantics are
needed, and a workflow when several complete operations depend on one another. The minimal example
below assumes `my_project.preprocessing.SurfaceTask` belongs to an installed consumer package.

Version 0.5 also accepts concise authoring. `inspect --resolved` shows the strict task generated
from it; validation and execution still use the same runner:

```yaml
name: prepare-surfaces
inputs: {raw: data/raw}
outputs: {surfaces: surfaces}
task:
  target: my_project.preprocessing.SurfaceTask
  params: {resolution: 1.0}
resources: {cpus: 4, memory: 8GiB}
```

## Minimal task

```yaml
schema_version: "1.0"
kind: task
name: prepare-surfaces
output_root: runs/tasks
resume: true
rerun_completed: false
inputs:
  - name: raw
    path: data/raw
required_artifacts: [surfaces]
task:
  target: my_project.preprocessing.SurfaceTask
  params: {resolution: 1.0}
execution: {mode: sequential}
metadata: {purpose: Build reusable molecular surfaces.}
```

Paths in `inputs` are resolved relative to the YAML file and content-hashed before planning. File
or directory bytes therefore participate in the task fingerprint; changing raw data selects a new
run directory even when its path is unchanged. Declare every local scientific input. Paths created
by the task stay below its fingerprinted run directory.

The existing commands dispatch by `kind`:

```bash
lambdaforge validate task.yaml
lambdaforge inspect task.yaml
lambdaforge run task.yaml --dry-run
lambdaforge run task.yaml
lambdaforge results task.yaml --write-index --fail-on-ambiguous
```

Validation checks Schema 1.0, imports, plugin contracts and the top-level constructor signature
without instantiating the task or creating outputs. Inspect and dry-run return the same immutable
`TaskExecutionPlan` and never import/construct user code.

## Python contract

```python
from lambdaforge.tasks import ArtifactDeclaration, ArtifactType, Task, TaskContext, TaskOutput


class SurfaceTask(Task):
    def __init__(self, resolution: float) -> None:
        self.resolution = resolution

    def run(self, context: TaskContext) -> TaskOutput:
        raw = context.input("raw")
        output = context.output("surfaces", create=True)
        output.mkdir(exist_ok=True)
        # Project-specific work reads raw and writes below output.
        return TaskOutput(
            outputs={"surface_dir": "surfaces"},
            metrics={"resolution": self.resolution},
            artifacts=[ArtifactDeclaration("surfaces", kind=ArtifactType.DATASET)],
        )
```

`context.input(name)` resolves and verifies a declared logical input. `context.output(name,
create=True)` resolves a configured relative output below the run directory and creates its parent.
Legacy path methods remain for strict pre-0.5 tasks. A task must never bypass these methods to infer
another run's fingerprint directory.

Inheritance is recommended for plugins but optional for a `target`: an external object with
`run(context)` is accepted, and a zero-argument `run()` is the concise duck-typed form. It may
return `TaskOutput`, a mapping (treated entirely as `outputs`) or `None`. Use `TaskOutput` whenever
metrics, artifacts or metadata are needed.

`TaskContext` supplies the attempt ID, task fingerprint, YAML source directory, materialized input
descriptors, resume flag, cancellation state and safe `input_path`/`output_path` helpers. A task
must not write outside `run_dir` and then declare the external path as an artifact.

## Results, artifacts and resume

One task identity is stored under:

```text
<output_root>/<task-name>/<fingerprint-prefix>/
├── config.yaml
├── environment.json
├── task.log
├── events.jsonl
├── result.json
└── .lambdaforge/attempts/result-<attempt-id>.json
```

`TaskResult` contains status, timestamps, duration, structured error, outputs, scalar metrics,
metadata and materialized `TaskArtifact` entries. Every artifact is run-relative, cannot escape
through `..` or symlinks, must exist, and records its role, byte size and deterministic SHA-256.
Directory hashes include ordered relative names and contents. `events.jsonl` is the append-only
structured start/finish stream; failed finishes include a conservative failure category. It
complements the human `task.log` and is not a result authority.

A matching successful result is skipped only while every recorded artifact still has the same
digest and every `required_artifacts` path exists. `rerun_completed: true` creates a new attempt;
the previous terminal result is archived before execution. Task results deliberately retain the
common experiment identity fields, so the existing `ResultCatalog` and `results` command can audit
both families without a second source of truth.

One task attempt executes locally and sequentially. Compose several task/experiment documents with
`kind: workflow`; the workflow runner bounds ready nodes while every node retains this
`TaskExecutionPlan`, fingerprint and resume contract. Training sweeps additionally support explicit
CPU/GPU process slots.

## Plugins and security

Reusable task classes can be published under `lambdaforge.tasks` and selected with:

```yaml
task:
  plugin: {kind: task, name: surface_builder}
  params: {resolution: 1.0}
```

Task plugins must subclass `lambdaforge.tasks.Task`. Fully qualified targets may use duck typing.
Both mechanisms import trusted Python code and are not a sandbox; never execute an untrusted YAML.
