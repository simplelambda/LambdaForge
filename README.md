# LambdaForge

LambdaForge is a managed scientific runtime for reproducible Python research. You write a normal
Python class, LambdaForge supplies execution context, and the same YAML can run locally, through
SSH, or through SLURM. It records inputs, code, parameters, resources, metrics, outputs, artifacts,
datasets, attempts, logs and environment provenance without putting infrastructure in scientific
code.

## Install

Use a virtual environment owned by the research project. Install a release wheel in reproducible
projects, or an editable checkout while developing LambdaForge:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.12.0
python -m pip install -e .
python -m pip check
lf --version
```

`lf init my-study` creates a complete installable example. LambdaForge requires Python 3.10 or
newer. A consumer project owns its scientific dependencies and its PyTorch build; managed cluster
bootstrap resolves a compatible remote Python/Torch environment without changing system Python,
drivers or CUDA.

## First Work

A YAML file executes only a class inheriting `lambdaforge.Work`. `run()` is its single lifecycle
method and its annotated Python signature is the source of parameter names, defaults and types.

```python
from pathlib import Path

import lambdaforge as lf


class CurateDNA(lf.Work):
    """Curate source records and keep a reproducible report."""

    def run(self, public_sources: Path, workers: int = 8) -> dict[str, int]:
        records = public_sources.read_text(encoding="utf-8").splitlines()
        curated = self.map(records, str.strip, key=str, workers=workers, executor="thread")
        report = self.run_dir / "report.txt"
        report.write_text("\n".join(curated), encoding="utf-8")
        self.outputs.artifact("report", report, role="report")
        self.metrics.log("accepted", len(curated))
        return {"accepted": len(curated)}
```

```yaml
name: wisdom-dna-curate
run: wisdom.preprocessing.CurateDNA
with:
  public_sources:
    file: ../data/dna/public-sources.json
  workers: 8
resources:
  cpu: 8
  memory: 8GiB
  time: 2h
```

Only `{file: ...}`, `{dataset: ...}` and `{from: step.output}` have special meaning. A plain YAML
string is always a string. Small typed files are hashed and automatically staged for remote work;
large data should be published/materialized as an immutable dataset instead of being copied into
every bundle.

Validate and understand the Work before executing it:

```bash
lf validate experiments/curate.yaml
lf explain experiments/curate.yaml
lf run experiments/curate.yaml --dry-run
lf run experiments/curate.yaml
lf run experiments/curate.yaml --on gpu-cluster
```

Remote submission is asynchronous by default: the terminal returns after a durable preparation
record is created. Use `--wait-for-submit` only when the caller deliberately wants to wait for SSH,
environment and scheduler acknowledgement.

## Runtime services

Inside `run()`, immutable planning information lives on `self.config`, `self.inputs`,
`self.resources`, `self.seed`, `self.trial`, `self.source_dir` and `self.resuming`. Managed mutable
services are `self.outputs`, `self.metrics`, `self.checkpoints`, `self.cache` and `self.progress`.
`self.run_dir` owns durable attempt files and `self.temp_dir` is deleted after the attempt.

- `self.outputs.value(name, value)` stores small JSON evidence.
- `self.outputs.artifact(name, path, ...)` safely owns and hashes a file/directory.
- `self.outputs.dataset(name=..., version=..., members=...)` streams and atomically publishes an
  immutable dataset version.
- `self.metrics.log(...)` and `log_many(...)` append scalar histories used by results and HPO.
- `self.checkpoints.path/save_json/load_json/exists` manages state shared by compatible attempts.
- `self.map(...)` provides bounded intra-job concurrency, stable keys, progress and JSON resume.

The return value is the primary result; named outputs are separate. An unhandled exception, invalid
artifact, failed dataset publication or failed result persistence makes the Attempt fail.

## Composition and experiments

YAML supports a sequence plus explicit parallel groups. The next level waits for the complete prior
level. Each member of a parallel group uses an isolated spawned process inside the enclosing Job;
seed/search Runs for one member remain serial so the declared resource request stays absolute. A
reference requires one unambiguous producing Run.

```yaml
name: study
steps:
  - name: prepare
    run: project.Prepare
  - parallel:
      - name: model-a
        run: project.TrainA
        with:
          data: {from: prepare.dataset}
      - name: model-b
        run: project.TrainB
        with:
          data: {from: prepare.dataset}
  - name: compare
    run: project.Compare
```

`seeds: [7, 17, 27]` creates independent Runs and exposes each value as `self.seed`. `search`
expands finite values or reproducible numeric ranges; `objective: {metric: val_score, mode: max}`
ranks the exact metric logged by the Work, averaging all seeds of the same variant. Resource fields
in YAML are the scheduler request, not capacity hints.

## Observe and operate

```bash
lf top
lf overview --json
lf show WORK
lf logs WORK --follow
lf cancel WORK
lf retry WORK
lf jobs list                 # advanced scheduler/process view
lf results list
lf datasets list
lf clean                     # preview only
lf clean --apply
```

`lf top` leads with semantic Work/Execution information; job IDs remain available for low-level
diagnostics. Normal execution reuses a verified successful scientific definition. `retry` creates a
new Attempt of the same Run, checkpoints make it resumable, and `--rerun` deliberately creates a
new Execution. Cleanup is preview-first and never treats published datasets, results or checkpoints
as reconstructible cache.

For clusters, datasets, HPO, result metadata, cleanup ownership and the complete CLI, read
[the manual](docs/MANUAL.md). Agent-assisted projects should give their agent [AGENTS.md](AGENTS.md)
as the concise operational contract; this avoids costly repository crawling and prevents invented
APIs.
