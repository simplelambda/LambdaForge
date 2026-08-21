<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="icons/lambdaforge-light.svg">
    <source media="(prefers-color-scheme: light)" srcset="icons/lambdaforge-dark.svg">
    <img src="icons/lambdaforge-dark.png" width="140" alt="LambdaForge logo">
  </picture>
</p>

# LambdaForge

**Reproducible AI workflows, from a laptop to GPU clusters.**

LambdaForge is an installable Python framework for research built on PyTorch and Lightning. It
turns YAML or Python descriptions into validated tasks, preprocessing pipelines, immutable
datasets, experiments, adaptive hyperparameter searches and durable local, SSH or SLURM jobs.
Your project keeps ownership of its models and data; LambdaForge supplies the execution,
provenance, reuse, result-management and safety machinery around them.

> **Status:** pre-1.0. Current YAML and documented public imports are supported, but minor
> releases may deliberately simplify APIs before 1.0. The repository currently has no licence
> file, so redistribution terms have not yet been granted.

## Why LambdaForge?

- Validate configurations and imports before spending GPU time.
- Run the same scientific definition locally or through SSH/SLURM.
- Reuse only results, stages and datasets whose identity and hashes match.
- Resume durable jobs without keeping a terminal or central server alive.
- Build immutable, versioned datasets through ordinary workflow stages.
- Compare seeds and variants without guessing which duplicated run is authoritative.
- Scale from finite sweeps to adaptive, resource-aware multi-fidelity HPO.
- Extend models, losses, metrics, tasks, callbacks and data code with normal Python imports.

```text
project code + YAML + logical data
              │
              ▼
      validate and materialize
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
    tasks   datasets  experiments/HPO
      └───────┼────────┘
              ▼
     local / SSH / SLURM jobs
              ▼
 results + artifacts + provenance
```

## Install

Python 3.10 or newer is required. Give each consuming project its own environment; do not copy
`src/lambdaforge`, share this repository's `.venv`, or patch `PYTHONPATH`.

```bash
cd /path/to/research-project
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .                # makes my_project.* importable
python -m pip check
lf --version
```

The consumer's `pyproject.toml` must allow the LambdaForge release being installed. For example,
`lambdaforge[parquet]>=0.10,<0.11` accepts compatible 0.10 releases, whereas `<0.10` rejects them.
Pip can finish an editable upgrade while warning that an already-installed consumer
is now incompatible; that environment is not healthy. Update the consumer bound deliberately,
reinstall it and require `python -m pip check` to succeed. Remote managed installation resolves both
wheels from scratch and therefore refuses an incompatible pair rather than ignoring its metadata.

For a reproducible installation, build and install a versioned wheel instead of an editable path:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-*.whl
```

The consumer project should select a PyTorch wheel compatible with its target hardware. Optional
extras include `hpo`, `adaptive-hpo`, `s3`, `parquet`, `onnx`, `viz`, `graph`, `viz3d`, tracking
providers, `cluster-password` and `dev`.

## Quick start

Generate a small consumer project and inspect what LambdaForge will execute:

```bash
lf init my-research-project --template preprocessing
cd my-research-project
python -m pip install -e .
lf validate experiments/preprocessing.yaml
lf inspect experiments/preprocessing.yaml --resolved
lf plan experiments/preprocessing.yaml
lf run experiments/preprocessing.yaml
```

`validate` checks structure and Python contracts without running work. `inspect --resolved` shows
the strict configuration produced from friendly YAML. `plan` is a read-only dry-run. `run` performs
the work and writes a content-addressed result with provenance.

A concise preprocessing task looks like this:

```yaml
name: normalize-records
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  key_field: id
  workers: 4
  workload: io
resources: {cpus: 4, memory: 8GiB, time: 30m}
```

`resources` is the scheduler request, so duplicate CLI flags are unnecessary; explicit flags are
optional per-field overrides. A workflow/dataset top-level mapping is exact. Without one,
LambdaForge derives safe CPU/RAM/GPU concurrency and total time from stage/node requests, DAG levels
and `max_parallel` because the complete DAG runs inside one fixed allocation.

YAML is trusted input: `target` constructs an importable Python object, `ref` imports a callable or
value, and `params` contains constructor arguments. Install the consumer package before validation
so paths such as `my_project.preprocessing.normalize_record` resolve normally.

## Core workflows

| Goal | Command |
|---|---|
| Validate without execution | `lf validate CONFIG` |
| See the strict materialized definition | `lf inspect CONFIG --resolved` |
| Preview resources, placement and actions | `lf plan CONFIG [--on CLUSTER]` |
| Run any configuration, including a dataset recipe | `lf run CONFIG [--on CLUSTER]` |
| Discover project configurations | `lf configs list` |
| Inspect experiments, runs and results | `lf experiments list/show/runs/results` |
| Inspect datasets or build by discovered name | `lf datasets list/show`; `lf datasets build NAME` |
| Reconnect to work | `lf jobs list`; `lf jobs show latest`; `lf jobs logs JOB --follow` |
| Watch all live work interactively | `lf top` (or `lf overview --json` for software) |
| Inspect cluster readiness | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Diagnose a failed command | rerun with `--debug`; use `--json` for automation |
| Audit and compare results | `lf results list`; `lf results compare A B --metric METRIC` |
| Inspect or plot evidence | `lf artifact inspect PATH`; `lf plot learning RUN --metric METRIC` |
| Preview safe cache collection | `lf storage gc --on CLUSTER` |
| Generate shell completion | `lf completion bash|zsh|fish` |

`lf` and `lambdaforge` are identical entry points. Commands that can remove data or collect cache
are preview-first and require an explicit `--apply` after review.

The normal hierarchy is `experiment → revision → execution → run → attempt → job`. Scientific
changes create a revision; cluster placement does not. The same active revision on the same cluster
is refused unless `--allow-duplicate` is explicit, while another cluster is allowed. `--rerun`
repeats terminal science, `--restart` discards continuation state, the default resumes compatible
state, and `jobs retry` creates an attempt only after failure, cancellation or timeout.

## Errors are diagnostics

Normal CLI failures explain what happened, why, impact, repair, next command and saved diagnostics.
A preflight error means no job was submitted; `EXECUTION FAILED` means code started. Dataset and
workflow reports distinguish one root `FAILED` stage from dependent `BLOCKED` stages.

Human diagnostics use stderr. Add `--json` for the stable category/exit/retry/context envelope and
`--debug` for internals; `--verbose` is operational detail. Redacted boundary records live below
`$XDG_STATE_HOME/lambdaforge/logs/errors`. See
[Understanding errors and diagnostics](docs/MANUAL.md#understanding-errors-and-diagnostics).

## Datasets, clusters and jobs

A managed dataset has four deliberately separate concepts:

```text
DatasetRecipe → DatasetBuild → DatasetVersion → DatasetPlacement
   how            execution       immutable         where verified
```

Recipe stages use the ordinary Workflow DAG, can reuse verified content-addressed outputs and
publish only after the final index and assets pass validation. The Registry owns managed
placements; a DataCatalog remains available for external or institutionally managed data.
`datasets list` prints the copyable `name@version` selector accepted by `datasets show`; an
unversioned name is accepted only when exactly one version exists.

Explicit local inputs of at most 10 MiB are content-hashed and copied automatically into the
execution bundle, including inputs declared inside an embedded dataset-recipe stage. Users never
copy them into hashed job directories. Larger inputs are refused before submission: register a
location for the target cluster in a DataCatalog, materialize a managed DatasetVersion, or perform
an explicit preview-first/institutional transfer instead of hiding hundreds of gigabytes in a
control bundle.

Remote execution is explicit. Register a cluster, diagnose it, preview the submission, then run:

```bash
lf clusters add atlas --host atlas-login --scheduler slurm --workspace /scratch/me/lf
lf doctor --on atlas
lf clusters bootstrap atlas --dry-run
lf clusters bootstrap atlas
lf run experiment.yaml --on atlas --dry-run
lf run experiment.yaml --on atlas
lf top
lf jobs logs latest --follow
```

A normal remote `run` or `datasets build` returns a durable job ID after a quick local hand-off; it
does not hold the terminal while LambdaForge builds wheels, hashes/copies bounded inputs, prepares
the remote environment and contacts the scheduler. The job is honestly reported as `preparing`
until the provider acknowledges it. Inspect `metadata.submission_phase` with `lf jobs show JOB
--json`, or use `lf top` to follow every cluster and job. Use `--wait-for-submit` only when a script
or diagnosis must synchronously wait for remote staging and scheduler acknowledgement.

`lf jobs logs JOB` labels three different kinds of evidence: LambdaForge lifecycle events,
submission-worker diagnostics and scientific output from the consumer. With `--follow`, durable
phase/state changes appear immediately and a quiet status observation appears every 30 seconds.
That observation proves that the controller/provider can still see the job; it does **not** prove
that a model, download or transform is advancing. LambdaForge workflows emit node start/end events
and generic preprocessing emits periodic completed-record checkpoints. Domain code should still
write bounded progress messages for a single long operation and flush them promptly.

The main `lf top` screen shows compact whole-cluster values and research work grouped by name,
revision and target; `v` switches to the advanced raw-job view. Clusters and work behave as one
vertical sequence: pressing `↑` on the first item selects the last cluster, and pressing `↓` on the
last cluster selects the first item. `Enter` opens the selected
cluster. Its detail screen shows clearer time charts, requested and measured personal use, and only
jobs on that cluster. `--history 120` controls the chart window. `l` opens the complete selected log
(`PgUp`/`PgDn`, `Home`/`End`, then `b`, `Esc` or `q` to return). Press `x`, then `x` again, `y` or
`Enter`, to cancel; the confirmation remains visible and does not block input. `r` refreshes and
`q` exits. Slow provider/log calls run in cancellable background processes, so navigation and exit
do not wait for a timeout.

The TUI has no private data source. `lf overview --json` includes per-job `timing` and requested
versus observed `usage`; `lf resources --all --json` includes each cluster's `personal` aggregate.
Direct jobs measure process-tree CPU/RAM/GPU memory; unsupported scheduler accounting remains
unavailable. These plus `lf jobs list --json` support wrappers. In a pipe, `lf top` prints once and
`lf top --json --follow` emits newline-delimited JSON snapshots.

Repeated submissions reuse a verified Python/CUDA/PyTorch receipt only while all host facts and
policy still match; otherwise compatibility is resolved again. SLURM first tries a safe copy-on-write
bundle clone, then a portable copy, reducing latency without sharing a mutable workspace.

OpenSSH reuses a private `ControlMaster`, avoiding authentication storms. Managed environments are
immutable user-space installations from exact wheels. `python.strategy: auto` tries compatible
Python/Conda-family runtimes, then verified micromamba; it never changes system Python, shell files,
drivers, CUDA or trust roots. Host CA trust is propagated with verification intact. Bootstrap
activates the verified environment before safely pruning only unreferenced LambdaForge environments.
Use `existing` for administrator runtimes. Migrate a legacy scalar profile with:

```bash
lf clusters set atlas python.strategy auto
lf clusters bootstrap atlas --dry-run
lf clusters bootstrap atlas
lf doctor --on atlas
```

## Python API

The small top-level facade covers common programmatic use:

```python
from lambdaforge import LambdaForge

report = LambdaForge.validate("experiment.yaml")
if not report.is_valid:
    raise ValueError(report.summary())

experiment = LambdaForge.experiment("experiment.yaml")
plan = experiment.inspect()
results = experiment.run()
```

Extension contracts and domain APIs live in documented namespaces such as `lambdaforge.tasks`,
`lambdaforge.preprocessing`, `lambdaforge.data`, `lambdaforge.training`, `lambdaforge.metrics`,
`lambdaforge.nn`, `lambdaforge.hpo`, `lambdaforge.controlplane`, `lambdaforge.diagnostics` and
`lambdaforge.artifacts`. Import from those namespaces, not private implementation modules.

## Documentation

- [Complete user and maintainer manual](docs/MANUAL.md)
- [Agent operating instructions](AGENTS.md)
- [Release history](CHANGELOG.md)
- [Security policy and threat model](SECURITY.md)

The README is intentionally a landing page. The manual is the single canonical explanation of
configuration, tasks, preprocessing, datasets, experiments, HPO, workflows, clusters, jobs,
results, storage, extensions, security and internal architecture.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src/lambdaforge
python -m pytest -q
python -m build
python -m twine check dist/*
```

For a release, change the version only in `src/lambdaforge/_version.py`; setuptools reads that same
constant for wheel/sdist metadata. Changelog headings remain historical release records.

See the maintainer section of the [manual](docs/MANUAL.md#19-architecture) before changing identity,
dataset publication, process control, storage deletion, transport or scheduler boundaries.
