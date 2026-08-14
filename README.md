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

> **Status:** `0.7.1` (pre-1.0). Current YAML and documented public imports are supported, but minor
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

For a reproducible installation, build and install a versioned wheel instead of an editable path:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-0.7.1-py3-none-any.whl
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

YAML is trusted input: `target` constructs an importable Python object, `ref` imports a callable or
value, and `params` contains constructor arguments. Install the consumer package before validation
so paths such as `my_project.preprocessing.normalize_record` resolve normally.

## Core workflows

| Goal | Command |
|---|---|
| Validate without execution | `lf validate CONFIG` |
| See the strict materialized definition | `lf inspect CONFIG --resolved` |
| Preview resources, placement and actions | `lf plan CONFIG [--on CLUSTER]` |
| Run a task, workflow, dataset recipe or experiment | `lf run CONFIG [--on CLUSTER]` |
| Discover project configurations | `lf configs list` |
| Build and inspect an immutable dataset | `lf datasets plan NAME`; `lf datasets build NAME` |
| Reconnect to work | `lf jobs list`; `lf jobs show latest`; `lf jobs logs JOB --follow` |
| Inspect cluster readiness | `lf doctor --on CLUSTER`; `lf resources --on CLUSTER` |
| Audit and compare results | `lf results list`; `lf results compare A B --metric METRIC` |
| Inspect or plot evidence | `lf artifact inspect PATH`; `lf plot learning RUN --metric METRIC` |
| Preview safe cache collection | `lf storage gc --on CLUSTER` |
| Generate shell completion | `lf completion bash|zsh|fish` |

`lf` and `lambdaforge` are identical entry points. Commands that can remove data or collect cache
are preview-first and require an explicit `--apply` after review.

## Datasets, clusters and jobs

A managed dataset has four deliberately separate concepts:

```text
DatasetRecipe → DatasetBuild → DatasetVersion → DatasetPlacement
   how            execution       immutable         where verified
```

Recipe stages use the ordinary Workflow DAG, can reuse verified content-addressed outputs and
publish only after the final index and assets pass validation. The Registry owns managed
placements; a DataCatalog remains available for external or institutionally managed data.

Remote execution is explicit. Register a cluster, diagnose it, preview the submission, then run:

```bash
lf clusters add atlas --host atlas-login --scheduler slurm --workspace /scratch/me/lf
lf doctor --on atlas
lf run experiment.yaml --on atlas --dry-run
lf run experiment.yaml --on atlas
lf jobs logs latest --follow
```

OpenSSH is the default and reuses a private `ControlMaster` socket for a bounded idle period, so
ordinary CLI use does not create an authentication storm. Managed environments are immutable
user-space virtual environments built from exact project/framework wheels. LambdaForge never
installs GPU drivers, system CUDA or Python itself. The configured cluster `python` must resolve to
Python 3.10 or newer; `doctor` reports an older interpreter as incompatible before bootstrap.

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
`lambdaforge.nn`, `lambdaforge.hpo`, `lambdaforge.controlplane` and `lambdaforge.artifacts`. Import
from those namespaces, not private implementation modules.

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

See the maintainer section of the [manual](docs/MANUAL.md#19-architecture) before changing identity,
dataset publication, process control, storage deletion, transport or scheduler boundaries.
