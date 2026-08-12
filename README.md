<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" type="image/svg+xml" srcset="icons/lambdaforge-light.svg">
    <source media="(prefers-color-scheme: dark)" type="image/png" srcset="icons/lambdaforge-light.png">
    <source media="(prefers-color-scheme: light)" type="image/svg+xml" srcset="icons/lambdaforge-dark.svg">
    <source media="(prefers-color-scheme: light)" type="image/png" srcset="icons/lambdaforge-dark.png">
    <img src="icons/lambdaforge-dark.png" width="140" alt="LambdaForge logo">
  </picture>
</p>

# LambdaForge

[Español](README.es.md) · English

LambdaForge is SimpleLambda's object-oriented framework for reproducible AI work. It combines
generic tasks, composable preprocessing, PyTorch, Lightning and a YAML engine behind one stable
Python package, so a research project can focus on its data and science instead of rebuilding
pipelines, training loops, provenance, result management and process scheduling.

> **Status:** `0.5.2`, usable but pre-1.0. The public namespaces documented below are the intended
> API; compatibility is not yet guaranteed between minor releases. The repository does not yet
> contain a licence file, so redistribution terms still need to be chosen by SimpleLambda.

## 0. Contents

- Getting started
  - [1. What LambdaForge provides](#1-what-lambdaforge-provides)
  - [2. Installation](#2-installation)
  - [3. Integrating into another project](#3-integrating-into-another-project)
  - [4. Quick start](#4-quick-start)
  - [5. Plain-language glossary](#5-plain-language-glossary)
  - [6. Friendly authoring and strict IR](#6-friendly-authoring-and-the-strict-internal-model)
- Core concepts
  - [7. Generic tasks and preprocessing](#7-generic-tasks-and-preprocessing)
  - [8. Scientific identity and reuse](#8-scientific-identity-reuse-and-explicit-reruns)
  - [9. Workflows and composition](#9-workflows-and-configuration-composition)
- Execution and data
  - [10. Multi-cluster control plane](#10-local-and-multi-cluster-control-plane)
  - [11. Persistent jobs and data placement](#11-persistent-jobs-and-data-placement)
  - [12. Operations and HPO](#12-inference-evaluation-export-and-hpo)
  - [13. Resources, backends and reliability](#13-resources-backends-and-reliability)
- Results and inspection
  - [14. Results, plots and artifacts](#14-artifact-stores-registry-and-reports)
  - [15. Observability and reproducibility](#15-observability-and-reproducibility)
- Reference and extension
  - [16. CLI reference](#16-cli-reference)
  - [17. Public API](#17-public-api)
  - [18. Conceptual model](#18-conceptual-execution-model)
  - [19. Architecture](#19-architecture)
  - [20–26. YAML, execution, outputs, components and extensions](#20-yaml-experiment-reference)
- Project information
  - [27. Review findings](#27-review-findings)
  - [28. Development and verification](#28-development-and-verification)
  - [29. Current limitations](#29-current-limitations)
  - [30. Why AGENTS.md exists](#30-why-agentsmd-exists)
  - [31. Documentation map](#31-documentation-map)
  - [32. Roadmap](#32-roadmap)
  - [33. Version 0.2 roadmap history](#33-02-roadmap-history)

## 1. What LambdaForge provides

- A generic Lightning training task for mapping-shaped batches, one or more losses and independent
  train/validation/test metrics.
- A separate, strict generic-task YAML family for preprocessing and other reproducible non-training
  work, with dry-run plans, content-addressed inputs, typed artifacts and attempt history.
- Composable source/transform/sink preprocessing, atomic per-record checkpoints, deterministic
  shards and a versioned `DatasetArtifact` manifest.
- Task/experiment workflow DAGs, safe YAML composition/interpolation, semantic provenance/diff and
  bounded CPU-only or heterogeneous resource planning.
- Reusable checkpoint inference/evaluation/ensemble/export tasks, finite and adaptive HPO and
  preview-first local/SLURM execution backends.
- Verified local/shared/S3-compatible artifact stores, distributed staging cache, a catalog-backed
  registry, factual reports/dashboard and structured observability/reproducibility profiles.
- Object construction from trusted YAML using fully qualified `target`/`ref` paths or installed,
  contract-checked entry-point plugins, including reusable datasets, callbacks and loggers.
- Draft 2020-12 schema validation of structure, expansion, resources and imports before execution.
- Versioned, preview-first configuration migrations with round-trip YAML and explicit atomic output.
- Cartesian parameter grids, named ablations and repeatable seed expansion.
- Sequential runs, multiple independent trainings per GPU, multi-GPU scheduling and one DDP job
  per device group.
- Cooperative cancellation, forced process-tree cleanup, Windows Job Objects, Linux parent-death
  guards and guarded PyTorch `DataLoader` workers.
- Materialised configurations, environment and run-scoped loaded-plugin provenance, logs, dense
  epoch CSV files, checkpoints, result manifests, cross-seed summaries, statistical comparisons
  and plots.
- Preview-first artifact retention with role-aware checkpoint selection, verified streaming ZIPs,
  explicit pruning rules, completion receipts and crash-recoverable transactions.
- Reusable MLP, 2D CNN, ECMP graph model, activations, normalisations, pooling operators, distances,
  binary/multiclass metrics and regression metrics.
- Bounded process-local dataset caching, multiprocess-coordinated disk/mmap quotas, explicit
  dataset/transform fingerprints, checksum or HMAC records and a safe NumPy/Torch codec, all
  selectable through the same recursive YAML object syntax.
- Fixed-memory binary and multiclass AUROC/AUPRC alternatives whose histogram resolution, averaging
  and logits policy are explicit experiment parameters.
- Optional, lazily loaded MLflow, TensorBoard and Weights & Biases logger adapters, selectable alone
  or together without adding a tracking service to the base installation.
- A small facade (`LambdaForge`), object APIs (`Experiment`, `TaskRun` and `Workflow`) and one CLI
  (`lambdaforge`).

LambdaForge is task-agnostic at the configuration and orchestration layers. A user project supplies
the domain-specific `Dataset`, optional collator and, when the default mapping contract is not
enough, its own model, task, data module or runner.

## 2. Installation

Python 3.10 or newer is required. From a clone, create an environment and install the project in
editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS activation is `source .venv/bin/activate`. Runtime dependencies include PyTorch,
Lightning, TorchMetrics, NumPy, Matplotlib, PyYAML, ruamel.yaml, JSON Schema, psutil and
threadpoolctl. `pywin32` is installed only on Windows. CUDA itself must match the installed PyTorch
build; LambdaForge does not install a CUDA driver.

The package prefers `lightning.pytorch` and retains runtime compatibility with the legacy
`pytorch_lightning` import name.

Tracking providers remain optional. Install `lambdaforge[mlflow]`,
`lambdaforge[tensorboard]`, `lambdaforge[wandb]` or the combined `lambdaforge[tracking]` extra; see
the [tracking guide](src/lambdaforge/tracking/README.md) before enabling remote publication.

Optional integrations never enter the base dependency set:

| Extra | Adds |
|---|---|
| `hpo` | Optuna finite-search adapter. |
| `adaptive-hpo` | BoTorch/GPyTorch Bayesian acquisition for adaptive HPO; Sobol/random remain available without it. |
| `s3` | Default boto3 client for `S3ArtifactStore`; an injected compatible client needs no extra. |
| `parquet` | Pandas/PyArrow registry export. |
| `onnx` | ONNX/ONNX Script model export. |
| `cluster-password` | Paramiko password SSH/SFTP plus OS keyring; default OpenSSH needs no extra. |
| `mlflow`, `tensorboard`, `wandb`, `tracking` | One tracking provider or all three. |
| `dev` | Tests, type checking and formatting tools for LambdaForge contributors. |

Install only what the consumer actually uses, for example
`python -m pip install "lambdaforge[adaptive-hpo,s3]"`.

## 3. Integrating into another project

LambdaForge is an installable library, not a source tree that must be copied into each study. Give
the consuming project its own virtual environment and install both projects into that environment.
For local framework development:

```bash
cd /path/to/my-research-project
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
python -m pip check
python -c "import lambdaforge; print(lambdaforge.__version__)"
```

The final `pip install -e .` is important: it makes paths such as
`my_project.models.ProjectModel` importable when LambdaForge resolves YAML. A normal consumer layout
is:

```text
my-research-project/
├── pyproject.toml
├── experiments/baseline.yaml
├── src/my_project/
│   ├── datasets.py
│   ├── losses.py
│   └── models.py
└── tests/
```

For a stable/offline installation, build a wheel in the LambdaForge checkout and install that
immutable artifact instead of an editable path:

```bash
python -m pip wheel /absolute/path/to/LambdaForge --no-deps --wheel-dir dist
python -m pip install dist/lambdaforge-0.5.2-py3-none-any.whl
```

Let the consumer project's lock file or constraints select a PyTorch build compatible with its
driver before installing LambdaForge; the framework accepts an already installed compatible
`torch`. Verify the target environment rather than assuming that `nvidia-smi` implies a CUDA-enabled
Python wheel:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
lambdaforge validate experiments/baseline.yaml
lambdaforge run experiments/baseline.yaml --dry-run
```

Do not copy `src/lambdaforge`, share LambdaForge's `.venv`, or patch `PYTHONPATH`: those approaches
hide dependency and import errors and make a published experiment difficult to reproduce. For
several independent projects, use a wheel/version per environment. For reusable third-party
extensions, publish entry-point plugins; for one project, ordinary installed `my_project.*` targets
are simpler. The [extension contracts](#26-extension-contracts) show both routes.

## 4. Quick start

### The ideas you need before running a command

LambdaForge reads a **configuration file**: a plain-text YAML document that says what work should
be performed, which Python objects implement it, which parameters they receive, where results go
and which resources may be used. YAML is indentation-sensitive; use spaces, never tabs.

There are three document types:

| Document | Use it for | Example |
|---|---|---|
| **Task** | One reproducible operation that is not necessarily training. | Preprocess data, export a model or generate a report. |
| **Experiment** | Train and evaluate models across configurations and seeds. | Compare two MLP widths with three random seeds. |
| **Workflow** | Connect complete tasks or experiments into a dependency graph. | Preprocess first, then train with the produced dataset. |

There are two views of the same configuration. The **authoring configuration** is what a user
writes: it may omit `kind` and `schema_version`, use named input/output mappings and use a Python
import string where the meaning is unambiguous. LambdaForge compiles it to a **materialized
configuration**: the strict, versioned object consumed by the established runners. A **Schema** is
the list of allowed fields, types and required values. Validation catches misspellings and invalid
combinations before expensive work starts. Existing strict YAML remains valid.

Object specifications use three recurring keys:

```yaml
task:
  target: my_project.tasks.ReportTask   # import this hypothetical class and construct it
  params:                              # pass these values to its constructor
    output_name: result.json
```

- `target` is the full Python import path of a class to construct.
- `params` is the mapping passed as keyword arguments to that class.
- `ref` imports a function, class or value without constructing it; for example
  `ref: torch.optim.AdamW` passes the optimizer class to the training task.

Configurations are trusted code because their imports may execute Python. Use files you created or
reviewed, and keep project-specific objects in the separately installed consumer package.

### Run the generated example

Start with the generated task because it works without requiring a dataset or GPU:

```bash
lambdaforge init my-ai-project
cd my-ai-project
python -m pip install -e .
```

`init` creates an installable `my_project` package, a small `ExampleTask`,
`experiments/task.yaml`, editor Schema settings and a suitable `.gitignore`. Installing the project
with `-e .` makes `my_project.tasks.ExampleTask` importable while you edit it.

The generated YAML means:

```yaml
kind: task                          # select the generic-task document family
schema_version: "1.0"              # validate against task Schema 1.0
name: example                       # stable human-readable task name
task:
  target: my_project.tasks.ExampleTask
required_artifacts: [output.json]   # success requires this file to exist
```

Use the commands in this order. `inspect --resolved` is the best first learning tool because it
shows defaults and expanded shorthand without running user code:

| Command | What it answers | Starts work? | Writes results? |
|---|---|---:|---:|
| `lambdaforge validate CONFIG` | Is the YAML structurally valid, and can referenced Python objects be imported? | No | No |
| `lambdaforge inspect CONFIG --resolved` | What strict configuration did my short YAML become? | No | No |
| `lambdaforge inspect CONFIG` | What exact run(s) or task plan would be used? | No | No |
| `lambdaforge run CONFIG --dry-run` | Can the execution layer prepare the same immutable plan without launching user work? | No | No |
| `lambdaforge run CONFIG` | Execute or safely resume the planned work. | Yes | Yes |
| `lambdaforge results CONFIG` | Which attempts exist and are any successful results ambiguous? | No | Only with `--write-index` |

Now exercise the complete safe path:

```bash
lambdaforge validate experiments/task.yaml
lambdaforge inspect experiments/task.yaml --resolved
lambdaforge inspect experiments/task.yaml
lambdaforge run experiments/task.yaml --dry-run
lambdaforge run experiments/task.yaml
lambdaforge results experiments/task.yaml --write-index --fail-on-ambiguous
```

The real run creates a fingerprinted directory below `runs/tasks/example/`. It contains the
materialized configuration, environment provenance, log, event stream, `result.json` and the
declared `output.json`. Repeating the command reuses a matching successful result only while its
identity and artifact digests remain valid.

### Move from the example task to model training

An experiment uses the same workflow, but it also needs datasets, a model, losses, metrics and
trainer settings. Copy [the complete experiment example](examples/experiment.yaml) into the
consumer project and replace every `your_project.*` path with classes from its installed package.
That example is a template, not a runnable built-in dataset: LambdaForge intentionally does not
guess the meaning or shape of domain data.

```bash
cp /path/to/LambdaForge/examples/experiment.yaml experiments/baseline.yaml
lambdaforge validate experiments/baseline.yaml
lambdaforge inspect experiments/baseline.yaml
lambdaforge run experiments/baseline.yaml --dry-run
lambdaforge run experiments/baseline.yaml
```

The equivalent Python calls use the same configuration and safety boundaries:

```python
from lambdaforge import LambdaForge

experiment = LambdaForge.experiment("experiments/baseline.yaml")
report = experiment.validate()  # ValidationReport; no training
planned_runs = experiment.expand()  # one materialized mapping per variant and seed
results = experiment.run(dry_run=True)
results = experiment.run()  # list of typed RunResult values
print(results[0].status, results[0].run_dir)
```

After training, `aggregate` rebuilds cross-seed tables and plots without retraining, `results`
audits attempts, and `retain` prepares an artifact-removal/compression plan. Retention remains
read-only unless `--apply` is explicit:

```bash
lambdaforge aggregate experiments/baseline.yaml
lambdaforge results experiments/baseline.yaml --write-index --fail-on-ambiguous
lambdaforge retain experiments/baseline.yaml
```

Resource flags such as `--mode parallel` or `--gpus 0,1` override only their corresponding YAML
execution fields. Add them after the sequential configuration works; the
[execution section](#22-execution-and-process-safety) explains their process and GPU semantics.

## 5. Plain-language glossary

| Term | Meaning in LambdaForge |
|---|---|
| **Configuration** | The YAML values that describe requested work. |
| **Schema** | Machine-readable rules used to reject invalid configuration before execution. It is not a dataset schema. |
| **Task** | One reproducible non-training operation. |
| **Experiment** | A training study that may expand into several runs. |
| **Workflow** | A dependency graph whose nodes are complete tasks or experiments. |
| **Suite** | All runs produced by one experiment configuration. |
| **Variant** | One concrete base/grid/ablation hyperparameter combination. |
| **Seed** | A recorded random initialization/repetition identifier. |
| **Run** | One variant and seed executed together. |
| **Attempt** | One try at a run or task; retries keep their own terminal metadata. |
| **Fingerprint** | A deterministic hash of scientific identity used to prevent incompatible reuse. |
| **Artifact** | A verified output file or directory, such as a checkpoint, dataset or report. |
| **Checkpoint** | Saved training state used for inference or exact continuation. |
| **Aggregation** | Combining completed run metrics into cross-seed summaries, comparisons and plots. |
| **Provenance** | Recorded configuration, code/environment and plugin information explaining how output was produced. |
| **HPO** | Hyperparameter optimization: choosing promising configurations from a declared search space. |
| **Fidelity** | The cumulative training budget already given to a candidate, normally epochs. |
| **Dry-run** | A read-only execution-plan check; it never calls the configured task or starts training. |

`target`, `ref` and `plugin` are object-resolution forms, not document types. A `target` constructs
an importable class, a `ref` imports an existing object, and a `plugin` resolves a named extension
published by another installed distribution.

## 6. Friendly authoring and the strict internal model

LambdaForge 0.5 separates ease of writing from strict execution. The flow is:

```text
short YAML -> AuthoringConfig -> AuthoringConfigNormalizer -> MaterializedConfig -> existing validator/runner
```

There is still one execution engine. The authoring layer only expands safe defaults and shorthand;
it never trains, imports a configured object or invents scientific choices. Authoring Schema 1.0
is packaged as `schemas/authoring.schema.json`; the materialized task, experiment and workflow keep
their own strict Schemas. Inspect both sides with:

```bash
lambdaforge inspect experiments/prepare.yaml --resolved
lambdaforge validate experiments/prepare.yaml
```

The first command explains what will be validated; the second validates the materialized document,
imports and constructor contracts. An old strict document and its short equivalent dispatch to the
same `TaskRunner`, `ExperimentRunner` or `WorkflowRunner`.

A complete short preprocessing document is:

```yaml
name: prepare-data
inputs:
  raw: ../data/raw.jsonl
outputs:
  processed: processed
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  key_field: id
  workers: 4
  workload: io
resources:
  cpus: 4
  memory: 8GiB
  time: 30m
```

`raw` and `processed` are logical names. The compiled source calls `context.input("raw")`; the sink
calls `context.output("processed")`. Physical paths remain provenance, not the task's programming
interface. `output_path`, `input_path` and `declared_input_path` remain compatible for strict older
tasks, but new project code should use the named methods. Worker semantics are explicit:
`workers: 1` is sequential, `io` uses threads, `cpu` uses spawn-safe child processes for transforms
while the parent owns sink/manifest, `auto` conservatively uses threads, and `gpu` requires one
worker. CPU transforms must be importable/picklable on Linux and Windows. Multiple GPUs use explicit
shards/jobs. `workers`, `workload` and checkpoint cadence are operational: they do not change the
built-in task/dataset scientific identity, and equivalent modes must produce identical content.
Sample without publishing a dataset with `lambdaforge debug CONFIG --records 3`; add
`--intermediates debug/stages` only when stage artifacts are useful.

Only unambiguous object fields accept strings. For example:

```yaml
model: my_project.models.ProjectModel
```

materializes to `model: {target: my_project.models.ProjectModel}`. Arbitrary strings are never
treated as imports. Use the full `target`/`ref`/`plugin` mapping whenever parameters or import
semantics need to be explicit.

## 7. Generic tasks and preprocessing

A generic task is the smallest reproducible unit in LambdaForge: one Python object receives a
`TaskContext`, performs bounded work and returns outputs, metrics and artifact declarations. It is
the right choice whenever the operation is not a Lightning training loop. Training experiments use
experiment Schema 1.1; generic tasks use the independent task Schema 1.0. Strict task YAML declares
`kind: task`; concise YAML is detected from its `task` or `preprocess` field and materializes that
declaration automatically.

Preprocessing is a specialized task assembled from three understandable roles:

1. A **source** yields records with stable identifiers.
2. Zero or more **transforms** change each record.
3. A **sink** writes results and can verify whether an earlier record is already complete.

The bundled example reads JSON Lines and writes one atomic JSON file per record without changing its
value. It is directly runnable from this checkout and deliberately needs no project code:

```bash
lambdaforge validate examples/preprocessing.yaml
lambdaforge inspect examples/preprocessing.yaml
lambdaforge run examples/preprocessing.yaml --dry-run
lambdaforge run examples/preprocessing.yaml
lambdaforge results examples/preprocessing.yaml --write-index --fail-on-ambiguous
```

Its core YAML is a recursively constructed source → transforms → sink pipeline. An empty transform
list means “copy/serialize the records”; the commented addition shows where project logic belongs:

```yaml
schema_version: "1.0"
kind: task
name: normalize-records
inputs:
  - {name: raw, path: data/raw.jsonl}
task:
  target: lambdaforge.preprocessing.PreprocessingTask
  params:
    source:
      target: lambdaforge.preprocessing.JsonLinesSource
      params: {path: data/raw.jsonl, key_field: id}
    transforms: []
    # To transform values, install the consumer package and replace [] with:
    # - target: lambdaforge.preprocessing.CallableTransform
    #   params: {function: {ref: my_project.preprocessing.normalize_record}}
    sink:
      target: lambdaforge.preprocessing.JsonDirectorySink
      params: {output_dir: processed}
```

Every top-level input is resolved relative to the YAML and uses strict content hashing by default.
Changing raw bytes selects a new fingerprinted run directory instead of silently reusing stale output. Each
terminal attempt records configuration, environment/plugins, logs, structured errors, scalar
metrics and SHA-256 artifacts. `PreprocessingTask` additionally checkpoints stable record keys for
safe retry, supports deterministic explicit shards and publishes a content-derived
`dataset-artifact.json`.

Project-specific logic remains in the installed consumer package. Use a `CallableTransform` for a
small function or implement the public `PreprocessingSource`, `PreprocessingTransform` and
`PreprocessingSink` contracts. For other batch work, implement `Task.run(TaskContext) -> TaskOutput`
and use the same task YAML/CLI. See the [generic-task guide](src/lambdaforge/tasks/README.md),
[preprocessing guide](src/lambdaforge/preprocessing/README.md) and
[complete example](examples/preprocessing.yaml).

## 8. Scientific identity, reuse and explicit reruns

LambdaForge now models identity in three separate pieces:

| Identity | Contains | Does not contain | Why |
|---|---|---|---|
| `DatasetIdentity` | Content hash, reviewed manifest hash, generated dataset ID or explicit external version. | Mount point or cluster path. | The same data at `/mnt/data` and `/scratch/data` is still the same science. |
| `CodeIdentity` | Clean Git commit; commit plus dirty diff hash; explicit release; or installable-project version plus available source hash. | Output directories and scheduler state. | A code change cannot silently reuse an incompatible result. |
| `ExecutionIdentity` | Cluster, resource and environment policy. | Model/data/code choices. | Moving equivalent work to another cluster does not pretend the science changed. |

Local task inputs default to the safest strategy:

```yaml
inputs:
  raw:
    path: ../data/raw
    identity: strict
```

For a very large immutable dataset, avoid rescanning every byte by choosing an auditable strategy:

```yaml
inputs:
  raw:
    path: /datasets/corpus
    identity: {strategy: manifest, manifest: ../data/corpus.sha256}
# or identity: {strategy: dataset_id}
# or identity: {strategy: version, namespace: lab/corpus, version: "2026-08-11"}
```

`strict` reads every byte. `manifest` hashes a reviewed manifest. `dataset_id` reads the
content-derived ID emitted by `PreprocessingTask`. `version` trusts an external immutable version;
changing bytes without changing that version is therefore a user error. The persisted input record
contains both logical identity and physical location for auditing, but only logical identity enters
the task fingerprint.

Execution is idempotent by default. If the same scientific identity already has a verified success,
`run` returns it without doing the work again. Lifecycle flags are deliberately explicit:

| Command | Completed success | Partial state/checkpoint |
|---|---|---|
| `run CONFIG` | Reuse/skip. | Resume when enabled. |
| `run CONFIG --no-resume` | Reuse/skip. | Start a new attempt without partial continuation. |
| `run CONFIG --force` | Create a new attempt. | May resume compatible state. |
| `run CONFIG --restart` | Create a new attempt. | Do not resume; clear checkpoint intent. |

Explain identities without exposing hashes as concepts the user must manage:

```bash
lambdaforge explain changes current.yaml
lambdaforge explain changes current.yaml --against previous.yaml
```

The second form reports the exact scientific paths that changed. Internal directories remain
collision-safe and content-addressed; normal commands show names, statuses and job IDs instead of
asking the user to select a directory hash manually.

## 9. Workflows and configuration composition

A workflow connects complete task or training documents; it does not introduce a second task
syntax. Start with one node per independently reproducible operation:

```yaml
kind: workflow
schema_version: "1.0"
name: prepare-and-train
output_root: runs/workflows
max_parallel: 2
nodes:
  preprocess:
    config: preprocessing.yaml
  train:
    config: experiment.yaml
    needs: [preprocess]
    bindings:
      data.train.params.dataset_manifest: >-
        ${nodes.preprocess.artifacts.dataset-artifact.json}
```

Validate/inspect before execution:

```bash
lambdaforge validate workflow.yaml
lambdaforge inspect workflow.yaml
lambdaforge run workflow.yaml --dry-run
lambdaforge run workflow.yaml
```

`needs` creates edges; cycles and unknown nodes are rejected. A failed node blocks only its
descendants, while independent branches continue. `continue_on_failure: true` is an explicit escape
hatch for a node that can consume failure state. Bindings accept exact
`${nodes.NAME.outputs.PATH}`, `.metrics.PATH` and `.artifacts.RUN_RELATIVE_PATH` references. Nodes
retain their own task/experiment fingerprints and resume rules, so rerunning the graph cannot reuse
scientifically different output. `max_parallel` bounds ready local nodes.

Any task or experiment YAML may use deterministic composition. `extends` files merge first,
`include` files next, and the leaf document last; paths are relative to the file declaring them.
Mappings merge recursively, lists replace and `{$delete: true}` removes an inherited key:

```yaml
# study.yaml
extends: configs/base.yaml
include: [configs/local-data.yaml]
model:
  params: {dropout: 0.2}
trainer:
  obsolete_option: {$delete: true}
data_root: ${env:DATA_ROOT}
run_root: ${config:experiment.output_root}
```

`lambdaforge compose study.yaml` prints the fully resolved, redacted values, source files and
per-path provenance. `lambdaforge diff left.yaml right.yaml` compares semantic leaves rather than
text or mapping order. Interpolation is limited to `${config:path}`, `${env:NAME}` and a full-value
`${secret:NAME}`; Python expressions are never evaluated. Secrets in generic tasks reach the target
at construction time but persist as `***`. A secret cannot be embedded in a larger string. Training
and workflow structure reject secrets because their materialized configurations are durable; let
tracking/provider code read credentials directly from its environment.

Python equivalents are `ConfigurationComposer.resolve()` and `ConfigurationDiff.compare()` from
`lambdaforge.configuration`, and `LambdaForge.workflow(path)` / `Workflow.from_yaml(path)`.

## 10. Local and multi-cluster control plane

The 0.5 control plane is a local coordinator, not a mandatory web service. The user's machine owns
configuration materialization, bundle identity and persistent job metadata. A cluster profile owns
four operational decisions: how to reach the machine, which scheduler accepts work, where small
execution bundles are cached, and which Python/environment command runs LambdaForge.

Catalogs merge by profile name with explicit precedence: user
`~/.config/lambdaforge/clusters.yaml`, project `lambdaforge.clusters.yaml`, then
`--clusters-file`/`--clusters` (or `LAMBDAFORGE_CLUSTERS`). `clusters add` writes the user scope by
default; use `--scope project` deliberately. `clusters inspect NAME` reports the winning and
overridden sources.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login                 # an OpenSSH config host; host keys are never disabled
    user: my-user                     # optional; the alias may already specify it
    auth: {mode: openssh}             # recommended: keys/agent/ProxyJump remain native
    scheduler: slurm
    workspace: /scratch/my-user/lambdaforge
    python: /shared/envs/research/bin/python
    environment: managed             # or existing for a user/admin-owned environment
    project_module: my_project       # doctor verifies this consumer import
    data_environment: atlas
    resource_mapping:
      gpu: {option: gres, value: "gpu:a100:{gpus}"}
      memory: {option: mem, value: "{memory_gib}G"}
    scheduler_directives: {partition: gpu, account: project123, exclusive: true}
  atlas-container:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/my-user/lambdaforge
    command_prefix: [apptainer, exec, /shared/images/project.sif]
    python: python
profiles:
  one-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, gpu_memory: 20GiB, time: 4h}
```

`command_prefix` is an argument vector, so containers/site wrappers can be selected without local
shell interpolation. OpenSSH is the default and preserves aliases, keys, agent, `known_hosts` and
ProxyJump. Optional password mode uses Paramiko with rejected unknown hosts and obtains its secret
interactively, from a `keyring:` reference in the OS keyring, or from an explicit `env:NAME`
reference. The value is never accepted through `--password`, persisted, logged, bundled or
fingerprinted. Install `lambdaforge[cluster-password]` only for that legacy mode.

Use the same configuration locally and remotely:

```bash
lambdaforge doctor
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/my-user/lambdaforge \
  --scheduler slurm --environment managed
lambdaforge clusters list
lambdaforge clusters inspect atlas
lambdaforge clusters test atlas
lambdaforge clusters bootstrap atlas
lambdaforge run experiment.yaml --on atlas --dry-run
lambdaforge run experiment.yaml --on atlas --cpus 8 --memory 32GiB --resource-gpus 1 --time 4h
lambdaforge run experiment.yaml --profile one-gpu
```

The portable `ResourceRequest` normalizes CPU cores, RAM, GPUs, GPU memory, duration, storage and
process count. One validated per-cluster translation layer emits standard `gpus`, generic/typed
`gres`, CPU, memory and time directives. Static directives support flags and repeated values;
submit/queue/accounting/cancel commands, job-ID regex, shell and trusted prologue/epilogue are also
configurable as safe argv/templates. Legacy `scheduler_options` remains compatible. Dry-run exposes
the exact script, resources, directives, warnings and submit argv; explicitly omitted resources
produce a warning. Units accept decimal `KB/MB/GB/TB`, binary `KiB/MiB/GiB/TiB`, and durations such
as `30m` or `4h`. The full schemas and security trade-offs are in the
[cluster guide](docs/CLUSTERS.md).

An `ExecutionBundle` contains materialized YAML, a manifest, exact wheels for LambdaForge and the
nearest consumer project, and only explicitly bounded small inputs. It is content-addressed under
`.lambdaforge/control/bundles`; dirty local source is built exactly as it exists, never replaced by
`git clone main`. In `managed` mode those wheel bytes and compatible dependency policy identify an
idempotent user-space venv under `WORKSPACE/.lambdaforge/environments`. In `existing` mode no
installation occurs and the configured Python must already contain the exact framework/project.
Offline clusters use a target-compatible `wheelhouse`/`--wheelhouse` and `--no-index`. LambdaForge
verifies PyTorch/CUDA but never installs drivers, system CUDA or cuDNN. The remote command remains
`python -m lambdaforge run config.yaml`, so there is no second runner.

`LocalTransport`, OpenSSH `SshTransport` and optional `PasswordSshTransport`, plus
`LocalScheduler`/`SlurmScheduler`, are independent public providers. `doctor` checks connection and
auth, workspace, Python/project/framework/PyTorch/CUDA, every configured scheduler executable,
resource mapping and partition without submitting a job. Provider boundaries are injectable, so
tests exercise credentials, transfers, submission, failures and reconnection without a real cluster.

## 11. Persistent jobs and data placement

Every control-plane submission returns a short LambdaForge `job_id` and stores one atomic JSON
record under the user's XDG state directory (`~/.local/state/lambdaforge/jobs` by default). The
record contains cluster, scheduler ID, exact argument vector, resources, bundle identity, timestamps
and retry lineage. It is independent from the research result: the job says how execution was
scheduled; `result.json` says what scientific attempt completed.

```bash
lambdaforge status --on atlas --state running --name baseline
lambdaforge status job-20260811120000-ab12cd34
lambdaforge logs job-20260811120000-ab12cd34 --follow
lambdaforge cancel job-20260811120000-ab12cd34
lambdaforge retry job-20260811120000-ab12cd34 --dry-run
```

`JobService` provides the same list/get/logs/cancel/retry operations to Python applications and a
future GUI. CLI JSON output is a direct serialization of the same `JobRecord`, `JobHandle`, doctor,
bundle and data service objects; there is no second GUI-only business layer.

Records also keep scientific/execution identities, exact environment and remote paths. A later
process refreshes non-terminal scheduler states, so a SLURM job survives the local CLI exiting.
Synchronize only small evidence or explicitly fetch one heavy artifact:

```bash
lambdaforge results sync JOB
lambdaforge plot learning JOB --follow --output plots/live.svg
lambdaforge artifact list JOB
lambdaforge artifact fetch JOB best-checkpoint --output checkpoints/best.ckpt
```

Sync allowlists result/metrics/environment/manifests/summaries/plots up to 16 MiB per file by
default. Checkpoints, datasets and other heavy artifacts are never implicit.

Large data is never copied merely because `run --on` was used. Register logical names in a data
catalog:

```yaml
datasets:
  raw-corpus:
    identity: {strategy: version, namespace: lab/raw-corpus, version: "2026-08-11"}
    locations:
      local: /data/raw-corpus
      atlas: /datasets/project/raw-corpus
```

Then reference it without a mount point in task authoring YAML:

```yaml
data_catalog: ../data-catalog.yaml
inputs: {raw: dataset:raw-corpus}
```

The target profile's `data_environment` selects a physical location. If it is missing, submission
fails before scheduling and tells the user to register or replicate the data. Ordinary local path
inputs up to 10 MiB are copied into the small execution bundle; larger implicit transfers are
refused. Data movement is a separate preview-first command:

```bash
lambdaforge data --catalog data-catalog.yaml list
lambdaforge data --catalog data-catalog.yaml locations raw-corpus
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas
# Review the exact source/destination, then:
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas --apply
```

The built-in replication provider uses `rsync` and requires both locations to exist in the catalog;
it does not guess destinations or rewrite the catalog. Object-store and institutional transfer
systems implement the `DataTransferProvider` boundary.

Training experiments use the same catalogue. A direct split such as
`data.train: dataset:raw-corpus/train` requires the entry to declare a dataset `loader` ObjectSpec
plus `path_parameter`; LambdaForge injects the selected location. Inside nested ObjectSpec params,
`{dataset: raw-corpus, subpath: train}` resolves only that typed marker. Ordinary strings are never
guessed. The physical mount can differ by cluster while the scientific fingerprint retains the
logical reference and declared dataset identity.

Workflow YAML may annotate a node with `on: atlas`, and dry-run plans display every placement.
Version 0.5 deliberately refuses to execute a mixed-cluster DAG in the in-process workflow runner:
downstream artifact transfer and durable coordinator recovery need stronger semantics than polling
remote logs. Submit such node configurations explicitly with `run --on`; local workflow execution
remains complete. This safety limitation avoids claiming a distributed workflow succeeded when its
artifacts were never materialized at the consumer node.

## 12. Inference, evaluation, export and HPO

Operational model work is a generic task and therefore receives the same input hashing, attempts,
provenance and artifacts as preprocessing. Declare every checkpoint in task `inputs`, then use its
YAML-relative path in `checkpoints`:

```yaml
kind: task
schema_version: "1.0"
name: test-inference
inputs:
  - {name: checkpoint, path: checkpoints/best.ckpt}
task:
  target: lambdaforge.operations.InferenceTask
  params:
    model: {target: my_project.models.Model, params: {features: 32}}
    checkpoints: checkpoints/best.ckpt
    data: {target: my_project.data.TestDataset}
    batch_size: 128
    model_input_key: x
    model_output_key: logits
```

`InferenceTask` writes CPU `predictions.pt`; several checkpoints form an ensemble whose matching
tensor outputs are averaged. `EvaluationTask` adds a `metrics` list and updates the ordinary
LambdaForge metric contract on a new dataset. `ExportTask` requires `example_inputs` and selects
`format: torchscript`, `torch_export`, `onnx`, or an injected object exposing
`export(model, args, path)`. Checkpoints load with `weights_only=True`; plain state dicts and
Lightning `state_dict` envelopes are accepted.

Deterministic grids/ablations remain the preferred exhaustive mechanism. For sampled search:

```python
from lambdaforge.hpo import RandomSearch

search = RandomSearch(
    {
        "optimizer.params.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "model.params.width": {"type": "choice", "values": [64, 128, 256]},
    },
    seed=17,
)
trial_configs = search.materialize(base_config, count=20)
```

Distributions are `choice`, `uniform`, `loguniform` and `int`; a `when` mapping makes a parameter
conditional on already sampled dotted paths. Trials have stable parameters, seed and SHA-256.
`OptunaSearch` is an optional adapter for reproducible TPE plus `asha` or `hyperband` pruning; install
Optuna in the consumer environment. It does not replace LambdaForge scheduling or result identity.

### Adaptive experiment optimization

Use an enabled top-level `hpo` block when a finite grid wastes training budget. Start from
[`examples/adaptive-hpo.yaml`](examples/adaptive-hpo.yaml), keep the ordinary model/data/loss/task
sections, and declare only dotted scientific paths:

```yaml
execution: {mode: sequential, gpus: [0, 1]}
hpo:
  enabled: true
  controller_seed: 2026
  max_concurrency: 4
  objective: {metric: val_auroc, direction: maximize}
  space:
    optimizer.params.lr: {type: float, low: 0.00001, high: 0.01, scale: log}
    model.params.hidden_features: {type: int, low: 64, high: 512}
    model.params.dropout: {type: float, low: 0.0, high: 0.5}
  initialization: {strategy: sobol, trials: auto}
  search: {strategy: bayesian, candidate_pool_size: 256}
  acquisition: {strategy: cost_aware_knowledge_gradient}
  fidelity: {unit: epochs, strategy: adaptive_learning_curve, min: 5, max: 100, step: 5}
  seeds:
    strategy: adaptive_racing
    values: [7, 17, 27]
    max_search_seeds: 3
    confirmation_values: [101, 211]
  pruning: {enabled: true, min_budget_before_drop: 10, probability_threshold: 0.01}
  memory:
    per_job_budget: 6GiB
    headroom: 512MiB
    allocator_cap: true
    resource_features:
      batch_size: data.datamodule.params.batch_size
      hidden_width: model.params.hidden_features
  budget: {max_actions: 50, max_total_epochs: 1500}
  confirmation: {top_k: 2}
```

Complete adaptive policy reference (defaults apply only when `hpo.enabled: true`):

| Field | Default | Meaning |
|---|---:|---|
| `controller_seed` / `max_concurrency` | `0` / `1` | Reproducible controller decisions and maximum live independent actions. |
| `objective.metric` / `direction` | `val_loss` / `minimize` | Exact epoch-CSV objective and its orientation. |
| `objective.risk.type` / `lambda` | `mean` / `0` | Scientific objective; `mean_minus_std` is opt-in and changes the question being optimized. |
| `space` | required | Dotted scientific paths with `float`, `int`, ordered `ordinal`, unordered `categorical` or `bool`; numeric scale is `linear` unless `log`; `when` adds a parent condition. |
| `initialization.strategy` / `trials` | `sobol` / `auto` | Initial design; `auto = max(4, 2 * (effective_dimensions + 1))`. Random is the baseline. |
| `search.strategy` | `bayesian` | `bayesian`, `sobol` or `random`. |
| `search.candidate_pool_size` / `refresh_interval` | `128` / `1` | Acquisition raw samples and scored-observation interval used to refit/cache the surrogate. |
| `acquisition.strategy` / `exploration_weight` | `cost_aware_knowledge_gradient` / `1` | KG, plain KG or expected improvement; global action scoring still accounts for predicted cost/feasibility. |
| `fidelity.strategy` | `adaptive_learning_curve` | Adaptive curves, `fixed`, or deterministic `successive_halving` baseline. |
| `fidelity.min` / `max` / `step` | `5` / `100` / `min` | Cumulative epoch boundaries; only `unit: epochs` is currently supported. |
| `seeds.strategy` / `values` | `adaptive_racing` / `[0]` | Shared ordered search seeds; `fixed` runs every declared seed. |
| `seeds.confirmation_values` / `max_search_seeds` | `[]` / number of search seeds | Disjoint final seeds and the search repetition ceiling. |
| `seeds.probability_threshold` | `0.9` | Competitiveness threshold for spending another adaptive seed. |
| `pruning.enabled` / `min_budget_before_drop` | `true` / fidelity minimum | Whether and when posterior competitive pruning may start. |
| `pruning.probability_threshold` / `equivalence_margin` | `0.01` / `0` | Conservative posterior drop threshold and practical-equivalence margin. |
| `memory.per_job_budget` / `headroom` | `0` / `0` | Logical cold-start reservation/optional allocator ceiling and extra scheduling headroom. Zero disables a logical byte limit. |
| `memory.safety_quantile` / `min_observations` | `0.99` / `3` | Conservative feature-aware reservation quantile and evidence needed before leaving cold start. |
| `memory.resource_features` | `{}` | Generic feature name → candidate/base dotted path, for example batch size, tokens or resolution. |
| `memory.allocator_cap` / `preflight` | `true` / `false` | Defensive child PyTorch ceiling and backwards-compatible switch for candidate-aware isolated probes. |
| `memory.probe_policy.mode` | `auto` if preflight, else `never` | `auto`, `always` or `never`; auto uses cold start, uncertainty, OOD, limit proximity and OOM probability. |
| `memory.unknown_capacity` | `declared_budget` | Use a positive declared budget when discovery fails, or `fail_closed`; UNKNOWN never means unbounded. |
| `memory.device_capacities` | discovered or UNBOUNDED CPU | Explicit usable bytes per listed GPU when cluster discovery is unavailable. `KNOWN(0)` remains a real zero. |
| `memory.structural.*` | parameter/gradient/optimizer state defaults | Optional parameter-count estimate; activations and workspaces remain outside it and trigger conservative probing. |
| `budget.max_actions` / `max_total_epochs` | `50` / actions × max fidelity | Hard study limits including pending commitments. |
| `budget.max_gpu_seconds` | unset | Optional measured/predicted GPU-time ceiling. |
| `confirmation.top_k` | `1` | Number of posterior configurations frozen for disjoint-seed confirmation. |
| `components.*` | built-ins | Trusted `target`/`params` replacements for the eight policy/model boundaries listed below. |

The controller compares `START_NEW`, real checkpoint `RESUME` and `ADD_SEED` through the same
one-step Gaussian moment approximation to Knowledge Gradient, divided by predicted cost and
multiplied by memory feasibility. This is explicitly recorded as `gaussian_moment_knowledge_gradient`,
not presented as exact BoTorch KG over heterogeneous actions. Confirmation remains a separate
scientific phase. Sobol initialization avoids correlated random starts. Optional Bayesian search
fits all observed curve points to `f(x,b)`: numeric spaces use a fidelity GP, mixed spaces use
Hamming categorical geometry plus an explicit fidelity feature, and multi-fidelity KG projects to
full budget with an inverse-cost utility. Categories are canonical and permutation-invariant;
conditional values have a distinct inactive state and activity mask. Pending actions enter
`X_pending`. A normal fit is retried with safer jitter before a named `HPO_SURROGATE_FALLBACK` to
Sobol. Install it with
`pip install "lambdaforge[adaptive-hpo]"`. `sobol` and `random` remain dependency-free baselines.
`PAUSE` is realized cooperatively at the selected epoch boundary and `DROP` prevents future
promotion; LambdaForge does not kill a promising process mid-optimizer-step merely to reshuffle a
slot. Small fidelity increments bound the latency before a newer decision can take effect.

Fidelity targets are cumulative epochs. Every partial action writes a normal last checkpoint and
full `metrics.csv`; promotion restores model, optimizer, scheduler, scaler and Lightning loop state,
then runs only missing epochs. `trainer.checkpoint_policy` must be `last`, `last_and_best` or `all`.
HPO pruning is separate from early stopping: it waits for `min_budget_before_drop` and drops only
below the configured posterior probability of practical competitiveness. Partial per-seed curves
use Bayesian basis posteriors rather than recent-slope extrapolation, preserving uncertainty for
warm-up, curvature, plateaus and non-monotonic schedules. Shared seeds are compared as paired
differences when possible.

Adaptive seed racing uses the same declared order across configurations and spends another seed
only where ranking uncertainty warrants it. Confirmation runs posterior top K configurations at
full budget on disjoint seeds that were not used to select those configurations.

`memory.per_job_budget` is a cold-start reservation and, with `allocator_cap: true`, a public
PyTorch child-process allocator ceiling. Exact peaks and consumer-declared `resource_features`
train a conservative non-negative log-linear predictor; out-of-distribution candidates inflate
uncertainty. An OOM under `L` bytes is retained as censored evidence `M(x,z) > L`, not discarded.
Explicit `device_capacities` support restricted clusters. LambdaForge requires no
`nvidia-smi`, custom environment variables, MIG/MPS or administrator service; allocator limits are
defensive rather than physical isolation, and batch size is never changed silently.
For representative checks, set `preflight: true`, provide `probe: {target: ...}` and optionally tune
`probe_policy`. The callable receives `(materialized_candidate, resource_context)` and must build
that candidate plus a representative batch, then perform forward/backward/optimizer work. It runs
in an isolated child on the selected logical GPU only when the deterministic policy requires it.
Legacy zero-argument probes remain accepted. This is project-supplied because LambdaForge cannot
infer a scientifically representative batch for an arbitrary dataset/model.
Adaptive actions currently use independent single-process CPU/GPU trials; `execution.mode: ddp` is
rejected for enabled HPO because group reservation and per-rank allocator ceilings would otherwise
be misleading. Static experiments retain normal DDP support.

```bash
lambdaforge validate examples/adaptive-hpo.yaml --no-imports
lambdaforge inspect examples/adaptive-hpo.yaml
lambdaforge run examples/adaptive-hpo.yaml --dry-run
lambdaforge run experiments/my-adaptive-study.yaml
```

`inspect` and `--dry-run` return a plan without state or training. A real study persists atomic
`state.json`, append-only `events.jsonl` and `summary.json` under
`SUITE/.lambdaforge/adaptive/STUDY_ID/`; individual actions remain normal auditable run directories.
Relaunching the same YAML reconciles terminal pending work and continues deterministically. Do not
edit state/result files manually. Scientific search changes create another study ID; concurrency or
declared capacity changes do not. Summaries report actual actions, epochs, full-training equivalents,
GPU seconds, OOMs and fallbacks—not invented counterfactual savings. They also contain per-config
search/confirmation seed usage, merged learning curves, peak-memory observations and confirmation
mean, sample deviation, standard error, normal 95% interval and paired differences over shared
seeds. A one-seed confirmation deliberately reports undefined dispersion/interval fields as null.
Use the existing aggregation/statistics layer for bootstrap or non-parametric publication analyses.
Enabled HPO and sweep are intentionally mutually exclusive.

Defaults are policies, not hard-coded domain assumptions. `hpo.components` accepts trusted
`target`/`params` specs for `searcher`, `fidelity_policy`, `seed_policy`, `learning_curve_model`,
`cost_model`, `memory_model`, `admission_controller` and `action_selector`. Implement the same
public method used by the built-in class (inspect its signature with `lambdaforge target ...`) and
keep the class in the consumer's installed package. Searchers implement `propose(space, state,
count)`; fidelity policies implement `resume_candidates(state)` and may add `dominated(state,
model)`; seed policies implement `candidates(state, model)`. Predictive models consume durable
state; memory models read action `parameters`/`resource_features`, preserve censored lower bounds
and return a conservative estimate. Admission returns feasibility/reservation and selectors rank
actions. These are duck-typed
policy boundaries: a custom object does not need to inherit a concrete built-in class. This keeps
domain priors replaceable without subclassing the runner or depending on Lightning internals.

## 13. Resources, backends and reliability

GPU parallel and DDP modes retain their previous contract. CPU-only parallel sweeps now omit
`gpus` and set an explicit slot count:

```yaml
execution:
  mode: parallel
  cpu_jobs: 4
  cpu_cores_per_job: 2
  cpu_threads_per_job: 2
  cpu_interop_threads_per_job: 1
  dataloader_num_workers_per_job: 0
```

LambdaForge rejects `cpu_jobs * cpu_cores_per_job` above the process's available affinity. Each CPU
slot hides CUDA and patches Lightning to `accelerator: cpu`; GPU slots continue to use
`jobs_per_gpu`, and DDP uses `devices_per_job`.

For portable planning use `ResourceRequest` and `ResourcePlanner`. Requests declare CPU, RAM, GPU
count/memory, storage and optional runtime. The planner either validates manual waves or performs
deterministic capacity-safe first-fit packing and returns peak capacity plus runtime/storage
estimates. Estimates are reported only when declared; the framework does not invent benchmarks.

`LocalExecutionBackend` and `SlurmExecutionBackend` implement `ExecutionBackend`. SLURM always
generates `submit.sbatch` first; `dry_run=False` is required to call `sbatch`. Constructor options
cover partition, nodes, array, dependency, container argv prefix, environment and requeue. Commands
are argument vectors and generated arguments are shell-quoted. `cancel(job_id)` and
`requeue_job(job_id)` require numeric IDs. Multi-node launcher details remain explicit in the command
rather than guessed by the framework.

`FailureClassifier` distinguishes cancellation, preemption, CPU/GPU OOM, transient, user and
unknown failures. `RetryPolicy` retries only configured categories for a bounded attempt count with
exponential backoff. `AttemptMode` names the semantic difference: resume reuses compatible state,
restart starts clean, retry repeats a failed intention, and fork creates a new identity. Unknown and
user errors are not retryable by default.

## 14. Artifact stores, registry and reports

Start with the stable result service rather than navigating hashed directories:

```bash
lambdaforge results list --root runs
lambdaforge results show baseline --root runs
lambdaforge results compare baseline ablation --metric val_loss --direction minimize
lambdaforge results export baseline --series --format csv --output analysis/curves.csv
```

Selectors accept a config/result path, attempt ID, fingerprint, run/experiment name or variant.
`show` returns all candidates and marks ambiguity; it never chooses by modification time. The old
`results SOURCE --write-index` syntax remains compatible as `results audit`. `MetricSeries` reads
the existing dense `metrics.csv` and normalizes run, seed, variant, split, metric, step, value and
timestamp without creating another log/database. JSON/CSV are core; Parquet uses the extra.

Plotting is a consumer of those results, never part of the training loop:

```bash
lambdaforge plot learning baseline --metric val_loss --aggregate mean --uncertainty std \
  --output plots/learning.svg
lambdaforge plot seeds baseline --metric val_accuracy --kind violin
lambdaforge plot sweep sweep.yaml --x optimizer.params.lr --metric val_loss
lambdaforge plot sweep sweep.yaml --x model.params.width --y optimizer.params.lr \
  --metric val_accuracy --output plots/sweep.html
lambdaforge plot sweep sweep.yaml --x model.params.width \
  --metric val_loss --metric val_accuracy --normalize
lambdaforge plot hpo runs/STUDY/.lambdaforge/adaptive/ID --parameter optimizer.params.lr
```

Mean curves/sweep cells report seed count and sample deviation or a normal mean CI. With one seed,
uncertainty is absent rather than zero. Missing 2-D cells remain missing unless interpolation is
explicit. `--normalize` applies per-metric min-max scaling over observed cells and retains raw
values in the `PlotSpec`; it is never implicit. Comparisons report deltas against the first selector
and label best/worst only with an explicit metric `--direction`. HPO/resource plots only expose
existing state/telemetry. `VisualizationService` first
builds a serializable `PlotSpec` (`--json`); Matplotlib renders PNG/SVG/PDF, optional `viz` renders
self-contained Plotly HTML. Atomic output plus `FIGURE.plot.json` makes generation reproducible and
cacheable. When placed below a run's `plots/`, both the figure and its timestamped specification
are returned by `artifact list`.

Artifacts are inspected without executing them:

```bash
lambdaforge artifact inspect predictions.npz --array logits --rows 20 --slice 0:100,:
lambdaforge artifact export predictions.npz --array logits --format csv
lambdaforge artifact validate graph.npz --require-array positions --shape positions=*,3 --finite
lambdaforge artifact visualize graph.npz --type graph --nodes positions --edges edge_index
lambdaforge artifact list baseline
```

NPY/NPZ always uses `allow_pickle=False`; previews are capped and large-array statistics use a
deterministic bounded sample. CSV/TSV/JSON/JSONL are supported. Graph/point-cloud/mesh geometry is
never inferred from shape: roles/type must be explicit, and optional meshes use `viz3d`/trimesh.
Public inspector, visualizer, schema, exporter and validator plugin boundaries keep lab-specific
formats outside core. Full guides: [results](docs/RESULTS.md), [artifacts](docs/ARTIFACTS.md),
[clusters](docs/CLUSTERS.md) and [preprocessing](docs/PREPROCESSING.md).

`ArtifactReference` is the portable tuple `(store, key, sha256, size, media_type)`. An
`ArtifactStore` publishes immutable content, verifies existence and stages a checked local copy.
`LocalArtifactStore` supports local/shared filesystems; `S3ArtifactStore` accepts an injected
S3-compatible client or optional `boto3`. Neither provider is selected implicitly. Store keys reject
absolute/traversal paths and every stage validates size and SHA-256.

`DistributedArtifactCache(root, upstream)` coordinates one cache copy per key using shared-filesystem
leases. It atomically publishes, detects/repairs corruption and invalidates only the cache, never the
authoritative store. The base store contract deliberately has no delete method, so referenced
content is not removed by local experiment retention.

Research discovery remains disk-backed:

```bash
lambdaforge results experiments/study.yaml --write-index --fail-on-ambiguous
lambdaforge registry runs --output analysis/registry.csv
lambdaforge dashboard runs --output analysis/dashboard.html
```

`ExperimentRegistry` reads `ResultCatalog` plus config/result metadata; it is not a second database.
`RegistryQuery` filters status, experiment name, required tags, exact metadata and fingerprint.
Export supports JSON/CSV and optional Parquet (`pandas` plus a Parquet engine). Select explicit
attempts before publication.

`ExperimentComparator.compare(groups, metric=...)` produces counts, means, standard deviations,
configured normal confidence intervals, mean effects and semantic config differences.
`ReportBuilder.write()` creates Markdown/HTML and a factual mean/interval plot. It never chooses a
winner or writes a scientific conclusion. `LocalDashboard` is a static, read-only table of registry
records, paths and metrics; tracking services remain optional complementary views.

## 15. Observability and reproducibility

Every generic task now writes bounded `events.jsonl` start/finish records alongside `task.log`;
failures include the explicit failure category. `EventLogger` is available for consumer events and
uses a cross-process lock. `ResourceMonitor.sample()` returns CPU, RSS, threads, optional CUDA
allocated/reserved/peak memory and optional throughput no faster than its configured interval.
`ProfilerAdapter` is the provider boundary; `TorchProfilerAdapter` uses a finite schedule and writes
TensorBoard-compatible traces.

```python
from lambdaforge.reproducibility import ReproducibilityProfile, SeedDeriver

profile = ReproducibilityProfile.named("strict", seed=7)
profile.apply()
loader_seed = SeedDeriver(7).derive("dataloader", "train", 0)
fingerprints = profile.fingerprints(materialized_config)
```

Profiles are `fast`, `repeatable` and `strict`; strict enables deterministic torch algorithms.
Seed derivation uses SHA-256, not process-randomized `hash()`. `fingerprints()` records full
infrastructure identity separately from science with execution/device fields removed.
`EnvironmentExporter.export(path, format=...)` supports `pip`, `conda` and a container-oriented JSON
snapshot without changing the environment.

## 16. CLI reference

| Command | Purpose | Writes by default |
|---|---|---:|
| `init DIRECTORY [--template minimal|preprocessing|training|full]` | Scaffold only the consumer pieces needed now; refuses collisions without `--force`. | yes |
| `doctor [--on CLUSTER]` | Check Python, LambdaForge, scheduler and PyTorch/CUDA visibility. | no |
| `validate CONFIG` | Schema/import/resource/DAG checks. | no |
| `inspect CONFIG --resolved` | Concise YAML compiled to strict runner configuration. | no |
| `inspect CONFIG` | Expanded runs or immutable task/workflow plan. | no |
| `run CONFIG --dry-run` | Exact execution plan. | no |
| `run CONFIG` | Execute experiment, task or workflow. | yes |
| `run CONFIG --force|--restart|--no-resume` | Explicitly control success reuse and partial continuation. | yes |
| `run CONFIG --on CLUSTER|--profile PROFILE` | Cache a small bundle and submit through the control plane. | job metadata; remote only without dry-run |
| `clusters add|list|show|inspect|export|credentials|test|bootstrap` | Manage layered profiles, external credentials, diagnostics and exact environments. | add/credentials/bootstrap |
| `status|logs|cancel|retry` (`jobs ...` also valid) | Filter/reconnect/follow/control persistent jobs. | cancel/retry only |
| `data --catalog FILE list|locations|inspect|replicate` | Inspect logical datasets/manifests; replication needs `--apply`. | only replicate `--apply` |
| `compose CONFIG` | Redacted materialization plus provenance. | no |
| `diff LEFT RIGHT` | Semantic configuration differences. | no |
| `explain authoring|experiment|task|workflow PATH` | JSON Schema fragment for a dotted property. | no |
| `explain changes CONFIG [--against OLD]` | Scientific identity and exact changed paths. | no |
| `target IMPORT.PATH` | Constructor signature and docstring. | no |
| `migrate CONFIG` | Preview migration; `--output` is explicit. | no |
| `plugins` | Entry-point metadata without provider import. | no |
| `results SOURCE` / `results audit SOURCE` | Compatible identity/duplicate audit; index writing is explicit. | no unless `--write-index` |
| `results list|show|compare|export|sync` | Human selectors, statistics, tabular export and lightweight remote evidence. | export/sync |
| `plot learning|sweep|seeds|hpo|resources` | Create `PlotSpec` JSON or atomic static/HTML figures. | only without `--json` |
| `artifact inspect|export|validate|visualize|list|fetch|plugins` | Safe bounded inspection and explicit retrieval/geometry. | export/visualize/fetch |
| `debug CONFIG --records N` | Sample preprocessing transforms without production sink/finalization. | only requested intermediates |
| `aggregate CONFIG` | Rebuild experiment aggregates. | yes |
| `retain CONFIG` | Retention preview; only `--apply` mutates artifacts. | no |
| `registry ROOT [--output FILE]` | Query JSON or export JSON/CSV/Parquet. | only with output |
| `dashboard ROOT --output FILE` | Static read-only HTML snapshot. | yes |

`lambdaforge init my-project --template preprocessing` is the fastest preprocessing path;
`training` creates a runnable toy baseline, `minimal` creates one generic task and `full` includes
both families. Rename `my_project`, implement the generated domain code, install it with
`pip install -e .`, and validate its YAML. The scaffold includes `.gitignore` rules for
environments, caches, builds and run output.

## 17. Public API

The supported entry points are deliberately narrow:

| Entry point | Purpose |
|---|---|
| `from lambdaforge import LambdaForge` | Load, run or construct objects through the facade. |
| `from lambdaforge import MaterializedConfig, JobHandle` | Inspect compiled authoring and durable submissions. |
| `from lambdaforge import Experiment` | Inspect, execute, aggregate and load one experiment suite. |
| `from lambdaforge import TaskRun, TaskResult, TaskExecutionPlan` | Validate, inspect, execute and audit one generic task. |
| `from lambdaforge import Workflow, WorkflowPlan, WorkflowResult, WorkflowValidationReport` | Validate, plan and run a task/experiment DAG. |
| `from lambdaforge import RunResult, AggregateResult` | Typed immutable results with legacy dict/JSON compatibility. |
| `from lambdaforge import ResultCatalog, ResultRecord` | Identity-aware discovery and explicit selection of attempt history. |
| `from lambdaforge import ResultService, VisualizationService, PlotSpec, ArtifactService` | Stable query, plotting and safe artifact application services. |
| `from lambdaforge import ArtifactRetentionPlan, ArtifactRetentionResult` | Typed immutable retention previews and outcomes. |
| `lambdaforge.data` | Logical identity/catalog/location, explicit transfers, dataset adapters and bounded caches. |
| `lambdaforge.tasks` | Generic task, context, plan, result and artifact contracts. |
| `lambdaforge.preprocessing` | Composable record preprocessing and dataset manifests. |
| `lambdaforge.configuration` | Authoring-to-IR compilation, includes, safe interpolation, redaction, provenance and diff. |
| `lambdaforge.controlplane` | Cluster/transport/scheduler providers, bundles, doctor and persistent job services. |
| `lambdaforge.results` | Human selectors, normalized metric series, comparison/export and remote sync. |
| `lambdaforge.visualization` | Renderer-neutral plot specifications and atomic rendering. |
| `lambdaforge.artifacts` | Inspector/visualizer/schema/validator contracts and local/remote services. |
| `lambdaforge.workflows` | DAG configuration, nodes, plans and results. |
| `lambdaforge.operations` | Inference, evaluation, ensembles and model export tasks. |
| `lambdaforge.hpo` | Finite random/Optuna and persistent adaptive multi-fidelity optimization. |
| `lambdaforge.execution` | Resource planning, local/SLURM backends and retry policy. |
| `lambdaforge.storage` | Artifact references, stores and distributed staging cache. |
| `lambdaforge.registry` | Catalog-backed queries, comparisons, reports and dashboard. |
| `lambdaforge.observability` | Structured events, monitoring and profiler adapters. |
| `lambdaforge.reproducibility` | Scientific/code/execution identities, profiles, seeds and environment export. |
| `lambdaforge.nn` | Models and the YAML component registry. |
| `lambdaforge.metrics` | Base metric contract and built-in metrics. |
| `lambdaforge.plugins` | Lazy discovery, run usage sessions, descriptors and resolution errors. |
| `lambdaforge.integrations` | Stable Lightning compatibility object for plugin authors. |
| `lambdaforge.tracking` | Lazy optional MLflow, TensorBoard and Weights & Biases logger adapters. |
| `lambdaforge.training` | Task, runner, configuration and process orchestration. |
| `lambdaforge.experiments` | Lower-level configuration, migrations, scheduling, aggregation and loading. |
| `python -m lambdaforge` / `lambdaforge` | CLI front end to the same object API. |

`LambdaForge.build(spec)` exposes the generic object factory:

```python
model = LambdaForge.build(
    {
        "target": "lambdaforge.nn.models.MLP",
        "params": {"in_features": 32, "out_features": 1, "hidden": [64, 32]},
    }
)
```

`LambdaForge.materialize(path)` returns the compiled strict document without execution;
`LambdaForge.submit(path, on="atlas", resources=...)` returns a persistent `JobHandle`. The lower
level `JobService`, `DataService` and `Doctor` are application services intended for CLIs, notebooks
and future graphical clients; their `to_dict()` results are stable JSON-facing envelopes.

Import from these namespaces instead of relying on file locations. Internal modules may move while
the public imports remain stable.

`LambdaForge.validate(path)` and `Experiment.validate(check_imports=True)` return an immutable
`ValidationReport`. The equivalent CLI supports `--json` for automation and `--no-imports` for
template files whose external project is not installed yet. Import checking never instantiates
configured objects, but importing a Python module can execute its top-level code, so configurations
must still be trusted.

`LambdaForge.preview_migration(path)` returns an immutable, non-writing
`ExperimentConfigMigrationResult`. Migration objects, exact Schema versions, catalogs and preview
formats are public from `lambdaforge.experiments`.

`LambdaForge.preview_retention(path)` and `Experiment.preview_retention()` build a strictly
read-only plan. Mutation is always explicit through `LambdaForge.apply_retention(path)`,
`Experiment.apply_retention()` or the CLI command `lambdaforge retain ... --apply`; the configured
`retention.mode: apply` may also run only after a successful final aggregation has published a
current completion receipt.

Installed plugin metadata can be inspected without importing provider modules:

```powershell
lambdaforge plugins
lambdaforge plugins --kind metric --json
```

## 18. Conceptual execution model

LambdaForge separates **scientific intent**, **operational scheduling** and **terminal evidence**.
This is the small mental model behind experiments, workflows and adaptive optimization.

Let `c` be a fully composed scientific configuration, `d` the logical dataset identities, `k` the
code identity and `o` its operational controls, such as output paths, placement, concurrency or
retention. Its identities are conceptually:

```text
scientific identity = SHA-256(canonical(c, d, k))
execution identity  = SHA-256(canonical(cluster, resources, environment policy))
```

Operational changes in `o` do not silently create a different scientific claim. Changing the model,
data, optimizer, loss, sampled hyperparameters or seed does. A run is skipped or resumed only when
this identity and the required artifacts/checkpoint contract still match; otherwise LambdaForge
creates another attempt and preserves the previous terminal result.

For an ordinary experiment, expansion produces a finite set of jobs:

```text
jobs = variants × seeds
```

where `V` is the set of base/grid/ablation variants and `S` the declared seeds. `sequential` executes
one job in the caller, `parallel` assigns independent jobs to explicit CPU/GPU slots, and `ddp`
assigns one job to a group of devices. Static experiment slots obey declared counts such as
`cpu_jobs`, `jobs_per_gpu` and `devices_per_job`; they do not invent a VRAM estimate. When portable
resource planning is requested, each job has a declared resource vector `r_j` and the planner seeks
an assignment `z_{jd}` satisfying, for every resource/device `d`:

```text
sum of resources reserved by jobs on device d ≤ declared capacity of device d
```

with `C_d` the declared capacity. The built-in planner uses deterministic first-fit/waves rather
than claiming an expensive globally optimal packing.

Remote placement adds a provider-neutral path without changing worker logic:

```text
authoring config -> materialized IR -> execution bundle -> transport -> scheduler -> ordinary runner
                                      \-> persistent JobRecord <- state/log/cancel/retry
```

Bundle caching removes repeated small control transfers; `DataCatalog` resolves large data where it
already lives. A scheduler ID is operational evidence, never a scientific result ID.

Adaptive HPO replaces the finite job list with repeated decisions. A hyperparameter configuration
is `x`, a seed is `s`, the cumulative fidelity is `b <= B`, and the observed objective is
`Y(x,s,b)`. The scientific target is the full-budget seed expectation:

```text
scientific value of configuration x = expected Y(x, seed, full budget) across seeds
```

At decision `t`, history `D_t` contains complete learning curves, seeds, pending work, elapsed cost,
memory peaks and failures. Candidate actions include starting `x`, resuming `(x,s)`, adding a seed or
confirming a finalist. Their common approximate utility is:

```text
utility(action | history)
    = value of information(action | history)
      / expected incremental cost(action | history)
      × probability(action fits available memory | history)
```

`I` is a Value of Information: BoTorch multi-fidelity KG proposes new `x`, while heterogeneous
START/RESUME/ADD_SEED actions use a documented one-step Gaussian moment KG approximation. It is not
the former `improvement + uncertainty` heuristic. `C` is predicted incremental time. Memory
admission reserves a conservative value
`R_M = Q_q(M | D_t) + headroom`; actions are packed only while their reservations fit the selected
device. This probability affects ranking, while the reservation is a hard scheduling constraint.

The optional Bayesian surrogate observes every available `Y(x,s,b)` rather than replacing a curve
with one extrapolated target. Unordered categories use Hamming geometry, ordinal categories retain
their order, and inactive conditional dimensions carry a separate state/mask. For seed estimates
with within-seed estimation variances `v₁, …, vₙ` and estimated population variance `tau²`,
LambdaForge propagates uncertainty as:

```text
variance of the estimated mean = tau² / n + (v₁ + ... + vₙ) / n²
```

Here `n` is the number of seeds. The first term is uncertainty caused by real variation from one
seed to another; the second is the remaining uncertainty inside the individual curve estimates.
For example, with two seeds, `tau² = 4` and `v₁ = v₂ = 1`, the result is
`4/2 + (1+1)/4 = 2.5`. The `v` term is divided by `n²` exactly once because the arithmetic mean
weights each seed by `1/n`.

Memory capacity is a tagged value—`UNKNOWN`, `UNBOUNDED` or `KNOWN(N)`—so failure to discover a GPU
cannot silently disable safeguards and `KNOWN(0)` cannot become unlimited.

The online loop is therefore: incorporate finished evidence → update curve/cost/memory beliefs →
generate new/resume/seed actions → reject unsafe or over-budget actions → rank by utility → best-fit
pack free resources → dispatch. A free slot can receive new work immediately; it does not wait for a
global rung. Pause is a normal checkpointed fidelity boundary, so later promotion continues from
`b` to `b + delta` instead of recomputing `0` to `b + delta`.

Uncertainty controls exploration and extra seeds but does not penalize the scientific mean unless
`mean_minus_std` is explicitly selected. Finally, search hyperparameters are frozen and disjoint
confirmation seeds estimate the result that should be analysed. These models are transparent,
replaceable approximations—not a promise that one heuristic is optimal for every research domain.

## 19. Architecture

```text
LambdaForge/
├── .github/workflows/             # hosted CPU CI plus opt-in self-hosted CUDA
├── examples/                     # runnable configuration templates
├── src/lambdaforge/
│   ├── EnvironmentManifest.py     # typed run provenance
│   ├── LambdaForge.py            # single discoverable facade
│   ├── cli/                      # command-line object
│   ├── configuration/            # authoring IR, composition, secrets, provenance and diff
│   ├── controlplane/             # clusters, transports, schedulers, jobs, bundles and doctor
│   ├── data/                     # logical identity/catalog/transfer, adapters and bounded caches
│   ├── execution/                # resource plans, backends and retry policies
│   ├── experiments/              # YAML, sweeps, execution, aggregation, retention
│   ├── integrations/             # third-party compatibility adapters
│   ├── metrics/                  # metric contracts; binary/multiclass/regression families
│   ├── nn/                       # models, losses and neural components
│   ├── hpo/                      # finite and action-centric adaptive optimization
│   ├── observability/            # JSONL events, resources and profilers
│   ├── operations/               # inference, evaluation and export tasks
│   ├── plugins/                  # lazy installed-package extension discovery
│   ├── preprocessing/            # source/transform/sink pipelines and dataset identity
│   ├── runtime/                  # shared cross-process filesystem locks
│   ├── registry/                 # result queries, comparisons, reports and dashboard
│   ├── reproducibility/          # profiles, seed derivation and environment exports
│   ├── schemas/                  # packaged experiment and generic-task JSON Schemas
│   ├── tasks/                    # non-training task plans, execution, results and artifacts
│   ├── storage/                  # artifact stores/references and distributed cache
│   ├── tracking/                 # optional provider logger adapters and dependency guards
│   ├── training/                 # Lightning core plus callbacks/data/orchestration
│   └── workflows/                # task/experiment DAG planning and local execution
├── tests/                        # unit, process and real training smoke tests
└── pyproject.toml                # package and tool configuration
```

The implementation follows the project's Java-influenced object philosophy:

- each implementation `.py` contains one class;
- reusable behaviour lives on objects, class methods or static methods rather than module-level
  utility functions;
- `__init__.py` and `__main__.py` are packaging entry points and are the intentional exceptions;
- enums replace closed sets of internal magic strings;
- YAML keys and fully qualified import paths remain strings because they are external protocol
  boundaries;
- responsibilities stay separated even when that means several small files.

PEP 8 normally favours short lowercase module names; matching class and module names is therefore an
intentional LambdaForge convention, enforced for consistency rather than presented as universal
Python style. Stable lowercase package namespaces and public re-exports keep that choice out of most
consumer imports.

Subpackages are introduced for stable conceptual boundaries, not merely after an arbitrary file
count. Classification is divided into `binary` and `multiclass`, while training separates
`callbacks`, `data` and `orchestration`. `nn.pooling` remains flat despite its size because all of its
classes implement one closely related contract; splitting it into tiny technique-based folders would
make comparison and discovery harder. Public imports are re-exported from `__init__.py`, so these
physical decisions do not leak into normal user code.

The previous root-level `models`, `metrics`, `distances`, `training` and `experiments` trees are now
one installable `src/lambdaforge` package. Imports no longer depend on an unrelated package called
`core` being present on `sys.path`.

## 20. YAML experiment reference

The example file is the canonical template. Top-level blocks are:

| Block | Required | Meaning |
|---|---:|---|
| `schema_version` | yes | Exact quoted Schema compatibility version; current value is `"1.1"`. Historical 1.0 and unversioned files migrate forward before validation. |
| `experiment` | yes | Name, output path, seeds, resume and completion policy. |
| `data` | yes | Train/validation/test objects and data-module configuration. |
| `model` | yes | Model object specification. |
| `losses` | yes | One or more loss objects. |
| `metrics` | no | Backwards-compatible metrics shared by stages unless overridden. |
| `train_metrics`, `val_metrics`, `test_metrics` | no | Explicit split-specific metric lists. |
| `optimizer` | no | Optimizer reference and parameters; defaults to AdamW. |
| `scheduler` | no | Scheduler reference, parameters and optional Lightning metadata. |
| `task` | no | Custom task target or default `LightningTask` parameters. |
| `trainer` | no | `LightningTrainConfig` fields and advanced `trainer_kwargs`. |
| `runner` | no | Custom runner target, callbacks or runner parameters. |
| `callbacks` | no | Additional callback objects constructed from YAML. |
| `sweep` | no | Base inclusion, Cartesian grid and named ablations. |
| `execution` | no | Sequential/parallel/DDP resources. |
| `aggregation` | no | Cross-seed confidence intervals, paired tests and reliability thresholds. |
| `retention` | no | Preview/apply policy for checkpoint roles, verified archives and explicit pruning rules. |

The packaged [JSON Schema](src/lambdaforge/schemas/experiment.schema.json) rejects unknown
framework-owned keys while `metadata` and `extensions` remain explicit task-defined escape hatches.
`lambdaforge validate` additionally checks sweep expansion, resource contradictions and every
`target`/`ref`/plugin import without constructing objects or creating output directories.

The canonical Schema requires `schema_version: "1.1"`. Schema 1.0 remains packaged for exact
historical validation. Files that omitted the field follow the deterministic
`unversioned -> 1.0 -> 1.1` chain; expanded and materialised configurations contain the current
version. `UnversionedToV1Migration` declares 1.0 and `ExperimentV1ToV1_1Migration` adds the
optional retention surface without enabling it.

### Object specifications

`target` imports a callable, recursively builds its parameters and calls it:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    activation: gelu
```

`ref` imports an object without calling it, which is useful for optimizers and collators:

```yaml
optimizer:
  ref: torch.optim.AdamW
  params: {lr: 0.001}
```

`plugin` resolves a named class published by an installed distribution. The explicit `kind` avoids
context-dependent magic strings and lets LambdaForge validate the class contract before creating a
fresh instance:

```yaml
model:
  plugin: {kind: model, name: acme_encoder}
  params: {hidden_features: 128}

val_metrics:
  - plugin: {kind: metric, name: calibrated_auc}
    params: {pred_key: probabilities}

data:
  train:
    plugin: {kind: dataset, name: acme_records}
callbacks:
  - plugin: {kind: callback, name: artifact_marker}
trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
```

Object specifications can be nested inside dictionaries and lists. They execute Python imports and
therefore YAML files must be trusted; this is configuration, not a sandbox.

### Experiment and sweep

```yaml
experiment:
  name: study_name
  output_root: runs/experiments
  seeds: [7, 17, 27]
  resume: true
  rerun_completed: false
  test_after_fit: true
  required_artifacts: [predictions.csv]

sweep:
  include_base: true
  grid:
    model.params.hidden: [[128, 64], [256, 128]]
    optimizer.params.lr: [0.001, 0.0003]
  ablations:
    - name: no_dropout
      set: {model.params.dropout: 0.0}
```

Grid keys are dotted paths and their Cartesian product is materialised for every seed. Ablations are
additional named configurations. Empty grid value lists and duplicate `(variant, seed)` runs are
rejected before training.

With `resume: true`, an incomplete run may resume from its last checkpoint. With
`rerun_completed: false`, a successful run is skipped only if `result.json`, the selected checkpoint
and every relative `required_artifacts` path exist. Required artifacts are generic paths; the
framework contains no domain-specific prediction-file assumption.

### Cross-seed statistical comparisons

Comparisons are paired by common seed. LambdaForge defines `delta = variant - baseline` and flips
its sign for metrics whose declared mode is `min`, so a positive `improvement` always favours the
variant. The complete nested YAML contract is:

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile  # normal or bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon              # sign or wilcoxon
      alternative: two_sided  # two_sided, greater, less, observed_direction
      calculation: auto             # auto, exact, asymptotic
      zero_method: wilcox            # wilcox, pratt, zsplit
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12             # null disables pre-ranking rounding
```

Omitting `aggregation` preserves the historical protocol: a 95% normal interval, the exact paired
sign test with `observed_direction`, `alpha: 0.05`, target power `0.80` and at least three pairs
before a verdict. Percentile bootstrap uses a deterministic stream derived from its base seed and
the comparison identity; batching bounds the transient resampling-index matrix while retaining
`O(resamples)` means for quantiles. Wilcoxon `auto` uses deterministic exact rank enumeration up to
`exact_max_pairs` and a normal approximation above it. Its zero conventions, alternatives,
calculation modes, artifact fields and Python objects are detailed in the
[statistical comparison guide](src/lambdaforge/experiments/statistics/README.md).

### Trainer and data-loader escape hatches

Common Trainer fields are explicit and validated. Future or uncommon Lightning options go under
`trainer_kwargs`:

```yaml
trainer:
  max_epochs: 100
  precision: bf16-mixed
  checkpoint_policy: last_and_best   # none, last, best, last_and_best, all
  checkpoint_monitor: val_auroc
  checkpoint_mode: max
  logger: csv                        # none, csv, lightning_csv, or object spec
  write_epoch_metrics_csv: true      # canonical input for reports/aggregation
  epoch_metrics_include: ["train_*", "val_*", "epoch_time_s"]
  epoch_console_exclude: ["*_loss_binary_cross_entropy_with_logits"]
  trainer_kwargs:
    limit_train_batches: 1.0
    enable_model_summary: true
```

`trainer.logger` may itself be a nested `target` specification for any compatible Lightning logger.
The same applies to top-level `callbacks`. `write_epoch_metrics_csv` keeps LambdaForge's canonical
dense artifact independently of that external logger; disable it only when aggregation is not
needed. The CSV and console table accept shell-style include/exclude patterns, while
`checkpoint_monitor`, `early_stopping_monitor` and their explicit `min`/`max` modes remove dependence
on metric-list ordering.

Task-level publication is independently configurable:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_key: x
    logging:
      log_total_loss: true
      log_individual_losses: true
      loss_prog_bar: false
      metric_prog_bar: true
      logger: false                    # true with an external/Lightning logger
```

The metric lists determine which values are computed. `TaskLoggingConfig` determines which loss
values are published and what reaches the progress bar/logger. `MetricAlias` gives two instances of
the same metric type distinct log names, for example accuracy at two thresholds.

Likewise `LightningDataModule` accepts shared and split-specific `dataloader_kwargs`. Keys controlled
by LambdaForge (`dataset`, `shuffle`, `worker_init_fn`, and the explicit constructor fields) cannot
be overridden through these dictionaries, preventing contradictory configuration.

## 21. Configuration migrations

Preview legacy normalisation before validation or execution:

```powershell
lambdaforge migrate legacy.yaml                 # unified diff, no writes
lambdaforge migrate legacy.yaml --format yaml   # complete resulting YAML
lambdaforge migrate legacy.yaml --format json   # stable result envelope
lambdaforge migrate legacy.yaml --check         # 1 if migration is required, otherwise 0
```

Persistence is always explicit and always targets a different path:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
lambdaforge validate experiment.v1_1.yaml
```

An existing destination requires `--force`; even then the source cannot be overwritten.
`--target-version` accepts an exact `MAJOR.MINOR` string and defaults to the current packaged
Schema. `--format` controls standard output, while `--output` always writes complete YAML
atomically. The default publication is no-clobber even with concurrent writers; `--force` switches
only a distinct destination to atomic replacement.

The current deterministic chain is `unversioned -> 1.0 -> 1.1`. The compatibility-only
`UnversionedToV1Migration` inserts the historical 1.0 declaration.
`ExperimentV1ToV1_1Migration` then advances that valid mapping to 1.1; because the new
`retention` block is optional and defaults to disabled, this step does not opt old experiments into
artifact mutation. Both exact Schemas remain packaged. The preview preserves comments, order,
quotes, anchors and newlines where the structural change permits, never imports user targets or
plugins, and never constructs experiment objects. Normal configuration loading applies the full
chain in memory without editing the source.

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.changed, preview.source_version, preview.target_version)
print(preview.render(MigrationPreviewFormat.DIFF))
```

See the [configuration migration guide](src/lambdaforge/experiments/migrations/README.md) for the
CLI exit-code contract, atomic-write guarantees, full object API, failure modes and the procedure
for adding a future Schema step.

## 22. Execution and process safety

| Mode | Scheduling |
|---|---|
| `sequential` | Runs execute in the caller, one after another. GPU slot fields are ignored. |
| `parallel` | Each run is a spawned process using one logical GPU; `jobs_per_gpu` permits concurrent independent jobs per GPU. |
| `ddp` | Each run receives a group of `devices_per_job` logical GPUs and Lightning runs DDP inside that process. |

GPU numbers are logical positions relative to the parent's `CUDA_VISIBLE_DEVICES`. If the parent has
`CUDA_VISIBLE_DEVICES=4,7`, a job requesting `[1]` sees physical GPU `7` as its local GPU `0`.
The direct object API makes inheritance and CPU-only execution unambiguous:

| `TrainingJob.devices` / scheduler slot | Child visibility |
|---|---|
| `None` | Inherit the complete parent-visible CUDA set unchanged. |
| `[]` or `()` | Set `CUDA_VISIBLE_DEVICES=""` and hide CUDA explicitly. |
| `[0]`, `[1]`, ... | Restrict the child to those logical positions; without a parent variable they are treated as physical IDs. |

Device assignments are frozen and reject strings, booleans, negative/fractional indices and
duplicates. LambdaForge may stage a restricted value while spawning a child, but restores the
parent environment immediately and does not leave `CUDA_VISIBLE_DEVICES` altered.

`ExecutionConfig` applies the same strictness before scheduling: parallel/DDP require a non-empty
GPU list, DDP groups must divide it exactly, counts must be integral and finite, and
`grace_seconds` must be a finite non-negative real number. Booleans are never accepted as numbers;
optional CPU limits use `null` for inheritance, must otherwise be positive, and only
`dataloader_num_workers_per_job` permits zero.

CPU thread, inter-op thread, affinity and data-loader worker limits are applied per job. Use
conservative values when several trainings share a GPU: process concurrency multiplies CPU and RAM
usage even if GPU memory fits.

Process cleanup is layered:

1. Public `TrainingOrchestrator.request_stop()` or managed SIGINT/SIGTERM/SIGBREAK sets the shared
   stop event idempotently; SIGBREAK is installed only where the platform exposes it.
2. Lightning callbacks stop at a train/validation/test batch boundary.
3. The orchestrator waits for `grace_seconds` using shared monotonic deadlines rather than granting
   the full timeout once per process.
4. Remaining descendant trees are terminated recursively with psutil, then killed if necessary;
   any survivor after the bounded escalation is reported as a `RuntimeError`.
5. Each child receives the launcher's exact `expected_parent_pid`. Linux adds `prctl` parent-death
   delivery and POSIX workers also verify/watch that PID, closing the reparenting race.
6. Windows jobs use a kill-on-close Job Object; `DataLoader` workers install the same descendant
   guard and thread limits.

Signal management is explicit. The default `manage_signals=True` is valid only in Python's main
thread and restores the previous handlers after the run. An embedding that runs the orchestrator in
a secondary thread must use `manage_signals=False` and call `request_stop()` from its own lifecycle
hook. Disabling handler ownership does not disable cooperative cancellation.

Windows isolation does not degrade silently. If a native Job Object cannot be initialised,
LambdaForge emits a `RuntimeWarning`, records the detail in `process_isolation_warnings` and retains
portable recursive cleanup. If an active Job Object cannot accept a new worker, that worker is
terminated and the run raises instead of continuing with weaker isolation.

This protects framework-created processes and descendants that remain in the same OS process tree.
No library can guarantee cleanup of a third-party program that deliberately detaches itself into an
unrelated service, so external launchers still need their own lifecycle contract. A
`TrainingOrchestrator` is stateful and not re-entrant: do not overlap `run`/`run_scheduled` calls on
the same object, and prefer one instance per independently owned run.

DDP metrics synchronise their accumulated state before computing non-linear values such as AUROC,
F1 and correlations. Custom metrics used with DDP must implement the distributed-state contract;
LambdaForge raises an error instead of silently averaging invalid per-rank scalars.

## 23. Outputs, resume and loading

Each concrete run has its own directory beneath `<output_root>/<experiment>/<variant>/<seed>/` and
can contain:

- `config.yaml`: fully materialised configuration;
- `environment.json`: UTC timestamp, Python/platform, core package versions, CUDA/cuDNN, visible GPU
  properties, selected CUDA variables, Git state and the distribution/version/group/value of every
  entry-point plugin successfully resolved by this run;
- `hparams.json`: compact hyperparameter summary;
- `train.log`: captured stdout/stderr;
- `metrics.csv`: one dense row per epoch;
- `checkpoints/`: files governed by `checkpoint_policy`;
- `result.json`: terminal status, attempt ID, scientific configuration fingerprint, UTC boundaries,
  paths, duration and best/final metrics;
- `.lambdaforge/attempts/result-*.json`: immutable terminal history retired before a retry;
- project-defined required artifacts.

The suite aggregate area contains per-variant epoch CSVs, seed statistics,
`baseline_comparisons.csv`, `reliability.json`, Benjamini-Hochberg q-values over the selected paired
test and optional PNG plots. The comparison CSV records the fully selected interval/test,
calculation status, effective sample sizes and bootstrap seed provenance. Historical
`ci95_improvement_*` and `p_value_sign_*` columns remain available beside the method-neutral fields.
Aggregation can be regenerated from disk and does not require model reconstruction.

Completion is identity-aware, not just path-aware. LambdaForge fingerprints the expanded
scientific configuration while excluding storage, retry, execution, aggregation and retention
controls. A changed model/data/loss/trainer configuration is therefore never skipped as an old
success and never resumes its incompatible checkpoint. Legacy results are matched against their
materialised `config.yaml` when possible and gain full identity metadata on archival.

Audit current and historical attempts before selecting values for a report or paper:

```bash
lambdaforge results experiment.yaml
lambdaforge results experiment.yaml --duplicates
lambdaforge results experiment.yaml --json --write-index
lambdaforge results experiment.yaml --fail-on-ambiguous
```

`--fail-on-ambiguous` returns 2 when one fingerprint has more than one successful attempt, making
the ambiguity enforceable in CI. `--write-index` atomically publishes
`.lambdaforge/result-index.json`; it is an index, not a second source of truth. Python exposes the
same fresh filesystem scan:

```python
records = experiment.results()  # includes archived attempts
duplicates = experiment.result_catalog().duplicate_groups()
chosen = experiment.result_catalog().select(attempt_id="20260722T...")
```

Never choose an arbitrary “latest” directory for publication. Record the explicit `attempt_id`,
`config_fingerprint`, seed, variant, checkpoint role and aggregate artifact in the paper's evidence
manifest. Multiple successful attempts are retained and reported as ambiguous until the researcher
makes that selection; LambdaForge does not silently decide which favourable result is canonical.

Load a checkpointed model through the suite:

```python
experiment = LambdaForge.experiment("experiment.yaml")
model = experiment.load_model(seed=17, variant="base", which="best")
```

`which` accepts `best`, `last` or `auto`. `auto` resolves best, then last, then the latest safe
local checkpoint; exact `best` and `last` requests never silently cross roles. Loading validates
checkpoint presence and understands both bare model state dictionaries and Lightning keys prefixed
with `model.`. A model class still needs to be importable from the materialised configuration.

## 24. Artifact retention

Schema 1.1 adds an optional, strict `retention` block. The safe starting point is preview mode:

```yaml
retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect: [reports/**, predictions/final.json]
  rules:
    - action: compress
      include: [artifacts/intermediate/**]
      exclude: []
      min_size_bytes: 1048576
      compression: {only_if_smaller: true}
    - action: prune
      include: [scratch/**]
      exclude: []
      min_size_bytes: 0
  archive: {name: artifacts.zip, compression_level: 6}
  lock_timeout_seconds: 60
```

Omitting this block is `disabled` and preserves every artifact. `preview` prints/plans but never
writes, archives or deletes; `apply` allows automatic retention only after a successful final
aggregation. The same boundary is explicit through the API and CLI:

```python
plan = experiment.preview_retention()
result = experiment.apply_retention()
```

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
```

Application requires a current `aggregate/aggregation_receipt.json` proving that every expected
variant/seed is terminal and successful and that committed run/aggregate fingerprints still match.
Core run files, required artifacts, protected globs, aggregate outputs, links/reparse points and
internal metadata cannot be selected by generic rules. Checkpoints use their own unambiguous
best/last policy. Compression streams into immutable per-run ZIPs, verifies member names, CRC,
sizes and SHA-256 before quarantine, and can retain incompressible sources. Pruning and compression
use a durable journal and reversible quarantine.

Training, final aggregation and retention coordinate with cross-process activity, aggregation and
retention locks in a fixed order. A crash before commit rolls back; a crash after the commit marker
finishes forward, and reapplying a committed plan is idempotent. See the
[artifact-retention guide](src/lambdaforge/experiments/retention/README.md) for the complete YAML
contract, eligibility receipt, transaction states, artifacts and limitations.

## 25. Built-in components

- Models: configurable `MLP`/`CNN2D`; `ECMP`; GCN, GraphSAGE, GAT, GATv2, relational GCN, PNA,
  sparse GraphTransformer, EGNN and GIN stacks plus `GraphReadout`; `GradTree`, `GRANDE`,
  oblivious trees and `NODE`; RNN/LSTM/GRU,
  Transformer encoder/decoder/seq2seq, Conformer, state-space adapter and temporal convolution;
  `DeepSets`/`SetTransformer`; residual MLP, `FTTransformer`, TabNet, SAINT, AutoInt and DeepFM;
  `ResNet2D`/`ConvNeXt2D`/`MobileNetV2`, `VisionTransformer2D`, `UNet2D` and
  `FeaturePyramidNetwork2D`; autoencoders, ensembles, mixture-of-experts, multitask and siamese
  composition; VQ-VAE and Gaussian diffusion; Neural ODE/CDE, DeepONet, Fourier neural operator,
  scalar/vector tensor fields and optional higher-order equivariant adapters; and `SIREN` implicit
  representations.
- Activations: CELU, ELU, GELU, hard sigmoid/swish, Identity, LeakyReLU, Mish, PReLU, ReLU/ReLU6,
  SELU, Sigmoid, SiLU, Softplus, Softsign, SquarePlus, Tanh, Entmax15/Entmoid15, Sine/Snake and the
  dimension-changing GLU/GEGLU/SwiGLU/ReGLU family.
- Normalisations: BatchNorm and InstanceNorm (1D/2D/3D), ChannelLayerNorm, GroupNorm, IdentityNorm,
  L2Norm, LayerNorm, RMSNorm and ScaleNorm.
- Pooling: basic and smooth reductions, statistics/concatenation, learned attention, top-k and
  probability operators for dense masked sets, plus indexed sparse sum/mean/max/attention.
- Pairwise components: Euclidean, squared Euclidean, Manhattan, Minkowski, Chebyshev, cosine,
  angular and Mahalanobis distances; dot, cosine and bilinear similarities; RBF, Laplacian and
  polynomial kernels.
- Losses: binary/multiclass cross-entropy and focal objectives, MSE, MAE, Smooth L1, Huber,
  Dice/Tversky, contrastive, triplet-margin, InfoNCE and a reusable beta-VAE objective. Every
  built-in training loss reduces to a scalar and exposes mapping keys, weight and stable name.
- Encodings and regularization: sinusoidal, learned, rotary and Fourier features; DropPath,
  feature dropout and Gaussian noise.
- Data and uncertainty: `CategoricalFeatureEncoder`, `FileDataset`, `NumpyMemmapDataset`,
  `DatasetCache`, fingerprinted safe NumPy/Torch
  serialization, checksum/HMAC envelopes and coordinated disk/mmap backends. Pickle remains an
  explicit trusted-local compatibility option. `TemperatureScaler` and
  `ConformalPredictionInterval` provide held-out post-hoc uncertainty components.
- Conformance: `ArchitectureConformanceCase` and `ArchitectureConformancePack` capture provenance,
  initialization state, parameter count, output tensor and tolerances in tiny weights-only
  references, then fail on numerical/shape drift.
- Binary metrics: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, Cohen's
  kappa, exact AUROC/AUPRC and fixed-memory `StreamingBinaryAUROC`/`StreamingBinaryAUPRC`.
- Multiclass metrics: accuracy, balanced accuracy, F1, exact AUROC/AUPRC and fixed-memory
  `StreamingMulticlassAUROC`/`StreamingMulticlassAUPRC` with macro, weighted or micro reduction.
- Regression metrics: MAE, MSE, RMSE, R², Pearson and Spearman correlation, plus `MeanMetric`.

Short shape-preserving activation and normalisation names are case-insensitive in YAML. Custom
aliases can be added through `ComponentRegistry`; Python model constructors also accept compatible
classes directly. GLU-family activations remain explicit objects because they halve a dimension and
cannot safely replace an ordinary per-layer activation.
See the [neural-components guide](src/lambdaforge/nn/README.md) and
[metrics guide](src/lambdaforge/metrics/README.md) for contracts and shapes.

### Advanced graph and equivariant models

The native graph path uses ordinary PyTorch tensors and sparse directed edges; PyG/DGL and dense
adjacency are not required. For every `edge_index=(2,E)`, row zero is `source` and row one is
`destination`.

| Input | Contract |
|---|---|
| `x` | Floating node features `(N,in_channels)`. |
| `edge_index` | Integer `(2,E)` with indices in `[0,N)`; floating and Boolean indices are rejected. |
| `edge_features` | Optional real `(E,edge_channels)`, required when `edge_channels > 0` and normalized to `x` device/dtype. |
| `edge_types` | Relational-GCN integer relation ids `(E,)` in `[0,num_relations)`. |
| `coordinates` | EGNN floating coordinates `(N,D)`, `D >= 1`, with exactly the same device/dtype as `x`. |

| Family | Result, controls and primary source |
|---|---|
| `GATv2` | `(N,out_channels)` dynamic multi-head attention with `hidden_channels`, per-layer `heads`, `concatenate_heads`, `share_weights`, feature/attention dropout, negative slope, self-loop policy/fill, residual and bias; shared `edge_channels`; hidden-only activation/normalization and kwargs. `GATv2Layer.forward_with_attention` returns aligned routed edges and `(E_routed,heads)` weights for one layer. Inspired by [Brody et al.](https://arxiv.org/abs/2105.14491). |
| `RelationalGCN` | `(N,out_channels)` from `num_relations` typed transforms; `num_bases`, `aggregation` (`sum`/`mean`), `message_chunk_size`, dropout, residual, root weight and bias are scalar/per-layer; activation/normalization and kwargs are hidden-only. Messages are grouped by relation and projected in bounded chunks instead of materializing per-edge matrices. Inspired by [Schlichtkrull et al.](https://arxiv.org/abs/1703.06103). |
| `PNA` | `(N,out_channels)` using non-empty, duplicate-free mean/min/max/std `aggregators` crossed with identity/amplification/attenuation/linear/inverse-linear `scalers`. It configures edge/message widths, pre/post MLP widths, degree statistics, epsilon, dropout, activation/kwargs and bias; normalization/residual are hidden-only. `layer_kwargs` can override all layer-owned choices per layer while stack input/output/edge widths remain reserved. Inspired by [Corso et al.](https://arxiv.org/abs/2004.05718). |
| `GraphTransformer` | `(N,out_channels)` from local sparse dot-product attention. Per-layer controls are heads/concatenation, feed-forward width, activation/normalization and kwargs, three dropouts, self-loop fill, pre/post norm, residual, beta gate and bias; edge features modify keys and values. Related to [Shi et al.](https://arxiv.org/abs/2009.03509). |
| `EGNN` | `(N,out_channels)` features or a mapping of features and updated `(N,D)` coordinates. Per-layer `message_channels`, feature dropout, residual, bias and `layer_kwargs` configure message/node/coordinate MLP widths, aggregation, dropout, displacement normalization/scaling, coordinate updates/tanh and optional message attention. Hidden activation/normalization are stack policies; output keys are configurable. Inspired by [Satorras et al.](https://arxiv.org/abs/2102.09844). |

A graph stack has `L_graph = len(hidden_channels) + 1` layers. A scalar per-layer option is
broadcast; a list must have exactly `L_graph` entries. Hidden-only lists have
`len(hidden_channels)` entries. GATv2 concatenated widths must be divisible by their head counts;
R-GCN basis counts cannot exceed `num_relations`; GraphTransformer `beta=true` requires a residual
path. Attention self-loops replace existing loops and synthesize aligned edge rows rather than
duplicating topology. Empty edge lists and isolated nodes remain finite.

R-GCN's default `message_chunk_size: 65536` bounds the projected message tensor while preserving
exact sparse `sum`/`mean` results. Set a smaller positive integer for constrained devices; `None`
removes that bound and should be used only when the edge count and memory budget are known.

PNA's `average_degree = mean(in_degree)` and
`average_log_degree = mean(log(in_degree + 1))` must be computed **only on the training
split/topology**, recorded in YAML and reused unchanged for validation, test and inference. Computing
them over held-out graphs leaks topology. Both statistics and `epsilon` must be positive and finite.

`EGNN.forward` returns only node features by default. With `output_mode: mapping` it returns the
configured `feature_output_key` and `coordinate_output_key`; `LightningTask` preserves that mapping,
so each loss can select its `output_key` and each metric its documented prediction/output key
(`pred_key` for most built-ins). `forward_with_coordinates` always returns
the pair. A complete [PNA YAML example](src/lambdaforge/nn/README.md#pna-with-training-only-degree-statistics)
and [EGNN mapping example](src/lambdaforge/nn/README.md#egnn-mapping-output) document named
`model_input_keys` and the relevant per-layer list lengths.

These implementations are dependency-light native cores, not reproductions of author training
pipelines. They make no checkpoint or benchmark parity claim. Sparse attention/message routing grows
broadly as `O(EH)` rather than allocating a global `N²` attention matrix; GraphTransformer therefore
attends only over supplied edges and has no implicit global or positional encoding. EGNN is
E(n)-equivariant for scalar node/edge features and coordinate updates, not scale-equivariant and not
a higher-order vector/tensor representation.

## 26. Extension contracts

### Model

Subclass `torch.nn.Module` or `lambdaforge.nn.models.Model`. Implement `forward(*args, **kwargs)`.
The base `predict` helper switches to evaluation/inference mode and restores the prior training
state. For the default task, either return a tensor (wrapped under `model_output_key`) or a mapping.

### Loss

Subclass `Loss` and implement `forward(outputs, batch, context=None) -> Tensor`. `LightningTask`
passes itself as `context`, while the default keeps the loss independently callable. Give every loss
a stable `name` and use mapping keys rather than task assumptions. Multiple losses are summed after
their configured weights are applied.

### Metric

Implement `update`, `compute` and `reset`. For DDP, also expose `distributed_state` and
`merge_distributed_state`, or deliberately use a framework metric. Metric instances are deep-copied
per stage so train, validation and test state never leaks.

Every stage requires unique metric names. Wrap a metric in `MetricAlias` when the same class is used
more than once with different parameters. Explicit stage lists avoid deepcopy requirements for a
project metric that owns a non-copyable external resource.

### Data and task

The default `LightningTask` expects mapping batches. `model_input_key` selects one tensor;
`model_input_keys` routes a sequence of positional inputs or a mapping of model argument names to
batch keys. Loss/metric objects select their own target/output keys. Tuple batches, multiple
optimizers, manual optimization or unusual control flow still use a custom `task.target`; all other
experiment, process and artifact machinery remains reusable.

`DatasetCache` is an optional map-style wrapper, not an implicit global cache. Its
`max_memory_bytes_per_process` budget counts retained serialized payload bytes and is paired with
`max_memory_entries`; it does not claim to cap total process RSS, live batches, prefetching or the
wrapped dataset. Worker RAM caching is off by default because every DataLoader worker owns a dataset
replica. Cache only deterministic loading/preprocessing, never random augmentation results. See the
[data guide](src/lambdaforge/data/README.md) before enabling it in parallel or DDP jobs.

### Installed plugins

External distributions can publish models, metrics, neural components, datasets, Lightning
callbacks/loggers and generic tasks through the canonical groups documented in the
[plugin guide](src/lambdaforge/plugins/README.md). Datasets must inherit PyTorch `Dataset`;
callbacks/loggers inherit the public `lambdaforge.integrations.Lightning` bases. Discovery reads
metadata only; resolving a selected plugin imports provider code and therefore has the same trust
boundary as `target`.
Built-ins and aliases registered explicitly in the current process keep precedence over
activation/normalisation plugins.

Each real run uses an isolated `PluginUsageSession` and atomically stores its canonically ordered
descriptors in `environment.json`. Earlier validation, sequential runs, installed-but-unused
providers and failed resolutions are excluded; cache hits and external component aliases actually
used by the run are included. Dry-runs record an empty list. See the
[provenance contract](src/lambdaforge/plugins/README.md#loaded-plugin-provenance).

### Tracking loggers

`trainer.logger` accepts the public `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` and `WeightsAndBiasesTrackingLogger` targets, or a non-empty list mixing
them with project loggers and installed logger plugins. Each adapter checks its own optional extra
only when constructed; importing LambdaForge remains provider-free. The canonical dense
`metrics.csv` is controlled separately by `write_epoch_metrics_csv`. Task losses/metrics reach the
provider only when `task.params.logging.logger` is enabled.

Provider credentials must stay outside YAML because materialized configurations are durable run
artifacts. Checkpoint upload is opt-in through `log_model` and independent of LambdaForge's local
retention transactions. See the [tracking guide](src/lambdaforge/tracking/README.md) for complete
parameters, local/remote and offline/online examples, privacy boundaries and failure behaviour.

### Runner

A custom runner must provide compatible `fit` and `test` methods. Configure it through
`runner.target`; its parameters are recursively object-built. Extra callbacks can therefore also be
declared as YAML object specifications.

All these extension objects are recursively constructed by `ObjectFactory`: models may be any
`torch.nn.Module`; losses subclass `Loss`; metrics subclass `Metric`; loggers and callbacks implement
their Lightning contracts. Custom batch structures, multiple optimizers or a different backend are
handled by replacing `task.target` or `runner.target`, without changing the experiment engine.

## 27. Review findings

The repository-wide review found and resolved the following structural or correctness risks:

| Finding in the previous layout | Resolution |
|---|---|
| Root folders were not an installable package and imports referenced external `core`/stale `coreold` namespaces. | Added `pyproject.toml`, a `src/lambdaforge` package and absolute self-contained imports. |
| Models, distances, experiments, metrics and training competed as root-level entry points. | Introduced the `LambdaForge`/`Experiment` API and cohesive public subpackages with stable re-exports. |
| Large modules contained many loose functions or several unrelated classes. | Converted behaviour to collaborator objects and split every implementation class into its own file. |
| `training` and `metrics.classification` had become visually dense. | Split only along stable contracts (`callbacks`, `orchestration`, `binary`, `multiclass`) while preserving public imports; cohesive families such as `pooling` remain flat. |
| Closed choices and component names relied on repeated string literals. | Added enums and `ComponentRegistry`; strings remain only at YAML/serialization boundaries. |
| Advanced Trainer/DataLoader settings required source edits. | Added validated `trainer_kwargs` and shared/per-split `dataloader_kwargs` forwarding. |
| One shared metric list and implicit monitor order limited experiment control. | Added per-stage metric lists, aliases, explicit monitor modes, loss publication policy and CSV/console filters. |
| Selecting a custom Lightning logger removed the canonical epoch CSV. | Separated external logger selection from `write_epoch_metrics_csv`, so reporting remains available by default. |
| YAML mistakes were discovered late during object construction. | Published a strict Draft 2020-12 schema and added `ExperimentValidator`, `ValidationReport` and `lambdaforge validate`. |
| Runs did not record enough software/hardware/plugin provenance. | Added typed, atomically written `EnvironmentManifest` artifacts and isolated successful-plugin usage sessions for real runs; dry-runs stay import-free. |
| DDP could average already-computed AUROC/F1/correlation scalars, which is mathematically invalid. | Metrics now gather and merge their state before computing; unsupported custom metrics fail explicitly. |
| Dataset reuse had no bounded memory contract and could silently multiply across workers. | Added an opt-in per-process serialized LRU, entry/byte limits, spawn/fork isolation and explicit worker policy. |
| Persistent cache writers could race, exceed quota after a crash, reuse stale transforms or deserialize unchecked bytes. | Added OS shared/exclusive locks, immutable namespace manifests, pre-eviction, crash reconciliation, generation tokens, explicit fingerprints, verified checksum/HMAC envelopes and a bounded non-pickle NumPy/Torch codec. |
| External classes required fully qualified paths even when distributed as reusable packages. | Added lazy entry-point discovery, explicit contracts for neural objects plus datasets/callbacks/loggers, YAML integration, conflict checks and a non-importing CLI listing. |
| Exact curve metrics retained every score and target. | Added binary and multiclass histogram AUROC/AP alternatives with fixed persistent state and bounded tensor `all_reduce` synchronization. |
| The built-in neural catalog was limited to a dense network, a CNN and one message-passing model. | Added task-neutral graph, differentiable-tree, sequence, set, tabular, vision, composition and implicit-representation families plus broader component categories. |
| The default task routed only one model tensor and offered no named optimizer groups. | Added positional/named `model_input_keys` and opt-in per-group optimizer settings exposed by models. |
| `CNN2D` selected 1D BatchNorm by default for NCHW tensors. | Its built-in default now creates `BatchNorm2d`. |
| `Model.predict` did not guarantee restoration of the previous training mode. | Inference is guarded by `try/finally` and restores the original state. |
| `test_after_fit` asked a new Trainer for an unknown `best` checkpoint and omitted its stop event. | It now uses the actual fitted checkpoint when available, otherwise current weights, and preserves cancellation. |
| Forced process termination focused on root workers and was fragile on Windows. | Added recursive psutil cleanup, Job Objects, parent-death guards and guarded data workers. |
| Completion assumed a domain-specific prediction artifact. | Replaced it with generic relative `required_artifacts`. |
| Documentation described obsolete modules/scripts and there was no root guide. | Replaced it with linked English/Spanish guides whose claims are checked against current code. |

These facilities remain explicit rather than automatic: researchers choose the cache budget, plugin
provider and exact-versus-streaming metric semantics in YAML, so adding convenience does not hide
resource or scientific trade-offs.

## 28. Development and verification

```powershell
ruff format --check src tests
ruff check src tests
mypy src\lambdaforge
pytest -q
```

The current suite covers configuration expansion, object/plugin construction, model validation,
aggregation, spawned process scheduling, the structural POO rules, a real one-epoch Lightning CPU
fit, schema/CLI validation, environment capture and external model/loss/metric/logger/callback
construction from YAML, installed dataset/callback/logger entry points, cache quotas/isolation,
HMAC corruption and substitution, spawn races,
crash recovery, mmap leases, lazy/mapped datasets, streaming metric state and retention
preview/apply, receipt, lock, ZIP-verification, rollback and concurrent-idempotence scenarios.
Plugin tests additionally cover exact distribution metadata, cache hits, validation/run isolation,
failed construction and a real `spawn` child manifest.
Adaptive-HPO tests cover categorical permutation/conditional masks, mixed multi-fidelity BoTorch,
pending points and safe fallback; analytical seed uncertainty, slow starters, paired racing,
probabilistic pruning and Gaussian action VoI; feature-aware/censored memory, explicit capacity
states, async dispatch, durable state and confirmation. Synthetic collaborators avoid neural
training for controller logic. Real CUDA paths cover cumulative checkpoint continuation without
repeated epochs, candidate-aware preflight, isolated OOM, allocator caps and concurrent single-GPU
trials. A two-GPU smoke test runs only where two logical devices are visible.
The process-integration tests create a real launcher/worker/descendant tree. POSIX delivers an
actual process-group `killpg(SIGINT)`; Windows asks the launcher to raise a targeted Python SIGBREAK
because a native console control event would affect the whole test group. A separate scenario
hard-terminates the launcher and verifies that every recorded descendant and temporary file is gone.
Emergency cleanup in each test also prevents a failed assertion from leaving workers behind.

### Repository and release hygiene

Git should contain the framework source, tests, Schemas, examples, human/agent documentation,
workflows, icons and packaging metadata. It must not contain virtual environments, credentials,
bytecode/tool caches, built wheels, profiler/test reports or local experiment/tracking output. The
root `.gitignore` covers those generated categories, including `.lambdaforge/`, `runs/`, provider
directories, dashboards and SLURM stdout/stderr. The generated consumer scaffold applies the same
essential rules.

Do not solve hygiene by ignoring broad scientific extensions such as `*.yaml`, `*.json`, `*.csv`,
`*.pt` or an entire `data/` tree: Schemas, experiment protocols, small test fixtures and reviewed
reference assets may legitimately need version control. Put runtime output below `runs/` and decide
explicitly whether every larger dataset/checkpoint is an external artifact or a reviewed repository
asset. `.env.example` may document variable names, but real `.env*` files are ignored and must never
contain committed credentials.

Before a release commit, review both visible and ignored state rather than relying on a clean-looking
IDE:

```bash
git status --short
git status --ignored --short
git diff --check
ruff format --check . && ruff check .
mypy src/lambdaforge && pytest -q
python -m pip wheel . --no-deps --wheel-dir /tmp/lambdaforge-wheel-check
```

The wheel must contain the packaged Schemas, specialist READMEs, `AGENTS.md`, changelog,
architecture documents and runnable examples. Selecting and adding a licence remains the only owner
legal decision before third-party redistribution; do not infer a licence from public source access.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs automatically for pull requests, pushes
to `main`, version tags and
manual dispatches. It grants only read access to repository contents, cancels stale runs for the
same ref, gives every job a timeout, caches pip downloads from `pyproject.toml` and retains reports
for 14 days. Its independent jobs are:

- formatting, linting and type checking on CPython 3.10;
- the complete CPU suite on Ubuntu and Windows for every currently stable supported CPython release,
  3.10 through 3.14, using the CPU-only PyTorch index and hiding CUDA explicitly;
- the complete suite with only `pytorch-lightning==2.2.*` installed, plus an assertion that the
  central adapter really selected the legacy namespace rather than modern `lightning`;
- source/wheel builds, `twine` metadata validation and an isolated wheel-content smoke test on
  CPython 3.14. JUnit reports and built distributions are uploaded as uniquely named artifacts.

[`.github/workflows/cuda.yml`](.github/workflows/cuda.yml) is deliberately manual and targets a
self-hosted runner labelled
`[self-hosted, linux, x64, cuda]`. The runner needs Actions Runner 2.327.1 or newer for the Node 24
[action versions](https://github.com/actions/setup-python/releases/tag/v6.0.0) used by these
workflows, a compatible NVIDIA driver and a CUDA-enabled PyTorch build. A preflight rejects CPU-only
or unusable installations and records the exact PyTorch, CUDA, cuDNN and device details. The
workflow then runs the `cuda`-marked one-epoch public-YAML training test and the remaining
regression suite. Merely defining, queuing or skipping this workflow is not CUDA coverage; only a
successful execution on the real runner is evidence.

Changes to GPU scheduling or cleanup should additionally be exercised on the target multi-GPU host
and interrupted manually at least once.

Every source module and class has a docstring. The repository audit also checks class/module name
matching, the one-class-per-file rule and absence of module-level helper functions in implementation
modules.

## 29. Current limitations

- `DatasetCache` bounds retained serialized payloads per process, not total RSS. DataLoader batches,
  prefetching, pinned memory, allocator overhead and the source dataset remain outside that budget;
  enabling worker caches multiplies the configured allowance across process replicas.
- Pickle remains the compatibility default and can execute code; select the safe NumPy/Torch codec
  for supported sample trees or keep pickle strictly local and trusted. A checksum is not
  authentication; HMAC must be configured explicitly and provides no encryption.
- Dataset/transform fingerprints remain explicit because arbitrary transform semantics cannot be
  inferred. `DatasetCache` coordinates cooperating processes on one filesystem; cross-machine
  staging uses `DistributedArtifactCache` with a shared lease directory and an explicit upstream.
- Lightning is the only built-in training backend.
- The default task assumes mapping-shaped batches; it routes one or several model inputs, while
  tuple batches and manual/multiple-optimizer flows need a custom task object.
- Exact binary and multiclass curve metrics still retain predictions. Their fixed-memory streaming
  alternatives introduce binning approximation; multiclass state grows as `O(num_classes * num_bins)`.
- Entry-point discovery covers reusable neural contracts, datasets, callbacks, loggers and generic
  tasks. Data modules and experiment runners remain fully supported through `target` and
  intentionally have no dedicated groups.
- Plugin provenance covers resolutions in the run process/context; user-created child processes
  require explicit IPC if their independently loaded plugins must be attributed to the parent.
- Statistical summaries are useful exploratory tools, not a substitute for a study-specific
  protocol. Normal intervals and asymptotic Wilcoxon remain approximations when explicitly selected
  or chosen by `auto` for larger paired samples.
- Experiment Schemas 1.0/1.1 and task/workflow 1.0 are current. Experiment migrations have no
  downgrade, in-place rewrite or remote source. Composed task secrets redact safely; experiment and
  workflow snapshots reject secrets and expect provider credentials to remain in the environment.
- Experiment retention is local-filesystem only and uses ZIP/Deflate. Preview can become stale by
  design; apply replans under locks. `ArtifactStore`/`S3ArtifactStore` provide publication/staging,
  but remote deletion and lifecycle policy deliberately remain provider-owned.
- MLflow, TensorBoard and Weights & Biases tracking adapters are optional. Provider
  authentication/network/storage, remote retention and service availability remain external;
  tracker failures fail the owning run, and LambdaForge retention cannot remove uploaded artifacts.
  Tracking is not the result source of truth.
- Workflow execution remains bounded and local. Workflow plans record `on` placement, and the
  control plane submits individual configs to local/SSH plus local/SLURM providers, but 0.5 does
  not pretend to coordinate mixed-cluster DAG artifact transfer or durable dependency recovery.
- Managed bootstrap installs exact framework/consumer wheels into a user-space venv; it cannot
  synthesize platform/CUDA dependency wheels, install drivers, load site modules or build a
  container. Offline sites need a compatible wheelhouse. Existing environments remain user-owned.
  Built-in replication is local/SSH rsync over predeclared locations.
- Cluster selection is explicit in 0.5.2. Profiles do not auto-discover total capacity, queue delay or
  monetary cost and the control plane does not claim optimal placement. `DataCatalog` resolves
  direct experiment splits and nested typed markers; arbitrary untyped strings stay project-owned.
- Remote result sync is allowlisted and per-file bounded, not a remote filesystem mirror. Heavy
  artifacts require explicit logical fetch. Live plotting polls small files; it is not a streaming
  server. Matplotlib is core, while interactive HTML/graph/mesh providers are optional extras.
- Finite random/Optuna remains available. Adaptive HPO dynamically schedules local independent
  trials; integration with workflow DAG resources, DDP actions and remote pruning callbacks is not
  implicit.
- Mixed BoTorch models fidelity and categorical Hamming geometry. The Gaussian action KG and
  Bayesian curve bases are documented approximations, not an exact universal joint posterior for
  every training domain. Resource features and representative probes remain consumer-owned.
- The S3-compatible store depends on client semantics and checksum metadata; it does not yet expose
  multipart-resume, provider-side leases or destructive lifecycle operations. The distributed cache
  requires a coherent shared filesystem for its lease directory.
- The local dashboard is a static read-only snapshot, not a long-running multi-user service. Its
  comparison intervals are objective normal approximations; use the experiment aggregation
  bootstrap/Wilcoxon tools and a study protocol for publication claims.
- Advanced graph families are native sparse cores without paper-checkpoint/benchmark parity.
  GraphTransformer is local to `edge_index`; PNA statistics are explicit training-split inputs; EGNN
  covers E(n) scalar-feature equivariance, not scale equivariance or higher-order tensor features.
- Hosted CI covers CPU behaviour on Ubuntu/Windows and CPython 3.10-3.14, including POSIX
  process-group SIGINT, targeted Windows Python SIGBREAK and hard-launcher-death scenarios. It does
  not exercise native Windows console-group CTRL_C/CTRL_BREAK delivery. A shared
  `TrainingOrchestrator` instance is not re-entrant, and a detached external daemon is outside its
  process-tree contract. Real CUDA and multi-GPU/DDP remain host-dependent; CUDA is covered only
  after the manual self-hosted workflow has succeeded.

## 30. Why AGENTS.md exists

An AI coding agent should not need to read hundreds of implementation modules and every specialist
README before it can configure a model or add a loss. That approach consumes context and money,
increases the chance that an early constraint is forgotten, and encourages the agent to infer APIs
from internal files that are not stable.

[AGENTS.md](AGENTS.md) and its synchronized [Spanish edition](AGENTS.es.md) are therefore the
framework's single, token-efficient operational manual. It
contains the complete capability map, the supported public boundaries, installation and YAML
workflow, extension recipes, result-publication rules, verification commands and a small routing
table for the rare case that deeper detail is needed. The intended agent workflow is:

1. Read `AGENTS.md` once.
2. Select an existing public object or extension contract from its catalogue.
3. Inspect only that object's signature/docstring or the one owner guide named by the routing table.
4. Validate and test through the documented public commands.

This does not replace the bilingual READMEs for humans or the precise class docstrings. It is a
compressed index and safety contract that prevents repository-wide context loading. Agents working
in this checkout discover it automatically; when LambdaForge is consumed from another workspace,
give the agent this file explicitly or reference its repository path from that project's own
`AGENTS.md`. Wheels install both editions under `share/lambdaforge`; obtain
its exact environment path without importing the framework with:

```bash
python -c "from importlib.metadata import distribution; print(distribution('lambdaforge').locate_file('share/lambdaforge/AGENTS.md'))"
```

## 31. Documentation map

- [Single-file agent manual](AGENTS.md) · [Español](AGENTS.es.md)
- [Technical architecture and class collaboration](docs/ARCHITECTURE.md) · [Español](docs/ARCHITECTURE.es.md)
- [Clusters and persistent jobs](docs/CLUSTERS.md) · [Español](docs/CLUSTERS.es.md)
- [Cluster credential/scheduler security](docs/SECURITY.md) · [Español](docs/SECURITY.es.md)
- [Results and plots](docs/RESULTS.md) · [Español](docs/RESULTS.es.md)
- [Artifact inspection](docs/ARTIFACTS.md) · [Español](docs/ARTIFACTS.es.md)
- [Preprocessing execution/debug](docs/PREPROCESSING.md) · [Español](docs/PREPROCESSING.es.md)
- [Authoring and configuration](src/lambdaforge/configuration/README.md) · [Español](src/lambdaforge/configuration/README.es.md)
- [Control plane](src/lambdaforge/controlplane/README.md) · [Español](src/lambdaforge/controlplane/README.es.md)
- [Adaptive optimizer internals](docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md) · [Español](docs/ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md)
- [Changelog](CHANGELOG.md) · [Versioning/deprecation](docs/GOVERNANCE.md) · [Español](docs/GOVERNANCE.es.md) · [Security](SECURITY.md)
- [Experiment system](src/lambdaforge/experiments/README.md) · [Español](src/lambdaforge/experiments/README.es.md)
- [Configuration migrations](src/lambdaforge/experiments/migrations/README.md) · [Español](src/lambdaforge/experiments/migrations/README.es.md)
- [Artifact retention](src/lambdaforge/experiments/retention/README.md) · [Español](src/lambdaforge/experiments/retention/README.es.md)
- [Statistical comparisons](src/lambdaforge/experiments/statistics/README.md) · [Español](src/lambdaforge/experiments/statistics/README.es.md)
- [Data and caching](src/lambdaforge/data/README.md) · [Español](src/lambdaforge/data/README.es.md)
- [Generic tasks](src/lambdaforge/tasks/README.md) · [Español](src/lambdaforge/tasks/README.es.md)
- [Preprocessing](src/lambdaforge/preprocessing/README.md) · [Español](src/lambdaforge/preprocessing/README.es.md)
- [Training and processes](src/lambdaforge/training/README.md) · [Español](src/lambdaforge/training/README.es.md)
- [Neural components](src/lambdaforge/nn/README.md) · [Español](src/lambdaforge/nn/README.es.md)
- [Metrics](src/lambdaforge/metrics/README.md) · [Español](src/lambdaforge/metrics/README.es.md)
- [Installed plugins](src/lambdaforge/plugins/README.md) · [Español](src/lambdaforge/plugins/README.es.md)
- [Optional experiment tracking](src/lambdaforge/tracking/README.md) · [Español](src/lambdaforge/tracking/README.es.md)
- [Complete YAML example](examples/experiment.yaml)
- [Runnable preprocessing example](examples/preprocessing.yaml)
- [Concise preprocessing example](examples/preprocessing-simple.yaml)
- [Cluster catalogue example](examples/lambdaforge.clusters.yaml) · [Data catalogue example](examples/data-catalog.yaml)
- [Workflow example](examples/workflow.yaml)
- [Adaptive HPO example](examples/adaptive-hpo.yaml)

Each sub-guide links back here and to its translation. Class docstrings are the most precise source
for individual constructor arguments.

## 32. Roadmap

The roadmap lives here so status cannot drift into a separate planning document. “Completed” means
a public implementation, documentation and focused tests; it does not claim that every external
provider or research method is built in.

| Priority | Capability | 0.5 status |
|---:|---|---|
| 1 | Generic task contract | Completed |
| 2 | Independent task Schema/configuration | Completed |
| 3 | Task validation | Completed |
| 4 | Immutable task plans | Completed |
| 5 | Unified facade and CLI dispatch | Completed |
| 6 | Task results and provenance | Completed |
| 7 | Typed, hashed artifacts | Completed |
| 8 | Composable preprocessing | Completed |
| 9 | Resume and deterministic shards | Completed |
| 10 | Versioned dataset artifacts | Completed |
| 11 | Workflow DAG | Completed: local bounded runner and task/experiment nodes |
| 12 | Configuration composition | Completed: include/extends/merge/delete/cycles |
| 13 | Safe interpolation and secrets | Completed with persist-safe task redaction |
| 14 | Provenance and semantic diff | Completed |
| 15 | First-class CPU scheduling | Completed |
| 17 | Execution backend contract | Completed |
| 18 | SLURM/HPC adapter | Completed at explicit plan/submission boundary |
| 19 | Failure/retry/preemption semantics | Completed: taxonomy, retry, attempt modes and SLURM requeue |
| 20 | Inference/evaluation/ensemble/export tasks | Completed |
| 21 | HPO | Completed: finite random/Optuna plus asynchronous action-centric multi-fidelity optimization, adaptive seeds, cost/VRAM admission, persistence and optional BoTorch |
| 22 | Distributed cache | Completed at shared-filesystem lease boundary |
| 23 | Artifact stores and references | Completed: local/shared/S3-compatible staging |
| 24 | Experiment registry and exports | Completed |
| 25 | Cross-experiment comparison/reports | Completed without generated conclusions |
| 26 | Optional local dashboard | Completed as static read-only HTML |
| 27 | Structured observability | Completed: events/resources/profiler/OOM taxonomy |
| 28 | Reproducibility profiles | Completed |
| 29 | CLI/IDE ergonomics and examples | Completed: init/explain/target/compose/diff and tested examples |
| 30 | Adoption/governance | Code/docs complete; owner licence selection remains a legal decision |
| 31 | Concise AuthoringConfig -> strict MaterializedConfig | Completed with Authoring Schema 1.0 and `inspect --resolved` |
| 32 | Named task inputs/outputs and concise preprocessing | Completed; legacy path APIs remain compatible |
| 33 | Logical dataset and code identity | Completed with strict/manifest/dataset-ID/version providers and Git/distribution/explicit code identity |
| 34 | Explicit idempotency policy | Completed with default reuse plus `--force`, `--restart` and `--no-resume` |
| 35 | Portable multi-cluster control plane | Completed for local/SSH transport and local/SLURM scheduling |
| 36 | Persistent job service | Completed with list/status/logs/cancel/retry and JSON records |
| 37 | Explicit data placement/replication | Completed with configurable catalogs, preflight refusal and rsync provider |
| 38 | Mixed-cluster workflow coordinator | Deferred intentionally; planning records placement but execution refuses until artifact transfer and durable recovery are sound |
| 39 | Stabilize cross-platform CI and installed-wheel verification | Completed: Windows fsync, dynamic refill, atomic metric CSV and isolated wheel smoke |
| 40 | Real preprocessing workload semantics | Completed: sequential, threaded I/O/auto, spawn CPU and safe single-worker GPU |
| 41 | Friendly training plus portable experiment datasets | Completed: concise aliases, resources, direct/nested logical references and path-independent identity |
| 42 | Managed/offline cluster environment | Completed: exact local wheels, content identity, user venv, bootstrap/doctor and wheelhouse |
| 43 | Durable job/result reconnection | Completed: filters/follow, lightweight sync and explicit artifact fetch |
| 44 | Stable ResultService and MetricSeries | Completed: human selectors, ambiguity, compare and JSON/CSV/Parquet export |
| 45 | Reproducible scientific plotting | Completed: learning/seeds/sweep/HPO/resources, PlotSpec, atomic render and sidecar cache |
| 46 | Safe artifact toolkit | Completed: bounded NumPy/tabular inspection, export, validators, explicit geometry and plugins |
| 47 | Preprocessing and dataset inspection | Completed: isolated N-record stage debug and DatasetArtifact report |
| 48 | Distributed workflow runtime and automatic placement | Deferred explicitly beyond 0.5.2 |

Future additions should be driven by demonstrated research needs and preserve the boundaries in the
[technical architecture](docs/ARCHITECTURE.md), rather than reopening this closed 1–30 checklist.

## 33. 0.2 roadmap history

Completed in this iteration: JSON Schema validation, typed environment and run/aggregate results, the extended CI
matrix, hardened `DatasetCache` plus file/mmap adapters, lazy entry-point discovery, run-isolated
loaded-plugin provenance, non-neural dataset/callback/logger contracts and binary/multiclass
streaming curve metrics, deterministic bootstrap intervals, paired Wilcoxon tests, plus the
categorized neural catalog documented above. Versioned configuration migrations now add safe
legacy normalisation, preview and explicit atomic persistence. Artifact retention now adds
completion receipts, role-aware checkpoint selection, verified streaming archives and recoverable
transactions. Optional tracking now adds lazily loaded MLflow, TensorBoard and Weights & Biases
logger objects without expanding the base dependency set. Native advanced graph support now adds
GATv2, relational GCN, PNA, local sparse GraphTransformer and scalar-feature EGNN with aligned edge
contracts and per-layer YAML configuration. The reviewed milestones and their current outcomes are:

1. **Typed result objects — completed**: terminal and aggregate objects now preserve direct
   dict/JSON compatibility, add typed attributes, version their envelopes and write atomically.
2. **Streaming multiclass curves — completed**: one-vs-rest histogram AUROC/AP requires
   `num_classes`, exposes macro/weighted/micro and per-class results, handles undefined classes
   explicitly and keeps persistent state bounded by `O(num_classes * num_bins)`.
3. **Persistent-cache hardening — completed**: canonical content/transform/configuration
   fingerprints, verified checksum/HMAC envelopes, a bounded non-pickle NumPy/Torch codec,
   immutable namespace manifests, atomic usage, generation-safe deletion and crash-recoverable
   multiprocess quotas are available through Python and recursive YAML objects.
4. **Loaded-plugin provenance — completed**: each run records the deterministic descriptor of every
   successfully resolved plugin, including cache hits and aliases, without contamination from
   discovery, validation, prior runs or parent processes; manifests are rewritten atomically on
   success and failure.
5. **Broader non-neural plugin contracts — completed**: dataset, callback and logger groups now
   enforce PyTorch/Lightning inheritance, work in their exact Schema positions, retain
   `target`/`ref` and logger-list compatibility, and participate in run provenance.
6. **Stronger comparison methods — completed**: YAML now selects deterministic bounded-memory
   percentile bootstrap or the legacy normal interval, and exact/asymptotic paired Wilcoxon or the
   legacy sign test. Typed results expose alternatives, zero handling, calculation provenance,
   effective sample sizes and explicit unavailable states while legacy aggregate columns remain.
7. **Configuration migrations — completed**: exact Schema value objects, a packaged 1.0/1.1
   catalog, immutable forward migration registry and validated
   `unversioned -> 1.0 -> 1.1` chain now back
   diff/YAML/JSON CLI previews, CI `--check`, transparent in-memory compatibility and explicit
   atomic output that cannot overwrite the source or race-clobber an unforced destination.
8. **Artifact retention policy — completed**: strict Schema 1.1 YAML selects disabled/preview/apply,
   best/last checkpoint roles, protected globs and verified streaming compression or explicit
   pruning. A fingerprinted final-aggregation receipt gates durable journal/quarantine
   transactions; ordered cross-process locks, rollback/forward recovery and immutable manifests
   make concurrent application safe and idempotent.
9. **Extended CI and interruption tests — completed**: the full CPU suite covers Ubuntu/Windows and
   CPython 3.10-3.14; strict time/device/resource tests, host-driven `request_stop()`, secondary-thread
   signal opt-out, exact-parent guards, bounded shutdown and visible Job Object degradation harden
   the process contract. An isolated job proves the `pytorch_lightning` 2.2 fallback, while real
   POSIX group SIGINT, targeted Windows SIGBREAK and hard launcher death check for residual children.
   A manual self-hosted CUDA workflow runs one public-YAML GPU epoch, records hardware evidence and
   then runs the remaining regressions without making CUDA a hosted-CI prerequisite.
10. **Optional tracking adapters — completed**: public lazy logger objects wrap MLflow,
   TensorBoard and Weights & Biases behind the existing single/list `trainer.logger` boundary.
   Separate and combined extras keep the base installation provider-free; explicit local/remote and
   offline/online parameters, opt-in checkpoint publication, actionable dependency errors and
   bilingual privacy/lifecycle guidance avoid making any service mandatory.
11. **Advanced graph layers — completed**: GATv2, relational GCN, PNA, local edge-aware
    GraphTransformer and E(n)-equivariant EGNN are dependency-light native stacks. Directed edge,
    edge-feature/relation/coordinate contracts, per-layer configuration, attention inspection,
    training-only PNA statistics and tensor/mapping EGNN outputs are documented and tested without
    claiming author-checkpoint or benchmark parity.
12. **Vision tasks beyond encoders — completed**: configurable U-Net dense prediction, a generic
   FPN over the shared hierarchical-backbone contract, variable-resolution Vision Transformer and
   MobileNetV2-style inverted-residual stages are public through Python and recursive YAML. Patch
   remainder policy, token/feature-map outputs, odd-size decoder alignment and fine-to-coarse
   feature channels are explicit and covered by gradient tests.
13. **Broader tabular research — completed**: deterministic categorical preprocessing plus native
   TabNet, SAINT, AutoInt and DeepFM objects are public from Python/YAML. Shape, range, mask and
   gradient tests cover the dependency-light cores; conformance cases provide the tree parity path.
14. **Long-sequence models — completed**: Transformer decoder/seq2seq and Conformer are native,
   batch-first models with explicit mask contracts. `StateSpaceAdapter` integrates S4/Mamba-like
   modules without forcing their compiled kernels into the base installation.
15. **Generative and uncertainty objects — completed**: a reusable beta-VAE objective, VQ-VAE,
   linear/cosine diffusion schedules, DDPM/DDIM sampling, temperature calibration and split-conformal
   prediction intervals are composable and tested.
16. **Scientific architectures — completed at the dependency-light boundary**: fixed-step Neural
   ODE/CDE, DeepONet, a 1D Fourier neural operator and native E(3) scalar/vector tensor-field message
   passing have shape/numerical/equivariance tests. Optional higher-order providers use a validated
   adapter; native `l>=2` irreps and adaptive stiff solvers remain future specialized work.
17. **Architecture conformance packs — completed**: source-linked cases capture initialization,
   parameter counts, expected outputs and tolerances, persist tiny weights-only references and group
   numerical/shape/checksum parity into a CI assertion without redistributing author checkpoints.
