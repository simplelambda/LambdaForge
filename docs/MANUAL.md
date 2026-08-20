<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" type="image/svg+xml" srcset="../icons/lambdaforge-light.svg">
    <source media="(prefers-color-scheme: dark)" type="image/png" srcset="../icons/lambdaforge-light.png">
    <source media="(prefers-color-scheme: light)" type="image/svg+xml" srcset="../icons/lambdaforge-dark.svg">
    <source media="(prefers-color-scheme: light)" type="image/png" srcset="../icons/lambdaforge-dark.png">
    <img src="../icons/lambdaforge-dark.png" width="140" alt="LambdaForge logo">
  </picture>
</p>

# LambdaForge Manual


LambdaForge is SimpleLambda's object-oriented framework for reproducible AI work. It combines
generic tasks, composable preprocessing, PyTorch, Lightning and a YAML engine behind one stable
Python package, so a research project can focus on its data and science instead of rebuilding
pipelines, training loops, provenance, result management and process scheduling.

> **Status:** usable but pre-1.0. The public namespaces documented below are the intended
> API; compatibility is not yet guaranteed between minor releases. The repository does not yet
> contain a licence file, so redistribution terms still need to be chosen by SimpleLambda.

## Contents

1. [What LambdaForge provides](#1-what-lambdaforge-provides)
2. [Installation and consumer integration](#2-installation)
3. [Quick start and mental model](#4-quick-start)
4. [Configuration and YAML](#6-friendly-authoring-and-the-strict-internal-model)
5. [Tasks and preprocessing](#7-generic-tasks-and-preprocessing)
6. [Identity, reuse and reruns](#8-scientific-identity-reuse-and-explicit-reruns)
7. [Workflows and composition](#9-workflows-and-configuration-composition)
8. [Clusters and managed environments](#10-local-and-multi-cluster-control-plane)
9. [Jobs and dataset placement](#11-persistent-jobs-and-data-placement)
10. [Operations and adaptive HPO](#12-inference-evaluation-export-and-hpo)
11. [Resources and reliability](#13-resources-backends-and-reliability)
12. [Results, plots and artifacts](#14-artifact-stores-registry-and-reports)
13. [Observability and reproducibility](#15-observability-and-reproducibility)
14. [CLI reference](#16-cli-reference)
15. [Understanding errors and diagnostics](#understanding-errors-and-diagnostics)
16. [Python API](#17-public-api)
17. [Conceptual execution model](#18-conceptual-execution-model)
18. [Architecture for maintainers](#19-architecture)
19. [Complete experiment YAML reference](#20-yaml-experiment-reference)
20. [Migrations, process safety and outputs](#21-configuration-migrations)
21. [Retention and built-in components](#24-artifact-retention)
22. [Extension contracts](#26-extension-contracts)
23. [Security model](#27-security-model)
24. [Current limitations](#28-current-limitations)

## 1. What LambdaForge provides

- A generic Lightning training task for mapping-shaped batches, one or more losses and independent
  train/validation/test metrics.
- A separate, strict generic-task YAML family for preprocessing and other reproducible non-training
  work, with dry-run plans, content-addressed inputs, typed artifacts and attempt history.
- Composable source/transform/sink preprocessing, atomic per-record checkpoints and deterministic
  shards; dataset publication is a separate explicit recipe/publication boundary.
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
- A small facade (`LambdaForge`), object APIs (`Experiment`, `TaskRun`, `Workflow` and
  `DatasetRecipe`) and one CLI exposed by equivalent `lambdaforge` and `lf` entry points.

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
the [tracking contract](#tracking-loggers) before enabling remote publication.

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

The consumer dependency range is an executable compatibility contract. A declaration such as
`lambdaforge[adaptive-hpo,parquet]>=0.8,<0.9` excludes every 0.9 release. Pip may still replace an
editable LambdaForge installation and only print a conflict warning because the consumer was
already present; `python -m pip check` then reports the broken environment. A clean managed cluster
solve sees both wheels together and refuses the contradiction. Do not work around it with
`--no-deps`: review the new release, change the consumer range (for example to `>=0.9,<0.10`),
reinstall the consumer and require `pip check` to pass. LambdaForge reads the built consumer wheel
and reports this incompatibility locally before transferring a remote execution bundle.

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
python -m pip install dist/lambdaforge-*.whl
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
trainer settings. Copy [the complete experiment example](../examples/experiment.yaml) into the
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

LambdaForge separates ease of writing from strict execution. The flow is:

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
safe retry and supports deterministic explicit shards. It publishes `dataset-artifact.json` only
when `publish_dataset: true` or legacy `dataset_name` is explicit; new multi-stage publication uses
the DatasetRecipe lifecycle in section 11.

Project-specific logic remains in the installed consumer package. Use a `CallableTransform` for a
small function or implement the public `PreprocessingSource`, `PreprocessingTransform` and
`PreprocessingSink` contracts. For other batch work, implement `Task.run(TaskContext) -> TaskOutput`
and use the same task YAML/CLI. See the [generic-task contract](#7-generic-tasks-and-preprocessing),
[preprocessing contract](#7-generic-tasks-and-preprocessing) and
[complete example](../examples/preprocessing.yaml).

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

The terminal is a unified, serverless control plane. A normal remote execution command first
persists a local submission record, starts one detached controller worker and returns its job ID;
that worker performs potentially slow bundle construction, input hashing/transfer, environment
preparation and scheduler submission. Direct-host scientific work is then owned by a detached
per-job supervisor and SLURM work remains owned by SLURM. `lambdaforge status`, `resources --all`, `top`, `configs`,
`datasets` and `storage` query the same application services. Read the
[control-plane architecture](#19-architecture) before changing provider boundaries.

The control plane is a local coordinator, not a mandatory web service. The user's machine owns
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
    python:
      strategy: auto                   # auto | existing | managed
      executable: python3              # tried first
      version: null                    # optionally pin a minor such as "3.13"
      allow_managed_install: true
    environment: managed             # or existing for a user/admin-owned environment
    pytorch: {channel: auto, require_cuda: auto}
    project_module: my_project       # doctor verifies this consumer import
    data_environment: atlas
    connection: {connect_timeout: 15s, keepalive: 30s, multiplex: true, persist: 2m}
    storage:
      state_root: /home/my-user/.lambdaforge/state
      cache_root: /scratch/my-user/lambdaforge/cache
      run_root: /scratch/my-user/lambdaforge/jobs
      dataset_root: /project/datasets
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
    python: {strategy: existing, executable: /usr/bin/python3.12}
profiles:
  one-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, gpu_memory: 20GiB, time: 4h}
```

`command_prefix` is an argument vector, so containers/site wrappers can be selected without local
shell interpolation. It is not a shell script: put `module load` in a reviewed scheduler prologue,
or configure the absolute Python/site wrapper exposed by the centre. OpenSSH is the default and
preserves aliases, keys, agent, `known_hosts` and
ProxyJump. A private `ControlMaster=auto` socket lets short operations reuse one authenticated
connection until `connection.persist` seconds of inactivity. Connection/auth/banner, keepalive and
command deadlines are independent; long scientific commands have no transport timeout unless one
is explicit. Optional password mode uses Paramiko with rejected unknown hosts and obtains its secret
interactively, from a `keyring:` reference in the OS keyring, or from an explicit `env:NAME`
reference. The value is never accepted through `--password`, persisted, logged, bundled or
fingerprinted. Install `lambdaforge[cluster-password]` only for that legacy mode.
Paramiko reuses one verified client inside a CLI operation; reuse across separate CLI invocations is
an OpenSSH multiplexing feature, so password mode establishes a new authenticated session each time.

Use the same configuration locally and remotely:

```bash
lambdaforge doctor
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/my-user/lambdaforge \
  --scheduler slurm --environment managed
lambdaforge clusters list
lambdaforge clusters inspect atlas
lambdaforge clusters test atlas
lambdaforge clusters bootstrap atlas --dry-run
lambdaforge clusters bootstrap atlas
lambdaforge run experiment.yaml --on atlas --dry-run
lambdaforge run experiment.yaml --on atlas --cpus 8 --memory 32GiB --resource-gpus 1 --time 4h
lambdaforge run experiment.yaml --profile one-gpu
lambdaforge top
```

Remote `run` and `datasets build` use asynchronous hand-off by default. The returned state is
`preparing`, not a false scheduler acknowledgement. The durable `metadata.submission_phase` moves
through `validation`, `runtime`, `bundle`, `staging`, `environment` and `scheduler`; after provider
acceptance the same job ID receives its real scheduler ID and `queued`/`staging` state. Preparation
failures and controller logs remain available through ordinary `jobs show` and `jobs logs`. A
pre-scheduler cancellation marks the request cancelled and the worker refuses the next phase. Use
`--wait-for-submit` for compatibility with a script that genuinely needs to wait for staging and
provider acknowledgement. Dry-runs remain synchronous and read-only so they can return the complete
plan.

`lf top` opens the interactive live view when stdin/stdout are a TTY. Use arrows or `j`/`k` to
select, `l` for recent logs, `x` plus confirmation to cancel, `r` to refresh and `q` to leave. It
collects slow provider snapshots in an isolated cancellable child, while terminal input and drawing
remain in the foreground; arrows and exit therefore do not wait for SSH/scheduler timeout. It shows
host CPU/RAM/GPU facts separately from job declarations and scheduler state; login-node usage
is not presented as SLURM partition capacity. No TUI-only backend exists: the versioned snapshot
from `lf overview --json`, job rows from `lf jobs list --json` and resource rows from `lf resources
--all --json` are the machine interface for a GUI or wrapper. Non-TTY `lf top` prints once;
`lf top --json --follow` is an NDJSON stream and `lf top --once` forces one human snapshot.

### Managed Python runtime

There are three separate layers. A **Python runtime** is the real CPython interpreter; a **Python
environment** is the isolated `venv`; **installed packages** are the exact LambdaForge, consumer,
PyTorch and dependency distributions inside that environment. A `venv` made with Python 3.9 is
still Python 3.9, so managed bootstrap resolves the runtime before it resolves CUDA/PyTorch or
computes the environment identity.

| Strategy | Meaning |
|---|---|
| `existing` | Use only `python.executable`; never provision a runtime. |
| `auto` | Try the configured executable, bounded supported Python minor alternatives, a cached runtime, an existing Conda/Mamba/Micromamba, then the managed fallback. |
| `managed` | Reuse or create a LambdaForge-owned runtime even if a compatible system Python exists. |

New `clusters add --environment managed` profiles default to `auto`. For backward compatibility,
the old scalar `python: python3` means strict `existing`; it never begins downloading software
silently. Migrate and optionally pin it without editing YAML:

```bash
lf clusters set atlas python.strategy auto
lf clusters set atlas python.version 3.13
lf clusters set atlas python.allow_managed_install false  # institutional no-install policy
```

`bootstrap --dry-run` performs discovery and reports the default interpreter, runtime action/path
and either the exact PyTorch plan or that its final wheel resolution awaits the planned runtime. It
does not create cluster directories, download the manager or install packages. Real bootstrap uses
this order:

```text
transport/platform -> Python constraints -> runtime -> PyTorch/CUDA wheel
                   -> host CA trust -> environment identity -> venv/packages
                   -> TLS/CUDA/framework verification -> active pointers
```

LambdaForge reads its own `Requires-Python` from installed release metadata and the nearest
consumer project's `project.requires-python`. It tries another candidate if the official selected
PyTorch channel has no wheel for an otherwise valid Python. The chosen runtime version,
architecture, provider/version and package fingerprint enter environment identity, so changing
Python cannot silently reuse an incompatible environment.

Existing `micromamba`, `mamba` or `conda` is reused only to create a dedicated prefix with `-p`
below `storage.cache_root`; global/base environments are not modified. If none exists and managed
installation is allowed, LambdaForge stages pinned micromamba 2.8.1-0 from the official mamba-org
release. The controller downloads over HTTPS, verifies a built-in SHA-256 for Linux x86-64,
AArch64 or ppc64le, transfers the single executable and verifies the checksum again. It never runs
`conda activate`, `conda init`, edits `.bashrc`, uses sudo or changes system Python/CUDA/drivers.
Micromamba supplies only the reusable CPython runtime; ordinary pip policy still installs the exact
framework/project/PyTorch packages into the content-addressed `venv`.

Before LambdaForge accepts a runtime it asks the configured host Python for its OpenSSL default CA
locations, selects a readable PEM bundle that can construct a non-empty verified context and records
only its absolute path. Managed Conda/Micromamba creation uses that trust, as do online pip/Torch
installation and the final task, experiment or dataset process through `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `PIP_CERT` and `CURL_CA_BUNDLE`. This deliberately reuses institutional roots
that already make system Python work. Installing `truststore` alone or Conda's public
`ca-certificates` cannot guarantee those local roots; disabling verification would be insecure.
Existing administrator-owned Python is not rewritten or given a managed trust policy. If no valid
host PEM bundle exists, provisioning fails closed and asks for the host trust to be repaired.

Runtimes, managers, Conda packages and environments are separate cache categories:

```text
storage.cache_root/
  runtime-managers/     pinned micromamba executable
  runtimes/             verified Python prefixes
  conda-pkgs/            reusable package cache
  runtime-packages/      staged offline solves
  environments/          exact LambdaForge/project/PyTorch venvs
```

With an offline `wheelhouse`, the controller uses micromamba to prefetch the target-platform Python
solve and transfers that package cache before remote `--offline` creation. The controller must be
Linux for this cross-target prefetch; otherwise provide an existing compatible runtime. Runtime and
environment creation use bounded locks, unique staging paths, verification and atomic publication.
A verified runtime may remain as reconstructible cache when later package installation fails, but
no active pointer references an incomplete runtime/environment. `storage status` accounts for all
categories and `storage gc` never selects a runtime referenced by the active pointer, a retained
environment or an active/queued job.

Once bootstrap has verified and activated the new environment, it prunes superseded directories
whose names belong to LambdaForge's content-addressed environment namespace. Cleanup runs under the
cache GC lock and retains the new environment, every environment referenced by a known non-terminal
job and references discoverable in durable direct-job state. If another environment build is in
progress, pruning is deferred and reported in human/JSON bootstrap output. This ordering guarantees
that a failed bootstrap never destroys the previous working environment. It does not delete Python
runtimes, package caches, datasets, results or administrator-owned environments; use preview-first
`storage gc` for the wider reconstructible cache lifecycle.

`doctor` reports the system/default Python separately from the active resolved runtime and managed
environment Python. Thus Python 3.9 can be a healthy system probe while bootstrap is still able to
provide Python 3.13; under `strategy: existing`, the same incompatibility remains an actionable
failure. `system-python-tls` and `managed-python-tls` also validate local trust-context construction
without depending on Zenodo or another external service. An old managed runtime with no recorded
host CA policy is reported as repairable by rerunning bootstrap.

The portable `ResourceRequest` normalizes CPU cores, RAM, GPUs, GPU memory, duration, storage and
process count. One validated per-cluster translation layer emits standard `gpus`, generic/typed
`gres`, CPU, memory and time directives. Static directives support flags and repeated values;
submit/queue/accounting/cancel commands, job-ID regex, shell and trusted prologue/epilogue are also
configurable as safe argv/templates. Legacy `scheduler_options` remains compatible. Dry-run exposes
the exact script, resources, directives, warnings and submit argv; explicitly omitted resources
produce a warning. Units accept decimal `KB/MB/GB/TB`, binary `KiB/MiB/GiB/TiB`, and durations such
as `30m` or `4h`. The full schemas and security trade-offs are in the
[cluster section](#10-local-and-multi-cluster-control-plane).

An `ExecutionBundle` contains materialized YAML, a manifest, exact wheels for LambdaForge and the
nearest consumer project, and only explicitly bounded small inputs. It is content-addressed under
`.lambdaforge/control/bundles`; dirty local source is built exactly as it exists, never replaced by
`git clone main`. An editable LambdaForge install resolves its PEP 610 source checkout. A normal
wheel install reuses its original local wheel when available or deterministically reconstructs a
pure-Python wheel from the installed package and metadata; bootstrap does not assume that
`pyproject.toml` exists inside `.venv` and does not require contacting an index for LambdaForge
itself. In `managed` mode those wheel bytes, resolved Python-runtime identity and compatible
dependency policy identify an idempotent user-space venv below `storage.cache_root/environments`.
In `existing` mode no
installation occurs and the configured Python must already contain the exact framework/project.
Offline clusters use a target-compatible `wheelhouse`/`--wheelhouse` and `--no-index`. LambdaForge
verifies PyTorch/CUDA but never installs drivers, system CUDA or cuDNN. The remote command remains
`python -m lambdaforge run config.yaml`, so there is no second runner.

Managed bootstrap resolves a constraint-compatible Python runtime, probes the NVIDIA driver and
compute capabilities, verifies wheel availability in official PyTorch indexes, and pins an exact
compatible Torch build before installing the framework. A legacy scalar Python remains strict;
`strategy: auto` can safely replace an incompatible login-node default in user space. Automatic mode
chooses the newest channel meeting its native toolkit driver floor:
for example, a 535-series H100 selects `cu121`, not `cu126`/`cu130`, while legacy Pascal-class GPUs
use `cu118` when a compatible wheel exists. It does not silently rely on minor-version compatibility
for CUDA 12/13; the documented CUDA 11.8 legacy floor is accepted and verified with an actual GPU
tensor operation. The exact plan is part of environment identity. If detection cannot
prove a safe choice, bootstrap fails with an actionable Python/channel/wheelhouse message rather
than guessing or silently installing CPU. Override with `pytorch.channel` only from reviewed centre
guidance. See the [cluster section](#10-local-and-multi-cluster-control-plane) for the complete decision table.

`LocalTransport`, OpenSSH `SshTransport` and optional `PasswordSshTransport`, plus
`LocalScheduler`/`SlurmScheduler`, are independent public providers. `doctor` checks connection and
auth, workspace, Python/project/framework/PyTorch/CUDA, every configured scheduler executable,
resource mapping and partition without submitting a job. Provider boundaries are injectable, so
tests exercise credentials, transfers, submission, failures and reconnection without a real cluster.

## 11. Persistent jobs and data placement

This section covers the durable state machine, safe process identity, CPU/RAM/GPU leases, pause
semantics, recovery guarantees and the dataset registry, profiling and lifecycle model.

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
lambdaforge jobs pause job-20260811120000-ab12cd34
lambdaforge jobs resume job-20260811120000-ab12cd34
lambdaforge jobs reconcile --all
```

`JobService` provides the same list/get/logs/pause/resume/cancel/retry/delete/reconcile operations
to Python applications and a
future GUI. CLI JSON output is a direct serialization of the same `JobRecord`, `JobHandle`, doctor,
bundle and data service objects; there is no second GUI-only business layer.

`preparing` means the local detached controller owns pre-scheduler work; `staging` means bundle or
direct-host supervisor staging; `queued` means the scheduler has accepted the job. These states are
not interchangeable. `overview --json` has `snapshot_version`, `generated_at_utc`, cluster resource
records, aggregate counts and the complete `jobs.items` records used by `top`. `top --json --follow`
emits one compact snapshot per line for stream consumers. Ordinary one-shot services remain the
recommended application API; a wrapper never needs to scrape terminal escape sequences.

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

Large data is never copied merely because `run --on` was used. LambdaForge 0.7 distinguishes:

```text
DatasetRecipe (how) -> DatasetBuild (execution) -> DatasetVersion (what)
                                                   -> DatasetPlacement (where)
```

A preprocessing Task produces ordinary artifacts unless publication is explicit. A
`kind: dataset` recipe reuses the Workflow DAG for content-addressed stages and atomically publishes
one immutable logical collection. `DatasetMember` and streaming `DatasetIndex` describe stable IDs,
arbitrary partitions/targets/metadata and any file/directory/record/URI asset layout. Artifact v2
keeps path-independent `content_id` separate from provenance `build_id`; v1 remains readable.

```bash
lf datasets plan example-records --on atlas
lf datasets plan example-records --on atlas --verbose
lf datasets build example-records --on atlas
lf datasets members example-records@1 --partition split=train --limit 50
lf datasets diff example-records@1 example-records@2
lf datasets materialize example-records@1 --on atlas --apply
```

Builds are durable jobs; reruns reuse verified stages, `--force-stage` invalidates downstream work,
and failed builds publish no DatasetVersion. Publication validates the index/assets/schema in
staging, renames atomically, and only then registers. Materialization really applies NOOP,
REPLICATE or BUILD rather than returning a producer command. The
[generic recipe](../examples/dataset-recipe.yaml) is a complete runnable starting point.
Remote controller-side plans never reuse the local cache as if it were remote: unobserved cache is
shown as `MISSING`, and the durable target worker rechecks the exact fingerprints before execution.

DatasetRegistry is the placement authority for managed versions, so a consumer can simply use
`dataset:example-records@1` without duplicating cluster paths. `DatasetResolver` pins the exact
version/content and records the selected placement outside scientific identity. DataCatalog remains
compatible for aliases, external/unmanaged data, loaders, explicit pins and institutional
overrides. For example, an external dataset can still use:

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

The target profile's `data_environment` selects an external physical location. If it is missing,
submission fails before scheduling and tells the user to register or replicate the data. Ordinary
local path inputs up to 10 MiB are content-hashed and copied into the small execution bundle. This
applies equally to a standalone Task YAML, a task YAML referenced by a dataset recipe and a task
mapping embedded directly under `stages.NAME.task`. LambdaForge rewrites the staged configuration
to a bundle-relative path, then the process scheduler copies the complete bundle into the durable
job workspace. The user never has to discover or populate a hashed bundle/job directory.

For example, this small manifest is transferred automatically when the recipe is submitted to a
remote cluster:

```yaml
stages:
  curate:
    task:
      kind: task
      schema_version: "1.0"
      name: curate
      inputs: {public_sources: ../data/public-sources.json}
      task: {target: my_project.tasks.CurateSources}
```

The 10 MiB limit covers each declared file or complete directory input. It is intentionally not a
transparent large-data transport: a ZIP or directory above the limit causes a configuration error
before bundle upload or scheduler submission. For hundreds of gigabytes, give the bytes a logical
identity and an explicit target placement, then reference that identity from the stage:

```yaml
data_catalog: ../data-catalog.yaml
inputs: {archive: dataset:raw-archive}
```

The catalogue can point `local` and `atlas` at different physical paths while the task fingerprint
retains one logical identity. If the target placement does not exist yet, create it through the
preview-first replication command below, `datasets materialize` for a managed DatasetVersion, an
object-store/transfer plugin, or the institution's supported data mover. The researcher chooses and
audits that placement/transfer policy; LambdaForge refuses to guess it or silently copy a massive
input. Data movement is a separate preview-first command:

```bash
lambdaforge data --catalog data-catalog.yaml list
lambdaforge data --catalog data-catalog.yaml locations raw-corpus
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas
# Review the exact source/destination, then:
lambdaforge data --catalog data-catalog.yaml replicate raw-corpus --from local --to atlas --apply
```

The built-in replication provider uses `rsync`, requires both source and destination locations to
be declared in the catalog and currently copies from a local source to a local or SSH destination.
It does not guess paths or rewrite the catalog. Object-store and institutional transfer systems
implement the `DataTransferProvider` boundary.

Training experiments use the same catalogue. A direct split such as
`data.train: dataset:raw-corpus/train` requires the entry to declare a dataset `loader` ObjectSpec
plus `path_parameter`; LambdaForge injects the selected location. Inside nested ObjectSpec params,
`{dataset: raw-corpus, subpath: train}` resolves only that typed marker. Ordinary strings are never
guessed. The physical mount can differ by cluster while the scientific fingerprint retains the
logical reference and declared dataset identity.

Managed versions are discovered without a catalogue for operational queries:

```bash
lf datasets list --all
lf datasets show raw-corpus@v3
lf datasets stats raw-corpus@v3 --on atlas
lf datasets verify raw-corpus@v3 --on atlas
lf datasets materialize raw-corpus@v3 --on gpu-lab
```

`remove` changes registration only. Physical `delete` is a matching-manifest, active-consumer-
checked preview and requires `--apply`. Storage GC can never delete dataset placements.

Workflow YAML may annotate a node with `on: atlas`, and dry-run plans display every placement.
LambdaForge deliberately refuses to execute a mixed-cluster DAG in the in-process workflow runner:
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
[`examples/adaptive-hpo.yaml`](../examples/adaptive-hpo.yaml), keep the ordinary model/data/loss/task
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
formats outside core. Full guides: [results](#14-artifact-stores-registry-and-reports), [artifacts](#14-artifact-stores-registry-and-reports),
[clusters](#10-local-and-multi-cluster-control-plane) and [preprocessing](#7-generic-tasks-and-preprocessing).

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
| `plan CONFIG [--on CLUSTER]` | Uniform shortcut for the best available dry-run/preflight. | no |
| `run CONFIG` | Execute experiment, task or workflow. | yes |
| `run CONFIG --force|--restart|--no-resume` | Explicitly control success reuse and partial continuation. | yes |
| `run CONFIG --on CLUSTER|--profile PROFILE` | Persist an ID immediately, then prepare/cache/submit remotely in a detached controller. `--wait-for-submit` waits for provider acknowledgement. | job metadata; remote only without dry-run |
| `clusters add|list|show|inspect|set|unset|remove|export|credentials|test|bootstrap|resources|storage` | Manage profiles and inspect their environment, resources and storage. | profile/credential/bootstrap mutations |
| `status`, `overview`, `top` | Global cluster/job/dataset view; `top` is interactive on a TTY, `top --once` is one-shot and `top --json --follow` is NDJSON. | no |
| `status|logs|cancel|retry` | Short aliases for the corresponding common persistent-job operations. | lifecycle commands only |
| `jobs list [--all]|status|show|logs|pause|resume|cancel|retry|delete|reconcile|groups`, `jobs group list|show` | Reconnect and safely control persistent work/groups. | lifecycle commands only |
| `resources [--on C|--all] [--processes]` | Separate observed host facts, scheduler view and declared job requests. | no |
| `configs|experiments|tasks list|show|validate|plan|run`; `experiments status|history|results` | Discover project YAML and operate it by unambiguous name. | only run |
| `datasets plan|build|list|show|members|member|diff|locations|stats|verify|lineage|add|remove|delete|materialize|replicate` | Recipe/build/version/placement lifecycle and inspection. | build/add/remove; delete/transfer only with `--apply` |
| `storage status|gc`, `environments list|show|gc` | Inspect bytes/file counts and collect only reconstructible unreferenced cache. | GC only with `--apply` |
| `data --catalog FILE list|locations|inspect|replicate` | Inspect logical datasets/manifests; replication needs `--apply`. | only replicate `--apply` |
| `compose CONFIG` | Redacted materialization plus provenance. | no |
| `diff LEFT RIGHT` | Semantic configuration differences. | no |
| `explain authoring|experiment|task|workflow PATH` | JSON Schema fragment for a dotted property. | no |
| `explain changes CONFIG [--against OLD]` | Scientific identity and exact changed paths. | no |
| `target IMPORT.PATH` | Constructor signature and docstring. | no |
| `migrate CONFIG` | Preview migration; `--output` is explicit. | no |
| `plugins` | Entry-point metadata without provider import. | no |
| `results audit SOURCE` | Compatible identity/duplicate audit; index writing is explicit. | no unless `--write-index` |
| `results list|show|compare|export|sync` | Human selectors, statistics, tabular export and lightweight remote evidence. | export/sync |
| `plot learning|sweep|seeds|hpo|resources` | Create `PlotSpec` JSON or atomic static/HTML figures. | only without `--json` |
| `artifact inspect|export|validate|visualize|list|fetch|plugins` | Safe bounded inspection and explicit retrieval/geometry. | export/visualize/fetch |
| `debug CONFIG --records N` | Sample preprocessing transforms without production sink/finalization. | only requested intermediates |
| `completion bash|zsh|fish` | Generate completion for both `lambdaforge` and `lf`. | no |
| `project status` | Project root/version/default cluster/configs/registry/active jobs. | no |
| `aggregate CONFIG` | Rebuild experiment aggregates. | yes |
| `retain CONFIG` | Retention preview; only `--apply` mutates artifacts. | no |
| `registry ROOT [--output FILE]` | Query JSON or export JSON/CSV/Parquet. | only with output |
| `dashboard ROOT --output FILE` | Static read-only HTML snapshot. | yes |

`lambdaforge init my-project --template preprocessing` is the fastest preprocessing path;
`training` creates a runnable toy baseline, `minimal` creates one generic task and `full` includes
both families. Rename `my_project`, implement the generated domain code, install it with
`pip install -e .`, and validate its YAML. The scaffold includes `.gitignore` rules for
environments, caches, builds and run output.

`lf` is an exact second entry point. Canonical grammar is
`lf <resource> <action> <object> [--on CONTEXT]`; moderate aliases are `ds`, `exp`, `env` and
`ls`. Project config names can replace paths when unambiguous. A top-level `default_cluster` in
project `lambdaforge.yaml`/`lambdaforge.clusters.yaml`, or the XDG user cluster catalog, supplies
`--on` only when the user omitted it; explicit `--on` always wins. Human and JSON output identify
whether the target was explicit, a project default or a user default.

### Understanding errors and diagnostics

An error is part of the operational interface, not just an exception string. The same immutable
diagnostic supplies terminal output, JSON and the persistent record. A normal failure answers:

1. **What:** the operation and concise problem;
2. **Why:** the cause LambdaForge can establish without guessing;
3. **Impact:** whether a job started and which work was retained or blocked;
4. **Fix:** a safe remediation;
5. **Next action:** a command that exists in the current CLI;
6. **Diagnostics:** the job log or redacted local record containing the full traceback.

The stable categories and process exit policy are:

| Human heading | JSON `category` | Exit | Meaning |
|---|---|---:|---|
| `CONFIGURATION ERROR` | `configuration` | 2 | A selector, path, option or required setting is missing/incompatible before execution. |
| `VALIDATION ERROR` | `validation` | 2 | Parsed YAML/configuration violates its schema or contract. |
| `ENVIRONMENT ERROR` | `environment` | 3 | Python, packages, managed environments, PyTorch or CUDA preparation failed. |
| `CONNECTION ERROR` / `AUTHENTICATION ERROR` | `connection` / `authentication` | 3 | The cluster is unreachable or rejected its credential/host policy. |
| `EXECUTION FAILED` | `execution` | 4 | A process, project task, stage or training run actually started and failed. |
| `DATA ERROR` | `data` | 4 | Dataset identity, location, integrity or scientific record constraints failed. |
| `RESOURCE ERROR` / `STORAGE ERROR` | `resource` / `storage` | 5 | Allocation, quota, capacity, permissions or filesystem cannot satisfy the request. |
| `OPERATION REFUSED` | `operation_refused` | 2 | Expected safety behavior prevented an unsafe overwrite/delete/GC action. |
| `INTERNAL ERROR` | `internal` | 10 | An unclassified invariant or probable LambdaForge bug occurred. |
| `WARNING` | `warning` | 0 | Execution can continue, but a legacy/surprising condition deserves attention. |
| `CANCELLED` | `cancelled` | 130 | The user/provider interrupted work; this is not a framework error. |

For example, a remote dataset build without permanent dataset storage is a configuration error. It
states that no job, remote computation, environment or bundle was created and proposes real
commands:

```bash
lf clusters set atlas storage.dataset_root /persistent/path/to/datasets
lf datasets build dataset-recipe --on atlas
```

In contrast, a terminal dataset-build job is an execution failure. `lf jobs show JOB` reads the
durable record and, when stage evidence exists, displays the root `FAILED` stage first, dependent
`BLOCKED` stages separately and completed reusable stages as preserved. Use:

```bash
lf jobs logs JOB --tail 300
lf jobs show JOB --json
lf jobs retry JOB                 # only after correcting the stated cause
```

LambdaForge does not claim that exit code 137 proves OOM or that an unknown third-party exception
is a user mistake. If evidence is insufficient, it says so and directs the user to logs. A rejected
immutable dataset overwrite is `OPERATION REFUSED`, not an internal/fatal error.

#### Human, JSON, verbose and debug modes

Human diagnostics are compact, have no ANSI dependency and go to stderr. Operational commands can
add `--verbose` for more planning/progress information. Add `--debug` anywhere to include the
underlying Python exception and full traceback; this is deliberately separate from verbose mode.
Known errors omit implementation paths and exception-class headings by default.

Add `--json` anywhere for one machine-readable object on stdout. It contains `status`, `category`,
stable category-level `code`, `exit_code`, `message`, `reason`, `impact`, `fixes`, structured
commands/context, `retryable`, optional `job_id` and `diagnostic_record`. Debug fields appear only
when `--debug` is also supplied. The meaning and exit code are identical to human output.

#### Persistent records and secrets

Failures at the CLI boundary are atomically recorded under
`$XDG_STATE_HOME/lambdaforge/logs/errors`, defaulting to
`~/.local/state/lambdaforge/logs/errors`. Directories use owner-only permissions where supported;
records include timestamp, sanitized command/operation, cluster/job, LambdaForge version,
exception chain, full traceback, displayed diagnostic and remediation. Job execution additionally
keeps durable stdout/stderr, so `lf jobs logs` remains usable for `FAILED` jobs and for the
`latest`/unambiguous selectors already supported by the job service.

Passwords, tokens, API keys, bearer headers, private-key blocks, credential URLs and secret-named
structured fields are redacted before terminal/JSON rendering or persistence. Do not attach raw
private project data when reporting a bug. For an internal error, retry once with `--debug` and
include the diagnostic record plus `lf doctor --on CLUSTER`; closing the CLI does not implicitly
cancel an already submitted remote job.

## 17. Public API

The supported entry points are deliberately narrow:

| Entry point | Purpose |
|---|---|
| `from lambdaforge import LambdaForge` | Load, run or construct objects through the facade. |
| `from lambdaforge import MaterializedConfig, JobHandle` | Inspect compiled authoring and durable submissions. |
| `from lambdaforge import Experiment` | Inspect, execute, aggregate and load one experiment suite. |
| `from lambdaforge import TaskRun, TaskResult, TaskExecutionPlan` | Validate, inspect, execute and audit one generic task. |
| `from lambdaforge import Workflow, WorkflowPlan, WorkflowResult, WorkflowValidationReport` | Validate, plan and run a task/experiment DAG. |
| `from lambdaforge import DatasetRecipe, DatasetBuildPlan, DatasetBuildResult` | Validate, plan and build one immutable dataset recipe. |
| `from lambdaforge import RunResult, AggregateResult` | Typed immutable results with legacy dict/JSON compatibility. |
| `from lambdaforge.experiments import PostRunAction, PostRunContext, PostRunResult` | Per-run final analysis contract with stable checkpoint/artifact context. |
| `from lambdaforge import ResultCatalog, ResultRecord` | Identity-aware discovery and explicit selection of attempt history. |
| `from lambdaforge import ResultService, VisualizationService, PlotSpec, ArtifactService` | Stable query, plotting and safe artifact application services. |
| `from lambdaforge import ArtifactRetentionPlan, ArtifactRetentionResult` | Typed immutable retention previews and outcomes. |
| `lambdaforge.data` | Dataset members/indexes, recipe/build/version/placement lifecycle, unified resolution, transfers, adapters and caches. |
| `lambdaforge.tasks` | Generic task, context, plan, result and artifact contracts. |
| `lambdaforge.preprocessing` | Composable record preprocessing and dataset manifests. |
| `lambdaforge.configuration` | Authoring-to-IR compilation, includes, safe interpolation, redaction, provenance and diff. |
| `lambdaforge.controlplane` | Cluster/transport/scheduler providers, bundles, doctor and persistent job services. |
| `lambdaforge.diagnostics` | Stable categories/model plus human/JSON rendering for application boundaries. |
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
| `python -m lambdaforge` / `lambdaforge` / `lf` | CLI front end to the same object API. |

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
level `JobService`, `DatasetService`, `ResourceService`, `StorageService`, `DataService` and `Doctor`
are application services intended for CLIs, notebooks
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
│   ├── _version.py               # sole package/runtime release-version constant
│   ├── cli/                      # command-line object
│   ├── configuration/            # authoring IR, composition, secrets, provenance and diff
│   ├── controlplane/             # clusters, transports, schedulers, jobs, bundles and doctor
│   ├── data/                     # logical identity/catalog/transfer, adapters and bounded caches
│   ├── diagnostics/              # categories, classification, rendering and redacted records
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

The implementation uses objects where they own state or a replaceable policy and small functions
where they are an entry point or pure transformation. A module is cohesive rather than forced to
contain exactly one class: `JobService` owns durable lifecycle transitions, `SubmissionService`
owns controller-side hand-off, `Transport` and `Scheduler` are provider boundaries,
`OverviewService` composes existing sources of truth, and `LiveJobMonitor` only handles terminal
interaction/rendering. The private `SubmissionWorker` is deliberately an internal process entry
point, not another public execution engine. This separation lets a future GUI call the same service
objects or JSON commands without importing a TUI.

Release identity has one writable source: `src/lambdaforge/_version.py`. Setuptools obtains dynamic
wheel/sdist metadata from it and `LambdaForgeVersion` exposes the same constant to CLI, bundles and
diagnostics. README installation commands intentionally use a wheel glob; changelog headings are
historical records, not competing version authorities.

Enums and typed immutable records replace closed magic-string/dictionary state. YAML keys, error
categories and fully qualified import paths remain strings where they are external protocol
boundaries. Public imports are re-exported from stable lowercase package namespaces, so physical
module layout and the existing matching class/module filenames do not become consumer contracts.

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
| `post_run` | no | Checkpoint-aware final actions with separate identity, artifacts and failure policy. |
| `sweep` | no | Base inclusion, Cartesian grid and named ablations. |
| `execution` | no | Sequential/parallel/DDP resources. |
| `aggregation` | no | Cross-seed confidence intervals, paired tests and reliability thresholds. |
| `retention` | no | Preview/apply policy for checkpoint roles, verified archives and explicit pruning rules. |

The packaged [JSON Schema](../src/lambdaforge/schemas/experiment.schema.json) rejects unknown
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

### Validation diagnostics and post-run analysis

LambdaForge deliberately has three different extension levels:

| Use | When | Examples |
|---|---|---|
| Lightning `callbacks` | The logic belongs to a validation/train batch or epoch | streaming diagnostics, a `val_*` HPO metric |
| `post_run` | One successful run needs bounded final analysis on the same allocation | predictions, a report, model interpretation |
| `Task`/`Workflow` | Work needs another allocation, cluster or independently scheduled lifecycle | large inference, export pipeline, multi-stage reconstruction |

A validation callback receives the ordinary detached `model_outputs` returned by
`LightningTask.validation_step`; it can summarize them without a second model forward and log an
exact `val_auxiliary_score`. That column enters `metrics.csv` and can be selected as the HPO
objective. The callback owns its memory strategy and distributed reduction. Any file-writing hook
must check `trainer.is_global_zero`.

For final per-run analysis, implement `run(PostRunContext) -> PostRunResult` and configure it:

```yaml
post_run:
  - name: predictions
    target: my_project.analysis.GeneratePredictions
    params: {split: test}
    checkpoint: best       # best (default), last, current or none
    required: true
    artifacts:
      - {name: predictions, path: analysis/predictions.npz, kind: predictions}
  - name: html-report
    target: my_project.analysis.GenerateReport
    params: {theme: paper}
    checkpoint: best
    required: false
```

The context contains the run directory, immutable materialized config, seed/variant, typed training
result, best/last/selected checkpoint, checkpoint SHA-256 and an identity-specific state directory.
It does not retain a live model or datamodule: project code reconstructs only what it needs from the
config and selected checkpoint. `best` is strict and never silently falls back to `last`; `current`
means the persisted final/current (`last`) checkpoint. Long actions can poll the live
`context.stop_requested` cooperative-cancellation state.

Training success is committed before actions begin. Each action then gets its own fingerprint and
atomic receipt. Receipts verify artifact path, logical name, type, size, SHA-256, media type,
producer and action identity using the shared task artifact implementation. Consequently an
interrupted action can resume or rerun without training again, and changing only a visualization
parameter invalidates only that action. A required failure makes `result.json` failed until the
action succeeds; an optional failure remains visible while the trained run stays usable.

Actions execute sequentially on global rank zero and reuse the training allocation. Adaptive HPO
defaults to `confirmed_runs`; the mapping form permits explicit `scope: all_runs`. Paused or
cancelled trials never run them. Use a callback—not a post-run metric—when HPO must optimize that
value. See the complete contracts and neutral Python example in the
[validation and post-run lifecycle](#validation-diagnostics-and-post-run-analysis).

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
[statistical comparison contract](#cross-seed-statistical-comparisons).

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

See the [configuration migration contract](#21-configuration-migrations) for the
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
[artifact-retention contract](#24-artifact-retention) for the complete YAML
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
See the [neural-component reference](#25-built-in-components) and
[metrics reference](#25-built-in-components) for contracts and shapes.

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
the pair. A complete PNA YAML contract
and EGNN mapping contract document named
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
[data contract](#11-persistent-jobs-and-data-placement) before enabling it in parallel or DDP jobs.

### Installed plugins

External distributions can publish models, metrics, neural components, datasets, Lightning
callbacks/loggers and generic tasks through the canonical groups documented in the
[plugin contract](#installed-plugins). Datasets must inherit PyTorch `Dataset`;
callbacks/loggers inherit the public `lambdaforge.integrations.Lightning` bases. Discovery reads
metadata only; resolving a selected plugin imports provider code and therefore has the same trust
boundary as `target`.
Built-ins and aliases registered explicitly in the current process keep precedence over
activation/normalisation plugins.

Each real run uses an isolated `PluginUsageSession` and atomically stores its canonically ordered
descriptors in `environment.json`. Earlier validation, sequential runs, installed-but-unused
providers and failed resolutions are excluded; cache hits and external component aliases actually
used by the run are included. Dry-runs record an empty list. See the
[provenance contract](#installed-plugins).

### Tracking loggers

`trainer.logger` accepts the public `MLflowTrackingLogger`,
`TensorBoardTrackingLogger` and `WeightsAndBiasesTrackingLogger` targets, or a non-empty list mixing
them with project loggers and installed logger plugins. Each adapter checks its own optional extra
only when constructed; importing LambdaForge remains provider-free. The canonical dense
`metrics.csv` is controlled separately by `write_epoch_metrics_csv`. Task losses/metrics reach the
provider only when `task.params.logging.logger` is enabled.

Provider credentials must stay outside YAML because materialized configurations are durable run
artifacts. Checkpoint upload is opt-in through `log_model` and independent of LambdaForge's local
retention transactions. See the [tracking contract](#tracking-loggers) for complete
parameters, local/remote and offline/online examples, privacy boundaries and failure behaviour.

### Runner

A custom runner must provide compatible `fit` and `test` methods. Configure it through
`runner.target`; its parameters are recursively object-built. Extra callbacks can therefore also be
declared as YAML object specifications.

All these extension objects are recursively constructed by `ObjectFactory`: models may be any
`torch.nn.Module`; losses subclass `Loss`; metrics subclass `Metric`; loggers and callbacks implement
their Lightning contracts. Custom batch structures, multiple optimizers or a different backend are
handled by replacing `task.target` or `runner.target`, without changing the experiment engine.

## 27. Security model

LambdaForge configurations are trusted code because `target`, `ref`, plugins and project callbacks
import Python. Validate and run only configurations and wheels you trust. Validation proves shape
and contracts; it is not a sandbox.

Secrets belong in OpenSSH, an operating-system keyring, hidden interactive input or an explicit
environment reference. They must not appear in YAML, command arguments, bundles, environment
identity, fingerprints, job records or logs. OpenSSH and Paramiko retain host-key verification;
LambdaForge never disables it for convenience.

Every path that copies, publishes, prunes or deletes data checks an exact owning root, rejects
symlink escape, stages before atomic replacement where the filesystem permits it and keeps
destructive application behind an explicit review step. Checksums detect accidental or malicious
content changes but do not encrypt data. HMAC-protected cache records require a separately managed
secret key.

Remote bootstrap is unprivileged. It installs reviewed wheels in a user-owned virtual environment
and may create a verified Python runtime only below the configured cache root. It never changes
system Python, shell profiles, drivers, system CUDA, modules, firewall or scheduler policy.
Scheduler prologues,
epilogues and custom commands are trusted administrator/user configuration and therefore use
validated placeholders rather than secret interpolation.

See the separate [security policy](../SECURITY.md) for reporting vulnerabilities and the compact
threat-model checklist. Security-sensitive changes require failure-path tests for path traversal,
secret redaction, process identity, host verification, dataset integrity or storage GC as relevant.

## 28. Current limitations

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
  control plane submits individual configs to local/SSH plus process/SLURM providers, but it does
  not pretend to coordinate mixed-cluster DAG artifact transfer or durable dependency recovery.
- Managed bootstrap can provision Linux CPython through a Conda-family prefix, then installs exact
  framework/consumer wheels into a user-space venv. It cannot synthesize missing platform/CUDA
  dependency wheels, install drivers, interpret interactive module functions or build a container.
  Offline sites need a compatible wheelhouse and a Linux controller for automatic Python-package
  prefetch. Existing environments remain user-owned.
  Built-in replication is local/SSH rsync over predeclared locations.
- Cluster selection is explicit. Profiles observe resources but cannot derive optimal
  placement from incomplete capacity, queue-delay or monetary-cost information. `DataCatalog` resolves
  direct experiment splits and nested typed markers; arbitrary untyped strings stay project-owned.
- Remote result sync is allowlisted and per-file bounded, not a remote filesystem mirror. Heavy
  artifacts require explicit logical fetch. Live plotting polls small files; it is not a streaming
  server. Matplotlib is core, while interactive HTML/graph/mesh providers are optional extras.
- Direct-host leases are cooperative admission and affinity, not cgroup/container isolation.
  Paused jobs retain memory/GPU resources. Built-in cluster-to-cluster dataset transfer is explicit
  and controller-relayed when the source is local; LambdaForge does not claim that such a relay survives
  controller loss or silently copy hundreds of gigabytes.
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
