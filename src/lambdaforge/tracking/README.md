# LambdaForge optional experiment tracking

[Repository guide](../../../README.md) · [Español](README.es.md) ·
[Training and processes](../training/README.md)

This package provides small, optional objects that adapt MLflow, TensorBoard and Weights & Biases
to LambdaForge's existing Lightning logger boundary. The base installation remains local and
service-free: no tracking SDK is imported until its adapter is constructed.

## Contents

- [Start here](#start-here)
- [Scope and public objects](#scope-and-public-objects)
- [Installation](#installation)
- [YAML selection](#yaml-selection)
- [MLflow](#mlflow)
- [TensorBoard](#tensorboard)
- [Weights & Biases](#weights--biases)
- [Complete parameter reference](#complete-parameter-reference)
- [Metrics, privacy, checkpoints and artifacts](#metrics-privacy-checkpoints-and-artifacts)
- [Custom loggers and plugins](#custom-loggers-and-plugins)
- [Dependency and provider failures](#dependency-and-provider-failures)
- [Official references](#official-references)

## Start here

Tracking copies selected metrics and metadata to a visualization provider. It does not replace
LambdaForge's local `config.yaml`, `metrics.csv`, `environment.json` or `result.json`, which remain
the reproducibility and result authority. Start with the built-in local CSV logger. Add one provider
only when the project has decided where data may be stored and how credentials are supplied.

TensorBoard is normally local. MLflow and W&B may be local/offline or remote depending on their
parameters. Installing an extra only makes its adapter available; no account, upload or network
publication happens until that adapter is configured and constructed.

## Scope and public objects

Import these objects from `lambdaforge.tracking`:

| Object | Responsibility |
|---|---|
| `MLflowTrackingLogger` | Construct Lightning's MLflow logger after checking the `mlflow` extra. |
| `TensorBoardTrackingLogger` | Construct Lightning's TensorBoard logger after checking the `tensorboard` extra. |
| `WeightsAndBiasesTrackingLogger` | Construct Lightning's W&B logger after checking the `wandb` extra. |
| `TrackingBackend` | Canonical `mlflow`, `tensorboard` and `wandb` backend identifiers. |
| `TrackingDependencyGuard` | Check availability without importing a tracking SDK, then require or import it explicitly. |
| `TrackingDependencyError` | Actionable `ImportError` with the exact optional-extra installation hint. |

The three adapters are ordinary Lightning loggers. The TensorBoard guard accepts `tensorboard`
(preferred) or an already installed `tensorboardX`; the documented `tensorboard` extra installs the
preferred backend. They do not create another experiment engine,
change the training loop or make a provider mandatory. `LightningRunner` passes one adapter, or a
non-empty list of adapters, directly to Lightning. LambdaForge still writes its materialized
configuration, result/environment manifests, captured log and—unless disabled—the canonical dense
`metrics.csv` independently of the selected external logger.
`EnvironmentManifest` also records installed `mlflow`, `tensorboard`, `tensorboardX` and `wandb`
versions when present.

## Installation

Install only the provider required by the project:

```powershell
python -m pip install "lambdaforge[mlflow]"
python -m pip install "lambdaforge[tensorboard]"
python -m pip install "lambdaforge[wandb]"
```

Install all three adapters with the combined extra:

```powershell
python -m pip install "lambdaforge[tracking]"
```

From an editable clone, use `-e ".[mlflow]"`, `-e ".[tensorboard]"`,
`-e ".[wandb]"` or `-e ".[tracking]"`. The base `lambdaforge` dependency set intentionally contains
none of these SDKs.

## YAML selection

Select one adapter with the same trusted recursive `target` syntax used by every other object:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: lambdaforge-research
      run_name: baseline-seed-7
      save_dir: ./mlruns
      log_model: false
  write_epoch_metrics_csv: true
```

To publish to several destinations, use a non-empty logger list. Each entry is constructed
independently and may mix built-in tracking adapters, project `target` classes and installed
`kind: logger` plugins:

```yaml
trainer:
  logger:
    - target: lambdaforge.tracking.TensorBoardTrackingLogger
      params:
        save_dir: ./tracking/tensorboard
        name: comparison
    - target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
      params:
        project: lambdaforge-research
        name: baseline-seed-7
        offline: true
        save_dir: ./tracking/wandb
        log_model: false
  write_epoch_metrics_csv: true
```

The existing strings `none`, `csv` and `lightning_csv` retain their original meaning. Provider names
are not new magic strings: adapters are explicit objects with fully configurable constructor
parameters.

Selecting a provider does not override the task publication policy. Set `logger: true` inside the
existing `task.params.logging` mapping when task losses and metrics should reach it:

```yaml
task:
  target: lambdaforge.training.LightningTask
  params:
    # ...the project's model routing and optimizer settings...
    logging:
      logger: true
```

`EpochStats` runtime scalars still use Lightning's logger path independently. Keep the task flag
`false` when the provider should receive runtime stats but not scientific loss/metric values.

## MLflow

### Local files

When `tracking_uri` is `null` and `MLFLOW_TRACKING_URI` is not set, Lightning uses `save_dir` for
local MLflow runs:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: local-study
      run_name: seed-7
      tracking_uri: null
      save_dir: ./tracking/mlruns
      tags: {stage: exploratory, owner: research-team}
      log_model: false
```

`save_dir` defaults to `./mlruns`. An explicit YAML `tracking_uri` takes precedence over
`MLFLOW_TRACKING_URI`; otherwise that environment variable takes precedence over local
`save_dir`. Use an explicit path when several launchers may have different working directories.
MLflow assigns the run when `run_id` is `null`; supplying a previous ID asks the provider to reuse
that run and is a scientific/lifecycle choice, not LambdaForge resume.

### Remote server

Set an HTTP(S) tracking URI for a remote server:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.MLflowTrackingLogger
    params:
      experiment_name: shared-study
      run_name: seed-7
      tracking_uri: https://mlflow.example.org
      artifact_location: null
      log_model: false
      synchronous: true
```

Provider credentials, tokens and object-store secrets belong in the process environment or the
provider's credential mechanism, never in YAML. `save_dir` has no effect when `tracking_uri` is set.
The remote server decides the default artifact location unless `artifact_location` is supplied.
Server authentication, TLS, authorization, backing database, artifact proxy and retention are
operator responsibilities outside LambdaForge.

## TensorBoard

TensorBoard writes event files and does not require a hosted tracking account:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.TensorBoardTrackingLogger
    params:
      save_dir: ./tracking/tensorboard
      name: local-study
      version: seed-7
      log_graph: false
      default_hp_metric: true
      prefix: ""
      sub_dir: null
      max_queue: 10
      flush_secs: 120
```

`save_dir` is required. Logs are placed under `save_dir/name/version`, with `sub_dir` appended when
present. Extra parameters are forwarded to the TensorBoard `SummaryWriter`; `max_queue` and
`flush_secs` above are examples. Launch the viewer separately:

```powershell
tensorboard --logdir ./tracking/tensorboard
```

Keep `version: null` for Lightning-generated versions, or ensure an explicit version is unique.
Concurrent jobs writing the same directory/version can corrupt or interleave an event stream.
Remote filesystem URLs depend on the provider support used by Lightning/fsspec and are not covered
by LambdaForge's local atomicity or retention guarantees.

## Weights & Biases

### Offline

Offline mode stores a self-contained W&B run directory for later review or synchronization:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
    params:
      project: lambdaforge-research
      name: seed-7
      offline: true
      save_dir: ./tracking/wandb
      log_model: false
      save_code: false
      tags: [baseline, offline]
```

Preserve the generated directory if it may later be synchronized with `wandb sync DIRECTORY`.
Lightning rejects `offline: true` together with `log_model: true` or `"all"`.

### Online

Online mode is the default and publishes during training:

```yaml
trainer:
  logger:
    target: lambdaforge.tracking.WeightsAndBiasesTrackingLogger
    params:
      project: lambdaforge-research
      entity: my-team
      name: seed-7
      group: comparison-2026
      job_type: train
      tags: [baseline]
      offline: false
      log_model: false
      save_code: false
```

Authenticate outside YAML, for example with the provider CLI or `WANDB_API_KEY`. `offline: false`
may perform network requests and provider-side metadata collection. To make an air-gapped policy
unambiguous, set `offline: true` in YAML and/or `WANDB_MODE=offline` in the launch environment.
`version` and `id` address the same provider run identity; set at most one. Likewise, `dir` is an
alias of `save_dir`.

## Complete parameter reference

The adapters mirror the installed Lightning logger constructors. Defaults below are LambdaForge's
public defaults.

### `MLflowTrackingLogger`

| Parameter | Default | Meaning |
|---|---|---|
| `experiment_name` | `"lightning_logs"` | MLflow experiment/container name. |
| `run_name` | `null` | Optional human-readable run name. |
| `tracking_uri` | `null` | Local/provider URI; `null` uses `MLFLOW_TRACKING_URI` when present, otherwise local `save_dir`. |
| `tags` | `null` | Metadata tags passed to MLflow. |
| `save_dir` | `"./mlruns"` | Local storage used when no tracking URI is set. |
| `log_model` | `false` | `false`, `true` or `"all"` checkpoint publication policy supported by Lightning. |
| `prefix` | `""` | Prefix added to published metric keys. |
| `artifact_location` | `null` | Explicit artifact root for a newly selected experiment. |
| `run_id` | `null` | Existing provider run to reuse; `null` creates/selects normally. |
| `synchronous` | `null` | Optional MLflow synchronous-logging hint; a non-`null` value raises on older supported Lightning versions that lack it. |

### `TensorBoardTrackingLogger`

| Parameter | Default | Meaning |
|---|---|---|
| `save_dir` | required | Root directory or supported filesystem URL. |
| `name` | `"lightning_logs"` | Experiment directory; an empty string omits this level. |
| `version` | `null` | Integer/string run directory or auto-generated version. |
| `log_graph` | `false` | Whether Lightning should record the model graph when inputs permit it. |
| `default_hp_metric` | `true` | Whether to add TensorBoard's default hyperparameter metric. |
| `prefix` | `""` | Prefix added to published metric keys. |
| `sub_dir` | `null` | Optional directory below the selected version. |
| `**kwargs` | none | Additional `SummaryWriter` settings such as `max_queue`, `flush_secs` or `filename_suffix`. In YAML, place them directly under `params`. |

### `WeightsAndBiasesTrackingLogger`

| Parameter | Default | Meaning |
|---|---|---|
| `name` | `null` | Human-readable W&B run name. |
| `save_dir` | `"."` | Local W&B metadata/run directory. |
| `version` | `null` | Provider run ID/resume identity; alias of `id`. |
| `offline` | `false` | Store locally instead of live synchronization. |
| `dir` | `null` | Alias of `save_dir`. |
| `id` | `null` | Alias of `version`. |
| `anonymous` | `null` | Provider anonymous-logging choice. |
| `project` | `null` | W&B project; provider/environment fallback applies when absent. |
| `log_model` | `false` | `false`, `true` or `"all"` checkpoint artifact publication. |
| `experiment` | `null` | Pre-existing W&B Run object, mainly for direct Python/injected use. |
| `prefix` | `""` | Prefix added to published metric keys. |
| `checkpoint_name` | `null` | Optional W&B checkpoint artifact name. |
| `add_file_policy` | `"mutable"` | Provider file-add policy; `"mutable"` is omitted on older supported Lightning versions, while requesting `"immutable"` there raises. |
| `**kwargs` | none | Additional `wandb.init` arguments such as `entity`, `group`, `tags`, `notes`, `job_type`, `save_code` or `settings`. In YAML, place them directly under `params`. |

Unsupported or mutually incompatible values are rejected by the installed Lightning/provider
version. LambdaForge does not silently discard arbitrary provider options.

## Metrics, privacy, checkpoints and artifacts

- `TaskLoggingConfig.logger` controls whether task losses/metrics reach the selected Lightning
  logger. Framework callbacks such as `EpochStats` also publish epoch time and memory scalars;
  progress-bar and dense-CSV filters do not redact a remote logger.
- `write_epoch_metrics_csv: true` remains recommended. It creates the framework's provider-neutral
  dense metric artifact even when a tracker is unavailable later.
- Any secret written in YAML is copied into materialized configuration/provenance artifacts. Use
  environment variables, secret stores or provider credential files for tokens and passwords.
- Review metric names, tags, run names, notes, hyperparameters, graph logging, source-code capture,
  system telemetry and provider settings before enabling a remote destination. Dataset samples are
  not uploaded by these adapters automatically, but custom callbacks/logger calls may do so.
- `log_model` defaults to `false` for every network-capable adapter. `true` or `"all"` can copy
  checkpoints to provider-managed storage, increase bandwidth/cost and retain a remote copy after
  local LambdaForge retention prunes the original.
- LambdaForge artifact retention governs paths inside its local suite tree. It cannot delete,
  roll back, verify or apply retention policy to MLflow/W&B artifacts, remote TensorBoard files or
  offline files placed outside that tree.
- A provider logger is created separately in each spawned training process. Use unique run IDs and
  directory/version choices; sharing one explicit ID or TensorBoard stream across simultaneous
  seeds mixes lifecycle and scientific provenance.

## Custom loggers and plugins

The adapters do not close the extension surface. A project-specific logger may still be selected
with:

```yaml
trainer:
  logger:
    target: my_project.logging.ProjectLogger
    params: {endpoint: local}
```

Reusable distributions can publish a class in the `lambdaforge.loggers` entry-point group and use:

```yaml
trainer:
  logger:
    plugin: {kind: logger, name: project_logger}
    params: {endpoint: local}
```

Both forms must satisfy the public `lambdaforge.integrations.Lightning.Logger` contract. The plugin
boundary adds lazy discovery, contract checking and loaded-plugin provenance; a direct `target` is
simpler for project-local code. LambdaForge's three built-in tracking adapters use direct public
targets and are not entry-point plugins. See the [plugin guide](../plugins/README.md).

Replace `runner.target` only when the training backend itself changes. A logger integration belongs
behind `trainer.logger` and should not own scheduling, checkpoints, aggregation or process cleanup.

## Dependency and provider failures

Importing `lambdaforge` or `lambdaforge.tracking` does not import `mlflow`, `tensorboard` or `wandb`.
Constructing a selected adapter first invokes `TrackingDependencyGuard`. If the SDK is absent,
`TrackingDependencyError`—a subclass of `ImportError`—names the backend and reports the smallest
installation command, for example:

```text
Tracking backend 'mlflow' requires the optional dependency 'mlflow'. Install the recommended backend with: pip install 'lambdaforge[mlflow]'
```

`lambdaforge validate` checks the class path without constructing it, so successful structural/import
validation does not prove that the optional runtime SDK, credentials, remote server or filesystem
is usable. Exercise a minimal real run in the deployment environment.

After the dependency check, authentication, network, permissions, invalid parameters, unavailable
servers and provider finalization errors remain provider/Lightning failures. They fail the owning
training run normally; LambdaForge does not hide them or silently fall back from online to local
tracking.

## Official references

- [Lightning MLflow logger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html)
- [MLflow tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [Lightning TensorBoard logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html)
- [PyTorch TensorBoard/`SummaryWriter`](https://docs.pytorch.org/docs/stable/tensorboard.html)
- [Lightning W&B logger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html)
- [W&B run initialization and modes](https://docs.wandb.ai/models/ref/python/functions/init)
- [W&B environment variables](https://docs.wandb.ai/models/track/environment-variables)
