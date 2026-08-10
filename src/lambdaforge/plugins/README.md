# LambdaForge plugin discovery

[Repository guide](../../../README.md) · [Español](README.es.md)

This package discovers classes published by separately installed distributions through standard
Python entry-point metadata. Plugins complement fully qualified YAML `target` paths; installing or
using a plugin never requires editing LambdaForge.

## Contents

- [Start here](#start-here)
- [Public objects](#public-objects)
- [Entry-point groups](#entry-point-groups)
- [Publishing a plugin](#publishing-a-plugin)
- [Using plugins in YAML](#using-plugins-in-yaml)
- [Python API](#python-api)
- [Command-line discovery](#command-line-discovery)
- [Validation](#validation)
- [Loaded-plugin provenance](#loaded-plugin-provenance)
- [Loading, caching and security](#loading-caching-and-security)
- [Contracts and precedence](#contracts-and-precedence)

## Start here

Most projects do **not** need a plugin. If a class lives in the same installed research project,
reference it directly with `target: my_project.module.Class`. Create a plugin only when another
Python distribution should publish a named reusable extension that consumers can select without
knowing its module path.

| Situation | Prefer |
|---|---|
| One project owns the class | Fully qualified YAML `target`. |
| Several projects install a reusable package | Entry-point `plugin`. |
| Pass an existing callable without constructing it | YAML `ref`. |

Plugin discovery reads installed distribution metadata lazily. Resolution imports third-party
Python code, so install only trusted packages and audit the recorded distribution/version
provenance with every run.

## Public objects

| Object | Responsibility |
|---|---|
| `PluginKind` | Closed set of supported contracts and their namespaced metadata groups. |
| `PluginReference` | Immutable, validated `(kind, name)` selection. |
| `PluginDescriptor` | Immutable installed/resolved metadata, including distribution and version. |
| `PluginRegistry` | Lazy discovery, contract checks, class caching and process diagnostics. |
| `PluginUsageSession` | Context-managed, run-local snapshot of successful resolutions. |
| `PluginResolutionError` | Contextual failure for missing, ambiguous, unloadable or invalid plugins. |

Import these objects from `lambdaforge.plugins`; individual class modules are implementation
details.

## Entry-point groups

| YAML kind | Distribution group | Required exported class |
|---|---|---|
| `model` | `lambdaforge.models` | `torch.nn.Module` subclass |
| `metric` | `lambdaforge.metrics` | `lambdaforge.metrics.Metric` subclass |
| `activation` | `lambdaforge.activations` | `lambdaforge.nn.activations.Activation` subclass |
| `normalization` | `lambdaforge.normalizations` | `lambdaforge.nn.normalizations.Normalization` subclass |
| `loss` | `lambdaforge.losses` | `lambdaforge.nn.losses.Loss` subclass |
| `distance` | `lambdaforge.distances` | `lambdaforge.nn.distances.Distance` subclass |
| `pooling` | `lambdaforge.pooling` | `lambdaforge.nn.pooling.Pooling` subclass |
| `similarity` | `lambdaforge.similarities` | `lambdaforge.nn.similarities.Similarity` subclass |
| `kernel` | `lambdaforge.kernels` | `lambdaforge.nn.kernels.Kernel` subclass |
| `encoding` | `lambdaforge.encodings` | `lambdaforge.nn.encodings.Encoding` subclass |
| `regularization` | `lambdaforge.regularization` | `lambdaforge.nn.regularization.Regularization` subclass |
| `dataset` | `lambdaforge.datasets` | `torch.utils.data.Dataset` subclass |
| `callback` | `lambdaforge.callbacks` | `lambdaforge.integrations.Lightning.Callback` subclass |
| `logger` | `lambdaforge.loggers` | `lambdaforge.integrations.Lightning.Logger` subclass |
| `task` | `lambdaforge.tasks` | `lambdaforge.tasks.Task` subclass |

Groups are deliberately separate: a name identifies one implementation within one contract, and
the same name may legitimately exist in different groups. Entry-point names are case-sensitive and
should contain only letters, numbers, underscores, dots and dashes.

## Publishing a plugin

An external package declares classes in its own `pyproject.toml`:

```toml
[project]
name = "acme-lambdaforge"
dependencies = ["lambdaforge>=0.4.1,<0.5"]

[project.entry-points."lambdaforge.models"]
acme_encoder = "acme_lambdaforge.models:AcmeEncoder"

[project.entry-points."lambdaforge.metrics"]
calibrated_auc = "acme_lambdaforge.metrics:CalibratedAUROC"

[project.entry-points."lambdaforge.activations"]
acmegelu = "acme_lambdaforge.activations:AcmeGELU"

[project.entry-points."lambdaforge.losses"]
calibrated_focal = "acme_lambdaforge.losses:CalibratedFocalLoss"

[project.entry-points."lambdaforge.datasets"]
acme_records = "acme_lambdaforge.data:AcmeRecords"

[project.entry-points."lambdaforge.callbacks"]
artifact_marker = "acme_lambdaforge.callbacks:ArtifactMarker"

[project.entry-points."lambdaforge.loggers"]
jsonl_logger = "acme_lambdaforge.logging:JsonLinesLogger"

[project.entry-points."lambdaforge.tasks"]
surface_builder = "acme_lambdaforge.tasks:SurfaceBuilder"
```

The value uses the standard entry-point `importable.module:attribute` syntax. Each attribute must be
a class satisfying its group contract. Install the distribution in the same Python environment as
LambdaForge before resolving it; editable installs are suitable during development.

Activation and normalization strings pass through `ComponentRegistry` normalization: case,
underscores and dashes are ignored. Publish their entry-point name in normalized form, for example
`acmegelu`; YAML may then use `acme-gelu`, `Acme_GELU` or `acmegelu`.

## Using plugins in YAML

A plugin specification is explicit and keeps `params` separate from discovery metadata:

```yaml
model:
  plugin:
    kind: model
    name: acme_encoder
  params:
    in_features: 32
    hidden_features: 128

val_metrics:
  - plugin:
      kind: metric
      name: calibrated_auc
    params:
      output_key: logits
      target_key: target

data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train, root: datasets/acme}
  val:
    plugin: {kind: dataset, name: acme_records}
    params: {split: validation, root: datasets/acme}

callbacks:
  - plugin: {kind: callback, name: artifact_marker}
    params: {filename: finished.txt}

trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
    params: {path: metrics.jsonl}
```

A generic task document selects the dedicated task contract at its root:

```yaml
schema_version: "1.0"
kind: task
name: build-surfaces
task:
  plugin: {kind: task, name: surface_builder}
  params: {resolution: 1.0}
```

`ObjectFactory` resolves the class and recursively builds `params` exactly as it does for `target`.
Every build creates a fresh instance. Existing `target` and `ref` forms remain supported and can be
nested inside plugin parameters. `trainer.logger` accepts one logger or a non-empty list mixing
`target`, `ref` and logger plugins; built-in `csv`/`lightning_csv`/`none` modes remain unchanged.

Activation and normalization plugins use their short alias in compatible model parameters:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    hidden: [64]
    activation: acme-gelu
```

## Python API

```python
from lambdaforge.experiments import ObjectFactory
from lambdaforge.plugins import PluginKind, PluginReference, PluginRegistry

registry = PluginRegistry.default()
metadata_only = registry.discover(PluginKind.MODEL)
with registry.usage_session() as usage:
    model_class = registry.resolve(PluginReference(PluginKind.MODEL, "acme_encoder"))
    model = ObjectFactory.build(
        {
            "plugin": {"kind": "model", "name": "acme_encoder"},
            "params": {"in_features": 32},
        },
        plugins=registry,
    )

used_by_this_context = usage.descriptors()
resolved_by_this_process = registry.resolved_plugins()
```

`discover()` returns immutable descriptors and does not load their classes. `resolve()` returns a
validated class, not an instance. Supplying a registry to `ObjectFactory` is useful for dependency
injection and tests; ordinary application code can use the process-local default.
`usage_session()` records only successful resolutions made inside its context, including cache
hits and activation/normalization plugin aliases. Repeated resolutions are deduplicated and its
snapshot is canonically ordered. `resolved_plugins()` is the wider process diagnostic ledger and
must not be substituted for a per-run session.

Call `registry.refresh()` after installing or removing distributions in a running interpreter.
`refresh(PluginKind.METRIC)` invalidates only one category.

## Command-line discovery

```powershell
lambdaforge plugins
lambdaforge plugins --kind metric
lambdaforge plugins --json
```

The command lists name, group, object reference, distribution and version from metadata without
importing plugin modules. Duplicate providers are all shown; attempting to resolve an ambiguous
`(kind, name)` fails rather than selecting one according to environment order.

## Validation

Normal validation checks the JSON Schema, confirms that the requested entry point exists, imports
its class and verifies the contract without instantiating it:

```powershell
lambdaforge validate experiment.yaml
```

The experiment schema constrains `kind: model` for `model`, `kind: metric` for metric lists,
`kind: loss` for loss lists, `kind: dataset` for `data.train/val/test`, `kind: callback` for
`callbacks` and `kind: logger` for `trainer.logger`. Every category remains valid in recursively
built parameters. A wrong top-level kind is therefore reported before execution, while nested
objects are contract-checked when `ObjectFactory` resolves them.
The independent task Schema 1.0 requires `kind: task` for its root plugin.

Template validation can deliberately skip every external `target`, `ref` and plugin load:

```powershell
lambdaforge validate experiment.yaml --no-imports
```

This mode validates structure and expansion but does not prove that referenced packages are
installed or satisfy their contracts. The report records that imports were not checked.
Import-aware validation also deliberately does not count as run usage: a later execution records
the plugin only if its object graph actually resolves it.

## Loaded-plugin provenance

Every real materialised run automatically records the entry points it successfully resolved in
`environment.json`:

```json
{
  "plugins": [
    {
      "kind": "model",
      "name": "acme_encoder",
      "group": "lambdaforge.models",
      "value": "acme_lambdaforge.models:AcmeEncoder",
      "distribution": "acme-lambdaforge",
      "version": "2.1.0"
    }
  ]
}
```

The list is deterministic and contains exactly the six fields shown above; distribution and
version are `null` when the installed metadata does not expose them. Installed-but-unused,
discovered-only, failed, ambiguous and contract-invalid entry points are absent. A class that
resolved successfully remains present if its constructor or the later training run fails.
Fully-qualified `target`/`ref` imports are not invented as plugin distributions.

Provenance is scoped to the run's execution context and process. Sequential runs, earlier
validation, the CLI metadata listing and the parent of a `spawn` worker cannot contaminate it. A
dry-run performs no imports and therefore writes `"plugins": []`. Resolutions performed only by
user-created child processes or DataLoader workers belong to those processes and are not attributed
to the parent without an explicit user IPC integration.

## Loading, caching and security

Discovery reads installed distribution metadata only. Resolution calls the selected entry point's
`load()` method, which imports its module and may execute module-level Python. Plugins must therefore
be treated as trusted installed code, exactly like fully qualified YAML targets; this mechanism is
not a process sandbox.

The registry caches discovered metadata and successfully resolved classes. It never caches model or
metric instances, datasets, tensors or user data. `refresh()` invalidates discovery/class caches but
does not rewrite the diagnostic fact that a descriptor was previously resolved. Each spawned Python
process owns its own default registry; a PID change after fork replaces inherited locks, caches,
diagnostic history and active usage contexts. No live plugin instance or open resource is
transferred between training jobs.

Failed imports retain their original exception as the cause. Missing names report available names,
and conflicting providers report every distribution and object reference. No first-provider-wins
fallback is used.

## Contracts and precedence

Models may be ordinary `torch.nn.Module` subclasses; inheriting LambdaForge's optional `Model` base
is not required. Metrics must implement the complete `Metric` lifecycle, including distributed state
when used with DDP. Losses, activations, normalizations, distances, pooling operators, similarities,
kernels, encodings and regularizers must inherit their corresponding LambdaForge base class.

Dataset plugins must expose a class derived from `torch.utils.data.Dataset`; this also admits
`IterableDataset`. The default LambdaForge data module shuffles its training split, so iterable or
streaming datasets should provide a compatible `data.datamodule.target`. Callback and logger authors
should inherit from `lambdaforge.integrations.Lightning.Callback` and
`lambdaforge.integrations.Lightning.Logger`, which follow LambdaForge's modern/legacy Lightning
selection. Entry points expose classes, not singleton instances or factories. Constructors should
remain spawn-safe and defer files, sockets and services until the normal runtime lifecycle.
Task plugins must inherit `lambdaforge.tasks.Task`; fully qualified task targets still accept duck
typing for concise project-local code.

For activation and normalization aliases, explicit process registration and built-in components are
checked before installed plugins. Consequently, a discovered package cannot silently replace
`relu`, `layernorm` or another existing alias. Explicit plugin references for every category state
their kind and exact name, eliminating contextual or magic-string inference.
