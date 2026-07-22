# LambdaForge experiment system

[Repository guide](../../../README.md) · [Español](README.es.md)

This package turns one trusted YAML document into reproducible variant/seed runs, schedules them and
reduces their on-disk results. It contains no task-specific model or dataset logic.

## Contents

- [Main objects](#main-objects)
- [Lifecycle](#lifecycle)
- [Configuration migrations](#configuration-migrations)
- [Expansion rules](#expansion-rules)
- [Execution](#execution)
- [Completion and resume](#completion-and-resume)
- [Typed results](#typed-results)
- [Attempt history and result selection](#attempt-history-and-result-selection)
- [Artifacts and aggregation](#artifacts-and-aggregation)
- [Artifact retention](#artifact-retention)
- [Statistical comparisons](#statistical-comparisons)
- [Loading](#loading)
- [Plugin specifications](#plugin-specifications)
- [Extension boundaries](#extension-boundaries)

## Main objects

| Object | Responsibility |
|---|---|
| `Experiment` | Public handle for expand, run, aggregate, retention and load operations. |
| `ExperimentConfig` | Own YAML loading, dotted paths, validation and suite expansion. |
| `ExperimentValidator` | Validate Schema, expansion, resources and imports without side effects. |
| `ValidationReport` | Immutable human/JSON-readable validation result. |
| `ExperimentConfigMigrator` | Plan, apply and validate exact forward Schema migrations without user imports. |
| `ExperimentConfigMigrationResult` | Immutable diff/YAML/JSON preview and explicit atomic YAML persistence. |
| `ExperimentConfigMigrationRegistry` | Immutable deterministic migration-chain registry. |
| `ExperimentSchemaVersion` / `ExperimentSchemaCatalog` | Exact version value and packaged Schema selection. |
| `ObjectFactory` | Recursively resolve `target`, `ref` and installed-plugin specifications. |
| `ExecutionConfig` | Validate resource settings and create logical GPU slots. |
| `ExperimentExecutor` | Select sequential, parallel or DDP execution. |
| `ExperimentRunner` | Materialise and run one configuration and write its result. |
| `ExperimentAggregator` | Read completed runs and create statistics, CSVs and plots. |
| `ArtifactRetentionPolicy` | Validate the strict Schema 1.1 checkpoint/archive/prune policy. |
| `ArtifactRetentionManager` | Preview or apply retention under ordered suite locks. |
| `ArtifactRetentionPlan` / `ArtifactRetentionResult` | Immutable mapping/JSON-compatible plan and transaction outcome. |
| `RunLoader` | Find run directories and reconstruct checkpointed models. |
| `RunResult` | Immutable, typed and JSON-compatible terminal result for one run. |
| `RunFingerprint` | Canonical identity of the expanded scientific configuration. |
| `ResultCatalog` / `ResultRecord` | Discover, audit and explicitly select current or archived attempts. |
| `AggregateResult` | Immutable mapping of typed per-variant aggregates. |
| `VariantAggregateResult` | Typed access to counts, state and metrics for one variant. |
| `StatisticalComparisonConfig` | Validate and materialise the nested comparison protocol. |
| `ConfidenceIntervalResult` | Immutable interval estimate, status and reproducibility metadata. |
| `PairedTestResult` | Immutable selected/diagnostic p-values, ranks and effective counts. |

Supporting classes such as `ExperimentWorker`, `StdIOCapture`, `TeeStream`, `CheckpointChoice` and
the status enums each live in their own module.

## Lifecycle

```text
YAML → ExperimentConfigMigrator → current Schema config + migration metadata
     → ExperimentValidator → ValidationReport
     → ExperimentConfig → expanded variant/seed configs
     → ExecutionConfig → process/device slots
     → ExperimentRunner → config, environment, log, metrics, checkpoints, result
     → ExperimentAggregator → cross-seed tables, plots and completion receipt
     → ResultCatalog → current/history audit and explicit attempt selection
     → ArtifactRetentionManager → read-only plan or gated transaction
     → RunLoader → reconstructed model
```

Use the high-level object unless writing an integration:

```python
from lambdaforge import Experiment

experiment = Experiment.from_yaml("experiment.yaml")
report = experiment.validate()
for run in experiment.expand():
    print(run["experiment"]["variant"], run["experiment"]["seed"])
results = experiment.run(dry_run=True)
print(results[0].status, results[0]["status"])
```

`lambdaforge validate experiment.yaml` performs the same validation. `--json` emits a stable report
and `--no-imports` keeps template validation useful before its external project or plugins are
installed. This option skips `target`, `ref` and entry-point loading. No object is instantiated and
no run directory is created; normal import checking can still execute module top-level code, so only
trusted configurations should be checked.

## Configuration migrations

Schema 1.1 requires the exact quoted declaration `schema_version: "1.1"`. Schema 1.0 remains
packaged for exact historical validation. YAML without a version is recognised as `unversioned` and
follows the deterministic `unversioned -> 1.0 -> 1.1` path. `UnversionedToV1Migration` declares
1.0 without changing experiment semantics; `ExperimentV1ToV1_1Migration` advances that valid
mapping to 1.1, whose optional retention block defaults to disabled.

`ExperimentConfig` applies the complete chain in memory at normal loading boundaries, so old source
files continue to expand, execute, aggregate and reload without being edited. Expanded and
materialised configurations use the canonical current version.

Use the dedicated command to inspect or persist the change:

```powershell
lambdaforge migrate legacy.yaml                   # default unified diff
lambdaforge migrate legacy.yaml --format yaml
lambdaforge migrate legacy.yaml --format json
lambdaforge migrate legacy.yaml --check           # 1 if stale, 0 if current
lambdaforge migrate legacy.yaml --target-version 1.1
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
```

`--target-version` defaults to the current packaged Schema. `--output` must be different from the
source and always receives complete YAML; an existing destination needs `--force`. The source is
still refused with `--force`. `--check` never writes and cannot be combined with `--output`.
Normal successful previews/writes return 0, migration or output errors return 1, and command syntax
errors return 2. In check mode, 1 specifically means that at least one valid step is required.

The migration path is presentation-aware and preview-first. It rejects duplicate YAML keys, keeps
the caller's mapping untouched, preserves programmatic Python value types and keeps
comments/order/quotes/anchors/newlines in files where possible. Each result is validated against
the exact target Schema. It does not call `ObjectFactory`, resolve plugins or import configured
`target`/`ref` objects. Writes use a synced temporary beside the destination, atomic no-clobber
publication by default and atomic replacement only with explicit `--force`; no in-place mode
exists.

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import ExperimentConfigMigrator, MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.render(MigrationPreviewFormat.DIFF))

mapping_preview = ExperimentConfigMigrator.default().preview_mapping(raw_config)
assert mapping_preview.target_version.value == "1.1"
```

`ValidationReport` exposes source/target versions and applied step descriptors. Migration Schema
validation deliberately excludes imports; run `lambdaforge validate` afterwards for expansion,
resources and optional import checks. The current registry contains the exact
`unversioned -> 1.0 -> 1.1` path and does not support downgrade, source overwrite, guessed versions
or skipped unregistered steps.

See the [complete migration guide](migrations/README.md) for preview envelopes, exit codes,
persistence guarantees, public objects, failure modes and the reviewed process for adding a future
Schema and consecutive migration object.

## Expansion rules

`experiment.seeds` accepts a scalar or list. `sweep.grid` maps dotted paths to non-empty value lists;
all grid dimensions form a Cartesian product. `sweep.include_base` controls whether the unmodified
configuration is included. Every `sweep.ablations` entry adds a named set of dotted-path overrides.

Expansion uses deep copies: one run cannot mutate another. Experiment names must be non-empty and
the final `(variant, seed)` identities must be unique. `lambdaforge inspect` prints the concrete
configurations without executing imported objects.

## Execution

`sequential` stays in the caller. `parallel` schedules one spawned process per independent run on
fixed one-GPU slots. `ddp` schedules a run on each group of `devices_per_job` GPUs and patches its
Lightning strategy to DDP. CLI overrides have precedence over YAML, which has precedence over
defaults.

The executor deliberately uses pickle-safe worker objects and the `spawn` start method. GPU indices
are logical relative to the parent `CUDA_VISIBLE_DEVICES`; CPU/thread/worker limits are patched into
each run rather than mutating the parent environment.

See the [training process guide](../training/README.md) for shutdown guarantees and caveats.

## Completion and resume

Machine-readable run states are `ok`, `failed`, `dry_run`, `interrupted` and `unknown`.

A run is complete only when:

1. `result.json` has status `ok`;
2. its stored or legacy-materialised scientific fingerprint matches the expanded config;
3. the result's selected checkpoint exists when checkpointing is required; and
4. every path in `experiment.required_artifacts` exists inside the run directory.

With `rerun_completed: false`, complete runs are skipped. With `resume: true`, incomplete runs use a
usable last checkpoint when one exists. Failure handling writes a terminal failure result and the
suite can be launched again without discarding successful seeds. Required artifacts must be relative
and are entirely project-defined.

A cooperative stop persists `interrupted`, skips `test_after_fit` and remains retryable; it is not a
successful terminal state for aggregation or retention. Before a real retry, the previous result is
retired into internal attempt history so a stale `ok` artifact cannot make a crashed attempt appear
successful.
An incomplete run resumes a checkpoint only when that same identity matches; changing model, data,
loss, metric, optimizer, trainer or extension settings starts cleanly instead of loading
incompatible weights.

## Typed results

`Experiment.run`, `LambdaForge.run`, `ExperimentRunner` and `ExperimentExecutor` return
`RunResult` objects. They preserve the historical dictionary contract—including direct
`json.dumps(result)`—while adding typed attributes such as `status: RunStatus`, `seconds`,
checkpoint paths and metric mappings. `result_version` versions the JSON envelope independently of
the YAML Schema. Version 2 also records `attempt_id`, `config_fingerprint`, `started_at_utc` and
`finished_at_utc`.

```python
import json
from lambdaforge import RunResult

result: RunResult = experiment.run()[0]
assert result["status"] == result.status.value
payload = result.to_dict()
same = RunResult.from_mapping(payload)
json_text = json.dumps(same)
```

Results reject item and attribute mutation. `with_updates(...)` creates a new object, unknown JSON
fields survive round trips, and old `result.json` files without `result_version` remain readable.
`write_json` uses a same-directory temporary file, flushes it and atomically replaces the target so
concurrent aggregation never observes a partially written object.

`Experiment.aggregate` and `ExperimentAggregator.write` return `AggregateResult`; legacy access such
as `aggregates["base"]["metrics"]` is unchanged. `aggregates.variant("base")` returns a
`VariantAggregateResult` with typed `complete`, `terminal`, `expected_runs`, `completed_runs` and
`metrics` properties. `AggregateResult.read_json` accepts both the historical variant map and the
complete `summary.json` envelope.

## Attempt history and result selection

`RunFingerprint` hashes a normalized expanded configuration. It deliberately excludes names,
output paths, checkpoint/retry controls, execution scheduling, aggregation, retention and
descriptive metadata; those change how an attempt is operated, not the scientific calculation.
Expanded seed, data, model, losses, metrics, optimizer, scheduler, task, trainer, runner, callbacks
and `extensions` remain identity-bearing. Changing one prevents stale completion reuse.

Before a retry replaces the canonical terminal marker, LambdaForge archives it under
`<run>/.lambdaforge/attempts/result-<attempt-id>.json`. The canonical `result.json` is the current
attempt; the archive is immutable history. Checkpoints/artifacts retain their normal run paths, so
archived terminal metadata must not be interpreted as an independently reloadable checkpoint copy.

Use the catalog instead of globbing for `latest`:

```python
experiment = Experiment.from_yaml("experiment.yaml")
records = experiment.results(status="ok")
catalog = experiment.result_catalog()
duplicates = catalog.duplicate_groups()
ambiguous = catalog.ambiguous_successes()
selected = catalog.select(attempt_id="20260722T120000000000Z-a1b2c3d4e5f6-acde1234")
index_path = catalog.write_index()
```

Every call to `records()` scans the current filesystem; `result-index.json` is only an atomic
interchange snapshot. `duplicate_groups()` includes retries of any status. `ambiguous_successes()`
specifically identifies a scientific identity with more than one successful result, where a paper
or downstream job must record an explicit `attempt_id` rather than let directory ordering decide.

```bash
lambdaforge results experiment.yaml
lambdaforge results runs/experiments/study --status ok --no-archived
lambdaforge results experiment.yaml --duplicates --json
lambdaforge results experiment.yaml --write-index --fail-on-ambiguous
```

The last option returns exit code 2 for ambiguity and is intended for CI/publication pipelines.
Malformed result files are ignored by discovery rather than treated as successful. The catalog is
local-filesystem metadata; remote object stores need an explicit synchronization/inventory layer.

## Artifacts and aggregation

The run directory contains the materialised `config.yaml`, typed provenance in `environment.json`,
`hparams.json`, captured `train.log`, `metrics.csv`, checkpoints, `result.json` and custom artifacts.
The environment artifact records Python/platform, core dependency versions, CUDA/cuDNN, visible GPU
properties, Git commit/branch/dirty state when available, and a deterministic `plugins` list. Each
plugin record contains `kind`, `name`, `group`, entry-point `value`, distribution and version.
Only entry points successfully resolved by that run appear: earlier validation, installed-but-unused
plugins and ordinary `target`/`ref` imports do not. Dry-runs write an empty list. The manifest is
written atomically before training and refreshed on normal or exceptional exit so constructor or
training failures preserve every resolution reached. Exact paths are derived from experiment name,
variant slug and seed.

`ExperimentAggregator.write` can reconstruct suite reports from disk. It emits per-variant terminal
and epoch summaries, wide/long CSV representations, seed statistics, pairwise directional tests,
Benjamini-Hochberg q-values and optional plots. Plot failure is recorded without losing numeric
tables. `lambdaforge aggregate --no-plots` is suitable for headless minimal environments.

A final aggregation publishes `aggregate/aggregation_receipt.json` last and only when every expected
run is `ok`, every variant is complete and terminal, required/core inputs are safe, and committed
run and aggregate fingerprints match. Incremental aggregation invalidates an older receipt and can
never trigger retention.

The aggregate statistics are exploratory. The code reports sample sizes and incomplete variants so
missing seeds are visible; study-specific inference decisions remain the researcher's responsibility.

## Artifact retention

Schema 1.1 adds an optional strict policy:

```yaml
retention:
  mode: preview                 # disabled, preview, apply
  checkpoints:
    keep: last_and_best         # all, best, last, last_and_best
    prune_unselected: true
  protect: [reports/**]
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

Omission is `disabled`. `preview` is read-only; `apply` allows automatic execution only after a
successful final aggregation. Manual application is still an explicit mutation request:

```python
plan = experiment.preview_retention()
result = experiment.apply_retention()
```

```powershell
lambdaforge retain experiment.yaml
lambdaforge retain experiment.yaml --json
lambdaforge retain experiment.yaml --apply
```

Application requires a current completion receipt and replans under locks. Core run files,
`required_artifacts`, protected globs, aggregate outputs, checkpoints selected by generic rules,
links/reparse points and internal transaction metadata are protected. Checkpoints use a separate,
role-aware policy; ambiguity or absence skips pruning for that run instead of guessing.

Compression streams into immutable per-run ZIPs, then verifies names, CRCs, sizes and SHA-256
before originals enter reversible quarantine. `only_if_smaller` preserves incompressible sources.
The durable journal rolls back before its commit marker and finishes forward after it; a committed
plan is idempotent. Training, final aggregation and retention coordinate through activity,
aggregation and retention cross-process locks in a fixed order.

See the [artifact-retention guide](retention/README.md) for the full eligibility contract,
protection rules, typed statuses, transaction artifacts, crash recovery and local-filesystem
limits.

## Statistical comparisons

`ExperimentAggregator` pairs each variant with its baseline using only seeds for which both values
exist. `delta` is `variant - baseline`; `improvement` equals that delta for `max` metrics and its
negative for `min` metrics. Thus all alternatives use one stable convention: positive means the
variant is better.

Select the inference protocol explicitly beneath `aggregation.comparisons`:

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon
      alternative: two_sided
      calculation: auto
      zero_method: wilcox
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12
```

Every nested mapping rejects unknown keys. If the whole block is absent, version-3 behaviour is
preserved: `normal` at 95%, exact `sign`, `observed_direction`, `alpha = 0.05`,
`target_power = 0.80` and `min_pairs_for_verdict = 3`. The selected p-value drives
Benjamini-Hochberg correction and the comparison verdict.

`bootstrap_percentile` estimates the paired mean by resampling with replacement. A SHA-256-derived
effective seed combines the configured base seed with
`(baseline_variant, variant, metric)`, so unrelated metric ordering does not perturb an existing
comparison. PCG64 generation is batched: `batch_size` is an upper request and
`max_batch_elements` limits transient index elements. The estimator retains only one mean per
resample (`O(resamples)`), returns explicit unavailable metadata below two pairs and marks constant
samples as degenerate.

`wilcoxon` offers `two_sided`, `greater`, `less` and `observed_direction`. `auto` uses exact
conditional sign enumeration while the number of non-zero pairs is at most `exact_max_pairs`, then
uses the normal approximation; requesting `exact` above the limit returns an explicit unavailable
result instead of silently changing methods. `wilcox` removes zeros before ranking, `pratt` includes
them in ranking but excludes their ranks from the random sign sum, and `zsplit` additionally splits
their rank contribution between the reported positive and negative statistics. Average ranks make
ties deterministic. Optional rounding occurs before zero detection and ranking.

`baseline_comparisons.csv` contains method-neutral interval and paired-test fields, selected and
diagnostic p-values, effective sample sizes, rank/zero diagnostics and bootstrap seed provenance.
`reliability.json` embeds the fully materialised `statistical_protocol` and every comparison;
`summary.json` records the protocol and artifact paths. Version-4 aggregation still writes the
historical normal-95% and sign-test columns, so existing consumers can migrate independently while
`p_value_directional`, its BH q-value and verdict follow the selected method.

The public configuration/enums/results are lazy exports from `lambdaforge.experiments`. Concrete
strategy objects and the composition engine are available from
`lambdaforge.experiments.statistics`:

```python
from lambdaforge.experiments import StatisticalComparisonConfig
from lambdaforge.experiments.statistics import StatisticalComparisonEngine

protocol = StatisticalComparisonConfig.from_mapping(config)
engine = StatisticalComparisonEngine(protocol)
interval = engine.confidence_interval(
    [0.02, 0.01, 0.03],
    identity=("base", "candidate", "val_auroc"),
)
test = engine.paired_test([0.02, 0.01, 0.03])
```

See the [statistics package guide](statistics/README.md) for every value, default, result field,
edge case and API object.

## Loading

```python
experiment = Experiment.from_yaml("experiment.yaml")
model = experiment.load_model(seed=7, variant="base", which="auto")
```

`CheckpointChoice` offers `best`, `last` and `auto`. `AUTO` resolves best, then last, then the latest
safe local checkpoint; exact `BEST` and `LAST` choices never silently cross roles. `RunLoader`
validates the run, imports the model from its materialised object specification, loads a direct
state mapping or strips Lightning's `model.` prefix, then returns the model in evaluation mode.

`RunLoader.load_result(run_dir)` reads the same run's terminal artifact as a `RunResult`.

## Plugin specifications

Installed models and metrics may be selected by explicit entry-point identity instead of an import
path:

```yaml
model:
  plugin:
    kind: model
    name: acme_encoder
  params:
    in_features: 32

val_metrics:
  - plugin:
      kind: metric
      name: calibrated_auc
    params:
      output_key: logits

data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train}

callbacks:
  - plugin: {kind: callback, name: artifact_marker}

trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
    params: {path: metrics.jsonl}
```

`plugin` contains only `kind` and `name`; constructor `params` remain its sibling and are built
recursively. The Schema restricts the model position to `kind: model` and metric lists to
`kind: metric`, loss lists to `kind: loss`, data splits to `kind: dataset`, callbacks to
`kind: callback` and trainer loggers to `kind: logger`. Fully qualified `target` specifications
remain valid and can coexist with plugins in one experiment. Logger lists may combine object,
reference and plugin forms.

Validation with import checks resolves and contract-checks the selected class without instantiating
it, but that validation is not reported as run usage. During execution, a per-run context captures
explicit plugin objects and plugin aliases resolved inside model constructors. Sequential runs and
spawned training processes have independent provenance; successful cache hits still count for the
current run. `--no-imports` validates the shape but deliberately does not require the external
distribution to be installed. See the [plugin guide](../plugins/README.md#loaded-plugin-provenance)
for the exact artifact contract, publishing groups, CLI discovery, cache behaviour, conflict rules
and trusted-code boundary.

## Extension boundaries

- Configure a custom `data.datamodule.target`, `task.target` or `runner.target` rather than forking
  the experiment engine.
- Models, losses and `train_metrics`, `val_metrics` and `test_metrics` accept either
  installed-plugin or fully qualified `target` specifications; nested `ref` values and object
  specifications are built recursively. The backward-compatible `metrics` key supplies all
  unspecified stages.
- Data splits, top-level `callbacks` and trainer loggers accept reusable installed plugins or local
  `target` objects. Logger lists and dataset `ref` objects remain supported; collators and other
  nested objects use the same recursive syntax.
- Checkpoint and early-stopping monitors and their `min`/`max` modes are explicit trainer settings;
  when omitted, the first validation metric and its declared direction are used.
- A custom runner must preserve compatible `fit` and `test` methods if it is used by
  `ExperimentRunner`.
- Treat YAML as trusted code because imported targets and resolved plugins can execute arbitrary
  Python.
- Plugin provenance covers the run process. Resolutions made only inside user-created child
  processes require an explicit user IPC integration if they must be attributed to the parent.
- Import public classes from `lambdaforge.experiments`; file names are implementation details.
- Add incompatible future configuration changes as packaged Schemas plus consecutive
  `ExperimentConfigMigration` objects; do not infer versions from document shape.
- Keep artifact cleanup behind `ArtifactRetentionPolicy`/`ArtifactRetentionManager` so receipt
  validation, path guards, lock ordering and recovery remain intact; do not delete run outputs from
  a custom aggregator.

The lifecycle classes remain together deliberately: unlike metrics and callbacks, their contracts
are tightly coupled and splitting them would create several tiny packages without an independent
public purpose. Revisit that boundary when a genuinely separate backend or storage family appears.
