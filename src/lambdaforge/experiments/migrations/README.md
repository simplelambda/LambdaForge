# Experiment configuration migrations

[Experiment guide](../README.md) · [Repository guide](../../../../README.md) ·
[Español](README.es.md)

This package owns version detection, deterministic forward migrations, exact JSON Schema selection
and non-destructive previews for LambdaForge experiment YAML. Migration does not construct
configured objects, resolve plugins, import user `target`/`ref` paths, start processes or create run
directories.

## Contents

- [Start here](#start-here)
- [Compatibility contract](#compatibility-contract)
- [Safe CLI workflow](#safe-cli-workflow)
- [Preview formats and exit codes](#preview-formats-and-exit-codes)
- [Python API](#python-api)
- [Object model](#object-model)
- [Round-trip and persistence guarantees](#round-trip-and-persistence-guarantees)
- [Validation and failure modes](#validation-and-failure-modes)
- [Adding a future migration](#adding-a-future-migration)
- [Current scope](#current-scope)

## Start here

The package version (`0.5.1`) and a YAML `schema_version` describe different things. The package
version identifies the installed LambdaForge release. The Schema version identifies the shape of a
configuration document. Upgrade a document only when `validate` reports an old/unsupported shape
or when reviewing a repository-wide framework upgrade.

`lambdaforge migrate old.yaml` is a read-only preview. It never trains and does not overwrite the
source. Persist the proposed YAML to a different file with `--output`, review the diff, validate the
new file and only then replace project references if desired.

## Compatibility contract

The current experiment Schema version is the quoted string `"1.1"`:

```yaml
schema_version: "1.1"
experiment:
  name: study
# ...
```

`schema_version` is required by both packaged JSON Schemas, historical 1.0 and current 1.1. It must
use the exact quoted `MAJOR.MINOR` form; an unquoted YAML value such as `1.1` is a number and is
rejected rather than silently coerced.

Historical LambdaForge configurations did not contain this field. They remain supported as legacy
`unversioned` input through the built-in `UnversionedToV1Migration`. That migration inserts
`schema_version: "1.0"` first and does not change experiment semantics. The consecutive
`ExperimentV1ToV1_1Migration` then changes only that declaration to 1.1. Schema 1.1 adds the
optional strict `retention` block; omission means disabled, so historical experiments do not opt
into artifact mutation. Every newly authored file should declare 1.1 explicitly.

`ExperimentConfig` normalises legacy input in memory at configuration boundaries, including
`Experiment.from_yaml`, runner/aggregator mappings and materialised-run loading. The original source
file is not modified. `ExperimentValidator` reports the source and target versions plus the applied
steps, validates each intermediate result against its exact Schema and finally validates the
normalised mapping against Schema 1.1. Use the explicit migration command to review and persist the
canonical YAML.

The default registry contains one deterministic consecutive path:

```text
unversioned --unversioned_to_1_0--> 1.0 --1_0_to_1_1--> 1.1
```

Downgrades, guessed versions and jumps without a registered consecutive path are rejected.
See the [artifact-retention guide](../retention/README.md) for the Schema 1.1 block's runtime
semantics; migration only declares compatibility and never applies retention.

## Safe CLI workflow

Start with the default unified diff:

```powershell
lambdaforge migrate legacy.yaml
```

Preview the complete migrated document or a machine-readable result:

```powershell
lambdaforge migrate legacy.yaml --format yaml
lambdaforge migrate legacy.yaml --format json
lambdaforge migrate legacy.yaml --target-version 1.1  # current; also the default
lambdaforge migrate legacy.yaml --target-version 1.0  # stop at the historical Schema
```

Nothing is written unless `--output` names a different path:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
```

An existing destination is refused. `--force` permits replacing that destination, but never the
source:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml --force
```

`--force` without `--output` is invalid. `--check` is intended for CI and cannot be combined with
`--output`:

```powershell
lambdaforge migrate experiment.yaml --check
```

The selected preview is still printed when `--output` or `--check` is used. `--format` controls that
standard output only; the file named by `--output` is always the complete migrated YAML. A
successful write confirmation and errors go to standard error, so JSON standard output remains
machine-readable.

## Preview formats and exit codes

| Format | Output |
|---|---|
| `diff` | Unified diff from the source to the target version; default. A current no-op prints a short status line. |
| `yaml` | Complete resulting YAML, including an unchanged current document. |
| `json` | Stable result envelope with versions, `changed`, steps, warnings, diff and resulting configuration. |

The JSON envelope has its own `migration_result_version`. This versions the result protocol and is
independent of the experiment `schema_version`. File-backed YAML already consists of portable
values. For a programmatic mapping, the semantic `result.config` preserves Python values such as
tuples, paths and dates, while the JSON envelope projects them deterministically to JSON arrays,
native path strings and ISO strings; unsupported opaque objects are represented by a stable type
label. This projection never changes the configuration used by the framework.

| Condition | Exit code |
|---|---:|
| Valid preview, or successful explicit output write | `0` |
| `--check` and no migration is required | `0` |
| `--check` and at least one migration step is required | `1` |
| Invalid YAML/Schema/version/path, unsafe output request or write failure | `1` |
| Command-line syntax rejected by `argparse` | `2` |

Because `--check` deliberately uses `1` to signal stale configuration, distinguish that expected
condition from an error by first running the normal preview when diagnosis is needed.

## Python API

The facade is the shortest read-only entry point:

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.source_version)  # unversioned
print(preview.target_version)  # 1.1
print(preview.changed)
print(preview.render(MigrationPreviewFormat.DIFF))
```

The lower-level public objects support file and mapping previews:

```python
from lambdaforge.experiments import ExperimentConfigMigrator

migrator = ExperimentConfigMigrator.default()
file_result = migrator.preview_file("legacy.yaml")
mapping_result = migrator.preview_mapping(raw_config)

payload = file_result.to_dict()
file_result.write_yaml("experiment.v1_1.yaml")
```

`preview_mapping` deep-copies its input, preserves its programmatic Python value types and never
mutates the caller's nested objects. The result's mapping-shaped JSON payload is deeply read-only;
`result.config` and `to_dict()` return independent defensive copies.
`write_yaml(path, overwrite=False)` is an explicit second operation; previewing alone has no
persistence side effect. Use `overwrite=True` only for a distinct, intentionally replaceable
destination.

The main migration abstractions are lazily exported from `lambdaforge.experiments`. Contributor-level
objects, including the built-in step and round-trip codec, are also exported from
`lambdaforge.experiments.migrations`. Class modules are implementation details.

## Object model

| Object | Responsibility |
|---|---|
| `ExperimentSchemaVersion` | Exact, ordered `MAJOR.MINOR` value object plus the internal `unversioned` marker. |
| `ExperimentSchemaCatalog` | Maps supported exact versions to packaged Draft 2020-12 Schemas, caches validators per object and checks declaration drift. |
| `ExperimentConfigMigration` | Abstract object contract for one exact forward transformation. |
| `UnversionedToV1Migration` | Compatibility-only built-in step that declares Schema 1.0. |
| `ExperimentV1ToV1_1Migration` | Consecutive built-in step that declares current Schema 1.1 while preserving every 1.0 field's semantics. |
| `ExperimentConfigMigrationStep` | Immutable identifier/version/description descriptor recorded in results. |
| `ExperimentConfigMigrationRegistry` | Immutable, deterministic, forward-only path planner. |
| `ExperimentConfigMigrator` | Copies, plans, applies, validates and renders a migration chain. |
| `ExperimentConfigMigrationResult` | Immutable mapping-compatible preview, renderers and explicit atomic YAML writer. |
| `MigrationPreviewFormat` | Enum for `diff`, `yaml` and `json`. |
| `RoundTripYamlCodec` | UTF-8, duplicate-key-rejecting, presentation-aware YAML codec. |

The registry rejects duplicate identifiers, more than one outgoing migration from the same version
and non-forward steps. `with_migration(...)` returns a new registry instead of mutating global
state.

## Round-trip and persistence guarantees

File previews use round-trip YAML handling to preserve comments, mapping order, quoted scalar style,
anchors and the dominant newline convention where the transformed structure permits it. A no-op
preview returns the exact original text. A changed document may still receive harmless presentation
normalisation, which is exposed in the result warnings; always inspect the diff.

The reader accepts exactly one UTF-8 mapping document and rejects duplicate keys. It does not invoke
LambdaForge's object factory, plugin registry or import validation, so a migration preview cannot
instantiate configured models, losses, datasets, callbacks or loggers. It is not a general sandbox:
the resulting text and JSON envelope can contain credentials or other values already present in the
configuration, and LambdaForge does not redact them.

Persistence is intentionally separate from migration planning:

1. The source and destination resolve to different paths, even with overwrite enabled.
2. Existing destinations require explicit `overwrite=True`/`--force`.
3. The destination parent is created only when a write was explicitly requested.
4. YAML is written and flushed to a unique temporary file beside the destination.
5. Without overwrite permission, the completed temporary is atomically linked into an absent
   destination, so concurrent writers cannot clobber one another. With explicit overwrite
   permission, atomic replacement is used instead.
6. Temporary residue is removed after a handled success or exception.

There is no in-place mode and no automatic source backup because the source is never a write target.
An abrupt process or machine termination can still leave the uniquely named temporary file, which
is never mistaken for the destination.

## Validation and failure modes

With the default `validate=True`, the migrator validates every applied step against the exact target
Schema. A current no-op is also validated. The `unversioned` marker has no standalone Schema:
legacy input is transformed and validated first as 1.0, then transformed and validated as 1.1.

Migration fails before persistence when:

- the YAML root is not a mapping, has duplicate keys or cannot be parsed;
- `schema_version` is not a quoted exact `MAJOR.MINOR` string;
- the requested target has no packaged Schema;
- the registry has no complete forward path or a downgrade is requested;
- a step emits the wrong version, a non-mapping or an invalid target document;
- the output is the source, already exists without overwrite permission or cannot be replaced.

Migration validation intentionally does not inspect `target`/`ref` imports or plugin availability.
Run `lambdaforge validate migrated.yaml` afterwards for expansion, resource and optional import
checks:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
lambdaforge validate experiment.v1_1.yaml
```

## Adding a future migration

A future incompatible experiment Schema change should be introduced as a reviewed release chain,
not inferred from document contents:

1. Package the new Draft 2020-12 Schema and declare its exact `schema_version`.
2. Add it to `ExperimentSchemaCatalog.DEFAULT_SCHEMA_FILES`.
3. Implement one `ExperimentConfigMigration` class with a stable identifier, exact source/target
   versions, a user-facing description and a deterministic `apply` method.
4. Add that object to the default immutable registry in consecutive version order.
5. Test no-op behaviour, the full chain, invalid intermediate output, round-trip presentation, CLI
   previews and atomic persistence.
6. Document semantic changes in both language guides before making the new version current.

Each step is checked immediately against its target Schema. The registry deliberately permits only
one outgoing step per source version, so future evolution remains a deterministic linear history
unless that contract is consciously redesigned.

## Current scope

Schemas 1.0 and 1.1 are both packaged, and the current real chain is
`unversioned → 1.0 → 1.1`. The second step introduces the optional Schema surface for artifact
retention without enabling it. No Schema downgrade, in-place rewrite, remote configuration source,
secret redaction or plugin-provided migration is implemented. These limits keep compatibility
changes reviewable and persistence local while future framework releases add concrete consecutive
steps only when the Schema actually changes.
