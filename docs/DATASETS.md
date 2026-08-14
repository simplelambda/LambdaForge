# Dataset lifecycle

[Español](DATASETS.es.md) | English | [Root guide](../README.md)

## 1. Mental model

LambdaForge 0.7 keeps four concepts separate:

```text
DatasetRecipe (how) -> DatasetBuild (execution) -> DatasetVersion (what)
                                                   -> DatasetPlacement (where)
```

A Task produces ordinary artifacts. Preprocessing publishes no dataset unless it receives
`publish_dataset: true` or the legacy `dataset_name`; a DatasetRecipe is the preferred explicit
publication boundary. A DatasetVersion is an immutable logical collection, not a directory or a
PyTorch `Dataset`. One version may have several placements without changing identity.

## 2. Logical content and identity

`DatasetIndex` streams canonical JSONL and yields `DatasetMember` objects. Every member has a stable
`id`, arbitrary `partitions`, arbitrary `targets`, scientific `metadata`, non-identity `display`
metadata and named `DatasetAsset` values. Assets may be files, multiple files, directories, records
inside containers or provider URIs; no per-sample directory layout is imposed.

```python
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember

members = [DatasetMember(
    "sample-001",
    partitions={"split": "train", "fold": 3, "cohort": "external"},
    targets={"class": 1, "affinity": 4.27},
    metadata={"source": "instrument-a"},
    display={"description": "Descriptive text; not scientific identity"},
    assets={
        "features": DatasetAsset("samples/001/features.npz", sha256="sha256:<real hash>"),
        "auxiliary": DatasetAsset("samples/001/aux", kind="directory"),
    },
)]
DatasetIndex.write("dataset/members.jsonl", members)
```

Use a real 64-character checksum. Counts and summaries derive from the index. DatasetArtifact v2
stores `content_id` (`dataset_id`), `build_id`, index identity, member count, partitions, optional
target schema, global assets, lineage and producer. Content identity uses member IDs and scientific
fields plus asset/global checksums—not physical paths, display text, cluster or provenance. Moving
the same bytes keeps `content_id`; another build of those bytes has another `build_id`. V1 remains
readable.

## 3. Write and build a recipe

Start with [the generic recipe](../examples/dataset-recipe.yaml). Each stage is an existing
`kind: task`; `needs` and `bindings` reuse the Workflow DAG. `required` asks whether publication is
scientifically valid without the stage. Independently, `reuse: auto|never` controls reuse of a
matching content-addressed Task result.

```yaml
kind: dataset
dataset:
  name: example-records
  version: "1"
  target_schema:
    type: object
    properties: {category: {type: string}}
    required: [category]
stages:
  discover: {task: tasks/discover.yaml, required: true, reuse: auto}
  normalize:
    task: tasks/normalize.yaml
    needs: [discover]
    bindings:
      task.params.roster: ${nodes.discover.artifacts.roster.jsonl}
publish: {from: normalize, root: dataset, index: members.jsonl}
```

```bash
lf validate dataset.yaml
lf datasets plan example-records --on atlas
lf datasets build example-records --on atlas
lf jobs show latest
```

Plan reports stage actions and `PUBLISH`/`NOOP`; add `--verbose` for each reason. A local plan can
prove `REUSE` from its exact cache. A controller-side remote plan reports unobserved stages as
`MISSING` instead of pretending the local cache is remote; the durable target worker authoritatively
rechecks exact Task fingerprints before it executes anything. `--force` forces everything;
`--force-stage NAME` also invalidates its downstream dependants. Builds are durable jobs. Completed
stage Tasks retain their fingerprint, result and integrity receipts in reconstructible stage cache,
so a later failure does not discard them. `storage gc` may collect unreferenced/stale stage cache,
but never published datasets, results or active job workspaces.

Publish validates unique IDs, safe/checksummed assets, index integrity, declared JSON target schema
and required stages in staging. It writes and verifies `dataset-artifact.json`, atomically renames
where supported, then updates Registry. A failed build publishes nothing. An existing
`name@version` with another content ID is rejected and published bytes are never overwritten.

## 4. Resolve and consume

For managed versions, DatasetRegistry is the operational placement authority. DatasetResolver pins
an exact version/content ID and selects the execution placement. DataCatalog remains for aliases,
external data, loader specs, explicit pins and institutional overrides; 0.6 catalogs still work,
but managed paths need not be duplicated.

```yaml
data:
  train: dataset:example-records@1/train
  val:
    target: my_project.data.MyDataset
    params:
      root: {dataset: example-records, version: "1", subpath: validation}
```

A direct split needs a `loader` ObjectSpec and `path_parameter` in Registry metadata or DataCatalog.
A nested marker injects its path into project params. Materialized evidence records name, version,
content/build identity and placement; scientific fingerprints use content identity, never path.
Unversioned references fail as ambiguous when several versions match.

## 5. Inspect, compare and profile

```bash
lf datasets ls --all
lf datasets show example-records@1
lf datasets stats example-records@1 --on atlas
lf datasets members example-records@1 --partition split=train --limit 50
lf datasets member example-records@1 sample-001
lf datasets diff example-records@1 example-records@2
lf datasets verify example-records@1 --on atlas
lf datasets lineage example-records@1
```

Inventory reads small registries and never scans filesystems. Member listing defaults to 100 and
supports offset. Diff reports added/removed/changed IDs plus partition, target and asset-identity
changes. Universal stats cover members, partitions, size, files, asset types and missing checksums.
Target meaning is used only with an explicit schema. A project profiler selected by `--schema`
runs beside a remote placement in the exact managed consumer environment; data is not downloaded.
Important read-only commands accept stable `--json`; human output is the default.

## 6. Materialize, replicate and remove

```bash
lf datasets materialize example-records@1 --on atlas              # preview
lf datasets materialize example-records@1 --on atlas --apply
lf datasets replicate example-records@1 --from local --to atlas   # preview
lf datasets replicate example-records@1 --from local --to atlas --apply
lf datasets remove example-records@1 --on atlas
lf datasets delete example-records@1 --on atlas                    # preview
lf datasets delete example-records@1 --on atlas --apply
```

Materialization returns `NOOP`, `REPLICATE` or multi-stage `BUILD`. Applying BUILD materializes
supported prerequisites and submits one durable `dataset-build` job—no manual producer-command
handoff. The built-in relay starts from a local source, uses verified staging/atomic publication and
requires the controller online; it does not pretend to be durable cluster-to-cluster transfer.
Shared/site providers remain extensions. `remove` only changes registration. `delete` requires an
exact verified placement, no active consumer and `--apply`. Normal GC cannot select versions.

## 7. CLI, API and migration

`lf` and `lambdaforge` are the same entry point. Grammar is
`lf <resource> <action> <object> [--on CONTEXT]`; concise aliases are `ds`, `exp`, `env` and `ls`.
`lf plan CONFIG --on CLUSTER` is the root dry-run shortcut. Use
`lf completion bash|zsh|fish` for dependency-free completion.

Import `DatasetAsset`, `DatasetMember`, `DatasetIndex`, DatasetRecipe/build plan/result,
DatasetRecord/Placement/Registry, DatasetResolver/Resolution, DatasetService, typed errors and
lifecycle plans from `lambdaforge.data`, never private modules.

Compatibility: Artifact/Record v1 and DataCatalog 0.6 remain readable. Legacy `dataset_name`
explicitly publishes an artifacts-only v2 manifest. Without it or `publish_dataset`, preprocessing
is only a Task. Legacy BUILD producers remain previewable; automatic apply needs a discoverable
`kind: dataset` recipe. Stage artifacts never appear in dataset listings unless deliberately
published.
