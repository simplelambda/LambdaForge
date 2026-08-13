# First-class datasets

[Español](DATASETS.es.md) | English | [Root guide](../README.md)

## 1. Entity and source of truth

A dataset record names one immutable version (`name@version`) and its content-derived
`dataset_id`. It contains sample/split counts, producer, lineage and zero or more physical
placements. A placement contains cluster, exact root, size, file count and verification status.
The `DatasetArtifact` beside the bytes remains authoritative; `.lambdaforge/datasets.json` is a
small atomic index that can be reconciled from cluster inventories.

Preprocessing that finishes successfully with `dataset-artifact.json` auto-registers it. This is
auxiliary: a registry write failure is an event and never rewrites a valid scientific manifest.
Existing `DataCatalog` YAML remains supported for declarative logical references, loaders and
environment-specific locations, but it is no longer required for list/show/stats/verify/lifecycle.

## 2. Discovery and inspection

```bash
lambdaforge datasets list
lambdaforge datasets list --on atlas
lambdaforge datasets list --all
lambdaforge datasets show corpus@v3
lambdaforge datasets locations corpus@v3
lambdaforge datasets stats corpus@v3 --on atlas
lambdaforge datasets verify corpus@v3 --on atlas
lambdaforge datasets lineage corpus@v3
```

Remote inventory runs a bounded command against the configured registry path; it never scans a
whole filesystem. `stats` always reports bytes, file count, manifest samples/splits and a simple
format summary. It does not guess labels, targets or domain meaning.

For classification, provide an explicit YAML schema, for example:

```yaml
task: classification
format: csv
file: examples.csv
target: label
classes: [cat, dog]
```

Then run `datasets stats corpus@v3 --schema classification.yaml`. The built-in profiler reports
class counts/proportions, missing targets and imbalance ratio. A project-specific schema can use
`profiler: {target: my_project.data.WisdomProfiler}`; that object implements
`profile(root, record, schema) -> mapping`. No WISDOM-specific assumptions enter the core.

## 3. Registration, remove and physical delete

```bash
lambdaforge datasets add RUN/dataset-artifact.json --root RUN
lambdaforge datasets remove corpus@v3 [--on atlas]
lambdaforge datasets delete corpus@v3 --on atlas       # preview
lambdaforge datasets delete corpus@v3 --on atlas --apply
```

`remove` changes only registration. `delete` targets one exact placement and defaults to preview.
Apply requires a matching valid manifest and artifact hashes, refuses broad/home/root paths and
refuses a declared active consumer. It then removes the placement registration. Dataset deletion
is separate from storage GC; GC can never select dataset roots.

## 4. Materialization and replication

```bash
lambdaforge datasets materialize corpus@v3 --on atlas
lambdaforge datasets materialize corpus@v3 --on atlas --strategy replicate --apply
lambdaforge datasets replicate corpus@v3 --from local --to atlas
lambdaforge datasets replicate corpus@v3 --from local --to atlas --apply
```

The deterministic plan is `NOOP` when present, `REPLICATE` when an immutable source placement
exists, or `BUILD` when a producer is registered. Plans show estimated bytes and whether the local
controller must remain online. Large data is never copied by `run --on` or by a preview.

The built-in transfer path supports a local source to local/SSH destination with explicit `rsync`.
It intentionally refuses to pretend that controller-relayed cluster-to-cluster transfer survives a
laptop outage. Shared filesystems and site transfer systems should implement a provider/durable
transfer task; BUILD plans tell the user to run the producer explicitly after its own inputs exist.

## 5. Python contracts

Public objects are `DatasetRecord`, `DatasetPlacement`, `DatasetRegistry`, `DatasetService`,
`DatasetMaterializationPlan`, `DatasetDeletionPlan` and `DatasetProfiler` under `lambdaforge.data`.
Use `DatasetService`, not registry-file globs. A profiler receives an exact registered root and
record and must not silently infer scientific semantics.
