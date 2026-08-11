# LambdaForge preprocessing

[Repository guide](../../../README.md) · [Español](README.es.md) · [Generic tasks](../tasks/README.md)

Preprocessing is implemented as a composition of generic public contracts rather than a
project-specific special case:

```text
PreprocessingSource → PreprocessingTransform(s) → PreprocessingSink
                              │
                              └→ PreprocessingTask → TaskResult + DatasetArtifact
```

The complete runnable example is [examples/preprocessing.yaml](../../../examples/preprocessing.yaml).
Validate and inspect it before execution:

```bash
lambdaforge validate examples/preprocessing.yaml
lambdaforge inspect examples/preprocessing.yaml
lambdaforge run examples/preprocessing.yaml --dry-run
lambdaforge run examples/preprocessing.yaml
```

## Start here

Think of preprocessing as a conveyor belt. The **source** decides which records exist and gives each
one a stable key. Each **transform** receives one record and returns its changed value. The **sink**
writes the final record and can later prove whether that key is already complete. `PreprocessingTask`
coordinates the belt, checkpoints progress and produces a `DatasetArtifact` manifest describing the
whole output dataset.

For ordinary JSONL work, start with the concise form and name every path once:

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
```

`workers: 1` is deterministic sequential processing. Larger values use a bounded thread pool;
`workload` records `auto`, `io`, `cpu` or `gpu` intent but does not silently choose processes,
devices or batch sizes. Threads suit waiting-heavy transforms. For CPU isolation, true parallel
Python compute or GPU batching, use explicit workflow shards or a project task with domain-specific
batching. Record manifests, atomic sink verification and error policy remain identical.
Custom transforms and sinks used with multiple workers must therefore be thread-safe.

The checked-in example is directly runnable with its bundled input and an empty transform list; it
needs no consumer package. The configuration shown below then demonstrates where an optional
project transform belongs. As with every LambdaForge configuration, `validate` checks it, `inspect`
explains the plan, `run --dry-run` exercises planning and plain `run` executes it. Start with one
transform and one process; introduce shards only after stable record keys and restart behaviour
have been verified.

## Built-in pipeline

`JsonLinesSource` reads one JSON value per non-empty line and uses either a configured mapping field
or the stable line number as its key. `FileTreeSource` yields sorted regular files below a root with
relative keys and refuses symlinks. `JsonDirectorySink` writes one atomic JSON envelope per record;
its SHA-256 filename prevents unsafe keys from becoming paths. `CallableTransform` explicitly wraps
a YAML `ref` callable that transforms only the record value.

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
    transforms:
      - target: lambdaforge.preprocessing.CallableTransform
        params:
          function: {ref: my_project.preprocessing.normalize_record}
    sink:
      target: lambdaforge.preprocessing.JsonDirectorySink
      params: {output_dir: processed}
    on_error: fail
    checkpoint_interval: 1
    dataset_name: normalized-records
    dataset_version: "1"
```

`target` constructs classes and `ref` imports the function unchanged. Project logic stays in the
installed consumer package. For richer objects, implement the three small contracts directly:

```python
from collections.abc import Iterable
from lambdaforge.preprocessing import (
    PreprocessingRecord,
    PreprocessingSource,
    PreprocessingTransform,
    PreprocessingSink,
)


class ProjectSource(PreprocessingSource):
    def records(self, context) -> Iterable[PreprocessingRecord]:
        yield PreprocessingRecord(key="stable-id", value={"raw": 1})


class ProjectTransform(PreprocessingTransform):
    def transform(self, record, context) -> PreprocessingRecord:
        return record.with_value({"feature": record.value["raw"] * 2})


class ProjectSink(PreprocessingSink):
    def write(self, record, context) -> None: ...
```

A sink should override `is_complete(key, context)` when it can verify per-record output and return
aggregate `ArtifactDeclaration` objects from `finalize(context)`. This lets resume skip only outputs
that still exist.

## Resume, failures and shards

The pipeline writes `preprocessing-manifest.json` atomically after
`checkpoint_interval` processed/failed records (default 1). Each entry has its stable key, latest
status, UTC update and structured error. A retry with the same task and input fingerprint reuses
successful entries only when the sink confirms their output. Failed entries run again.

`on_error: fail` aborts at the first failed record after checkpointing it. `skip` records failures
and permits the task to finish; the resulting metrics make the partial count explicit. LambdaForge
never hides a record failure or silently changes the input.

`shard_count: N` and `shard_index: i` assign each key using a stable SHA-256 modulo. Shards are
deterministic, disjoint and cover the source. Run each shard as an explicit task configuration with
its own name/output; this release does not launch or merge shards automatically. General local/HPC
workflow scheduling remains a subsequent roadmap item.

## DatasetArtifact

Every successful `PreprocessingTask` writes `dataset-artifact.json`. It records:

- a content-derived `dataset_id`;
- human dataset name/version;
- sample and optional split counts;
- preprocessing/task fingerprint;
- source/input descriptors;
- SHA-256 and size of every sink artifact;
- creation time, LambdaForge version, environment-manifest link and user metadata.

The ID excludes creation time and absolute execution location. Equal declared science and equal
artifact bytes produce the same identity. Split counts cannot be negative or exceed total samples.
The manifest links to `environment.json`, which owns Git, Python, package, CUDA and plugin
provenance.

## Boundaries

- Schema 1.0 preprocessing is sequential inside one task. Use explicit shards for independent
  parallel work; CPU process pools and DAG scheduling are intentionally not hidden in this release.
- Local inputs and outputs are supported. Shared/remote artifact stores need future lease and
  atomicity contracts.
- Arbitrary Python targets/refs are trusted code. JSON output validation is not sandboxing.
- The framework cannot infer the scientific meaning of an undeclared external input. Built-in
  `PreprocessingTask` therefore requires at least one top-level content-hashed `inputs` entry, and
  built-in JSONL/file-tree source paths must match it or lie below a declared input directory.
