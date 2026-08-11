[English](PREPROCESSING.md) | [Español](PREPROCESSING.es.md)

# Preprocessing execution and debugging

## 1. Small path first

```yaml
name: prepare
inputs: {raw: data/raw.jsonl}
outputs: {processed: processed}
preprocess:
  function: my_project.preprocessing.normalize_record
  key_field: id
  workers: 4
  workload: io
```

Run `validate`, `inspect`, `run --dry-run`, then `run`. The concise form compiles to the same strict
source/transform/sink task described in the package preprocessing README. Inputs are content-hashed;
changed bytes create a new task identity. Stable keys drive deterministic shards, manifests and
resume. A successful record is reused only if the sink verifies its bytes.

## 2. Workload semantics

- `workers: 1`: sequential reference behavior for every workload.
- `workload: io`: a bounded thread pool; transforms and sink writes share the process.
- `workload: cpu`: a spawn process pool runs only picklable transforms; the parent writes the sink,
  manifest and dataset artifact. User objects must be importable and spawn-serializable.
- `workload: auto`: conservative threads, avoiding surprise process serialization.
- `workload: gpu`: exactly one worker. Use explicit shards/jobs/resources for multiple GPUs.

`workers`, `workload` and checkpoint cadence are execution policy, so the built-in preprocessing
fingerprint excludes them. Sequential/thread/process modes must yield the same content hashes and
`DatasetArtifact` identity for the same inputs and transforms.

The parent checkpoints only completed futures. FAIL cancels pending work and persists failures;
SKIP records the error and continues. The same stable keys/shard hash work on Linux and Windows.

## 3. Sample debugging

```bash
lambdaforge debug preprocessing.yaml --records 3 --json
lambdaforge debug preprocessing.yaml --records 3 --intermediates debug/stages
```

Debug constructs the configured source/transforms, reads at most N deterministic source records and
reports source key/type, every transform/type/preview/duration and full exception evidence. It does
not call the production sink, finalize a dataset or create the normal output root. Optional
intermediate JSON belongs to the requested debug directory and the `debug:` identity can never be
reused as a complete `DatasetArtifact`.

## 4. Dataset inspection

Register the result manifest in a `DataCatalog`, then use
`lambdaforge data --catalog data-catalog.yaml inspect dataset:processed-v3`. The report includes
logical identity, physical locations/reachability/size, dataset ID, producer/config identity,
sample/split counts, artifact declarations and validation state.
