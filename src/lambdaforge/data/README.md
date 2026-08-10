# LambdaForge data and dataset cache

[Repository guide](../../../README.md) · [Español](README.es.md)

The <code>lambdaforge.data</code> package provides task-agnostic, map-style
PyTorch dataset adapters and an optional bounded cache. Caching is never
implicit: a project chooses which deterministic stage to wrap, how much
serialized data each process may retain, and whether a persistent backend is
appropriate.

## Contents

- [Start here](#start-here)
- [Object map](#object-map)
- [DatasetCache](#datasetcache)
- [RAM quota semantics](#ram-quota-semantics)
- [Persistent backends](#persistent-backends)
- [Fingerprints, integrity and safe serialization](#fingerprints-integrity-and-safe-serialization)
- [Multiprocess coordination and recovery](#multiprocess-coordination-and-recovery)
- [DataLoader workers and capacity planning](#dataloader-workers-and-capacity-planning)
- [FileDataset](#filedataset)
- [NumpyMemmapDataset](#numpymemmapdataset)
- [Dataset plugins](#dataset-plugins)
- [Complete YAML example](#complete-yaml-example)
- [Statistics](#statistics)
- [Invalidation and lifecycle](#invalidation-and-lifecycle)
- [Determinism and cache keys](#determinism-and-cache-keys)
- [Security and extension contracts](#security-and-extension-contracts)

## Start here

A **dataset** is the object that answers “how many samples exist?” and “give me sample number
`i`”. LambdaForge follows PyTorch's map-style `Dataset` contract and does not prescribe what a
sample means. A **cache** is an optional wrapper around such a dataset: after a deterministic sample
has been loaded once, serialized bytes may be reused instead of repeating expensive I/O or parsing.

Begin with an ordinary dataset and `num_workers: 0`. Add `DatasetCache` only after measuring that
sample loading is a bottleneck and identifying a deterministic stage safe to reuse. RAM cache is
private to each process; disk/mmap backends coordinate a shared byte quota. Neither replaces the
operating-system page cache, preprocessing artifacts or correct DataLoader sizing.

## Object map

| Object | Responsibility |
|---|---|
| <code>DatasetCache</code> | Wrap a map-style dataset with a process-local RAM LRU and an optional byte backend. |
| <code>CacheStats</code> | Immutable process-local statistics and current backend usage snapshot. |
| <code>CacheUsage</code> | Atomic entry/byte snapshot for one coordinated backend namespace. |
| <code>DatasetFingerprint</code> | Canonical content, deterministic-transform and configuration identity. |
| <code>DiskCacheBackend</code> | Store verified records atomically under one multiprocess disk quota. |
| <code>MemoryMappedCacheBackend</code> | Hold a shared filesystem lease while reading a verified mmap record. |
| <code>CacheRecordCodec</code> | Versioned SHA-256 checksum or HMAC-SHA256 record envelope. |
| <code>CacheIntegrityMode</code> | Closed checksum/authentication choice; avoids magic strings in Python. |
| <code>CacheNamespaceManifest</code> | Immutable shared quota and record-format contract. |
| <code>DatasetSerializer</code> | Abstract conversion contract between keys/samples and bytes. |
| <code>NumpyDatasetSerializer</code> | Deterministic, bounded NumPy/Torch tree codec with no pickle. |
| <code>PickleDatasetSerializer</code> | Compatibility codec for explicitly trusted local data only. |
| <code>CacheBackend</code> | Abstract byte-store contract for project backends. |
| <code>CacheRecord</code> | Own a backend payload and its idempotent close callback. |
| <code>FileDataset</code> | Lazily load an explicit ordered list of files with a project callable. |
| <code>NumpyMemmapDataset</code> | Read aligned <code>.npy</code> arrays without loading the complete arrays into RAM. |

All of these names are public from <code>lambdaforge.data</code>. The
<code>lambdaforge.training.data</code> package has a different responsibility:
it contains the Lightning data-module and guarded DataLoader worker adapter.

## DatasetCache

The constructor is:

~~~python
DatasetCache(
    dataset,
    max_memory_bytes_per_process,
    max_memory_entries=10_000,
    backend=None,
    serializer=None,
    key_fn=None,
    cache_in_workers=False,
    strict=False,
    fingerprint=None,
)
~~~

An access follows this order:

1. Hash the versioned result of <code>key_fn(index)</code>, or the index itself.
2. Look in the process-local RAM LRU when RAM caching is enabled.
3. Look in the optional backend.
4. Load the wrapped dataset on a miss, serialize the result, and admit it to
   every configured layer whose quota permits it.

RAM and backend hits are deserialized into new objects. This isolates mutable
lists, mappings, NumPy arrays and CPU tensors from later callers. The first
miss is serialized before the original dataset result is returned, so a caller
mutating that result cannot change the stored bytes.

The default remains <code>PickleDatasetSerializer</code> for compatibility.
Persistent research caches should select <code>NumpyDatasetSerializer</code>
explicitly whenever their sample tree is supported. The serializer must encode
both the versioned key tuple and every sample. Nested CUDA tensors are
deliberately not cached: datasets should return CPU samples and let the
training/device-transfer layer own accelerator memory.

With <code>strict=False</code>, serialization and backend failures are counted
and the wrapped dataset remains usable. With <code>strict=True</code>, these
failures become descriptive <code>RuntimeError</code> exceptions. The
<code>cache_in_workers</code> and <code>strict</code> parameters require real
boolean values.

## RAM quota semantics

Two independent limits bound the process-local LRU:

- <code>max_memory_bytes_per_process</code> is the sum of immutable serialized
  payload lengths retained by one process replica. Zero disables the RAM
  layer while still allowing a backend.
- <code>max_memory_entries</code> prevents millions of tiny records from
  exhausting memory through mapping/key overhead. It must be positive.

The least recently used records are evicted until both constraints are
satisfied. Reading a record refreshes its position. A payload larger than the
whole byte budget is skipped without flushing useful smaller records.

The byte limit is exact for retained serialized payloads, but it is **not a
hard RSS limit**. It excludes:

- SHA-256 strings, the ordered mapping, locks and Python allocator overhead;
- the wrapped dataset and any data it already owns;
- temporary serialization/deserialization buffers;
- the live sample returned to the caller;
- DataLoader prefetch queues, collated batches and pinned memory;
- model, optimizer and metric state;
- filesystem and memory-map page caches.

Choose a budget with headroom. A process may not immediately return freed
allocator pages to the operating system even after an LRU eviction.

## Persistent backends

Both built-in disk backends use this constructor:

~~~python
DiskCacheBackend(
    root,
    namespace,
    max_bytes,
    max_entries=100_000,
    record_codec=None,
    lock_timeout_seconds=60.0,
    lock_poll_interval_seconds=0.01,
    remove_invalid_records=True,
)

MemoryMappedCacheBackend(
    root,
    namespace,
    max_bytes,
    max_entries=100_000,
)
~~~

<code>namespace</code> is hashed into an isolated directory. Records use opaque
SHA-256 names. The directory contains a versioned manifest fixing
<code>namespace</code>, <code>max_bytes</code>, <code>max_entries</code> and
the record-codec fingerprint. Opening the same namespace with a different
quota, integrity mode or HMAC key identifier fails before mutating records.
Use a new namespace for a new shared contract.

Writes encode and flush one same-directory temporary, reserve space by evicting
old complete records, then publish with <code>os.replace</code>. Quota
reservation happens before publication; a process killed immediately after
the replace therefore cannot leave completed records over quota. LRU order is
stable by nanosecond mtime and filename. The directory entry is fsynced where
the operating system supports it.

<code>MemoryMappedCacheBackend</code> changes only the read path: it maps one
serialized cache record instead of allocating a separate
<code>read_bytes()</code> result. <code>DatasetCache</code> closes that mapping
immediately after deserialization. The reconstructed sample still consumes its
ordinary Python/Tensor memory.

Disk quotas count each complete <code>.lfcache</code> envelope plus payload,
not only the serialized sample. One in-progress temporary, the manifest, lock
file, directory metadata and filesystem allocation units are outside that
logical quota. Because writers are serialized, at most one cooperating
temporary exists per namespace. A sample whose complete envelope cannot fit is
rejected without evicting useful records.

### Fingerprints, integrity and safe serialization

<code>DatasetFingerprint(content, transform, configuration=None)</code>
canonicalizes a JSON/YAML configuration and combines it with explicit content
and deterministic-transform identifiers. When supplied to
<code>DatasetCache</code>, its digest and the serializer-format fingerprint
enter a versioned cache key. Changing any declared component creates a miss
without deleting the older generation.

Python callers can compute a full ordered content snapshot with:

~~~python
fingerprint = DatasetFingerprint.from_files(
    ["data/features.npy", "data/targets.npy"],
    transform="standardize-v4",
    configuration={"epsilon": 1e-6},
)
~~~

This reads files in bounded chunks. It is a snapshot, not a file watcher, and
it cannot infer the meaning of arbitrary callables. YAML constructs
<code>DatasetFingerprint</code> directly, so a file digest used there must be
precomputed. Omitting <code>fingerprint</code> preserves the legacy key
derivation for compatibility; persistent experiments should make the identity
explicit.

<code>CacheRecordCodec</code> verifies the envelope before any
<code>DatasetSerializer.loads</code> call:

- <code>checksum_sha256</code> is the default and detects accidental
  corruption, but does not authenticate a writer;
- <code>hmac_sha256</code> authenticates with a key of at least 32 bytes by
  default. Prefer <code>authentication_key_env</code> in YAML so secrets are
  not committed.

The signature binds format, mode, namespace, record key, length and payload,
so moving a valid record to a different key or namespace fails. Comparison
uses a constant-time primitive. HMAC provides authenticity and integrity, not
encryption, deletion resistance or protection from replaying an older valid
record.

<code>NumpyDatasetSerializer</code> supports nested string-key mappings,
lists/tuples, JSON-like scalars, bytes, ordinary non-object NumPy arrays and
dense non-quantized CPU tensors. Its deterministic ZIP/NPY format uses
<code>allow_pickle=False</code>. Limits for archive bytes, manifest bytes,
array count, decoded bytes and nesting depth are constructor parameters.
Member sizes and NPY headers are validated before array materialization.
Object/structured arrays, arbitrary classes, CUDA, sparse and quantized
tensors are rejected.

### Multiprocess coordination and recovery

Every built-in persistent backend uses an OS lock: shared for copied reads,
exclusive for mutation, reconciliation and quota reservation. An mmap
<code>CacheRecord</code> retains its shared lease until <code>close()</code>,
preventing another process from evicting an open Windows mapping. Lock
acquisition has explicit timeout and polling parameters, and the OS releases a
lease after normal or abrupt process exit.

Construction and <code>recover()</code> remove orphan <code>*.tmp</code>
files, validate the namespace manifest and reconcile existing complete records
to quota. <code>usage()</code> returns entry count and bytes from one
coordinated scan. Generation tokens make
<code>remove_if_unchanged</code> conditional, so a corrupt read cannot delete
a newer concurrent replacement (the ABA case).

These guarantees apply to cooperating LambdaForge processes on a local
filesystem whose lock and atomic-replace semantics match the operating-system
contract. This is not a network cache, distributed consensus protocol or hard
filesystem-space reservation; validate NFS/SMB behavior separately.

## DataLoader workers and capacity planning

<code>DatasetCache</code> deliberately drops RAM entries, counters and locks
when pickled for <code>spawn</code>. It also detects a PID change after
<code>fork</code> and resets inherited RAM before access. No
<code>multiprocessing.Manager</code>, shared-memory segment or auxiliary cache
process is created.

The safe default is <code>cache_in_workers: false</code>:

- with <code>num_workers: 0</code>, access happens in the training process and
  its RAM cache is active;
- with <code>num_workers > 0</code>, worker RAM caching is disabled and only
  the optional disk backend is used;
- with <code>cache_in_workers: true</code>, every worker owns a separate empty
  RAM LRU with the full per-process byte and entry limits.

Let <code>J</code> be simultaneous independent training jobs, <code>R</code>
the DDP ranks per job, and <code>S</code> the simultaneously live DataLoader
pools that wrap a <code>DatasetCache</code>. For split <code>s</code>, let
<code>W_s</code> be its workers and <code>C_s</code> its byte budget:

~~~text
replicas_s = 1                              when W_s = 0
replicas_s = W_s                            when W_s > 0 and cache_in_workers
replicas_s = 0                              when W_s > 0 and worker caching is off

conservative serialized-RAM ceiling
    = J × R × sum(replicas_s × C_s for every live split s)
~~~

Add any manually pre-warmed parent cache separately. Train, validation and test
DataLoaders may own different worker pools, and persistent pools can coexist.
<code>persistent_workers: true</code> makes worker caches useful across epochs
but keeps their memory resident; with it disabled, worker caches disappear
when their process exits.

The disk backend may be shared across workers/ranks when all use exactly the
same trusted namespace and deterministic source contract.

## FileDataset

The constructor is:

~~~python
FileDataset(files, loader, root=None)
~~~

<code>files</code> is an ordered explicit sequence, not a glob or a single
string. Paths are normalized at construction, but sample files are opened only
by <code>__getitem__</code>. If <code>root</code> is provided, relative paths
are resolved under it and paths escaping it are rejected. Missing files fail
at access time.

The loader receives a resolved <code>pathlib.Path</code> and should be an
importable, pickle-safe callable when DataLoader workers use <code>spawn</code>.
<code>FileDataset</code> itself does not retain samples; wrap it in
<code>DatasetCache</code> when deterministic decoding is expensive.

## NumpyMemmapDataset

The constructor is:

~~~python
NumpyMemmapDataset(arrays, as_tensors=True)
~~~

<code>arrays</code> maps non-empty output names to <code>.npy</code> paths.
Construction does not open the files. The first <code>len(dataset)</code> or
indexed access opens them with:

~~~python
np.load(path, mmap_mode="r", allow_pickle=False)
~~~

All arrays must be non-scalar and have the same first dimension. Object arrays
and pickled data are refused. Each indexed row is copied into writable memory
before it is returned; <code>as_tensors=True</code> converts that copy with
<code>torch.from_numpy</code>, while false returns independent NumPy arrays.
Mutating a result therefore cannot modify the read-only file or later samples.
The parameter requires a real boolean value.

Mappings are process-local. Spawn serialization drops handles, and a forked
child reopens after detecting its PID. Call <code>close()</code> explicitly,
use the dataset as a context manager, or rely on process exit as the final
fallback. Explicit close is important before replacing or deleting mapped
files on Windows.

This adapter and <code>MemoryMappedCacheBackend</code> are different:
<code>NumpyMemmapDataset</code> maps source arrays and copies one sample;
<code>MemoryMappedCacheBackend</code> maps a serialized cache record and then
deserializes it. Neither places a hard limit on operating-system page cache.

## Dataset plugins

Reusable distributions may publish Dataset classes through the
<code>lambdaforge.datasets</code> entry-point group. A split selects the class
explicitly and passes every constructor option through YAML:

~~~yaml
data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {root: datasets/acme, split: train}
~~~

The class must inherit <code>torch.utils.data.Dataset</code>; each build creates
a fresh instance and the registry retains neither samples nor dataset objects.
Plugins can be nested as the <code>dataset</code> parameter of
<code>DatasetCache</code>, but doing so never enables caching implicitly.
Keep constructors importable, lazy and spawn-safe. <code>IterableDataset</code>
satisfies the plugin contract, but requires a custom data module because the
default training adapter shuffles its map-style training split. See the
[plugin guide](../plugins/README.md) for publishing and provenance.

## Complete YAML example

The following is a complete hardened experiment configuration. The project
loader must return a supported mapping containing <code>x</code> and
<code>target</code>; the key callable receives one dataset index. Set
<code>LAMBDAFORGE_TRAIN_CACHE_HMAC_KEY</code> to a secret of at least 32 bytes
outside the YAML. The content digest is an example and must be replaced with
the real precomputed snapshot.

~~~yaml
schema_version: "1.0"

experiment:
  name: cached_file_training
  output_root: runs/experiments
  seeds: [7, 17]
  resume: true

data:
  train:
    target: lambdaforge.data.DatasetCache
    params:
      dataset:
        target: lambdaforge.data.FileDataset
        params:
          root: data/train
          files:
            - sample-000.npy
            - sample-001.npy
          loader:
            ref: my_project.data.load_training_sample
      max_memory_bytes_per_process: 67108864  # 64 MiB per enabled process
      max_memory_entries: 4096
      backend:
        target: lambdaforge.data.MemoryMappedCacheBackend
        params:
          root: .cache/lambdaforge
          namespace: my-project/train-decoder-v3
          max_bytes: 4294967296               # 4 GiB complete-record quota
          max_entries: 100000
          record_codec:
            target: lambdaforge.data.CacheRecordCodec
            params:
              integrity: hmac_sha256
              authentication_key_env: LAMBDAFORGE_TRAIN_CACHE_HMAC_KEY
      serializer:
        target: lambdaforge.data.NumpyDatasetSerializer
        params:
          compressed: false
          max_arrays: 1024
          max_decoded_bytes: 1073741824
      fingerprint:
        target: lambdaforge.data.DatasetFingerprint
        params:
          content: "sha256:replace-with-real-dataset-content-digest"
          transform: "my_project.data.load_training_sample:v3"
          configuration:
            normalization: none
      key_fn:
        ref: my_project.data.training_cache_key
      cache_in_workers: true
      strict: true

  val:
    target: lambdaforge.data.NumpyMemmapDataset
    params:
      arrays:
        x: data/validation/features.npy
        target: data/validation/targets.npy
      as_tensors: true

  test:
    target: lambdaforge.data.NumpyMemmapDataset
    params:
      arrays:
        x: data/test/features.npy
        target: data/test/targets.npy
      as_tensors: true

  datamodule:
    target: lambdaforge.training.data.LightningDataModule
    params:
      batch_size: 64
      num_workers: 4
      persistent_workers: true
      prefetch_factor: 2

model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    hidden: [128, 64]

losses:
  - target: lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss
    params:
      output_key: logits
      target_key: target

val_metrics:
  - target: lambdaforge.metrics.classification.BinaryAUROC
    params:
      pred_key: logits
      target_key: target

task:
  params:
    model_input_key: x
    model_output_key: logits

trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  checkpoint_policy: last_and_best
  enable_progress_bar: false

execution:
  mode: sequential
  dataloader_num_workers_per_job: 4
~~~

With one job, one rank, four workers and only the training cache enabled in
workers, the configured serialized-RAM ceiling is
<code>4 × 64 MiB = 256 MiB</code>. Prefetching and live batches are additional.

## Statistics

<code>cache.stats()</code> returns a frozen <code>CacheStats</code>:

| Field | Meaning |
|---|---|
| <code>memory_hits</code> | Reads served from this process RAM LRU. |
| <code>backend_hits</code> | Reads served from the backend. |
| <code>misses</code> | Requests that loaded the wrapped dataset. |
| <code>writes</code> | Misses admitted to at least one configured layer. |
| <code>evictions</code> | Process-local RAM LRU removals. |
| <code>skipped_oversize</code> | Serialized samples that fitted no enabled layer. |
| <code>serialization_errors</code> | Key/value serialization failures, corrupt payloads or rejected CUDA samples. |
| <code>backend_errors</code> | Backend operation failures, including rejected checksum/HMAC records. |
| <code>memory_entries</code>, <code>memory_bytes</code> | Current process RAM payload usage. |
| <code>max_memory_bytes_per_process</code>, <code>max_memory_entries</code> | Configured RAM limits. |
| <code>backend_entries</code>, <code>backend_bytes</code> | One coordinated backend usage snapshot. |
| <code>process_id</code> | Process owning the counters and RAM LRU. |

Counters are process-local and reset after spawn/fork state initialization.
They are not automatically aggregated across workers. Clearing records does
not reset historical counters.

## Invalidation and lifecycle

- <code>cache.invalidate(index)</code> removes one key from RAM and, by
  default, the backend. Pass <code>include_backend=False</code> for RAM only.
- <code>cache.clear()</code> clears only this process RAM. Use
  <code>cache.clear(include_backend=True)</code> to clear the configured
  backend namespace too.
- Prefer a new backend <code>namespace</code> for a new dataset or transform
  version. Index-by-index invalidation cannot prove that every stale record was
  found.
- RAM is released with cache/process lifetime. Disk records intentionally
  survive process exit until quota eviction or explicit clearing.
- <code>NumpyMemmapDataset.close()</code> is idempotent and should be called
  deterministically when files will be moved, replaced or deleted.
- A direct backend caller must close every returned <code>CacheRecord</code>;
  <code>DatasetCache</code> already does this with a context manager.

## Determinism and cache keys

Caching a random augmentation freezes whichever result wins the first miss.
Cache deterministic loading/decoding first and apply stochastic augmentation
afterward.

The default key is the dataset index. A <code>key_fn</code> may provide a
stable sample identifier, but it must be deterministic, spawn-safe and
supported by the serializer. Avoid lambdas and process-random values.
<code>DatasetFingerprint</code> owns dataset/transform/configuration identity
separately from per-sample identity. With no fingerprint, LambdaForge preserves
the legacy key envelope; with one, it uses the versioned fingerprint-aware
envelope and serializer-format identity.

Concurrent workers may calculate the same miss. Atomic replacement keeps the
record complete, but the last writer wins. Therefore every producer sharing a
namespace must implement the same deterministic contract.

## Security and extension contracts

<code>PickleDatasetSerializer</code> can execute code while loading. A
checksum does not make pickle safe against a malicious writer. Use pickle only
with a cache root controlled by the same trusted user/project, and select it
explicitly when <code>NumpyDatasetSerializer</code> cannot represent a sample.
HMAC proves that a holder of the configured secret produced the record; it
does not make arbitrary producer code trustworthy. Never point pickle at
downloaded, shared-untrusted or attacker-writable directories.

The compatibility constructor is
<code>PickleDatasetSerializer(protocol=pickle.HIGHEST_PROTOCOL)</code>. Its
<code>format_fingerprint</code> includes the protocol. A serializer change is
also isolated in fingerprint-aware keys, but a new namespace remains the
clearest operational boundary.

A custom <code>DatasetSerializer</code> implements:

~~~python
dumps(value) -> bytes
loads(payload) -> object
~~~

A custom <code>CacheBackend</code> implements:

~~~python
read(key) -> CacheRecord | None
write(key, payload) -> bool
remove(key) -> None
clear() -> None
current_bytes: int
entry_count: int
usage() -> CacheUsage
remove_if_unchanged(key, token) -> bool
~~~

Backend keys are lowercase SHA-256 digests supplied by
<code>DatasetCache</code>. Backends own byte persistence and resource closure;
they must not infer task semantics. Keep custom objects importable and
pickle-safe if an experiment constructs them through YAML and DataLoader
workers use spawn. The base class supplies compatible fallbacks for
<code>usage</code> and <code>remove_if_unchanged</code>; override them to
provide atomic usage and generation-safe deletion.

The hardened disk format is intentionally fail-closed. If a namespace contains
legacy <code>.lfcache</code> records but no manifest, construction raises
instead of guessing their format or executing them. Choose a new namespace or
clear that disposable old cache explicitly.
