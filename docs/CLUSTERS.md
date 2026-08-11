[English](CLUSTERS.md) | [Español](CLUSTERS.es.md)

# Cluster runtime guide

LambdaForge 0.5.1 submits one explicit task or experiment to one named cluster. It keeps the same
scientific YAML and changes only physical data locations and execution policy.

## 1. Register and diagnose

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/project \
  --scheduler slurm --environment managed
lambdaforge clusters list
lambdaforge doctor --on atlas
lambdaforge clusters bootstrap atlas
```

The profile is stored in `lambdaforge.clusters.yaml` by default. OpenSSH still controls keys,
`known_hosts`, agent and ProxyJump. LambdaForge stores no password and does not disable host-key
checking. The remote workspace must be writable without root.

`managed` builds exact wheels for LambdaForge and the nearest consumer `pyproject.toml`, stages
them in a content-addressed bundle and creates
`WORKSPACE/.lambdaforge/environments/ENVIRONMENT_ID`. Repeating bootstrap or submission reuses a
verified environment. Local dirty source is built as it exists; remote execution never pulls
`main`. `existing` performs no install and requires the configured Python to contain exactly this
LambdaForge release; set `project_module` so `doctor` can verify the consumer.

For a cluster without Internet, prepare compatible dependency wheels on the target platform and
set `wheelhouse` or pass `clusters bootstrap --wheelhouse PATH`. Offline pip uses `--no-index` and
fails clearly if incomplete. LambdaForge never installs NVIDIA drivers, a system CUDA toolkit or
cuDNN; `doctor` reports the PyTorch/CUDA view of the selected Python.

## 2. Portable data and resources

```yaml
data_catalog: data-catalog.yaml
environment: local
data:
  train: dataset:raw-corpus/train
resources: {cpus: 8, memory: 32GiB, gpus: 1, gpu_memory: 20GiB, time: 4h}
```

Each catalog entry declares stable `identity`, a `loader` with explicit `path_parameter`, and
`locations` such as `local` and `atlas`. The bundle replaces the authoring environment with the
target profile's `data_environment`; the logical reference and fingerprint remain unchanged.
Nested `{dataset: raw-corpus, subpath: train}` markers work inside object params. Ordinary strings
are never interpreted as datasets. Small declared task inputs may be staged; large bytes fail closed
and must use a catalog or explicit `lambdaforge data replicate ... --apply`.

## 3. Submit and reconnect

```bash
lambdaforge run experiments/study.yaml --on atlas --dry-run
lambdaforge run experiments/study.yaml --on atlas
lambdaforge status --on atlas --state running --name study
lambdaforge logs JOB --follow
lambdaforge cancel JOB
lambdaforge retry JOB
```

The persistent local `JobRecord` contains scheduler ID, scientific/execution identities, bundle,
remote paths and timestamps. Closing the laptop does not cancel a SLURM job; a later `status`
reconnects to the scheduler. A local-scheduler process is not made remote merely by persistence.

## 4. Results and limits

```bash
lambdaforge results sync JOB
lambdaforge plot learning JOB --follow --output learning.svg
lambdaforge artifact list JOB
lambdaforge artifact fetch JOB best-checkpoint --output checkpoints/best.ckpt
```

Sync retrieves only allowlisted small metadata, metrics, manifests, summaries and plots (16 MiB per
file by default). Checkpoints/datasets are never implicit. Fetch selects one logical artifact and
rejects paths outside the recorded remote job directory.

Not supported in 0.5.1: automatic cluster placement, one workflow spanning clusters, a resident
coordinator, remote driver installation, or automatic construction of platform/CUDA dependency
wheels. Submit workflow nodes explicitly if they need different clusters.
