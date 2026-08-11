# LambdaForge control plane

[Root guide](../../../README.md) · [Español](README.es.md)

## 0. Contents

- [1. Mental model](#1-mental-model)
- [2. Cluster catalogue](#2-cluster-catalogue)
- [3. Submission and bundles](#3-submission-and-bundles)
- [4. Jobs](#4-jobs)
- [5. Providers](#5-providers)
- [6. Safety and limits](#6-safety-and-limits)

## 1. Mental model

The control plane is a local application service. It materializes authoring YAML, stages a small
content-addressed `ExecutionBundle`, submits the ordinary LambdaForge command through a `Transport`
and `Scheduler`, and persists a local `JobRecord`. It does not replace experiment/task runners and
does not require a server.

```text
ControlPlane -> ExecutionBundleBuilder -> Transport -> Scheduler
            \-> JobService -> JobStore -> JobRecord/JobHandle
```

Scientific results remain in each run's `result.json`; scheduler lifecycle remains in the job
store. `ExecutionIdentity` records placement/resources without changing `ScientificIdentity`.

## 2. Cluster catalogue

`ClusterCatalog.load()` searches an explicit path, `LAMBDAFORGE_CLUSTERS`, the project
`lambdaforge.clusters.yaml`, then the user config path. `local` always exists. An SSH profile must
declare an absolute workspace.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/user/lambdaforge
    python: /shared/env/bin/python
    environment: managed
    project_module: my_project
    data_environment: atlas
    command_prefix: [apptainer, exec, /images/project.sif]
    scheduler_options: {partition: gpu}
profiles:
  one-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, time: 4h}
```

`command_prefix` is an argv prefix for containers/site wrappers. Do not place credentials in this
file. Verify with `lambdaforge doctor --on atlas` or `clusters test atlas`.

## 3. Submission and bundles

```bash
lambdaforge run config.yaml --on atlas --dry-run
lambdaforge run config.yaml --profile one-gpu
```

`ExecutionBundleBuilder` caches strict YAML, manifest, exact local framework/consumer wheels and
small path inputs (10 MiB maximum). Large inputs use `DataCatalog`. `managed` creates an idempotent
user venv keyed by wheel/Python/wheelhouse identity; `existing` only verifies. Offline mode needs a
target-compatible wheelhouse. No branch clone, driver or system CUDA installation occurs. See the
[complete cluster guide](../../../docs/CLUSTERS.md).

Python:

```python
from lambdaforge.controlplane import ControlPlane
from lambdaforge.execution import ResourceRequest

handle, bundle = ControlPlane().submit(
    "experiment.yaml",
    cluster="atlas",
    resources=ResourceRequest.from_mapping({"cpus": 8, "memory": "32GiB", "gpus": 1}),
    dry_run=True,
)
```

## 4. Jobs

`JobStore` writes atomic JSON below `$XDG_STATE_HOME/lambdaforge/jobs` or
`~/.local/state/lambdaforge/jobs`. `JobService` is restart-safe for scheduler-backed jobs:

```bash
lambdaforge status --on atlas --state running --name study
lambdaforge status JOB_ID
lambdaforge logs JOB_ID --follow
lambdaforge cancel JOB_ID
lambdaforge retry JOB_ID --dry-run
```

Retry creates a new job ID and `retry_of` link. It never overwrites the prior job or scientific
attempt. `results sync JOB` retrieves small evidence; `artifact fetch JOB NAME` explicitly retrieves
one heavy artifact.

## 5. Providers

- `LocalTransport`: local subprocess and filesystem staging.
- `SshTransport`: OpenSSH/scp with normal host-key policy and quoted remote argv.
- `LocalScheduler`: synchronous execution through a transport.
- `SlurmScheduler`: script generation, `sbatch`, `squeue`/`sacct`, logs and `scancel`.
- `ControlPlaneFactory`: default provider selection; inject another factory/provider in services.

Implement `Transport` and `Scheduler` for another platform. Return `CommandResult`,
`SchedulerSubmission` and portable `JobState`; never make the runner provider-aware.

## 6. Safety and limits

- Remote actions occur only with `run --on` and no `--dry-run`; data replication needs `--apply`.
- SSH host verification and credentials belong to OpenSSH; LambdaForge never disables them.
- Scheduler options reject unsafe names/newlines; commands remain argument vectors.
- Mixed-cluster workflow placement is visible in dry-run but execution is refused until durable DAG
  recovery and artifact transfer are implemented soundly.
- Cluster choice is explicit; capacity/queue/cost discovery and automatic placement are not claimed.
- `DataCatalog` resolves named task inputs plus direct/nested typed experiment references; ordinary
  strings remain a consumer concern.
- Local scheduler execution is synchronous. SLURM jobs reconnect across CLI processes.
