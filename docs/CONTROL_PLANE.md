# LambdaForge 0.6 terminal control plane

[Español](CONTROL_PLANE.es.md) | English | [Root guide](../README.md)

## 1. Mental model

LambdaForge has no mandatory server. Each CLI invocation is a short-lived controller that reads
small local metadata, contacts only the requested clusters and exits. Long work is owned by either
SLURM or one detached `ProcessSupervisor` per job on the execution host. Therefore closing the
laptop or losing the SSH session does not kill a correctly acknowledged job.

```text
YAML source -> materialized config -> immutable bundle cache
                                         |
local JobRecord <- scheduler identity <- submit
                                         |
                           durable job/work + state + logs
```

The local `JobRecord` is a reconnectable index, not the scientific result and not the authoritative
remote process state. `result.json`, metrics and checkpoints remain scientific evidence. Use
`jobs reconcile` when local metadata was lost.

## 2. First use

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/lambdaforge \
  --scheduler local --environment managed --cache-root /scratch/me/lf-cache \
  --run-root /scratch/me/lf-jobs --dataset-root /project/data
lambdaforge clusters bootstrap atlas
lambdaforge doctor --on atlas
lambdaforge datasets list
lambdaforge experiments list
lambdaforge experiments run baseline --on atlas --dry-run
lambdaforge experiments run baseline --on atlas
lambdaforge jobs list --all
```

`scheduler: local` means the durable process scheduler in 0.6; it no longer means that the calling
CLI waits for training. `scheduler: slurm` keeps SLURM authoritative. Both use the same `JobService`.

## 3. SSH connection reuse and timeouts

OpenSSH is recommended. LambdaForge starts a small `ssh` client process for each remote operation,
but by default those clients reuse one authenticated OpenSSH master socket for 60 seconds after the
last use. Consecutive commands and even consecutive CLI invocations therefore avoid repeated TCP,
key exchange and authentication. The private socket directory is mode `0700` below the user's XDG
cache. OpenSSH host aliases, agent, keys, `known_hosts` and `ProxyJump` still apply.

```yaml
connection:
  connect_timeout: 15s
  auth_timeout: 30s        # Paramiko password mode
  banner_timeout: 30s      # Paramiko password mode
  keepalive: 30s
  multiplex: true          # OpenSSH ControlMaster=auto
  persist: 2m              # close after this idle period
  command_timeout: null    # long scientific commands have no transport deadline
```

Connection/authentication deadlines are not command deadlines. Doctor/probes/inventory use explicit
short deadlines. Scheduling and long scientific commands have no implicit timeout. Scientific
runtime is controlled by `resources.time`; for the process scheduler its supervisor changes the job
to `timeout` and terminates the verified process group. Password/Paramiko mode reuses its connection
within one CLI process but cannot persist it between CLI invocations; prefer OpenSSH for frequent
cluster use.

Disable multiplexing only when site policy requires it: `connection.multiplex: false`. Increasing
`persist` reduces authentication churn but keeps the local master connection available longer.

## 4. Storage layout and migration

Each cluster can place small state, reconstructible cache, mutable runs and scientific datasets on
different filesystems:

```yaml
storage:
  state_root: /home/me/.lambdaforge/state
  cache_root: /scratch/me/lambdaforge/cache
  run_root: /scratch/me/lambdaforge/jobs
  dataset_root: /project/datasets
  cache_max_size: 50GiB
  cache_max_age: 30d
```

Bundles and environments are below `cache_root`; per-job state/logs/work are below `run_root`;
registry state is below `state_root`. Dataset bytes are never forced into `.lambdaforge`. Existing
0.5 managed environments and active-environment pointers remain readable. New data uses the 0.6
layout. GC never treats results, datasets or retained checkpoints as cache.

## 5. Global views and honest failure states

```bash
lambdaforge status                 # global overview
lambdaforge overview --json
lambdaforge resources --all
lambdaforge top --follow
lambdaforge storage status --all
```

Cluster queries are bounded and parallel. An unreachable cluster is reported as offline with its
last observed resource snapshot where available; this is not converted into a failed job. A
provider error during job refresh produces `unknown` plus `last_known_state` and an error message.
LambdaForge never searches or signals arbitrary remote PIDs.

## 6. Explicit non-goals

0.6 does not add a central daemon, automatic cluster placement, coordinated multi-cluster HPO or a
durable mixed-cluster workflow DAG. Repeating `--on` creates named independent job-group replicas;
HPO requires `--independent-hpo` to make the lack of shared optimizer state explicit.
