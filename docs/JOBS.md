# Durable jobs and the process scheduler

[Español](JOBS.es.md) | English | [Root guide](../README.md)

## 1. Lifecycle and files

Portable states are `created`, `staging`, `queued`, `running`, `paused`, `succeeded`, `failed`,
`cancelled`, `timeout` and `unknown`. `staging` is observable work, not a hidden pre-submit delay.
For a direct/process cluster, one exact directory contains:

```text
RUN_ROOT/job-ID/
  request.json       immutable submitted command/resources
  state.json         atomically replaced authoritative state
  heartbeat          updated independently of stdout
  stdout.log
  stderr.log
  usage.jsonl        CPU/RSS/thread/GPU-memory samples
  work/              mutable scientific workspace
  control/           reserved control requests
```

The supervisor and scientific child have recorded PID, process-group, creation time and command
hash identities. Cancel/pause/resume verify all of them before signalling, so PID reuse cannot make
LambdaForge control an unrelated process. Cancellation targets the process group and descendants,
then escalates after a grace period. Inventory reads only directories with matching LambdaForge
request/state job IDs.

## 2. Commands

```bash
lambdaforge jobs list [--on atlas] [--state running] [--name baseline] [--json]
lambdaforge jobs show JOB
lambdaforge jobs logs JOB [--tail 200] [--follow]
lambdaforge jobs pause JOB
lambdaforge jobs resume JOB
lambdaforge jobs cancel JOB
lambdaforge jobs retry JOB [--dry-run]
lambdaforge jobs delete JOB          # local terminal metadata only
lambdaforge jobs reconcile --on atlas
lambdaforge jobs reconcile --all
lambdaforge jobs group list
lambdaforge jobs group show GROUP
```

`delete` never deletes remote work or results. `retry` creates a new job and records `retry_of`.
`reconcile` enumerates the configured provider's LambdaForge inventory and can rebuild a missing
local record. `logs --follow` reconnects; it does not keep the original submission process alive.

## 3. Pause, queues and resources

Pause/resume are capabilities. The process scheduler supports POSIX `SIGSTOP`/`SIGCONT`; a SLURM
profile supports them only when trusted site commands were configured. Pausing process work stops
CPU execution but deliberately retains RAM, VRAM, CPU/GPU leases and workspace. The CLI reports
that warning.

Direct hosts have cooperative, locked leases. A job waits in `queued` until its declared CPU/RAM
and GPU request can be admitted. CPU affinity and thread limits are applied where supported.
Selected GPUs are exposed through `CUDA_VISIBLE_DEVICES`. LambdaForge leases never claim physical
isolation and conservatively avoid GPUs with externally observable compute processes. SLURM remains
responsible for capacity/isolation on SLURM clusters.

`resources.time` is a wall-clock supervisor deadline. It is distinct from SSH timeouts and ends in
`timeout`. Heartbeats continue for silent jobs. If the supervisor itself is force-killed, 0.6 does
not falsely claim that it can always recover/kill an already orphaned arbitrary process: state may
become stale/unknown and requires operator inspection.

## 4. Multi-cluster groups

```bash
lambdaforge experiments run baseline --on atlas --on gpu-lab
lambdaforge jobs group list
```

This creates independent jobs linked by a persistent `group_id`; it is not distributed training or
a shared adaptive optimizer. An HPO config is rejected unless `--independent-hpo` is given. Partial
submission can leave a group with jobs already accepted by earlier clusters; those job IDs remain
visible and cancellable rather than being hidden by rollback fiction.
