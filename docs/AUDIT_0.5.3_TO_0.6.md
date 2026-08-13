[English](AUDIT_0.5.3_TO_0.6.md) | [Español](AUDIT_0.5.3_TO_0.6.es.md)

# LambdaForge 0.5.3 to 0.6 control-plane audit

This audit was performed against the 0.5.3 source before implementing 0.6. It distinguishes facts
observed in code from requests in the 0.6 specification.

## Existing reusable capabilities

- `ClusterCatalog` already merges user, project and explicit YAML profiles and keeps source
  provenance. `ClusterService`, `ControlPlane` and `ControlPlaneFactory` already separate
  application services from `Transport`, `Scheduler` and `EnvironmentProvider` contracts.
- OpenSSH and host-key-verifying password transports already avoid `shell=True`; the Paramiko
  transport reuses one client during one CLI invocation. SLURM commands/resources are configurable
  through `SlurmProfile` and submitted jobs already survive the controlling PC.
- `JobStore` atomically persists backward-readable JSON records. `JobService` supports list,
  refresh, logs, cancel and retry. Scientific execution already has process-tree cleanup,
  parent-death guards and Windows Job Objects that can be reused rather than duplicated.
- Managed environments and bundles are content-addressed. 0.5.3 resolves an exact remote-compatible
  Torch plan, validates CUDA and includes dependency policy in environment identity.
- `ResourceRequest` distinguishes requested CPU/RAM/GPU/storage/runtime/processes. Resource
  monitoring already records process and CUDA observations during scientific execution.
- `DataCatalog`, `DataService`, `DatasetArtifact`, identity providers and transfer providers already
  provide logical identity, declared placements, content manifests, inspection and explicit
  replication. Preprocessing writes a content-derived `dataset-artifact.json`.
- `ResultService`, `RemoteResultService`, artifact services, `ExperimentRegistry`, retention,
  configuration materialization/plugins and workflow validation are reusable object-level APIs.

## Confirmed limitations

- `LocalScheduler.submit()` runs synchronously through `Transport.run()`, stores state/logs only in
  its Python object and cannot reconnect, pause or enforce runtime. Remote hosts without SLURM keep
  the SSH command open for the complete run.
- `JobService.submit()` creates its local record only after scheduler submission returns. A crash or
  long synchronous submit therefore has no durable CREATED/STAGING record.
- `SshTransport` starts a new `ssh`/`scp` process for every operation. It benefits from a user's
  existing OpenSSH `ControlMaster`, but 0.5.3 does not enable or bound multiplexing itself.
  `PasswordSshTransport` reuses its Paramiko connection within the process, but one `ssh_timeout`
  controls connect, banner, authentication and `exec_command`.
- `Transport.run()` has no command-timeout argument. Connection establishment, command duration and
  requested job runtime are not represented as independent policies.
- The local store is the only source of truth for direct/process jobs. There is no remote job
  inventory, supervisor, heartbeat, safe PID identity, GPU lease, reconciliation or global
  bounded-concurrency query. Network failure can be confused with scheduler state.
- Job states omit STAGING, PAUSED and TIMEOUT. Scheduler pause/resume capabilities and the fact that
  paused CUDA work may retain VRAM are not represented.
- Resource capacity has planning types but no cluster probe/service, global resource view or
  terminal `top` view. Requested, scheduler-allocated, observed and available values are not shown
  together as distinct concepts.
- Dataset operations require an explicit catalog. There is no automatic project/user registry,
  remote placement index, profiler contract, safe delete, lineage/materialization planner or
  automatic registration of successful preprocessing output.
- Configs are path-oriented. There is no project config index, run-by-unambiguous-name,
  experiments/tasks entity service or persistent job group.
- Bundle cache, remote work directory and output location are physically coupled during remote
  execution. Bundles, managed environments, bootstrap staging and failed temporaries have no
  reference-aware global GC. Cluster storage roots and cache budgets are not modeled.
- `CommandLineInterface` is 1,700+ lines and contains material domain dispatch. It is already backed
  by services in places, but further domains should use dedicated services/CLI modules.

## False assumptions from the specification

- `DataService.inspect()` is not missing from the CLI: `lambdaforge data --catalog C inspect NAME`
  already calls it. The 0.6 `datasets` interface can reuse it and remove the mandatory catalog UX.
- SLURM execution is already asynchronous and durable at the scheduler. The fundamental gap is
  `scheduler: local`/direct SSH, plus global recovery and uniform state.
- Result selection no longer universally requires hash-directory hunting: `ResultService` already
  resolves names, fingerprints and paths locally, and small remote evidence has explicit sync.
- Managed environment construction already verifies/publishes a marker and can reject incomplete
  cache entries. It is not, however, built under a separate temporary directory and lacks global
  cleanup/ref tracking.
- A mandatory SQLite database is not justified yet. Entity volumes are small and atomic JSON plus
  locks provides simpler migration, remote portability and inspectability.
- Splitting the runtime environment from every consumer wheel is not safe without an exact project
  dependency lock/layer contract. 0.6 should first share downloads, improve reuse and implement GC.

## Backward compatibility concerns

- Existing 0.5 job JSON must remain readable; new fields need defaults and record versions.
- `scheduler: local` remains accepted but changes from blocking completion to durable process
  submission. Scripts that assumed the CLI return meant scientific completion must use
  `jobs status/logs` or an explicit local foreground mode.
- Existing `ssh_timeout` remains accepted and becomes the deprecated alias for connect timeout only.
  It must never become a command or scientific runtime limit.
- Existing workspace directories (`bundles`, `environments`) remain discoverable and GC-classified;
  they are never silently deleted. `data` remains an alias while `datasets` becomes primary.
- Existing path-based run/results/data commands and scientific YAML schemas remain valid. 0.6 adds
  operational indexes; it does not move scientific configuration into CLI flags.
- Mixed-cluster workflow DAGs, distributed adaptive-HPO controllers and automatic placement remain
  rejected because durable coordination and objective placement policy are separate designs.

## Proposed 0.6 architecture

The control plane keeps no required daemon or server. Sources of truth are explicit:

| Entity | Source of truth | Local role |
|---|---|---|
| SLURM job | scheduler/accounting | cached `JobRecord` and metadata |
| Process job | per-job remote supervisor state | cached/indexed `JobRecord` |
| Dataset placement | `DatasetArtifact` plus atomic placement entry | merged project/user query index |
| Scientific result | result directory and `result.json` | small synchronized result index |
| Bundle/environment | immutable cache bytes plus completion marker | reference/GC index |
| Config | project YAML materialized by `AuthoringConfig` | deterministic project config index |

`ProcessScheduler` launches one detached, per-job `ProcessSupervisor`; it owns state, logs,
heartbeat, usage, process identity, runtime enforcement and GPU leases. `JobService` writes CREATED
before submission, refreshes from the authoritative provider and reconciles remote inventories.
Scheduler capabilities expose pause/resume instead of guessing support.

`SshConnectionPolicy` separates connect/auth/banner/keepalive/command semantics. OpenSSH uses a
private bounded `ControlMaster`/`ControlPersist` socket cache so repeated operations and consecutive
CLI invocations reuse one authenticated connection for a configured idle period. Paramiko keeps one
client per invocation and uses keepalive; command timeout is independently optional.

`ResourceService`, `DatasetService`, `ProjectConfigService`, `StorageService` and
`EnvironmentService` are object APIs. Multi-cluster reads run with bounded concurrency and preserve
UNREACHABLE/last-known state. Destructive plans are immutable previews and require `--apply`.
Datasets and results are evidence, never cache.

## Migration plan

1. Add versioned connection/storage policies and timeout-aware transports while retaining legacy
   profile fields.
2. Add process supervisor/state values/capabilities; make local/direct submission durable and write
   the initial local record before scheduler submission.
3. Add reconciliation, job groups and bounded global queries. Keep SLURM authoritative.
4. Add provider-neutral resource snapshots and direct/SLURM probes, then overview/top.
5. Add atomic dataset registry, auto-registration hooks, profilers and remote inventory merge.
6. Add safe dataset plans for verify/remove/delete/replicate/materialize and explicit large transfer
   application.
7. Add deterministic config discovery and experiment/task facades without duplicating runners.
8. Separate mutable job work from immutable bundle cache; add storage/environment reports and
   reference-aware preview-first GC over cache/tmp only.
9. Preserve aliases/layout readers, update bilingual beginner-first manuals and run focused tests,
   POSIX process integration, package smoke and the complete CI suite.
