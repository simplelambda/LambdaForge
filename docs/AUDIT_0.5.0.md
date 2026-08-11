[English](AUDIT_0.5.0.md) | [Español](AUDIT_0.5.0.es.md)

# LambdaForge 0.5.0 audit

This audit was completed before the 0.5.1 implementation. It combines the source review with the
actual failed logs from GitHub Actions run `31499871979` at commit `e2fe74d`.

## Confirmed bugs

- `ArchitectureConformanceCase.write_reference()` calls `fsync` on a read-only descriptor. Windows
  reports `OSError: [Errno 9] Bad file descriptor` in every supported Python job.
- `TrainingOrchestrator.run_dynamic()` reports every process found complete in one polling pass
  before refilling the first released slot. Under loaded Linux runners this violates the adaptive
  controller's observe-then-decide contract and made the timing test fail on Python 3.10/3.11.
- The distribution smoke job installs the wheel with `--no-deps` and then imports the package. The
  wheel correctly declares `jsonschema`, but it is absent by construction, so this is not an
  installed-wheel test and fails with `ModuleNotFoundError`.
- `PreprocessingTask` selects `ThreadPoolExecutor` for every `workers > 1`; `workload: cpu` has no
  runtime effect.
- Dense epoch CSV replacement is not atomic, so a concurrent plot/result reader can observe a
  partially rewritten file.

## Potential bugs not reproduced

- The public CI log contains no independent failures for Python 3.12-3.14 on Ubuntu. The observed
  pattern was caused by timing and platform-specific defects above, not a version compatibility
  issue.
- No evidence supports changing or reducing the supported Python/Windows matrix.
- The existing SSH transport preserves OpenSSH host-key and agent behaviour; no security bypass was
  found.

## Missing features

- CPU-process preprocessing with a spawn-safe parent-owned sink protocol.
- Friendly training aliases and typed dataset references inside experiment object specifications.
- Managed, content-addressed environments containing the exact LambdaForge and consumer project
  builds, including an explicit offline wheelhouse policy.
- Lightweight remote result synchronization and logical artifact retrieval.
- A reusable result query/metric-series layer, declarative plots, safe artifact inspection,
  preprocessing sampling and dataset inspection.
- Complete English/Spanish pairs for maintained architecture and operational guides.

## Existing features that already solve requested items

- Strict/current experiment, task and workflow schemas remain the execution IR.
- `ResultCatalog` is already the authoritative attempt source and `ExperimentRegistry` already
  exports JSON/CSV/Parquet. The new result service must compose these rather than create a database.
- `metrics.csv`, aggregation statistics, HPO state, resource columns and persistent `JobStore`
  already provide the source evidence needed by analysis services.
- Local/SSH transports, local/SLURM schedulers, bundle caching, status/log/cancel/retry and explicit
  data replication already provide the provider-neutral control-plane foundation.
- Workflow plans already expose placement and correctly refuse non-local execution.

## Backward-compatibility constraints

- Every valid 0.5.0 strict YAML remains valid; friendly syntax only compiles into that IR.
- Existing task/experiment runners, result JSON and `lambdaforge results SOURCE` remain available.
- `existing` remains a supported environment policy; managed environments are opt-in per cluster.
- Dataset references are resolved only in explicit dataset fields/specifications. Arbitrary strings
  are never treated as paths.
- 0.5.1 does not add automatic placement, distributed multi-cluster workflows, implicit large-data
  replication, system CUDA installation or a GUI.

## Implementation plan

1. Correct the four CI/runtime defects and add isolated wheel/process tests.
2. Extend authoring and dataset resolution without changing the strict runner path.
3. Add environment identity/providers, exact bundles, bootstrap diagnostics and small remote sync.
4. Add `ResultService`, `MetricSeries`, `PlotSpec`/visualization and artifact/debug services.
5. Route CLI commands through those services, synchronize bilingual documentation and run the full
   release verification matrix available on the development host.
