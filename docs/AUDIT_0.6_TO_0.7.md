[English](AUDIT_0.6_TO_0.7.md) | [Español](AUDIT_0.6_TO_0.7.es.md)

# LambdaForge 0.6 to 0.7 audit

This audit records the state of the real `main` branch before the 0.7 implementation. It is not a
roadmap inferred from documentation. Each conclusion below was checked against the implementation
and its tests on 2026-08-14.

## Classification

| Area | Classification | Evidence and 0.7 decision |
|---|---|---|
| Preprocessing publication | CONFIRMED | `PreprocessingTask.run()` always writes `dataset-artifact.json` and falls back from `dataset_name` to the task name. Make publication explicit; retain `dataset_name` as the legacy opt-in. |
| Task versus dataset boundary | DESIGN ISSUE | `TaskRunner` registers any dataset manifest it finds. Keep this compatibility behavior, but ordinary preprocessing must no longer manufacture that manifest. |
| Dataset recipe | CONFIRMED | There is no `kind: dataset` document or recipe/build entity. Add a typed recipe that compiles its stages to the existing Workflow DAG rather than adding another DAG engine. |
| Stage reuse | PARTIALLY SOLVED | `TaskFingerprint`, verified `TaskResult` artifacts and Workflow bindings already provide content-addressed reuse. Add recipe-level decisions, `required`/`reuse`, force propagation and durable receipts; do not add a parallel cache key. |
| Durable builds | PARTIALLY SOLVED | `JobService` and the 0.6 schedulers already persist durable jobs and `job_type`. Dataset build submission should use them with `job_type=dataset-build`. |
| Dataset v1 identity | DESIGN ISSUE | `DatasetArtifact.create()` hashes name, version, preprocessing fingerprint, source and metadata together with bytes. Introduce v2 `content_id` and separate `build_id`; continue reading v1. |
| Dataset logical members | CONFIRMED | v1 has only aggregate sample/split counts and task artifacts. Add `DatasetMember` and a streaming JSONL `DatasetIndex`; physical layout remains unrestricted. |
| Partitions and targets | CONFIRMED | v1 exposes one rigid `splits` map and no target schema. v2 derives arbitrary partition summaries and keeps generic targets with optional explicit schema. |
| Dataset placements | ALREADY SOLVED | `DatasetPlacement` is independent from `DatasetRecord.dataset_id`, supports multiple clusters and registry reconciliation. Preserve it. |
| Immutable aliases | ALREADY SOLVED | `DatasetRegistry.register()` rejects the same `name@version` with a different identity. Preserve and improve the typed error. |
| Registry/DataCatalog resolution | CONFIRMED | managed placements live in `DatasetRegistry`, while `DatasetReferenceResolver`, `TaskInput` and `ExecutionBundleBuilder` require `DataCatalog.locations`. Add one `DatasetResolver`; DataCatalog remains for loaders, aliases and external data. |
| Versioned references | PARTIALLY SOLVED | parsing supports `dataset:name/subpath`, while registry selectors support `name@version`; bindings do not pin both consistently. Extend `DatasetReference` and record exact content/placement bindings. |
| Dataset lockfile | NOT WORTH CHANGING | Materialized bindings plus immutable registry IDs can pin runs without another mutable source of truth. Re-evaluate only if alias resolution must be shared without registry state. |
| Materialize `BUILD` | CONFIRMED | `DatasetService.materialize(..., apply=True)` raises an instruction to run the producer manually. Route recipes/producers through durable submission or local build execution. |
| Atomic publication | PARTIALLY SOLVED | Registry writes are atomic but dataset publication is not a staging/validate/rename transaction. Add a publisher boundary; never register partial bytes. |
| Stage-cache GC | PARTIALLY SOLVED | `StorageOperations` already restricts GC to cache roots and excludes dataset roots. Put recipe stage cache below the cache root and supply active/published references. |
| Dataset validation | PARTIALLY SOLVED | v1 verifies manifest artifact checksums and safe paths. Extend validation to member IDs, index syntax, asset existence/integrity, partitions and declared target schema. |
| Dataset profiling | PARTIALLY SOLVED | filesystem statistics and an explicit classification profiler exist. Base profiling must use `DatasetIndex`; project profilers currently refuse remote execution. |
| Members and diff CLI | CONFIRMED | no member listing/detail or content diff API exists. Add bounded streaming queries and identity-aware diff. |
| Lineage | PARTIALLY SOLVED | `DatasetRecord.lineage` is a flat tuple. Preserve reads and add structured recipe/build/stage/input provenance in v2. |
| CLI entry point `lf` | CONFIRMED | only `lambdaforge` is installed. Add `lf` to the same callable. |
| CLI grammar and aliases | PARTIALLY SOLVED | resource namespaces and root shortcuts already exist, but output/options differ. Add moderate aliases and shared rendering without a second command implementation. |
| Run by config name | ALREADY SOLVED | root `run`/`validate`/`inspect` and resource commands already use `ProjectConfigService` when a path does not exist. Preserve ambiguity checks. |
| Default cluster | CONFIRMED | no user/project default is applied to run commands. Add an explicit, visible preference with `--on` precedence. |
| Root `plan` | CONFIRMED | entity `plan` forwards to `inspect`, but there is no uniform root shortcut or data-readiness envelope. Add it without removing `run --dry-run`. |
| Typed CLI errors | CONFIRMED | handlers generally render `ERROR: KeyError/RuntimeError`. Add dataset/control-plane error types and actionable rendering. |
| CLI architecture | DESIGN ISSUE | `CommandLineInterface.py` contains parser and behavior for every domain (over 2300 lines). Move new dataset behavior, rendering, errors and completion into focused CLI modules; do not rewrite stable commands solely for file layout. |
| Human versus JSON output | DOCUMENTATION DRIFT | several read-only commands accept `--json` but print JSON in both modes; jobs list lacks headers. Add stable human renderers while preserving JSON. |
| Active/dry-run states | CONFIRMED | `CREATED` is non-terminal and dry-run submission persists provider preview state, so it can appear active indefinitely. Introduce `PLANNED` and an explicit activity breakdown while reading legacy `CREATED`. |
| `top` telemetry | PARTIALLY SOLVED | resource/job snapshots exist; presentation compresses them into an ambiguous active count. Improve the view, preserving `unknown` rather than inventing values. |
| Friendly job selectors | CONFIRMED | `JobService.get()` only accepts exact IDs. Add deterministic `latest`/unambiguous name selection. |
| Completion | CONFIRMED | no shell completion command exists. Generate dependency-free bash/zsh/fish scripts from known resources. |
| Authoring metadata | DOCUMENTATION DRIFT | concise top-level `data_catalog`, `environment` and `resources` are accepted, but strict materialization stores them under `extensions.authoring` and some docs expose that IR. Keep the internal form and document only concise input for normal use. |
| Artifact and result services | ALREADY SOLVED | first-class artifact/result services, remote evidence sync and explicit heavy fetch already exist. Add aliases only; do not create another registry. |
| Version consistency | CONFIRMED | the v0.6.1 commit still declares `0.6.0` in `pyproject.toml`, and generated templates contain older constraints. Add one consistency test and update 0.7 declarations/templates. |

## Architectural boundary retained for 0.7

The implementation must keep four identities separate:

```text
DatasetRecipe (how) -> DatasetBuild (execution) -> DatasetVersion (what)
                                                   -> DatasetPlacement (where)
```

Workflow owns DAG execution, Task owns stage scientific identity and verified outputs, JobService
owns durable scheduling, DatasetArtifact/Index own immutable content, DatasetRegistry owns managed
placements, and DataCatalog describes aliases/loaders/external datasets. No daemon, distributed DAG
scheduler, database or automatic scientific placement is justified by this audit.
