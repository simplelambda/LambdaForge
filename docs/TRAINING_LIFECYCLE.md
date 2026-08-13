# Training lifecycle and post-run action audit

[Repository guide](../README.md) · [Español](TRAINING_LIFECYCLE.es.md)

## 1. Decision

Most loop-bound requirements already belonged to configurable Lightning callbacks. Existing generic
Tasks, model operations and Workflows already covered independently scheduled downstream work. The
real missing capability was narrower: declarative, checkpoint-aware work after one concrete run,
with completion/failure semantics and restart-safe artifact provenance, without refitting when only
that work changed.

LambdaForge therefore keeps callbacks unchanged, adds `PostRunAction` for that gap, and retains
Task/Workflow as the resource/lifecycle boundary. No domain-specific model, MIL, protein, geometry
or reconstruction concept was added.

## 2. Audit of the provisional local changes

| Provisional change | Classification | Resolution |
|---|---|---|
| `TerminalEvaluationContext` with config/run/checkpoint | USEFUL BUT NEEDS GENERALIZATION | Replaced by immutable `PostRunContext` with result, seed/variant, strict roles, digest, artifact path and resumable state. |
| `TerminalEvaluationService` best-then-last evaluator | BADLY PLACED and USEFUL BUT NEEDS GENERALIZATION | Replaced by `PostRunService`; selection is explicit, no fallback, actions have receipts/artifacts/failure policy. |
| Calling terminal evaluation after writing successful `result.json` | BADLY PLACED | Training is committed separately first; canonical success is published only after required actions succeed. |
| `fail_on_error` evaluator flag | USEFUL BUT NEEDS GENERALIZATION | Replaced per action by `required`; optional failures remain recorded and visible. |
| One `terminal-evaluation.json` mapping | DUPLICATES EXISTING API | Removed; actions return shared artifact declarations and receipts embed `TaskArtifact` records. |
| Adding terminal evaluator YAML to `RunFingerprint` | BADLY PLACED | `post_run` has a sub-fingerprint and is excluded from neural training identity. |
| Adaptive optimizer tracking paused checkpoints for later pruning evaluation | BADLY PLACED | Removed. Pauses are not final successes; default actions run only for confirmation, or all successful terminal runs explicitly. |
| Special handling for completed/early-stopped/cancelled/pruned HPO states | USEFUL BUT NEEDS GENERALIZATION | Reduced to a generic scope gate; cancellation and multi-fidelity pause never execute successful-run actions. |
| `callbacks:` construction through `ObjectFactory`/`LightningRunner` | NECESSARY | Preserved. No LambdaForge wrappers were added around Lightning hooks. |
| Callback-specific scientific logic in core | TOO DOMAIN-SPECIFIC | None retained or introduced. Validation exposes generic detached outputs only. |

## 3. Existing capabilities and the confirmed gap

| Requirement | Callback | Task/Workflow | Previous internal hook | PostRunAction |
|---|---:|---:|---:|---:|
| Batch/epoch hook | yes | no | no | no |
| Reuse the validation forward | yes, now through `model_outputs` | no | no | no |
| Different resources/cluster | no | yes | no | no |
| Stable concrete run/checkpoint/result context | partial/live | explicit bindings, separate run | private/programmatic | yes |
| Required vs optional completion | no | node-level | no | yes |
| Artifact hashes and provenance | project-owned | yes | no | yes, shared artifact types |
| Resume only downstream work | callback state depends on Lightning | yes, but separate DAG config | no | yes |
| Change report YAML without refitting | not applicable | yes | no declarative API | yes |

`on_run_finished` remains a programmatic suite/executor notification used for aggregate refresh; it
is not a user-facing scientific lifecycle. `InferenceTask`, `EvaluationTask` and `ExportTask` remain
the right choice when downstream work deserves its own task identity or allocation.

## 4. Completion and identity sequence

```text
fit/test succeeds on rank zero
  -> commit .lambdaforge/post-run/training-result.json
  -> for each action, select exact checkpoint and compute action identity
  -> reuse a matching verified receipt or execute action
  -> hash declared artifacts with TaskArtifact
  -> required failure: publish failed result, preserve training commit
  -> all required succeed: atomically publish reusable result.json
```

The training fingerprint excludes `post_run`. An action identity contains its target, parameters,
checkpoint policy, required policy, declared artifacts, run scientific identity and selected
checkpoint digest. An interrupted action has no success receipt; its identity-specific state
directory remains for project-managed continuation. Relaunching reconciles actions before any fit.

## 5. Explicit boundaries

- `current` means a persisted final/current (`last`) checkpoint; live in-memory weights are not a
  reproducible resume input.
- Actions are sequential and reuse the allocation. They are not schedulers.
- Actions execute only on global rank zero. Distributed scientific reduction belongs to callback or
  project code.
- HPO objectives must be logged during validation. Post-run results cannot alter an already observed
  trial decision.
- Paused/pruned intermediate HPO checkpoints are not presented as successful final runs.

