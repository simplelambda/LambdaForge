---
name: lambdaforge-recovery
description: "Inspect a failed, cancelled or interrupted LambdaForge Study and prepare recovery of its original Execution without repeating successful Runs. Use when retry/resume is blocked, identity changed, or checkpoint continuation is uncertain."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Recover evidence, not a fresh experiment

## Prerequisites and input contract

Inputs: pinned project, exact original Job/Execution, failure, placement and explicit researcher
decision about compatible code changes. Names are labels, not identities.

## Procedure

1. Read saved Job and bounded structured failure. Request current provider health before
   declaring ownership lost; unknown/stale is not permission to duplicate work.
2. Obtain the native recovery preview. Tabulate reused/retried/pending logical Runs, latest
   Attempts, checkpoints and already-spent budgets.
3. Fixed/repeated designs use authoritative design/Run/Attempt evidence. Adaptive designs
   require valid controller state; missing state must not become a fresh experiment.
4. Check scientific identity: parameters, seeds, typed inputs, datasets, objective, design and
   original owned Execution. Code acknowledgement does not waive those checks.
5. Hand the exact plan to the researcher for native CLI/Console apply. Agents currently prepare
   recovery only; approved notes or cancellation requests cannot launch recovery.

Read [recovery cases and commands](references/procedure.md). Checkpoint presence is not proof
of epoch-level continuation: the Work must explicitly support compatible restoration.
Imported packages remain read-only; unsupported Fleet recovery must not replay on one host.

## Validation

Deliver an exact reuse/retry/pending plan, blockers and whether training restarts or restores.
Never refund failed Attempt cost, edit immutable records or retry consumer exceptions indefinitely.

