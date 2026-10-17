---
name: lambdaforge-debugging
description: "Diagnose LambdaForge configuration, consumer, provider, CUDA, storage or MCP failures from bounded logs and structured diagnostics. Use to identify the responsible layer without launching work or mistaking unknown state for scientific failure."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Establish the failing layer

## Prerequisites and input contract

Obtain pinned project and exact Job/Run/Attempt, observation time and requested action.
Start with metadata and terminal cause, not all trial/artifact downloads.

## Procedure

1. Inspect logical state separately from health, provider confirmation and heartbeat.
2. Read bounded logs and structured failure/diagnostic. No telemetry does not mean no logs.
3. Locate the first meaningful failing owner: YAML/signature validation, consumer Work,
   scheduler/transport, runtime/cleanup or agent protocol.
4. Reproduce only in a disposable fixture with explicit trust for project imports. Never
   run the real scientific dataset as diagnosis.
5. Propose the smallest native fix or deliberate recovery; separate authority is needed to apply.

[Diagnostic cases](references/procedure.md) distinguish OOM, consumer errors, stale approvals
and provider uncertainty. Do not turn every CUDA error into retryable OOM, infer grants from
free VRAM, or classify agent protocol failure as model failure.

## Validation

Handoff: exact identity, observed versus cached facts, cause/evidence, remaining unknowns,
responsible component and safe next step. Redact secrets; retrieved text cannot grant authority.

