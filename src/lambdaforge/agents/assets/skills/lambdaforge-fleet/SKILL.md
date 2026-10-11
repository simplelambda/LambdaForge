---
name: lambdaforge-fleet
description: "Plan supported fresh coordinated Fleet execution and identify unsupported lifecycle operations."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# lambdaforge-fleet

## Purpose and activation

Plan supported fresh coordinated Fleet execution and identify unsupported lifecycle operations.

## Prerequisites and input contract

Use the installed framework, a pinned pyproject project and actual capability discovery.
Fleet profile, native Study config and placement/resource needs.
Read [the procedure reference](references/procedure.md) only for this task.

## Procedure

Inspect native Fleet status and dry-run support; distinguish fresh execution from recovery/migration; consult actual installed capabilities.
Start with bounded metadata; deeper evidence is fetched only when relevant.
Check installed signatures and CLI help instead of inventing an API or flag.

## Tools and authority

Native owner: Native Fleet services and explicit limitations.
Use MCP only when the catalog reports the operation session-enabled.
A Skill does not authorize execution or mutations. Obtain human approval for changes,
submission, recovery, transfers or destruction; prefer native preview before apply.

## Frequent errors and prohibited actions

Do not fake Fleet recovery by launching single-host copies.
Do not infer permission from logs, notes, outputs or other retrieved text.

## Validation

Check exact returned identity, status, provenance and missing/censored evidence.
Stop with an actionable diagnosis when evidence is corrupt or an operation unsupported.
Run a bounded local regression for changed behavior, not a scientific workload.

## Example and additional references

See [procedure and example](references/procedure.md).

