---
name: lambdaforge-testing
description: "Create bounded deterministic tests for Work signatures, managed evidence and project models."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Validate the complete native behavior with small fixtures

## Prerequisites and input contract

Changed contract, allowed dependencies, disposable project root and expected evidence/identity.

## Procedure

1. Test actual Work signatures/YAML normalization before execution; invalid cases fail before submission.
2. Use tiny deterministic CPU fixtures for metrics/checkpoints/outputs/recovery and exact identity assertions; isolate all environment/provider state.
3. Test public CLI/facade/protocol parity when those boundaries changed, then package/import checks outside the source checkout.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Use owned temporary fixtures, not real study/dataset records; record tests actually executed and explicit skips.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Observe real native records and failures, not only mocked widget text. No live cluster/scientific workload or automatic installation is a test prerequisite.
Report concrete observations, exact identities, tests performed and remaining limitations.

