---
name: lambdaforge-metrics
description: "Record scientific metrics and define objective components without mixing incompatible checkpoints."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Record coherent metric evidence

## Prerequisites and input contract

Metric names/units, phase, epoch/step, objective direction and any authored guardrails.

## Procedure

1. Inspect installed self.metrics signatures and objective normalization; keep the raw metric identity stable even with display names.
2. Record metrics at explicit comparable points, with timing separate from scientific scores.
3. Ensure each utility component refers to the same accepted checkpoint/evidence point; test missing/nonfinite and class-support cases.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

The consumer owns metric computation; one authored native utility owns HPO decisions.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

No cross-epoch synthetic optimum, test-set optimization or guessed multi-metric weights. Missing utility stays missing, not zero or success.
Report concrete observations, exact identities, tests performed and remaining limitations.

