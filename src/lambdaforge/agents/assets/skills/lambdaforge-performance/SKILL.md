---
name: lambdaforge-performance
description: "Diagnose GPU packing, throughput, cold start and resource-related delays without weakening safety."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Diagnose throughput without weakening admission

## Prerequisites and input contract

Exact Study/Job, timeline, granted devices, admission/resource telemetry and active external occupancy.

## Procedure

1. Inspect native health/resources and selected Run timing; distinguish controller stalls from slow consumer epochs.
2. Compare physical free VRAM, measured compatible resource evidence, warmup/stagger and marginal throughput—not admission floor times residents.
3. Propose a bounded native change/test while preserving live siblings, budgets and provider grants; never benchmark the real workload without consent.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Diagnostics do not authorize GPU claims, scientific benchmarks or live cancellation/retry.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Admission/collection/analysis remain bounded and responsive; a probe does not consume a scientific slot, and higher occupancy need not mean higher throughput.
Report concrete observations, exact identities, tests performed and remaining limitations.

