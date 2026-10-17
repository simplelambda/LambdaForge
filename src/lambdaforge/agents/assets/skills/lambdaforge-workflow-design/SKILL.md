---
name: lambdaforge-workflow-design
description: "Compose native Work steps and parallel levels with explicit output dependencies."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Compose native steps and one parallel level

## Prerequisites and input contract

Step Works, typed output contracts, dependency bindings and intended ordering.

## Procedure

1. Identify independently executable Work classes and the exact data passed between them.
2. Express sequence and one-level parallel composition in native WorkConfig; use explicit typed dependencies, not shell paths.
3. Validate the composition and dependency outputs with a tiny fixture; retain Work/Execution/Run/Attempt identities for each step.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Composition remains the existing execution model, not an alternate Task engine.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Dependencies bind declared outputs and compatible contracts; unsupported arbitrary DAGs or nested parallelism fail explicitly rather than being flattened.
Report concrete observations, exact identities, tests performed and remaining limitations.

