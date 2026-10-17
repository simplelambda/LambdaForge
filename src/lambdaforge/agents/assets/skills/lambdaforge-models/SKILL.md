---
name: lambdaforge-models
description: "Find or design neural components using LambdaForge tensor contracts rather than inventing a redundant model family."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Find a component before inventing a model family

## Prerequisites and input contract

Input/output tensor semantics, dimensions, dtype/device, architecture goal and installed exports.

## Procedure

1. Query list_model_catalog/get_model_contract statically; read actual constructor/forward signatures and declared tensor contracts.
2. Compare available primitives to the scientific need; keep absent semantics explicit rather than inferred from a class name.
3. Implement only the missing component with focused shape/dtype/device tests and a tiny forward/backward fixture when authorized.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Static discovery does not execute models. A real forward/backward test needs the appropriate dependency and local fixture.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Test declared shapes, masked/empty cases, dtype/device preservation and gradients; do not claim unsupported tensor contracts.
Report concrete observations, exact identities, tests performed and remaining limitations.

