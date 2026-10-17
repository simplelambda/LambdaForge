---
name: lambdaforge-datasets
description: "Consume, verify, version, reconstruct or replicate immutable managed datasets through native services."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Respect immutable content and explicit reconstruction

## Prerequisites and input contract

Dataset name/version, exact content ID, known placements and whether the goal is reuse, reproduce or publish.

## Procedure

1. Inspect metadata and all registered placements; identify local/remote content conflicts without selecting a winner automatically.
2. Run native preflight before costly reconstruction; reuse/materialize exact published bytes when that is the research intent.
3. For reproducibility use an explicit project verifier and variable-specific tolerances; preview authorized replication/reconciliation/deletion through native services.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Replication/reconciliation/destruction need explicit authority; scientific equivalence never waives checksums.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Transport hashes remain exact. Scientific equivalence is separate, explicit and never permission to relabel/replace immutable bytes.
Report concrete observations, exact identities, tests performed and remaining limitations.

