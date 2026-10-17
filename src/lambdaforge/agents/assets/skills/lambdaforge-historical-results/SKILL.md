---
name: lambdaforge-historical-results
description: "Read an exact prior Execution or use ResultInput/ProductInput as a typed dependency."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Use exact historical evidence read-only

## Prerequisites and input contract

Original Execution/content IDs, relevant Run/Attempt and expected result/product contract.

## Procedure

1. Page the native result catalog; select exact Execution rather than display name.
2. Fetch only the required summary/Run/curves/artifact preview through public result queries.
3. Use ResultInput for exact historical Execution evidence or ProductInput for sealed contracted content; retain explicit consumption/provenance.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Metadata selection does not authorize training, heavy artifact downloads or mutable result aliases.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Imported packages remain read-only, references are exact and failed/pruned/missing evidence remains distinguished.
Report concrete observations, exact identities, tests performed and remaining limitations.

