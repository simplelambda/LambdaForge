---
name: lambdaforge-work
description: "Create or modify a LambdaForge Work class or its YAML arguments and managed outputs."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Author the single native Work lifecycle

## Prerequisites and input contract

Scientific inputs/outputs, trusted consumer source, installed Work signature and intended YAML.

## Procedure

1. Implement an importable subclass of lambdaforge.Work with Work.run(...); keep scientific arguments in that signature and docstring.
2. Declare typed inputs explicitly and use self.metrics, self.checkpoints and self.outputs for managed evidence; return strict JSON.
3. Validate YAML against the actual signature, resolve native policy and test a tiny disposable CPU fixture before any authorized submission.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Authoring permission does not grant submission. Native validation imports project Python and needs explicit trust.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Reject unknown/missing arguments, constructor parameters, free-function targets and non-JSON results. Assert durable output/provenance using the native ResultStore.
Report concrete observations, exact identities, tests performed and remaining limitations.

