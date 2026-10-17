---
name: lambdaforge-analysis
description: "Interpret result/Study coverage, parameter effects and uncertainty without confusing censored evidence with final scores."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Interpret native Study evidence honestly

## Prerequisites and input contract

Exact Study/Execution, scientific question, metric/fidelity, completed and censored support.

## Procedure

1. Load the selected native analysis panel lazily; do not fit a second HPO model in the adapter.
2. Separate marginal description, joint prediction, formal comparison, coverage and uncertainty.
3. Inspect missing/pruned/confirmation/seed-noise evidence before recommending the next question or graph.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

An analysis recommendation is not approval to change the objective or launch science.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Coverage/effect magnitude is not scientific confidence or causality; partial evidence never becomes a completed-seed mean.
Report concrete observations, exact identities, tests performed and remaining limitations.

