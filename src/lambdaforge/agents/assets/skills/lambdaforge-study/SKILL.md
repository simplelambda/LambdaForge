---
name: lambdaforge-study
description: "Design adaptive HPO, repeated-seed studies or exhaustive sweeps, and review compatible fixed Study extensions with explicit evidence requirements."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Choose design, replication and scientific evidence

## Prerequisites and input contract

Parameter domain, metric/units, authored objectives, seeds and physical time/Run budgets.

## Procedure

1. Choose repeated-seed, fixed sweep or adaptive search according to the research question, not available GPU count.
2. Resolve native StudyDesign/EvidencePlan and distinguish candidate ceiling from required seed/confirmation/support obligations.
3. Inspect utility/guardrails, conditional domains, resource ceilings and stopping semantics before authorizing any computation.

Read [the domain procedure and cases](references/procedure.md) when this task applies.
For complete authoring examples, design selection and the current extension/recovery boundary,
read [design and inspection workflows](references/design-workflows.md).

## Tools and authority

A proposal or confidence label does not authorize launching Runs or changing the scientific objective.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

All required planned evidence is accounted for, failed cost is not refunded, and incomplete/pruned evidence is not a final survivor mean.
Report concrete observations, exact identities, tests performed and remaining limitations.

