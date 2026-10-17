---
name: lambdaforge-checkpointing
description: "Add safe resumable state or diagnose whether recovery can continue a training checkpoint."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Restore explicit compatible Run state

## Prerequisites and input contract

Exact Run/Attempt, consumer checkpoint contract, immutable inputs/parameters and recovery intent.

## Procedure

1. Separate Study-state recovery from training-state restoration; identify which lifecycle owns each.
2. Declare resumable state through managed checkpoints; use serializable dependency state and explicit consumer restoration of model/optimizer/epoch/RNG when supported.
3. Test an interrupted tiny Run and compatible restoration, then inspect native recovery without resetting successful evidence or spent budgets.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Do not open untrusted pickle/model bytes as metadata; continuation must be an explicit consumer capability.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

A checkpoint reference alone never proves epoch-level resume. Incompatibility is explicit; failed Attempt history and original Execution remain intact.
Report concrete observations, exact identities, tests performed and remaining limitations.

