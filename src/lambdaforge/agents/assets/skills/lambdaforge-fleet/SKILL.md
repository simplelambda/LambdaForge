---
name: lambdaforge-fleet
description: "Plan supported fresh coordinated Fleet execution and identify unsupported lifecycle operations."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Use coordinated Studies within verified limits

## Prerequisites and input contract

Native Fleet definition, exact coordinator/member identities and fresh execution versus recovery intent.

## Procedure

1. Inspect get_fleet_info and native capabilities before authoring a multi-placement Study.
2. Keep one native optimizer/coordinator with exact member leases/fences, bounded streams and original budgets.
3. Distinguish supported fresh coordinated execution from unsupported migration/recovery/continuation/export; report limitations explicitly.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Dry-run must not acquire grants or start members. Unsupported policy cannot be approximated silently.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

No single-host replay, independent pseudo-optimizers, invented offers, broadened grants or local resolution of member-host paths.
Report concrete observations, exact identities, tests performed and remaining limitations.

