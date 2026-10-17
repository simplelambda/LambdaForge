---
name: lambdaforge-clusters
description: "Configure or diagnose provider profiles, SSH/SLURM/GPU access with explicit operator authority."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Inspect providers and edit safe native profiles

## Prerequisites and input contract

Pinned project, exact registered profile, intended transport/scheduler/GPU policy and operator authority.

## Procedure

1. List/get native cluster metadata; distinguish Direct/SLURM scheduling from command/shared/exclusive GPU access.
2. When authorized, request bounded doctor/resources; separate fresh physical observations from provider-owned opaque grants.
3. Use native guided setup/edit for durable policy changes, retaining credentials as references and explicit project mirror/dataset roots.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Profile edits, bootstrap, claims, release and credentials require explicit operator consent; never place passwords in notes/argv.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

No broadened CUDA tokens, system Python/CUDA mutation or guessed host identity. All configured fields round-trip through the native profile contract.
Report concrete observations, exact identities, tests performed and remaining limitations.

