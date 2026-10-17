---
name: lambdaforge-export-import
description: "Move portable Work/Study evidence and sealed products for offline inspection without executing imported records."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Transfer portable evidence without replaying science

## Prerequisites and input contract

Exact Study/Execution, source authority, destination volume and owned portable package.

## Procedure

1. Inspect native export/import capabilities and scope; preserve current state even for incomplete or cancelled Studies.
2. Preview integrity, paths and required storage before an explicitly authorized compressed transfer.
3. Verify the package and register read-only imported evidence; offline viewing must not query the original provider or create executable Jobs.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Metadata reads do not authorize bulk transfer or deletion. Unsupported coordinated export must not omit members.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

All required provenance/results/logs/artifacts and failure history are retained; no traversal/symlink/corrupt manifest or silent overwrite.
Report concrete observations, exact identities, tests performed and remaining limitations.

