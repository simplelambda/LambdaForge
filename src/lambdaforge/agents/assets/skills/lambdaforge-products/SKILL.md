---
name: lambdaforge-products
description: "Declare contracted products from durable outputs or consume a sealed product with exact scientific expectations."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Publish and consume sealed contracted products

## Prerequisites and input contract

Declared product contract, output members, exact producer evidence and intended consumer expectations.

## Procedure

1. Inspect list_products/get_product/get_product_evidence metadata before opening any asset bytes.
2. Declare complete contracted durable members and promote only from accepted native evidence; verify content identity separately from display name.
3. Bind ProductInput to exact sealed content and record consumption; retry publication failures without recalculating science.

Read [the domain procedure and cases](references/procedure.md) when this task applies.

## Tools and authority

Product publication/transfer requires explicit operator authority; agent read capabilities do not imply mutation support.
Installed signatures/capability availability remain authoritative; a Skill grants no permission.

## Validation

Exact checksums, contract/member completeness, producer provenance and consumption survive. No mutable path or incomplete confirmation is promoted as a selected product.
Report concrete observations, exact identities, tests performed and remaining limitations.

