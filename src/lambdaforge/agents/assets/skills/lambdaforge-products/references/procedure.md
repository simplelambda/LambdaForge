# Publish and consume sealed contracted products · procedure

## Native discovery/example

```bash
lf products list --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Inspect the product publication/consumption in packaged work.py; compare native contract and provenance metadata before requesting artifact transfer.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Guess checkpoint filename from output directory | Inspect the declared member; do not fabricate a product. |
| Publication failed after expensive Run | Preserve outputs and use native publication retry. |
| Best surviving seed only | Do not promote a survivor-only mean over an incomplete planned block. |
| Same product label, different bytes | Use exact content identity and explicit policy, never overwrite silently. |

## Verification and handoff

Exact checksums, contract/member completeness, producer provenance and consumption survive. No mutable path or incomplete confirmation is promoted as a selected product.
Product publication/transfer requires explicit operator authority; agent read capabilities do not imply mutation support.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

