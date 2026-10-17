# Respect immutable content and explicit reconstruction · procedure

## Native discovery/example

```bash
lf datasets list --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Use get_dataset/get_dataset_placements/get_dataset_members with exact content identity. Listing metadata must not scan NPZ/model/asset bytes.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Same name/version has three hashes | Diagnose authority conflicts and require an exact chosen content/placement. |
| Operational metadata differs inside NPZ | The project must declare its scientific contract; LF does not strip arbitrary fields. |
| Eigenvector sign/base changes | Use appropriate explicit invariants, not indiscriminate rounding. |
| Placement folder is gone | Native index removal differs from owned-content deletion; preview exact scope. |

## Verification and handoff

Transport hashes remain exact. Scientific equivalence is separate, explicit and never permission to relabel/replace immutable bytes.
Replication/reconciliation/destruction need explicit authority; scientific equivalence never waives checksums.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

