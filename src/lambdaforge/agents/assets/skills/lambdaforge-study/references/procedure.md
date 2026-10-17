# Choose design, replication and scientific evidence · procedure

## Native discovery/example

```bash
lf config resolve experiments/study.yaml --format json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Compare the packaged repeated.yaml and study.yaml using native config resolve, then explain which evidence is required to select a candidate.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Want all combinations | Use an explicit fixed design rather than silently disabling selected optimizer features. |
| Early score is strong but ancillary metrics poor | Only authored utility/guardrails influence decisions; do not invent weights. |
| Pruned candidate has good best epoch | Inspect comparable fidelity/evidence, not just the best scalar. |
| Resource slots idle | Inspect scientific feasibility/admission and physical budgets separately. |

## Verification and handoff

All required planned evidence is accounted for, failed cost is not refunded, and incomplete/pruned evidence is not a final survivor mean.
A proposal or confidence label does not authorize launching Runs or changing the scientific objective.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

