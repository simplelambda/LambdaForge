# Interpret native Study evidence honestly · procedure

## Native discovery/example

```bash
lf results list --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Use get_study_panel(job_id, view='analysis') explicitly after listing the Study, then inspect only one relevant Trial/Run.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Two candidates only | Show limited descriptive data without strong inferential claims. |
| Parameter active only under another value | Interpret conditional/joint support, not global marginal importance alone. |
| Pruned run has good intermediate score | Mark censored and compare fidelity; preserve display data without treating it as final. |
| Analysis fails | Report the native diagnostic; do not recompute a speculative replacement client-side. |

## Verification and handoff

Coverage/effect magnitude is not scientific confidence or causality; partial evidence never becomes a completed-seed mean.
An analysis recommendation is not approval to change the objective or launch science.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

