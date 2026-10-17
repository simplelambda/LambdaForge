# Diagnose throughput without weakening admission · procedure

## Native discovery/example

```bash
lf health EXACT_JOB --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Start with lf health and the selected resources panel, compare epoch/wall-time and admission reasons before changing packing policy.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| 20GiB safety floor and light residents | Floor is a start threshold, not an assumed per-resident allocation. |
| Controller busy while workers idle | Inspect bounded collection/analysis liveness, not only CUDA utilization. |
| GPU disappeared from grant | Stop only owned workers on revoked tokens; do not use newly reported devices. |
| External process causes OOM | Attribute failed Attempt resource evidence; healthy siblings survive. |

## Verification and handoff

Admission/collection/analysis remain bounded and responsive; a probe does not consume a scientific slot, and higher occupancy need not mean higher throughput.
Diagnostics do not authorize GPU claims, scientific benchmarks or live cancellation/retry.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

