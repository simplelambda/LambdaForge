# Recovery procedure and cases

```bash
lf jobs show EXACT_JOB --json
lf jobs logs EXACT_JOB --tail 80
lf health EXACT_JOB --refresh --json
lf retry EXACT_JOB --dry-run --json
```

Placeholders are exact project-owned IDs. Health/log/retry inspection may query the selected
provider; read-only science inspection is not necessarily offline. Facade tools are `get_job`,
`get_work_logs`, `get_execution_health`, `prepare_recovery` and `get_recovery_history`;
check installed signatures and availability. Native recovery validation imports project Python,
so `prepare_recovery` needs separate trusted project-code consent.

| Case | Correct recovery |
|---|---|
| Ten repeated seeds: nine succeeded, one OOM | Reuse nine exact Runs; new Attempt for the failed logical Run only. |
| Fixed sweep partially interrupted | Preserve successful points; retry incomplete owned Runs. |
| Valid adaptive controller | Restore original decisions, pending work and spent budgets. |
| Adaptive controller missing/corrupt | Diagnose; never fabricate state or restart silently. |
| Consumer bug fixed | Explicit compatible-code acknowledgement plus unchanged science identity. |
| Owner possibly alive | Fresh native ownership checks block duplication. |
| Imported evidence / unsupported Fleet | Read-only or unsupported, never single-host replay. |

OOM is failed resource evidence, not an objective or permission for endless retry. Recovery
accounts for physical costs. Compatible checkpoints reduce lost training only when the Work
supports restoration. Otherwise restart the failed Run while preserving history and siblings.

Human apply uses installed `lf retry --help` options or Console **Resume Study**. Inspect any
required code acknowledgement and validation feedback. Cancellation approval is not recovery.

Acceptance: original Execution/identities retained, successful evidence not duplicated, new
Attempts linked, aggregate state based on latest valid outcomes, cost/history preserved.
If blocked, report exact missing evidence/ownership uncertainty and next safe read—not a
hidden-directory edit, kill command or complete rerun.

