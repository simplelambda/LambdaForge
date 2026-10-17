# Diagnostic procedure

```bash
lf jobs show EXACT_JOB --json
lf jobs logs EXACT_JOB --tail 80
lf health EXACT_JOB --json
lf health EXACT_JOB --refresh --json
```

The last read deliberately contacts the selected provider. Logs must retain the terminal
cause even when stdout lacks a traceback. Request detailed traceback only when needed;
never copy credentials, arbitrary record bodies or entire artifact directories.

| Evidence | Interpretation / next step |
|---|---|
| Typed torch OOM or scheduler OUT_OF_MEMORY | Resource failure; inspect exact Attempt and occupancy. Not success or objective. |
| Consumer bf16/float32 mismatch | Application contract error unless native code proves transformation; not OOM retry. |
| Provider unreachable / UNKNOWN | Observation uncertainty, not lost processes; fresh ownership evidence required. |
| No telemetry with failed Job | Read native logs/failure; do not automatically call it preparing. |
| Recovery identity mismatch | Compare normalized science and seed policy; code acknowledgement is insufficient. |
| MCP isError / INTERNAL_ERROR | Protocol/adapter failure; scientific completion is not established. |
| PLAN_STALE / OPERATION_CONFLICT | Re-preview or inspect claimed outcome; never replay from retryability alone. |

Facade reads include `get_job`, `get_execution_health`, `get_work_logs`, `get_study_run` and
selected `get_study_panel`; check installed signatures. Metadata discovery must not import
consumer code or allocate GPUs. Use native diagnostics, not another exception classifier.

If LambdaForge is responsible, add a small native behavioral regression. If the consumer is
responsible, report its location/contract without modifying it under a framework-only task.
Preserve scientific identity, healthy siblings and immutable failure history.

