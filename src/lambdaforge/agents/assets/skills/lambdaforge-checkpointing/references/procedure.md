# Restore explicit compatible Run state · procedure

## Native discovery/example

```bash
lf retry EXACT_JOB --dry-run --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Inspect the checkpoint use in packaged work.py and the native retry dry-run before deciding whether application restoration is implemented.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Checkpoint exists but Work does not restore it | Restart only the failed logical Run; do not promise epoch continuation. |
| Model or data identity changes | Reject incompatible state; code acknowledgement does not waive science checks. |
| Resumable map dependency state includes machine paths | Use content/scientific identity, never pickle or host paths. |
| Adaptive controller missing | Refuse silent restart; application checkpoint is not controller state. |

## Verification and handoff

A checkpoint reference alone never proves epoch-level resume. Incompatibility is explicit; failed Attempt history and original Execution remain intact.
Do not open untrusted pickle/model bytes as metadata; continuation must be an explicit consumer capability.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

