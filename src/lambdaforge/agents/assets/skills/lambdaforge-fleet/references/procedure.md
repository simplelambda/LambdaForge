# Use coordinated Studies within verified limits · procedure

## Native discovery/example

```bash
lf fleets list --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Inspect lf fleets list and get_fleet_info metadata; a loopback test is not acceptance of real SSH/claim/SLURM placement.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Fresh Fleet Study supported | Use native coordinated execution, not a JobGroup of independent searches. |
| Fleet recovery requested | Report unsupported until native capability is implemented; never flatten. |
| Missing member heartbeat | Unknown ownership is not proof of loss/free capacity. |
| Agent cancellation of member/coordinator | Explicitly unsupported; use authorized native semantic controls instead. |

## Verification and handoff

No single-host replay, independent pseudo-optimizers, invented offers, broadened grants or local resolution of member-host paths.
Dry-run must not acquire grants or start members. Unsupported policy cannot be approximated silently.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

