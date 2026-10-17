# Orient without executing the project · procedure

## Native discovery/example

```bash
lf ai status
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Start with metadata from an unfamiliar project, then inspect one YAML before requesting permission to validate it.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Changing cwd | Keep the original pinned project, not the nearest new directory. |
| Same Work name on two clusters | Use exact Job/Execution identity and placement, never a name key. |
| Nested project or secret path | Refuse traversal; request the intended root instead. |
| Consumer modules print or allocate GPU | Do not import them for metadata discovery. |

## Verification and handoff

Return relevant Work/YAML paths, installed version, exact historical selectors and gaps. Merely finding a class does not validate it.
Discovery authorizes neither project writes nor scientific execution.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

