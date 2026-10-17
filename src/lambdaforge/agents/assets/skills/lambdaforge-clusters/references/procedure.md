# Inspect providers and edit safe native profiles · procedure

## Native discovery/example

```bash
lf clusters list --json
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Use list_clusters/get_cluster for cached profile inspection, then request get_cluster_resources only for the selected host.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Site uses gpu exec | Direct backend plus command GPU access; never invent --gpu-slot. |
| Provider exposes fewer GPUs than YAML asks | Use only valid inherited grants under supported scaling rules. |
| More free VRAM exists on an ungranted GPU | Do not use it. |
| Long provisioning | Show native action progress, avoid duplicate bootstrap requests. |

## Verification and handoff

No broadened CUDA tokens, system Python/CUDA mutation or guessed host identity. All configured fields round-trip through the native profile contract.
Profile edits, bootstrap, claims, release and credentials require explicit operator consent; never place passwords in notes/argv.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

