# Compose native steps and one parallel level · procedure

## Native discovery/example

```bash
lf validate experiments/sequence.yaml
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Inspect the native sequence/parallel recipes in the installed catalog and resolve the authored YAML before running a disposable fixture.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Step consumes prior output | Use a declared typed dependency, not a guessed file path. |
| Parallel sibling fails | Preserve healthy sibling evidence under native failure semantics. |
| Need an arbitrary DAG | Report unsupported; do not add another runner. |
| Required constructor object graph | Reject; each executable target is a Work.run class. |

## Verification and handoff

Dependencies bind declared outputs and compatible contracts; unsupported arbitrary DAGs or nested parallelism fail explicitly rather than being flattened.
Composition remains the existing execution model, not an alternate Task engine.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

