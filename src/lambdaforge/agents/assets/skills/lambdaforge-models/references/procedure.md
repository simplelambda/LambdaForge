# Find a component before inventing a model family · procedure

## Native discovery/example

```bash
lf ai api
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Inspect get_model_contract('MLP') before importing the optional backend; use its returned declaration rather than a guessed constructor.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Listing imports Torch/CUDA | Use installed static catalog discovery instead. |
| Many similar model aliases | Prefer the existing component whose contract fits. |
| Generic GNN factory requested | Do not introduce recursive construction or a redundant model family. |
| Dtype mismatch in consumer forward | Verify the actual boundary; do not blame scheduler or treat it as OOM. |

## Verification and handoff

Test declared shapes, masked/empty cases, dtype/device preservation and gradients; do not claim unsupported tensor contracts.
Static discovery does not execute models. A real forward/backward test needs the appropriate dependency and local fixture.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

