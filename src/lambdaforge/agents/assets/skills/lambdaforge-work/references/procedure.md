# Author the single native Work lifecycle · procedure

## Native discovery/example

```bash
lf validate experiments/work.yaml
```

Run in the intended project. Uppercase selectors/paths are placeholders, not a Work-name guess.
Inspect installed help first. These commands are metadata/planning; parsing them does not execute science.

Use lf ai skills examples to inspect the packaged work.py/work.yaml; adapt the scientific run signature, not framework-owned directory layouts.

Authoring route (no scientific execution):

```bash
lf help --search "validate experiment"
lf config schema --output work.schema.json
lf config explain experiments/work.yaml --field execution.max_parallel
lf validate experiments/work.yaml
lf explain experiments/work.yaml
lf run experiments/work.yaml --dry-run
```

The schema knows infrastructure fields, not arbitrary Work.run parameter truth. Use the actual
signature for `with` arguments. A plain string is never a typed input declaration. Static
field help avoids project imports; native validate/explain/dry-run require trusted code imports.
Use `lf init DIRECTORY --template minimal` for a new installable consumer; its distribution,
package/import names and YAML are derived from DIRECTORY. Existing projects are not overwritten
by default. Do not scaffold over scientific records to fix a module import.

## Decision cases

| Situation | Native interpretation / next step |
|---|---|
| Plain string resembles a path | Do not infer a typed file input; declare the type explicitly. |
| Output publication fails | Preserve outputs/checkpoints and retry publication, not science. |
| Work constructor requires arguments | Move scientific parameters to run; do not build recursive objects. |
| Test wants a real expensive dataset | Use a small isolated fixture instead. |

## Verification and handoff

Reject unknown/missing arguments, constructor parameters, free-function targets and non-JSON results. Assert durable output/provenance using the native ResultStore.
Authoring permission does not grant submission. Native validation imports project Python and needs explicit trust.

If a capability is missing, stale or unsupported, retain evidence and explain the exact blocker.
Do not create a CLI subprocess adapter, a second runner/result store or an application-specific workaround.

