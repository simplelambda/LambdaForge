# lambdaforge-checkpointing procedure reference

## Native entry point

```bash
lf retry EXACT_JOB --dry-run --json
```

Run this in the intended project. Uppercase selectors/paths are placeholders:
replace them with an exact owned identifier, not a historical display-name guess.
Queries may contact a provider only when explicitly requested. Do not run mutations
or science as a discovery step. Read each command's installed `--help` first.

## Acceptance

Use checkpoint file/save_json/load_json; distinguish durable acknowledgement from a checkpoint request; implement explicit Work resuming behavior.
Do not claim epoch continuation when only study recovery is implemented.

The shared public-API examples installed with these Skills are in
`lf ai skills examples`. The tiny Work demonstrates metrics, JSON checkpoints,
managed outputs and a self-contained HTML section. The YAML fixtures demonstrate
native repeated and adaptive studies. Native docs remain authoritative for product
contracts, exact historical bindings and provider-specific operations.

## Recovery of uncertainty

If the installed operation is absent, report unsupported. Do not synthesize a shell
workaround or a second result store. Preserve evidence and ask for the missing
researcher decision. Distinguish observed behavior from hypotheses in the final report.

