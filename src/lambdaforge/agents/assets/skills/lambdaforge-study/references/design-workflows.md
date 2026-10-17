# Choose the evidence before scheduling

Use the same importable consumer `Work` for all designs. The following complete toy fixture
belongs in `src/my_project/__init__.py`; it demonstrates design contracts, not real training.

```python
from lambdaforge import Work


class Training(Work):
    def run(self, strength: float, epochs: int = 50) -> dict[str, float]:
        score = 1.0 / (1.0 + strength)
        for epoch in range(epochs):
            self.metrics.log("score", score, step=epoch)
        return {"score": score}
```

Use this YAML at `experiments/study.yaml`. Replace the toy computation with the project's
science; changing a result after a completed Study is not a compatible-code acknowledgement.

```yaml
name: regularization
run: my_project.Training
with:
  epochs: 50
seeds: [11, 23, 47]
sweep:
  space:
    strength: [0.0, 0.1, 0.3]
objective: {metric: score, mode: max}
execution: {max_parallel: 3}
resources: {cpu: 1}
```

- Work: one scientific computation. No search or sweep is required.
- Repeated seeds: remove sweep and supply one fixed `with.strength`; estimate variation for
  that candidate. This does not measure coverage of a parameter space.
- Sweep: finite authored combinations × required seeds. Use when completeness of these cells
  is the question; a concurrency ceiling does not reduce required evidence.
- Adaptive HPO: replace `sweep` with `search: {space: {strength: {range: [0.0, 1.0]}}}`.
  Sampling/pruning and replication are native policies; do not simulate them with a shell loop.
- Exploratory analysis: selection after seeing evidence does not become a confirmatory test.
  Follow the native evidence/confirmation contract rather than selecting a survivor-only mean.

Authoring and inspection:

```bash
lf config schema --output work.schema.json
lf config explain experiments/study.yaml --field execution.max_parallel
lf validate experiments/study.yaml
lf explain experiments/study.yaml
lf config resolve experiments/study.yaml --format json
lf config diff experiments/old.yaml experiments/study.yaml --json
lf inspect execution-EXACT --kind execution --json
lf diagnose job-EXACT --json
lf retry job-EXACT --dry-run --json
```

Static schema guidance does not import consumer code. Validate/explain/resolve/diff do, so obtain
trust for project imports. A diff reports semantic changes; it is not proof that recovery can
reuse results or that file bytes are identical. `diagnose` does not infer safe recovery from
FAILED/UNKNOWN; use the exact native retry preview. Training checkpoint continuation additionally
requires explicit consumer support.

Incremental fixed Study extensions are a separate native capability, not retry:

```bash
lf studies extend execution-EXACT --config experiments/expanded.yaml --dry-run --json
```

Inspect the real plan's `apply_supported`, target, evidence/code/runtime proof and budget.
Supported fixed designs run only new cells with `--apply --plan-hash HASH`; preview never launches.
The Console's Extend Study dialog preselects the edited original YAML and opens the cumulative
series after confirmation. Local CPU and same-placement retained native managed CPU/GPU contracts
are supported; absent/mixed provenance is not a compatibility waiver. Subsequent revisions use
the Series ID and full accumulated design, not a delta replayed as another origin. Inspect with
`studies show/history/refresh/analyze/report`; `studies export` preserves a read-only collection.
Failed deltas use native Job retry, retaining earlier revisions and spent Attempt cost. A duplicate
accepted apply returns its existing Job. Adaptive/Fleet/migration/automatic-replication extensions
remain excluded. Consult the installed extension guide and obtain explicit submission authority
immediately before apply; neither a plan nor this Skill authorizes computation.

When results disagree, report exact candidate/Run/Attempt identities, metric, split, fidelity,
seed counts and missing/censored cells. Do not silently replace the primary metric, add
guardrails from incidental metrics or interpret marginal effects as causal responsibility.
