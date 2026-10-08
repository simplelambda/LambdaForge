# Scientific correctness audit

[Español](SCIENTIFIC_AUDIT.es.md)

Baseline: `c8f2d2e`, LambdaForge 0.17.0. This closes the ordered-evidence and
conclusion-consistency corrections below, **not the entire requested engineering audit**.
No WISDOM code, real cluster, historical scientific manifest or GitHub CI is changed.

## Confirmed defects and corrections

| Classification | Authority / previous behavior | Correction and evidence |
| --- | --- | --- |
| CONFIRMED_BUG | `SequentialSweep.evaluate` sorted the intersection of numeric seed IDs; newly completed blocks could reorder previous weighted observations. | Requires explicit persisted `seed_order`, consumes only its complete prefix, records exact inference seeds. Tests use the real affine ProjectSeedStream, reversed worker arrivals and missing early cells. |
| CONFIRMED_BUG | `runner._execute_fixed_evidence_group` initialized `committed_lookahead=None` even during recovery. | Restores the existing `sweep-blocks.json` commitment through shared read-only validation; a partially completed speculative block prevents opening another. An integration test preserves original Run records, manifest and physical spend across interruption. |
| CONFIRMED_BUG | Block telemetry reported min/max completed ordinals even across holes. | Reports the complete contiguous prefix only. Initial block inventory is durable before dispatch, including interruption before the first terminal result. |
| CONFIRMED_BUG | `WorkRunner._summary` counted only initial authored obligations, potentially calling a sweep succeeded with a committed lookahead cell missing. | Recovery and native finalization share `recovery.fixed_requirements`: created blocks remain required without rewriting the immutable design. Integration preserves three valid Runs and exposes the fourth as pending, rather than restarting them or hiding debt. |
| CONFIRMED_BUG | ScientificDesign changed unsupported predictions to `UNRESOLVED` while retaining another hypothesis's stability. | Unsupported descriptive conclusion has zero realization support; predictive conclusion/stability and missing-support reason remain separate. No coverage is relabelled confidence. |
| CONFIRMED_BUG | `_exact_conclusion` and sweep post-hoc analysis assigned `1-modal_probability` to an unresolved event; empty distributions claimed stability one. | Uses the actual unresolved token mass, or zero without realizations. Weak numerical leaders no longer rewrite observed equivalence realizations into an unrelated winner probability product. |
| CONFIRMED_BUG | Study Analysis treated any sequential `stop=true`, including `INCOMPLETE`, as formal resolution. | Shared formal-state reader requires a scientific decision; operational stopping remains unresolved. Absence of a sequential procedure is `None`, not formal approval. |

The remote retry reader embeds the same pure block validator from its authoritative source, like
the existing recovery/seed readers. A regression executes it with isolated Python (`-I -S`),
without an installed LambdaForge runtime; it does not claim real SSH acceptance.

`ScientificQuestionState` now exposes descriptive and formal resolution separately. Its legacy
`resolved` accessor retains exploratory descriptive applicability when no formal procedure exists;
it must not be read as a formal confidence guarantee.

## Statistical contract

The estimator formula remains the predictable plug-in empirical-Bernstein sequence from
[Waudby-Smith and Ramdas, Theorem 2 and equation 15](https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf).
Its weights depend only on the preceding observations. Paired differences are bounded and the
theorem assumes a common conditional mean; deterministic seed generation alone does not prove
that scientific assumption. Primary-family Bonferroni and authored practical margins are unchanged.
Monte Carlo tests check implementation regressions, not the theorem or real-model calibration.

## Compatibility and ownership

- Runtime inference is `paired-pm-eb-cs-v2`, with `committed-acquisition-prefix-v1` and exact
  `evidence_seeds`. Historical v1 records remain inspectable, never rewritten or relabelled v2.
- Immutable historical `StudyDesign.replication_policy` declarations remain unchanged; the actual
  evaluation record owns its runtime policy version. No Run/Attempt/scientific IDs change.
- Block inventory remains version 1: it already contains ordinals and commitment. Recovery
  rejects duplicate/discontinuous ordinals, conflicting commitments and inconsistent seed metadata.
  Initial coordinates come from `seed_source.resolved`, not guessed numeric order. Missing
  authoritative coordinates fail before recovery writes. Read previews remain read-only.
- Scientific-question snapshots are version 4; parameter conclusion semantics are version 3.
  Existing `confidence` aliases mean descriptive stability of the exact displayed event, never
  probability of truth or formal coverage. Old persisted analyses are not silently recomputed.
- Explicit finite seeds keep authored ordinal order; fixed sweeps do not gain adaptive pruning.
  Retry/continuation remain new Attempts of the same logical Run. Imported evidence remains inert.

## Reproducible verification

```bash
pytest -q tests/hpo tests/work tests/analysis tests/products tests/tui
python benchmarks/sequential_sweep_audit.py
python benchmarks/hpo_scientific_design.py
python benchmarks/resource_scheduling.py
ruff check .
mypy src/lambdaforge
```

The audit benchmark peeks at every prefix of 200 paths, up to 96 blocks, with three reference
comparisons in degenerate, low-variance and high-variance null scenarios. The resource benchmark
is an existing synthetic regression, not real-GPU validation or a new scheduler improvement.
The existing scientific-design benchmark covers only three small landscapes; it is insufficient
to claim full eight-scenario scientific acceptance.

Local results for this change:

- Broad retained HPO/Work/Analysis/products/Console suites: **718 passed, 628.84 s**, before the
  final committed-debt and isolated-reader additions; then the final affected backend suites:
  **144 passed, 43.70 s**. This is not a claim that full repository pytest was rerun.
- Ruff, mypy (**581 source files**) and whitespace checks passed.
- Installed wheel outside the checkout: packaging/import/assets smoke, CLI help/scaffold/validation,
  native spawned CPU automatic sweep with four Runs and read-only recovery preview passed.
  Dependencies were reused in a temporary prefix; no user project environment was replaced.
- Calibration: **0/200** family violations in each of three null scenarios, 96 possible looks.
  The three-landscape benchmark retained mean regret approximately **3.70e-17** for both policies.
  Existing synthetic resource benchmark: adaptive makespan **58 s**, fixed **109 s**, both zero
  OOMs. These are regression results of existing scheduling, not a new scheduling improvement.
- CUDA hardware, real SSH/SLURM and new trace-derived resource experiments were **not run**.

## Explicit unfinished boundaries

| Classification | Remaining work | Acceptance needed |
| --- | --- | --- |
| CORRECTNESS_RISK | Full lifecycle/continuation fault injection and physical overhead accounting across all readers. | Interrupted checkpoint/publication, delayed terminal delivery, heterogeneous rungs and no-information continuation tests through all native paths. |
| DEFERRED_FEATURE | Demand-learned unequal CPU leases and additional ARI calibration. | Real owned-process demand/progress experiments and trace-derived replay superiority, preserving grant/affinity safety. Existing residency-based leases and ARI are not replaced speculatively. |
| DEFERRED_FEATURE | Revised scientific action valuation across correlated questions and costs. | Expanded eight-landscape comparisons with support, calibration, cost and regret; existing question/optimization authorities remain intact. |
| DEFERRED_FEATURE | Broader WorkRunner extraction, contract-aware dataset substitution and product dependency waiting. | Characterization and end-to-end ownership/identity/recovery tests; explicit certificate policy, no inferred transitivity. |
| DEFERRED_FEATURE | Public Fleet recovery/adoption, resumable evidence transfer and distributed export. | Complete lease/fencing/dispatch/acknowledgment/recovery/export acceptance chain. Existing distributed gates stay explicit. |

Existing products, certificates, import/export, resource replay and failure classification are
covered by retained suites; that is regression coverage, not completion of their pending features.
