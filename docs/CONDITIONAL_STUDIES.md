# Conditional scientific Studies

[Español](CONDITIONAL_STUDIES.es.md) · [Research HTML](RESEARCH_ANALYSIS.md)

## Contents

1. [One question, one Study](#1-one-question-one-study)
2. [Activation grammar](#2-activation-grammar)
3. [Validate and understand the plan](#3-validate-and-understand-the-plan)
4. [Time and target capacity](#4-time-and-target-capacity)
5. [Identity and existing evidence](#5-identity-and-existing-evidence)

## 1. One question, one Study

Use one `sweep.space` when comparing alternatives with different applicable parameters. All cells
belong to one Study and share the same authored seeds. Use `steps` for a real workflow, not to
partition comparable experimental branches. Steps remain separate invocations, not merged Studies.

```yaml
name: optimizer-comparison
run: my_project.Compare
seeds: [4, 7, 32, 54]
sweep:
  reference: {optimizer: baseline}
  space:
    momentum:
      values: [0.8, 0.9]
      when: {optimizer: {in: [sgd, accelerated]}}
    optimizer: [baseline, sgd, accelerated]
resources: {gpu: 2, time: 24h}
execution: {max_parallel: 4, max_time: 12h}
```

This creates five candidates and twenty required Runs, not twelve candidates or one Study per
optimizer. `baseline` has no `momentum` argument; the Work signature's default applies. A reference
must select exactly one active candidate. Zero matches or multiple matches are authoring errors.

## 2. Activation grammar

Each descriptor optionally has `when`. Multiple parents are combined with AND:

```yaml
when:
  optimizer: {in: [sgd, accelerated]}
  use_momentum: true
```

`{parent: value}` is the historical equality syntax; `{parent: {eq: value}}` means exactly the same
thing. Membership is explicit: `{parent: {in: [value1, value2]}}`. Membership values must be
non-empty, unique JSON scalars belonging to the parent's finite authored domain. Bare lists,
unknown operators, missing parents, cycles and out-of-domain values fail validation. No expressions,
`or`, `not` or recursive condition language are supported. Historical equality on numeric ranges
remains supported; membership requires a finite domain.

Parents may appear after children in YAML. One shared `ActivationCondition`/`ParameterSpace`
authority validates and topologically orders dependencies for sweeps, random/Sobol/adaptive
generation, encoding and analysis. Inactive keys are absent, never `null`, never multiplied and
then deduplicated. Duplicate authored sweep values are rejected. Domain order remains authored;
membership ordering does not change scientific identity. Conditional coverage uses active observed
values, not a fabricated inactive category.

## 3. Validate and understand the plan

```bash
lf validate comparison.yaml
lf explain comparison.yaml
lf run comparison.yaml --on CLUSTER --dry-run --json
```

Validation/explanation include a bounded `preflight`: candidate count, shared seeds, required Runs,
first independent categorical branch counts, reference, requested GPUs, parallel limit, Study time
budget and scheduler wall-time. They do not print hundreds of individual identities in human
output. `--json` retains the full planning model for automation. Local validation explicitly says
target capacity has not been checked; a read-only target plan may also require an already prepared
runtime/environment. A branch summary is descriptive, not a domain-specific family API.
For adaptive searches these counts describe the current bounded proposal window (subject to the
candidate ceiling), not a promise to execute every proposal. Required replication becomes an
obligation only after a candidate is proposed. Fixed-sweep counts cover the complete authored design.

Root `resources` are inherited by steps unless overridden. Root `with`, `seeds`, `replicates`,
`search`, `sweep`, `execution`, `objective` and `analysis` are rejected for a steps composition:
declare them on the relevant Work step. No root experimental policy is silently ignored.
Preflight validates Work inheritance/signatures without constructing a Work or running scientific
code; it cannot detect arbitrary semantic bugs inside a consumer's `run()` implementation.

## 4. Time and target capacity

`resources.time` is the scheduler wall-time ceiling of a schedulable unit. Sequential levels add;
members of one parallel level contribute their maximum. Six sequential steps of 168 h request
1008 h (42 days); three sequential steps of 24 h request 72 h. This is a ceiling, not a prediction
of training duration. `execution.max_time` is the logical Study's dispatch budget and is reported
separately; it does not replace scheduler wall-time.

For a direct host, a successful live GPU UUID inventory together with an unambiguous inherited
visibility restriction can reject a request exceeding known capacity before bundle preparation,
including dry-run. Occupancy is not capacity. A failed/ambiguous probe stays unknown, never an
invented count. Scheduler login hosts are not allocation inventories. A site-command launcher's
grants use the existing runtime check: adaptive Studies may scale down to fewer granted tokens,
never broaden grants or claim additional devices during preflight. Local validation does not
contact any execution target.

## 5. Identity and existing evidence

Normalized scalar equality preserves historical serialization and identity. New membership syntax
is additive to the current schema; it requires no new YAML version or compatibility runner.
Predicates remain in Study initialization/design and therefore travel through analysis, export and
retry. Existing persisted equality documents are read, not rewritten. A changed condition, domain,
seed or objective remains a scientific change, not an automatic retry waiver. Fixed sweeps still
execute every required cell/seed without adaptive pruning or scientific replanning.

The generic acceptance fixture in `tests/work/test_conditional_membership.py` has branch counts
1, 2, 10, 16, 9 and 18: exactly 56 candidates × four authored seeds = 224 required Runs. Its unified
conditional design equals the intended union of independently described branches, without
importing or modifying any consumer project.
