# Study observations in desktop clients

`JobService.study_run` accepts local scientific evidence outside the provider allocation
directory only when the nearest project, project id, Work name, full scientific
fingerprint and sealed `execution.json` all match the recorded Job. Evidence must
remain inside that exact Execution's `runs/` tree; symbolic paths are rejected.
This does not broaden remote ownership or change recovery provenance.
The sealed execution must identify this Job. Historical executions without a
Job stamp are accepted only at the exact canonical fingerprint path; unstamped
reruns and evidence stamped with another Job remain inaccessible.

`lambdaforge.analysis.ObservedStudyChart.summarize` is a read-only descriptive
projection of a native Study Analysis. Select metric names, an X parameter (or
`trial`), optionally a second parameter, `center="mean"|"median"`, and
`spread="none"|"sd"|"se"|"iqr"|"range"`. Standard error of a mean is rejected
for a median. Missing and censored final objectives are never replaced with best
partial observations. Configurations receive equal weight, independently of seed
counts. The response includes individual observations, grouped estimates, support,
exact intervals and their scope. Across-configuration dispersion is distinct from
within-seed uncertainty; a single observation does not imply zero variance.

These charts do not fit a surrogate, select a winner or modify HPO decisions.
Use native `parameter_conclusions` and `interaction_conclusions` for the
framework's scientific interpretation, including its evidence/stability warnings.
