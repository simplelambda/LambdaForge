# Work results, historical evidence and independent products

[Español](WORK_RESULTS.es.md) · [Manual](MANUAL.md) · [Products](PRODUCTS.md)

A Work is a scientific execution, not necessarily a training experiment. A Study adds an authored
parameter/seed design and scientific coordination. Both use the same runner, Execution/Run/Attempt
records, managed outputs and export/import transaction. Ordinary Work reports never fit a surrogate,
invent an objective or aggregate different Attempts as if they were one observation.

## Inspect and report

Open a Work in the Research Console and select **Results**. The Run/Attempt selector keeps failures,
retries and their evidence separate. Overview, Metrics, Outputs, Visualizations, Products, Logs, Resources and
Provenance load only the selected panel. Root Results lists use compact indexes; they do not fetch
scalar history, models or project HTML. Opening Metrics reads bounded curves of the selected Attempt;
opening Outputs reads descriptors, structured values and checkpoint names, not model bytes. Checkpoints
are explicitly labelled as current shared logical Run state, not archived snapshots of an old Attempt. Missing
retained files are shown as such. An in-progress Job needs its persisted Execution attestation before
deep evidence is available; Job logs and controls remain available during preparation.

Provenance includes a selectable table of exact historical-result/product inputs. Enter opens the
recorded source when available in the local catalog, never another execution with the same name.
A missing source gives an explicit export/import instruction rather than launching its producer.

```bash
lf results list --json
lf results show EXECUTION --view overview --json
lf results show EXECUTION --view metrics --run RUN_ID --attempt 1 --json
lf results show EXECUTION --view outputs --run RUN_ID --json
lf results preview-output EXECUTION OUTPUT_NAME --run RUN_ID --attempt 1 --json
lf results report EXECUTION --output work-results.html
```

In Outputs, Enter explicitly previews a selected retained regular file only when it is at most
64 KiB. The preview verifies its exact size and managed checksum; UTF-8 is shown as inert text,
including HTML. Larger files, directories and unavailable artifacts show metadata and an explanation,
without downloading their bytes. Use the report's isolated viewer to run a registered HTML visualization.

**Open HTML report** explicitly builds an offline report from saved records. It includes bounded log
tails, downsampled scalar curves, exact Attempt metadata, outputs, failure, configuration and provenance.
The metric library is embedded once, with separate Run/Attempt groups and step/observation axes. Different
Attempts are never silently combined, and cross-Attempt correlations are not presented as measured evidence.
Exports also retain the complete original logs/scalar files and retained checkpoints/artifacts. No old
scientific package is imported to display evidence. A Work without metrics or HTML still has a report.
Failed reports label partial evidence honestly; compacted failed artifacts are not reconstructed.

Register project visualizations with the existing API:

```python
viewer = self.outputs.html_section("predictions", title="Predictions")
viewer.write_text("<html><body><h1>Saved scientific view</h1></body></html>")
```

The project owns its visualization logic. LambdaForge verifies the registered file and embeds it in
an opaque-origin sandbox using the same security policy as Study reports. HTML must be self-contained;
use included/data-URL assets. External requests and privileged filesystem access remain blocked.
Limits remain 16 MiB per project document and 64 MiB per report. Reading a list never opens these files.

## Choose an exact historical dependency

Human names are selectors, never identities. Repeating a name is allowed. One match resolves directly;
multiple matches require an exact Execution ID. The Console shows deterministic local `#n` labels with
the original identity. These labels are not portable scientific selectors. Imports retain their original
Execution and origin; they never become executable Jobs.

```bash
lf results reference previous-work --json
lf results reference EXECUTION --run RUN_ID --attempt 1 --json
lf results reference EXECUTION --product selected-models --json
lf results reference EXECUTION --product selected-models --artifact model-1 --json
```

These metadata-only commands return a typed YAML/JSON marker with exact IDs fixed internally. A later
same-name execution cannot change that marker. In **Run Work**, enter the consumer parameter name and
choose **Choose historical input…**. Select an exact catalog row, then **Use exact reference**.
The picker also exposes exact Run/Attempt selection and registered product-artifact selection.
Submit validates, explains and prepares the bound configuration without editing the authored YAML. The same
operation is available through `lf run CONFIG --input-ref 'PARAMETER=JSON_MARKER'`. Launch bindings apply
to a single Work/Study, not a composed or Fleet configuration; those declare typed inputs in their YAML.
An input changed between selection and preparation fails explicitly rather than selecting another source.

### ResultInput: evidence of one Execution

```yaml
with:
  previous:
    result:
      execution: previous-work
      # run: run-...        # required to read an individual Run of a multi-Run Execution
      # attempt: 1          # optional exact Attempt; otherwise latest is pinned by the envelope
```

The resolver adds `evidence_id`, a canonical SHA-256 of the exact original result/configuration.
The consumer receives `lambdaforge.work.ResultInput`: immutable `metadata`, `configuration`, `metrics`,
`result`, `outputs`, bounded `metric_curves()` and explicit verified `artifact(NAME)`. It is pickle-safe
for spawned workers. Its identity is part of the consumer fingerprint and recorded inputs. It accepts
finalized native/imported snapshots (including failures), never a mutable live result. Multi-Run evidence
requires explicit Run selection before reading individual metrics/results/artifacts. Deleting its owning
evidence can invalidate this dependency: it is not an autonomous model contract.

### ProductInput: durable scientific content

Existing declarations remain supported:

```yaml
with:
  model: {product: {name: selected-models, contract: example/models:v1}}
```

For an exact published output of a known Execution:

```yaml
with:
  model:
    product:
      from: {execution: previous-study, output: selected-models}
      artifact: model-1  # optional exact registered member; not a filesystem path
      expect: {dataset: example-data@1}
```

The contract can be omitted only when the exact publication receipt declares it unambiguously. It is
still validated; expectations remain explicit. Native preparation replaces the source selector with the
sealed content ID. Direct product references work independently even after deleting the producer.
Unpublished temporary outputs are not products. Dataset inputs and same-execution `from` references
keep their existing semantics.

With `artifact` declared, `ProductInput.selected_artifact` exposes the choice and `artifact()` verifies
and opens that exact member. Requesting another member fails instead of substituting it. Without it,
the existing explicit `artifact(NAME)` API is unchanged. This is a selection within the same sealed
product and contract, not a fabricated sub-product. Its logical member name participates in the
consumer identity and survives pinning, remote/Fleet preparation and export/import.

## Explicit promotion from an ordinary Work

```yaml
products:
  evaluation-report:
    kind: ScientificReport
    contract: example/evaluation:v1
    scientific_meaning: {protocol: evaluated-on-fixed-test-v1}
    outputs: [report, counts]
```

Supported ordinary kinds are ScientificReport, AnalysisResult, ModelArtifact and Selection. Meaning is
a publisher declaration, not an independently proven equivalence. List registered structured values
and retained regular files explicitly. Directory members must be registered individually; a directory
is not implicitly guessed as a model. Promotion currently requires one logical Run with a successful
latest Attempt; multi-Run model selection continues using the existing ModelSet declaration.

Finalization seals immutable independent bytes with the native registry and locks. Scientific results
are persisted first. A storage/publication failure leaves successful scientific evidence unchanged and
preserves the outputs needed by `lf products finalize EXECUTION --apply`; that retries only publication.
Concurrent retries reuse the exact publication. Retiring the producer does not remove sealed products.

## Portable evidence and remote availability

```bash
lf export WORK_OR_EXECUTION --output ./exports
lf import ./exports/EXPORTED_PACKAGE --json
lf import ./exports/EXPORTED_PACKAGE --apply
```

Works use the existing native version-2 export/import format, with additive `execution_kind` metadata.
Older Study packages remain readable. Original records stay in `portable/`; local placement/indexes
stay separate. **Results** and imported **Work** entries open the same explorer without contacting the
original cluster. Re-export of an import verifies and preserves its original package/provenance.

Resolving an ID does not materialize its bytes elsewhere:

```bash
lf products materialize EXACT_CONTENT_ID --on cluster
lf products materialize EXACT_CONTENT_ID --on cluster --apply
lf results materialize EXECUTION --on cluster
lf results materialize EXECUTION --on cluster --apply
```

Console product/result explorers provide **Materialize…**, destination selection, preview and explicit
confirmation. Preview reads source metadata only. Apply uses a compressed ZIP64 package and native
safe extraction/import, verifies original content/evidence identity and reuses an exact existing copy.
No producer is launched. Dependencies must already be available on every supported target, including
Fleet members. Fleet preparation includes pinned historical dependencies in its native input stratum;
each member rechecks them before executing its concrete Runs. Existing distributed dataset/continuation
gates remain unchanged. Preparation/worker checks fail closed if absent. A remote-only source should first be
explicitly exported/imported into the controller catalog; this operation is not an implicit inter-cluster
orchestration service. Destination Python must provide the current LambdaForge runtime and sufficient
staging/durable space. Source scratch is project-owned, not the system temporary partition. SSH credentials,
reachability and site permissions remain external requirements.

See [the executable example](../examples/work_results/README.md) for Study → selected ModelSet →
independent visualization Work → custom HTML → export/import, without repeating training.
