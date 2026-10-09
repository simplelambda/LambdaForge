# Durable scientific products: implemented foundation

[Español](PRODUCTS.es.md)

This is the working **product model/catalog/transport and explicit Work-input layer**, not the
completed Study orchestration system. Native Study decisions, local model selection, typed Work
dependencies and actual consumer records are available. A single native Study may declare automatic
post-Study decision/model publication. Native single-host Study export/import includes published
products. The Console includes a metadata/audit browser and confirmed Study import.
Dependency waiting/replanning and complete Fleet export remain pending.
See the [reform ledger](ARCHITECTURAL_REFORM.md). Existing Work execution/recovery and exact Dataset
publication remain unchanged.

## Identities and contracts

`lambdaforge.products` exports `ProductContract`, `ProductArtifact`, `StudyProduct`,
`ProductRegistry`, `ProductBundle`, `SelectionPolicy`, `ModelSelection`, `select_models`,
`build_study_decision`, `ProductRequirement`, `ProductInput`, `ProductPublication` and
`publish_declared_products`. A product has:

- an explicit versioned contract, such as `project/report:v1`, and an exact set of scientific fields;
- `scientific_id`, computed from kind, contract declaration and those declared meanings;
- `content_id`, computed from the canonical payload/meaning and exact artifact manifests;
- immutable original producer provenance, with separately appended attestations for later producers;
- a physical catalog placement that does not alter either identity.

The producer's full configuration may be provenance; it is **not** the consumer compatibility key.
Names are immutable aliases. The same contract identifier cannot acquire another field declaration;
use a new contract version. Changing scientific objective, input/label semantics, design or selection
rule must change the declared meaning. Nothing automatically guesses exclusions inside NPZ, model
checkpoints or arbitrary formats. Equal declared scientific meaning does not certify equivalence of
different bytes.

Every artifact is an independent regular file with a raw SHA-256 and size. Publication copies and
verifies it before committing. It never links to a disposable Attempt, removes the source, or scans
unselected files. Metadata reads are bounded at 512 KiB each. Listing/showing does not open large
artifacts; `verify` is explicit. Missing reads and previews do not create directories or locks.

## Explicit publication API

Use this API from a postprocessing/operator script, not as a hidden replacement runner. Native
post-Study declaration and explicit post-Study model selection are available below.

```python
from lambdaforge.products import ProductContract, ProductRegistry, StudyProduct

contract = ProductContract("my_project/count-report:v1", ("sources", "labels", "method"))
report = StudyProduct(
    name="benchmark-counts-v1",
    kind="ScientificReport",
    contract=contract,
    payload={"train": 317, "validation": 272},
    scientific_meaning={
        "sources": {"dataset": "sha256:<exact-published-content>"},
        "labels": {"positive": "binding", "partitions": "audited-v1"},
        "method": "count-audited-members-v1",
    },
    producer={
        "execution_id": "<actual-original-execution>",
        "evidence_fingerprint": "<actual-persisted-evidence-fingerprint>",
    },
)
registry = ProductRegistry()  # Current pyproject project/.lambdaforge/products
plan = registry.publish(report)  # Read-only; does not publish.
registry.publish(report, apply=True)
resolved = registry.resolve(
    "benchmark-counts-v1", contract=contract,
    scientific_expectations={"method": "count-audited-members-v1"},
)
```

Placeholders in the example must be replaced by actual evidence; the generic envelope does not
derive or certify that evidence. Kind labels such as `StudyDecision`/`ModelSet` do not automatically
select models or prove their payload semantics. The project owns its explicit declaration.

For artifacts, pass `ProductArtifact(name, relative_path, raw_sha256, size_bytes, role)` descriptors
and `files={artifact_name: source_path}` to `publish`. `artifact_path(selector, name)` verifies the
selected promoted bytes by default before returning their path. Do not substitute the framework's
historical filename-aware managed-file fingerprint for the artifact's raw byte SHA-256.

## Native CLI

```bash
lf products list --json
lf products show benchmark-counts-v1 --json
lf products provenance benchmark-counts-v1 --json
lf products consumers benchmark-counts-v1 --json
lf products verify benchmark-counts-v1 --json
```

All operations accept an explicit `--root DIR`; otherwise use the current project's catalog or
`LAMBDAFORGE_PRODUCT_ROOT`. Metadata list/provenance support `--offset`/`--limit` pages. Scientific
code should not guess remote Job paths. Workers receive a project-owned product root from the
existing control plane, separate from disposable Jobs and caches. The remote default is
`STATE_ROOT/products` for the scoped project; local execution uses `PROJECT/.lambdaforge/products`.
Unsafe remote roots nested in the Job or cache root are rejected.

An operator may serialize `product.to_dict()` and publish it through
`lf products publish MANIFEST.json --file ARTIFACT=SOURCE [--apply]`. This is an advanced explicit
declaration, not automatic selection and not an application-specific workaround. Preview checks
source shape/size; apply checks exact bytes. A corrupt source never becomes a valid published
product. A conflicting immutable name requires another name/version, never an overwrite flag.

## Post-Study ModelSet selection

Register a scored model snapshot from the ordinary Work lifecycle:

```python
self.outputs.from_checkpoint(
    "model", "best.ckpt", role="model",
    metadata={"metrics": {"auprc": measured_auprc, "accuracy": measured_accuracy}, "step": epoch},
)
```

The project must supply values evaluated for **those exact weights**, not an unrelated last epoch.
This keeps an ordinary independent Attempt artifact. Promotion to a durable catalog is deliberate.
Alternatively use `outputs.file` with model role and the same explicit metadata. Directory models
and externally published artifacts are not yet supported by the local selector.

Create an ordinary selection YAML, for example `selection.yaml`:

```yaml
artifact: model
rank_by: auprc
mode: max
group_by: [hidden_dim]  # Omit for a global top_k.
top_k: 1
constraints:
  accuracy: {min: 0.6}  # Same registered snapshot, never latest Run metrics.
tie_policy: stable
```

```bash
lf products select EXECUTION_ID --name best-per-width --contract my_project/models:v1 \
  --policy selection.yaml
# After reviewing, repeat with --apply.
```

Use `--results-root DIR` for an explicit local ResultStore and `--root DIR` for the product catalog.
The same operation is exposed by `select_models(source, execution_dir, policy, name=..., contract=...)`.
It selects latest valid Attempts, excludes failed/pruned/partial-fidelity outcomes, applies explicit
same-snapshot constraints and deterministically ranks each exact conditional group. Inactive and
explicit null parameter values are distinct. `tie_policy: include_equivalent` optionally includes
models within an authored `practical_margin` of the top-k boundary; it is not a confidence claim.

Selection reads/hashes **selected model files only**, so even preview may perform substantial
local I/O for large selected weights; it creates no files/locks. Apply promotes exact independent
copies. Source evidence/failure history stays unchanged. The ModelSet records chosen logical Runs,
Attempts, seeds, parameters, snapshot metrics/steps, grouping, rationale, content checksums and
producer evidence. A failed unrelated Run does not invalidate completed eligible snapshots.
The ranking is explicitly **among eligible observed snapshots**, not proof of a global optimum,
fresh-seed confirmation or a redecision by HPO. Missing snapshot-bound metrics fail with guidance;
the selector never guesses that Run best/last metrics refer to arbitrary checkpoint bytes.

Remote/Fleet selection and publication inside composed `steps` remain pending. The local selector
never reaches outside its verified owned Execution to guess model paths.

## Native StudyDecision

```bash
lf products decide EXECUTION_ID --name pooling-decision --contract my_project/pooling:v1
# Review the native decision, then repeat with --apply.
```

This seals the native finalized Study selection, objective/design/input meanings, selected
Run/Attempt identities and evidence fingerprint. It does not rank candidates again or fit HPO.
Selection and scientific resolution remain distinct: missing final Analysis leaves explicit
unresolved questions, and incomplete fresh-seed confirmation cannot become a survivor-only
selection. A failed auxiliary Run does not invalidate an otherwise complete eligible selection.

If `analysis.json` exists, it must match the exact finalized Execution/evidence identity; stale,
provisional or mismatched conclusions are rejected with guidance to recompute Analysis explicitly.
No scalar histories, checkpoints or model bytes are read for this metadata operation. The Python
routes are `ResultStore.decision(...)` and `build_study_decision(...)`.

## Declared post-Study publication and publication-only retry

A single Study (`search`, `sweep` or repeated `seeds`) may declare native products alongside its
ordinary Work fields:

```yaml
products:
  pooling-decision:
    kind: StudyDecision
    contract: my_project/pooling:v1
  best-per-width:
    kind: ModelSet
    contract: my_project/models:v1
    select:
      artifact: model
      rank_by: auprc
      mode: max
      group_by: [hidden_dim]
      top_k: 1
```

Validation checks this closed declaration before computation. `lf explain` includes it, and the
Execution records its expected products before Runs start. Finalization persists training evidence
first, then uses the same decision/selection/catalog services to publish independent products
before compaction. `StudyDecision` requires an objective; a ModelSet requires explicitly scored
artifacts. These are not arbitrary constructors, executable callbacks or a second runner.

Publication failure preserves every native result/checkpoint. `products.json` records pending,
published or failed products with the exception/phase; `product-publication-history.jsonl` retains
performed publication attempts. Run status/objective evidence is not rewritten as a training
failure. Inspect publication separately:

```bash
lf products status EXECUTION_ID --json       # Bounded metadata; no weight reads.
lf products finalize EXECUTION_ID --json     # Read-only preview.
lf products finalize EXECUTION_ID --apply    # Publication only; no Runs launched.
```

Publication-only apply acquires the native Execution ownership lock, then the publication/catalog
locks. Successful receipts are reused without requiring original Attempt artifacts; promoted bytes
are reverified on apply. Concurrent repetition is idempotent; corrupt receipts/evidence and changed
declarations fail closed. Immutable-name conflicts require a deliberate new product name/version,
not overwriting an existing model set. `--results-root` and `--root` select local evidence/catalog.
The CLI does not currently dispatch publication-only recovery to a remote host; execute it on the
host owning that Execution/catalog. Automatic Fleet promotion is not implemented.

## Typed Work dependencies

Declare the product as an ordinary Work argument, not a producer-status condition:

```yaml
name: model-visualization
run: my_project.Visualize
with:
  models:
    product:
      name: best-per-width
      contract: my_project/models:v1
```

The project may add `expect: {FIELD: EXACT_VALUE}` to require explicit scientific meanings.
The resolver checks the declared contract/expectations and supplies a pickle-safe `ProductInput`:

```python
import lambdaforge as lf
from lambdaforge.products import ProductInput

class Visualize(lf.Work):
    def run(self, models: ProductInput) -> dict[str, int]:
        for model in models.payload["models"]:
            weights = models.artifact(model["artifact"])
            # Read these verified weights and register ordinary managed visualization outputs.
        return {"models": len(models.payload["models"])}
```

Metadata access does not hash/download model bytes. `artifact(name)` verifies that exact promoted
file before returning its read-only input path. Compatibility and the consumer's input identity
use sealed contract/science/content, not the full producer YAML hash or current producer status.
Validation/preview create nothing; actual Run Attempts append immutable consumer records, exposed
by `lf products consumers`. Cached Execution reuse does not fabricate a new consumption event.

Remote bundles pin aliases to exact content IDs and verify materialization in the destination
project catalog before staging. Import the explicit product bundle there first; LambdaForge does
not implicitly transfer large weights, expand a name to another representation or fall back to a
Job-local catalog. Missing/incompatible products fail with a diagnostic; automatic dependency
waiting or producer replanning is not yet implemented.

## Portable product export/import

```bash
lf products export benchmark-counts-v1 --output ./report-export
lf products export benchmark-counts-v1 --output ./report-export --apply
lf products import ./report-export --root ./independent-products
lf products import ./report-export --root ./independent-products --apply
```

These operations transport **one product**, not the whole producer Study. Export writes a verified
directory containing `bundle.json`, `product.json`, selected file bytes and complete producer
attestations (not downstream consumer audit records). Existing destinations are protected. Import checks manifest SHA/identity, exact
artifact bytes and the complete provenance inventory; it preserves the original producer and
never launches computation. Repeated/concurrent imports of the same product are idempotent. Even
when the destination already contains the product, a corrupted import source is rejected.

Publication/import are serialized with the existing native cross-process lock. Interrupted copies
remove only their exact unpublished transaction directory; original files stay untouched. A crash
after object commit but before catalog binding may leave an unbound immutable object: repeating
the same operation verifies and completes its binding, not a silent new execution.

An exported/imported product survives deletion of its original Study. This does **not** implement
a dependency DAG, signed producer authenticity, certificate-based Dataset equivalence or complete
Fleet export.

## Import a portable Study without executing it

```bash
lf import ./portable-study--execution-ID --json
lf import ./portable-study--execution-ID --apply --json
lf results list --json
lf products show models --json
```

The default verifies/previews; `--apply` registers evidence in the current project's ResultStore
and sealed products in its native ProductRegistry. `--results-root`/`--products-root` optionally
override placement, never scientific identity. Native version-2 exports are accepted: final,
failed/cancelled and running snapshots retain captured status. A snapshot never becomes completed
evidence. Pre-execution Job-only packages without an owned Execution are not Study imports.

Verification covers all listed files, SHA-256/size/count, absence of extra files, metadata versions,
Execution/Run/Attempt identity/ownership and product/producer attestations. Traversal, symbolic,
special or hard-linked content, duplicate inventory and corrupt incoming/existing evidence fail
closed. Concurrent imports are serialized/idempotent. A native Execution or another snapshot with
the same Execution ID is never overwritten: use a separate results root to retain multiple captures.
Checksums do not establish cryptographic authenticity of an untrusted author's scientific claims.

Original files/provenance remain intact in `portable/`; `import.json` records the new placement.
ResultStore lists the entry, relocates logical Run paths only in its log read model and reads included
Analysis without refitting. No application imports, computation, checkpoint continuation or automatic
HTML opening occur. Imported configuration cannot become native recovery state. An imported
archive's `portable/` directory is already the reusable package; it needs no new producer export.

In the Research Console, **Studies → Import Study…** (also accessible from Products and Ctrl+P)
selects a package directory, verifies it in a worker, previews its exact identity and asks for
confirmation. Registration adds a labelled read-only snapshot to Studies/Overview in the current
project. Trials/seeds use local hierarchical projections and the shared selected-Run dashboard;
retained metrics/logs/artifacts are read locally, not from the original cluster. Captured running
snapshots are not live controllers. Cancel, retry, history deletion and producer export controls
are hidden; imported archive deletion remains an explicit Results operation. HTMLs included in the
package remain accessible in its `portable/reports/` directory.

The native aggregate `execution/result.json` may exceed 64 MiB in a large valid Study. Import first
verifies its exact declared size and SHA-256 along with every file, then strictly parses/validates
that aggregate without applying the individual-metadata size ceiling. Checksums, path/ownership,
duplicate-field and non-finite-value checks are unchanged. Root-screen refresh only reads small
`import-study.json` indexes, not the aggregate. Reapply an older verified import once to build its
`import-view/` presentation files under the import lock; scientific records and the original receipt
remain untouched. Import copies retained bytes and therefore needs local space for the package and
its local index; it never downloads artifacts again.

Export includes products published through the native declaration/receipt, their independent bytes
and original producer history. Provider transfer fetches only those objects, never the whole catalog;
telemetry sampling cannot alter product bytes. Missing published products fail explicitly. Undeclared
operator publications use `products export`, not discovery by sweeping unrelated catalogs.

Archive registration commits atomically before independently locked product bindings. A later
product transaction failure leaves the verified archive available: retry the same import to finish
binding. This is not a cross-store distributed transaction. Deleting imported Study evidence preserves
its independently promoted products.
