# Published datasets and scientific reconstruction

[Español](DATASET_RECONSTRUCTION.es.md) · English

A published `NAME@VERSION` identifies one immutable content ID. A scientific validation **PASS**
does not mean that a reconstruction has identical bytes, or that all members are scientifically
equivalent. LambdaForge never overwrites this version, relaxes transport checksums, rounds arrays,
or guesses which fields inside an NPZ/other format are operational.

## Choose the intent before computing

```bash
lf datasets preflight corpus@6 --intent publish --on gpu12 --json
lf datasets preflight corpus@6 --intent reuse --on local --json
lf datasets preflight corpus@6 --intent rebuild --on gpu16 --json
```

Preflight reads the controller and all configured remote indexes, without changing them, downloading
assets or starting a Work. It returns locations, exact content IDs, discovery failures and an
`allowed` decision. A new publication is refused if the version exists; reuse/rebuild requires an
unambiguous reference. Incomplete discovery cannot prove absence. Preflight is not a reservation:
the final publisher still checks identity under the registry lock.

1. **Reuse**: `lf datasets materialize corpus@6 --on TARGET` previews native exact placement;
   review and add `--apply`. `replicate` explicitly selects source/destination. This copies/verifies
   published content; it does not rebuild scientific arrays. Native compressed replication supports
   local/remote placements, preferring direct site SSH; see below.
2. **Rebuild**: compute an independently sealed, unregistered reconstruction and compare it.
3. **Publish**: use a fresh version for changed bytes, even after an equivalence report.
   Additional representations under the same version are not supported; there is no silent
   alternative-content placement policy.

## Publish once, replicate exact content

```bash
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --json
# Review the route, size and destination, then apply:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --apply
```

In bare `lf`, open **Datasets → DatasetVersion → Replicate…**, choose endpoints, review the native
preview and confirm. The Console stays responsive; inline phases/elapsed time and compressed byte
counts (on the relay route) show progress. Keep the session open until completion. Locations updates
after registration. The target needs `storage.dataset_root` and a usable LambdaForge runtime
(existing or bootstrapped); no scientific Work or GPU claim runs. Effective roots remain project-scoped.

`--route auto` probes site SSH from source to destination with trusted host-key checking, without
prompts or agent/credential forwarding. If available, bytes travel directly between hosts.
Configure site-authorized noninteractive authentication and trusted host keys on the source host
to enable it; LambdaForge never installs keys or copies controller credentials. `--route direct`
requires that connection. `--route relay` uses existing authenticated controller transports with
a bounded **compressed in-memory stream**, not an archive staged on your computer or in `/tmp`.

Streaming tar/gzip (level 3) verifies the complete source, rejects symbolic/special entries and
unsafe archive paths, uses owned destination staging/storage admission, and checks exact content
before locked atomic promotion/registration. Both indexes expose the same content ID to typed
Dataset inputs. An identical destination is verified/reused; different existing content is refused.
Interrupted copies never register partial content. Deliberate retry reuses an already committed
exact placement or transfers again; it is not byte-offset resume. Legacy registered source paths
remain usable. Replication never rebuilds/relabels/deletes source data or automatically reconciles
conflicting versions; inspect existing divergence or publish a fresh version first.

LambdaForge cannot infer an output name/version from arbitrary Python before calling `Work.run()`.
The producing project must perform the public preflight as its first operation:

```python
def run(self, version: str, intent: str = "publish"):
    self.outputs.dataset_preflight(name="corpus", version=version, intent=intent)
    # Expensive computation only after preflight. Checkpoint results as usual.
    rows = self.compute_members()
    return self.outputs.dataset(
        name="corpus", version=version, members=rows, intent=intent,
        scientific_identity={
            "sources": {"source-id": "sha256:<exact source checksum>"},
            "selection": ["member-id"],
            "labels": {"member-id": 1},
            "configuration": {"modes": 64},
            "algorithm": "my-project/spectral-contract-v1",
        },
    )
```

The Work preflight checks its execution host's registry. Use the native preflight above to also
detect cross-host divergence before submission. `intent="rebuild"` stores the candidate in a pinned
named checkpoint collection and returns its content ID and placement path as an ordinary output
value (`publication_status="reconstructed-unregistered"`), never as a registered DatasetVersion.
`reuse` is a preflight/materialization intent, not a `members` publication option.

## Three separate identities

- **Scientific**: explicit project `scientific_identity` with required sources, selection, labels,
  configuration and algorithm contract. The manifest/record exposes `scientific_id`. It is a
  declaration, not proof of reproducibility. Historical manifests have no scientific declaration;
  they remain readable, but non-identical content is unresolved until explicitly declared.
- **Content**: existing `dataset_id == content_id`, exact per-asset checksums, sizes and canonical
  logical index. Nothing about transport or verification is weakened. Identical content cannot
  silently acquire a different scientific declaration in an existing registry version.
- **Operational**: manifest producer/build provenance, paths, Execution/Run/Attempt, hardware and
  timestamps. These are not content identity. Putting them *inside* an asset still changes its bytes.

## Compare entire sealed reconstructions

```bash
lf datasets compare /reference/root /candidate/root --on gpu16 \
  --verifier my_project.dataset_checks:compare_member --policy comparison-policy.json \
  --output comparison-report.json --json
```

Both roots must be on the selected host. Policy and output-report paths are on the controller;
only policy/JSON reports cross the transport, not assets. A verifier is explicitly trusted project
code imported on that host; LambdaForge does not load code specified by a dataset manifest.

Python API: `from lambdaforge.data import DatasetComparison, DatasetComparisonContext`;
`DatasetComparison.compare(left, right, verifier=callback, verifier_id="project/check-v1", policy=policy)`.
The callback receives one `DatasetComparisonContext` with validated roots, both `DatasetMember`
descriptors and the explicit policy. Return strict JSON with `equivalent: bool`,
`checked_assets: [logical_asset_names]` and optional `details`. Every changed asset must be covered.
Use each descriptor's relative asset path below its root; never infer paths from machine provenance.

Historical reference without a contract: use `--contracts contracts.json` (Python:
`scientific_contracts={exact_content_id: declaration, ...}`). Each declaration is explicitly supplied
by the project and bound to the exact verified content ID; it must reflect independently checked
source/selection/label/algorithm evidence, not an assertion that two datasets are equivalent.
The report preserves the original absent `scientific_id`, records the assertions and separate
`comparison_scientific_id`, and still requires the full member verifier. No historical manifest or
registry is edited. An assertion cannot override a conflicting persisted declaration.

Example policy (illustrative; choose tolerances from the actual scientific contract):

```json
{
  "diffusion_eigenvectors": {"method": "spectral-projector", "atol": 1e-8, "rtol": 0.0},
  "metadata_json": {"method": "exact-scientific-fields-operational-differences-reported"}
}
```

LambdaForge validates policy shape/finite nonnegative tolerances, not numerical formats. Your
verifier must implement every method and report operational differences explicitly. It must compare
embedded identifiers, labels, partitions and source bytes **exactly**, never with tolerances. Sign
flips or rotations of degenerate spectral spaces may require subspace projectors, residuals and
appropriate eigenvalue grouping, not elementwise rounding or a global `allclose`.

Before invoking the callback, LambdaForge verifies both complete datasets with their own exact IDs,
then requires equal scientific contracts, member identifiers, partitions, targets/member metadata,
asset names, target schema and global assets. Changed URI assets must be locally materialized first.
Corruption yields `invalid`; a missing declaration/verifier yields `unresolved`; scientific changes
yield `different`. A complete approved comparison yields `equivalent`, while retaining both distinct
content IDs and `byte_equal=false`. It never registers/replaces content. Byte-identical content is
`exact` unless declared scientific contracts conflict. No single-member comparison proves a dataset.

## Publication failure without recomputation

`outputs.dataset` preserves a fully sealed rejected candidate and prints its exact path. Failed or
interrupted publication also protects Attempt artifacts, metrics/logs/results and checkpoints from
automatic compaction. If the recovery volume cannot accept a copy, the sealed tree is retained on
its original volume and its path is reported. These retained bytes intentionally consume storage;
inspect them before explicit lifecycle deletion.

```bash
lf datasets publish-candidate /preserved/candidate --on gpu16 --version 7 --json
# Review exact source, content/scientific IDs, target and version, then:
lf datasets publish-candidate /preserved/candidate --on gpu16 --version 7 --apply
```

This verifies saved bytes and atomically publishes/registers only that candidate. It does not run
the Work or repeat calculation. Omit `--version` only when the original registration is absent or
already has the exact content/declaration. Repeated apply is idempotent; corrupt/unsafe candidates
are rejected. The original failed Attempt history remains failed: publication recovery is not a
fabricated successful scientific execution. Candidate, original published version and checkpoints
are not deleted by this command. Ordinary `lf retry` still invokes Work code; it is not this
publication-only operation. Local publication uses configured `storage.dataset_root`, or the local
registry's sibling `datasets/published`; remote publication requires a configured permanent root.

### Failures before sealing (0.17.1)

Before the first copy, `outputs.dataset` now stores `publication-request.json` and a streaming
`members.jsonl` with exact source checksums in its publication checkpoint collection. If that
volume cannot accept the initial metadata, it uses an owned `.publication-requests` directory on
the publication volume. A failure prints the exact request directory. The same
`lf datasets publish-candidate REQUEST_DIRECTORY [--on CLUSTER] [--apply]` handles this request:
preview checks every source byte and the declaration; apply only copies/seals/registers those
bytes under existing locks. Source mutation, unsafe paths and immutable-version conflicts fail
closed. It does not invoke a Work or rewrite the original failed Attempt as succeeded.

Older failures may have retained validated source files and their member index, but no complete
publication request. Do not invent missing scientific metadata. Explicitly restore the original
member/asset declaration, schema, metadata and provenance, then use the public preparation API:

```python
from lambdaforge.data import DatasetIndex, DatasetPublisher

# original_members must match the producer's declaration, including any extra design assets.
# An existing validated index can provide the basic members, without recalculating geometry.
original_members = (member.to_dict() for member in DatasetIndex(source / "members.jsonl"))
request = DatasetPublisher().prepare_publication(
    name, version, original_members,
    source_root=source, request_root=recovery_directory,
    build_provenance=original_provenance,
    metadata=original_metadata, target_schema=original_target_schema,
    scientific_identity=original_scientific_identity,
)
print(request)  # Use publish-candidate on this directory, preview first.
```

For WISDOM's older preprocessing failure, its retained `attempt-0001/dataset/members.jsonl`
and `dna-validation/dna-validation-report.json` are the starting evidence. Preserve its original
first-member `dataset_design` asset and publication declaration too; do not infer them from NPZ
internals. This repeats only inventory verification/publication, not geometry or annotation.

## Conflicting local and remote registries

`lf datasets list --all` and the Console Datasets screen discover every configured cluster's small
registry. Equal name/version/content merges locations; distinct content stays in separate visible
rows, marked **CONFLICT** in the Console. Listing never changes indexes and incomplete discovery is
visible. Member reads and whole-version deletion never guess an identity in a conflict.
**Datasets → exact identity row → Manage copies…** remains enabled: choose a target and
**Keep as project reference**, **Remove registration · keep files**, or **Delete managed copy**.
Every operation previews the exact hash/root and requires confirmation. REFERENCE marks the
controller reference; other divergent rows remain CONFLICT until explicitly retired.

The same native CLI operations are preview-only unless `--apply` is supplied:

```bash
lf datasets list --on gpu16 --json  # obtain the full exact content ID
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:FULL_ID_16
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:FULL_ID_16 --apply
lf datasets delete corpus@6 --on gpu12 --content-id sha256:FULL_ID_12  # preview
lf datasets delete corpus@6 --on gpu12 --content-id sha256:FULL_ID_12 --apply
# Index-only alternative, including a broken/missing copy or legacy external root:
lf datasets remove corpus@6 --on gpu12 --content-id sha256:FULL_ID_12 --apply
```

Replace the example IDs with complete observed hashes. Adoption verifies all source checksums
on its host before archiving/replacing the controller declaration. It chooses **future resolution**,
not a scientific equivalence certificate: it does not rewrite other registries, relabel bytes,
overwrite immutable publications or change previous Runs, pinned input IDs, checkpoints or provenance.
To retain a divergent representation as published evidence, use a distinct explicit version.

Exact deletion independently selects the target registry identity, even if the controller points
elsewhere. It requires a matching manifest, managed path and no active consumers. Index-only removal
does not delete bytes or free disk space and can retire a corrupt/missing physical copy; the registry
must still be readable and valid. Empty entries are retired so they do not reappear in inventory.
Both operations reject a changed hash/root, lock the target registry, and archive the previous
declaration in `dataset-registry-history/change-*.json` before mutation. Archive entries record
before/requested state, not proof a subsequent write succeeded. Unreachable/corrupt registries cannot
be treated as absent. If remote retirement succeeds but controller cleanup fails, inspect both
inventories again; never retry against an assumed identity. Historical evidence is never deleted.
Ordinary unpinned operations still fail closed; `reconcile` remains identity-preserving.

## Durable equivalence certificates

`DatasetEquivalenceCertificate` reuses the immutable ProductRegistry as a `ScientificReport` with
the reserved `lambdaforge/dataset-equivalence:v1` contract. It runs the existing whole-dataset
comparison before sealing approval; it does not accept an arbitrary approval JSON as proof.
The certificate binds both exact content IDs, the complete scientific declaration, verifier ID,
variable policy and comparison evidence. Operational roots, comparison timestamp and producing
Execution/operation provenance are separate from scientific/content identity.

```python
from lambdaforge.data import DatasetEquivalenceCertificate
from lambdaforge.products import ProductRegistry
from my_project.dataset_checks import verify_member

certificate = DatasetEquivalenceCertificate.build(
    "/published/reference", "/sealed/reconstruction",
    name="corpus-reconstruction-check-v1",
    scientific_contract=science_declaration,  # The full explicit dataset contract.
    verifier=verify_member,
    verifier_id="my_project.verify_member:v1",
    policy=variable_specific_policy,
    producer={
        "execution_id": "<recorded producing Execution or comparison operation>",
        "evidence_fingerprint": "<recorded source/comparison identity>",
    },
)
registry = ProductRegistry()
preview = certificate.publish(registry)  # No locks/files/registry changes.
certificate.publish(registry, apply=True)
restored = DatasetEquivalenceCertificate.load(registry, certificate.product.name)
```

Unresolved, scientifically different or corrupt comparisons cannot be certified. Byte-identical
copies still need an explicit scientific declaration for a scientific certificate. Historical
manifests need the same explicit content-bound `scientific_contracts` assertions as `compare`.
Reports above the catalog's 512 KiB metadata bound fail explicitly; no evidence is truncated.
`lf products show/provenance/export/import` and Console Products inspect/transport these records.
The recorded verifier is never executed on load/import.

Certificates are trusted project assertions, not cryptographic authenticity or transitive equality.
`accepts(reference_content_id=..., candidate_content_id=..., scientific_contract=...)` tests only
that exact recorded pair and contract; it is not a current integrity check. Verify the materialized
candidate separately with `DatasetOperations.verify(root, candidate_content_id)` before use.
Never substitute the reference's content ID for the candidate's or overwrite an immutable version.
Automatic contract-aware Work YAML resolution is **not implemented yet**: ordinary typed dataset
inputs still require exact content. No certificate silently changes that behavior.

## Required WISDOM changes (not performed by LambdaForge)

For `wisdom-dna-reduced@6`, the three reported hashes are distinct exact representations. Matching
45/47 NPZ entries and 27 tiny eigenvector differences for `10FI_Y` do not establish equivalence of
589 proteins. WISDOM should call `dataset_preflight` before geometries/annotations, expose its
publish/rebuild intent, checkpoint expensive results, declare exact sources/design/labels plus
algorithm contract, and move absolute paths/framework runtime provenance out of NPZ scientific
payloads into manifest provenance/sidecars. Relevant algorithm changes still belong in the contract.
Provide a versioned member verifier that checks all variables, protects labels/splits/source bytes,
and uses justified variable-specific tolerances and spectral invariants. Compare **all** members,
save the report, and publish a new version if bytes differ. Do not modify historical registered NPZs
in place, reuse a conflicting `@6`, or infer a universal tolerance from this one protein.
