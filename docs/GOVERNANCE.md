[English](GOVERNANCE.md) | [Español](GOVERNANCE.es.md)

# Versioning, deprecation and releases

LambdaForge follows Semantic Versioning. Before 1.0, a minor release may contain a documented
breaking change; patch releases must remain compatible with the public API of their minor series.
After 1.0, breaking public API or Schema changes require a major release.

The public API is the set of names re-exported from documented `lambdaforge.*` namespaces plus the
documented YAML/JSON result contracts. Physical module paths not re-exported or documented are
private even when Python can import them.

Deprecations use this lifecycle:

1. Introduce the replacement, emit `DeprecationWarning`, document migration and add a changelog
   entry.
2. Preserve the old public contract for at least one subsequent minor release when technically and
   scientifically safe.
3. Remove it only in a release permitted by the SemVer stage, recording the removal and migration.

Schema changes never silently reinterpret valid science. A new Schema version, a previewable
forward migration and tests for the old form are required. Stored terminal results are append-only;
new readers must tolerate known older result versions.

A release candidate is ready only when version declarations agree, changelog and README are
current, lint/type checks and selected/full tests pass as appropriate, wheel and sdist build, wheel
contents are audited, and the repository contains no caches, environments, credentials or generated
run data. CUDA claims additionally require successful execution on a CUDA-enabled Python build.

After committing 0.5.3, the owner should push the commit, wait for the complete GitHub Actions
matrix, and only then create/push the annotated tag and GitHub Release:

```bash
git push origin main
# Wait for the CI run on this exact commit to be green.
git tag -a v0.5.3 -m "LambdaForge 0.5.3"
git push origin v0.5.3
gh release create v0.5.3 --verify-tag --generate-notes \
  --title "LambdaForge 0.5.3"
```

Do not run these publication commands before CI is green; this repository preparation does not
claim that a remote tag or release has already been created.

The owner must select and approve the repository licence before public redistribution. Adding an
OSI or proprietary licence grants or withholds legal rights and is deliberately not inferred by a
code change.
