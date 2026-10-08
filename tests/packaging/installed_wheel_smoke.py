"""Smoke an installed LambdaForge wheel from outside its source checkout."""

from __future__ import annotations

from importlib.metadata import distribution
from importlib.resources import files

import lambdaforge
from lambdaforge import Work


def main() -> None:
    """Verify imports, packaged resources and one minimal authoring operation."""
    installed = distribution("lambdaforge")
    assert lambdaforge.__version__ == installed.version
    assert files("lambdaforge").joinpath("schemas/work.schema.json").is_file()
    installed_files = tuple(installed.files or ())
    for relative in (
        "share/lambdaforge/AGENTS.md",
        "share/lambdaforge/AGENTS.es.md",
        "share/lambdaforge/CHANGELOG.md",
        "share/lambdaforge/CHANGELOG.es.md",
        "share/lambdaforge/README.md",
        "share/lambdaforge/README.es.md",
        "share/lambdaforge/SECURITY.md",
        "share/lambdaforge/SECURITY.es.md",
        "share/lambdaforge/examples/work.yaml",
        "share/lambdaforge/examples/sequence.yaml",
        "share/lambdaforge/docs/MANUAL.md",
        "share/lambdaforge/docs/MANUAL.es.md",
        "share/lambdaforge/docs/STORAGE.md",
        "share/lambdaforge/docs/STORAGE.es.md",
        "share/lambdaforge/docs/DATASET_RECONSTRUCTION.md",
        "share/lambdaforge/docs/DATASET_RECONSTRUCTION.es.md",
        "share/lambdaforge/docs/ARCHITECTURAL_REFORM.md",
        "share/lambdaforge/docs/ARCHITECTURAL_REFORM.es.md",
        "share/lambdaforge/docs/SCIENTIFIC_AUDIT.md",
        "share/lambdaforge/docs/SCIENTIFIC_AUDIT.es.md",
        "share/lambdaforge/docs/PRODUCTS.md",
        "share/lambdaforge/docs/PRODUCTS.es.md",
    ):
        matches = tuple(
            item
            for item in installed_files
            if item.as_posix().endswith(relative) and installed.locate_file(item).is_file()
        )
        assert len(matches) == 1, relative
    scripts = {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["lf"] == scripts["lambdaforge"]
    assert lambdaforge.__all__ == ["Work", "__version__", "clustering"]
    assert Work.__module__ == "lambdaforge.work.Work"
    from lambdaforge.products import (
        ProductBundle,
        ProductContract,
        ProductInput,
        ProductPublication,
        ProductRegistry,
        StudyProduct,
        build_study_decision,
        publish_declared_products,
    )

    assert ProductContract("example/report:v1", ("sources",)).identifier == "example/report:v1"
    assert callable(ProductBundle.import_bundle)
    assert callable(ProductRegistry.resolve)
    assert callable(StudyProduct.from_dict)
    assert callable(ProductInput.artifact)
    assert callable(ProductPublication.from_mapping)
    assert callable(build_study_decision)
    assert callable(publish_declared_products)
    from lambdaforge.data import DatasetEquivalenceCertificate
    from lambdaforge.work import ResultStore

    assert callable(DatasetEquivalenceCertificate.build)
    assert callable(ResultStore.import_export)
    from lambdaforge.hpo.SequentialSweep import PairedSweepSequentialAnalyzer
    from lambdaforge.work.sweep_blocks import sweep_block_inventory

    decision = PairedSweepSequentialAnalyzer(mode="max", bounds=(0, 1)).evaluate(
        {1: {54: 0.5}, 2: {54: 0.6}}, seed_order=[54]
    )
    assert decision.policy_version == "paired-pm-eb-cs-v2"
    assert decision.evidence_seeds == (54,)
    assert callable(sweep_block_inventory)
    print(lambdaforge.__version__)


if __name__ == "__main__":
    main()
