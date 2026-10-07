"""Lazy scientific product exports; catalog reads do not load scientific Analysis/HPO."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.products.bundle import ProductBundle
    from lambdaforge.products.decision import build_study_decision
    from lambdaforge.products.dependency import ProductInput, ProductRequirement
    from lambdaforge.products.models import ProductArtifact, ProductContract, StudyProduct
    from lambdaforge.products.publication import ProductPublication, publish_declared_products
    from lambdaforge.products.registry import ProductRegistry
    from lambdaforge.products.selection import ModelSelection, SelectionPolicy, select_models

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.products.{module}", name)
        for module, names in {
            "bundle": ("ProductBundle",),
            "decision": ("build_study_decision",),
            "dependency": ("ProductInput", "ProductRequirement"),
            "models": ("ProductArtifact", "ProductContract", "StudyProduct"),
            "publication": ("ProductPublication", "publish_declared_products"),
            "registry": ("ProductRegistry",),
            "selection": ("ModelSelection", "SelectionPolicy", "select_models"),
        }.items()
        for name in names
    },
)

__all__ = [
    "ModelSelection",
    "ProductArtifact",
    "ProductBundle",
    "ProductContract",
    "ProductInput",
    "ProductRegistry",
    "ProductPublication",
    "ProductRequirement",
    "SelectionPolicy",
    "StudyProduct",
    "build_study_decision",
    "select_models",
    "publish_declared_products",
]
