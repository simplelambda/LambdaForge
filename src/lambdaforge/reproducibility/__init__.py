"""Reproducibility profiles, identities, seeds and environment exports."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
    from lambdaforge.reproducibility.EnvironmentExporter import EnvironmentExporter
    from lambdaforge.reproducibility.ExecutionIdentity import ExecutionIdentity
    from lambdaforge.reproducibility.IdentityExplainer import IdentityExplainer
    from lambdaforge.reproducibility.IdentityExplanation import IdentityExplanation
    from lambdaforge.reproducibility.ReproducibilityProfile import ReproducibilityProfile
    from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
    from lambdaforge.reproducibility.SeedDeriver import SeedDeriver

LazyExports.install(
    __name__,
    {
        "CodeIdentity": ("lambdaforge.reproducibility.CodeIdentity", "CodeIdentity"),
        "EnvironmentExporter": (
            "lambdaforge.reproducibility.EnvironmentExporter",
            "EnvironmentExporter",
        ),
        "ExecutionIdentity": (
            "lambdaforge.reproducibility.ExecutionIdentity",
            "ExecutionIdentity",
        ),
        "IdentityExplainer": (
            "lambdaforge.reproducibility.IdentityExplainer",
            "IdentityExplainer",
        ),
        "IdentityExplanation": (
            "lambdaforge.reproducibility.IdentityExplanation",
            "IdentityExplanation",
        ),
        "ReproducibilityProfile": (
            "lambdaforge.reproducibility.ReproducibilityProfile",
            "ReproducibilityProfile",
        ),
        "ScientificIdentity": (
            "lambdaforge.reproducibility.ScientificIdentity",
            "ScientificIdentity",
        ),
        "SeedDeriver": ("lambdaforge.reproducibility.SeedDeriver", "SeedDeriver"),
    },
)

__all__ = [
    "CodeIdentity",
    "EnvironmentExporter",
    "ExecutionIdentity",
    "IdentityExplainer",
    "IdentityExplanation",
    "ReproducibilityProfile",
    "ScientificIdentity",
    "SeedDeriver",
]
