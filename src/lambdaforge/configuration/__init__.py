"""Composable, inspectable and secret-aware configuration."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
    from lambdaforge.configuration.AuthoringConfigNormalizer import AuthoringConfigNormalizer
    from lambdaforge.configuration.AuthoringSchemaCatalog import AuthoringSchemaCatalog
    from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer
    from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff
    from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
    from lambdaforge.configuration.MaterializedConfig import MaterializedConfig
    from lambdaforge.configuration.ProjectConfigRecord import ProjectConfigRecord
    from lambdaforge.configuration.ProjectConfigService import ProjectConfigService
    from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration
    from lambdaforge.configuration.SecretValue import SecretValue

LazyExports.install(
    __name__,
    {
        "AuthoringConfig": ("lambdaforge.configuration.AuthoringConfig", "AuthoringConfig"),
        "AuthoringConfigNormalizer": (
            "lambdaforge.configuration.AuthoringConfigNormalizer",
            "AuthoringConfigNormalizer",
        ),
        "AuthoringSchemaCatalog": (
            "lambdaforge.configuration.AuthoringSchemaCatalog",
            "AuthoringSchemaCatalog",
        ),
        "ConfigurationComposer": (
            "lambdaforge.configuration.ConfigurationComposer",
            "ConfigurationComposer",
        ),
        "ConfigurationDiff": (
            "lambdaforge.configuration.ConfigurationDiff",
            "ConfigurationDiff",
        ),
        "ConfigurationKind": (
            "lambdaforge.configuration.ConfigurationKind",
            "ConfigurationKind",
        ),
        "MaterializedConfig": (
            "lambdaforge.configuration.MaterializedConfig",
            "MaterializedConfig",
        ),
        "ProjectConfigRecord": (
            "lambdaforge.configuration.ProjectConfigRecord",
            "ProjectConfigRecord",
        ),
        "ProjectConfigService": (
            "lambdaforge.configuration.ProjectConfigService",
            "ProjectConfigService",
        ),
        "ResolvedConfiguration": (
            "lambdaforge.configuration.ResolvedConfiguration",
            "ResolvedConfiguration",
        ),
        "SecretValue": ("lambdaforge.configuration.SecretValue", "SecretValue"),
    },
)

__all__ = [
    "AuthoringConfig",
    "AuthoringConfigNormalizer",
    "AuthoringSchemaCatalog",
    "ConfigurationComposer",
    "ConfigurationDiff",
    "ConfigurationKind",
    "MaterializedConfig",
    "ProjectConfigRecord",
    "ProjectConfigService",
    "ResolvedConfiguration",
    "SecretValue",
]
