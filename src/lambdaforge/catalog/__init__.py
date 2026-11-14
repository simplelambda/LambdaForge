"""Metadata-only scientific operation discovery; no model imports on discovery."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.catalog.ComponentFactory import ComponentFactory
    from lambdaforge.catalog.errors import ContractError
    from lambdaforge.catalog.OperationCatalog import OperationCatalog

LazyExports.install(
    __name__,
    {
        "OperationCatalog": ("lambdaforge.catalog.OperationCatalog", "OperationCatalog"),
        "ComponentFactory": ("lambdaforge.catalog.ComponentFactory", "ComponentFactory"),
        "ContractError": ("lambdaforge.catalog.errors", "ContractError"),
    },
)
__all__ = ["OperationCatalog", "ComponentFactory", "ContractError"]
