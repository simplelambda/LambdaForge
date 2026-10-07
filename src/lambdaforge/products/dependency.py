"""Explicit Work product inputs: immutable contract/meaning resolution, not a second runner."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.products.models import ProductContract, StudyProduct, _mapping
from lambdaforge.products.registry import ProductRegistry
from lambdaforge.ProjectContext import ProjectContext


@dataclass(frozen=True, slots=True)
class ProductRequirement:
    """Exact immutable product selector and explicit semantic compatibility expectations."""

    name: str
    contract: str
    expect: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 256:
            raise ValueError("Product requirement name must be an exact non-empty selector.")
        ProductContract(self.contract, ("declaration",))
        object.__setattr__(self, "expect", _mapping(self.expect, field="expectations"))

    @classmethod
    def from_mapping(cls, value: Any) -> ProductRequirement:
        if (
            not isinstance(value, Mapping)
            or set(value) - {"name", "contract", "expect"}
            or not {"name", "contract"}.issubset(value)
        ):
            raise ValueError("Product input requires {product: {name: NAME, contract: CONTRACT}}.")
        return cls(value["name"], value["contract"], value.get("expect", {}))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "contract": self.contract, "expect": dict(self.expect)}


@dataclass(frozen=True, slots=True)
class ProductInput:
    """Read-only metadata and explicit verified artifact access supplied to Work.run().

    Metadata access does not read model bytes. ``artifact(name)`` verifies precisely that promoted
    file before returning its path. Treat input paths as read-only, like ordinary file inputs.
    Do not guess producer Job paths or use the producer YAML/status as dependency authority.
    """

    product: StudyProduct
    _root: Path

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.product.payload

    @property
    def content_id(self) -> str:
        return self.product.content_id

    def artifact(self, name: str) -> Path:
        return ProductRegistry(self._root).artifact_path(self.content_id, name)


def resolve_product_input(value: Any, source_dir: Path) -> tuple[ProductInput, Path]:
    """Resolve declared compatibility from the execution project's catalog, without writes."""
    requirement = ProductRequirement.from_mapping(value)
    root = os.environ.get("LAMBDAFORGE_PRODUCT_ROOT")
    if root is None:
        if os.environ.get("LAMBDAFORGE_EXECUTION_MODE") == "worker":
            raise ValueError(
                "Worker product root was not supplied by the control plane; "
                "prepare this Job with the current LambdaForge runtime."
            )
        root = str(ProjectContext.discover(source_dir).root / ".lambdaforge" / "products")
    registry = ProductRegistry(root)
    product = registry.resolve(
        requirement.name,
        contract=requirement.contract,
        scientific_expectations=requirement.expect,
    )
    manifest = (
        registry.root / "objects" / product.content_id.removeprefix("sha256:") / "manifest.json"
    )
    return ProductInput(product, registry.root), manifest


def product_identity(product: ProductInput) -> Mapping[str, Any]:
    """Compatibility depends on sealed science/content, never the full producer YAML hash."""
    return FrozenJsonMapping(
        {
            "product": product.content_id,
            "contract": product.product.contract.identifier,
            "scientific_id": product.product.scientific_id,
        }
    )
