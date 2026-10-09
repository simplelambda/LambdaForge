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
    artifact: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 256:
            raise ValueError("Product requirement name must be an exact non-empty selector.")
        ProductContract(self.contract, ("declaration",))
        if self.artifact is not None and (
            not isinstance(self.artifact, str)
            or not self.artifact.strip()
            or len(self.artifact) > 256
        ):
            raise ValueError("Product artifact must be an exact non-empty registered name.")
        object.__setattr__(self, "expect", _mapping(self.expect, field="expectations"))

    @classmethod
    def from_mapping(cls, value: Any, *, source_dir: Path | None = None) -> ProductRequirement:
        if isinstance(value, Mapping) and "from" in value:
            return historical_product_requirement(value, source_dir)
        if (
            not isinstance(value, Mapping)
            or set(value) - {"name", "contract", "expect", "artifact"}
            or not {"name", "contract"}.issubset(value)
        ):
            raise ValueError("Product input requires {product: {name: NAME, contract: CONTRACT}}.")
        return cls(value["name"], value["contract"], value.get("expect", {}), value.get("artifact"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "contract": self.contract, "expect": dict(self.expect)} | (
            {"artifact": self.artifact} if self.artifact is not None else {}
        )


@dataclass(frozen=True, slots=True)
class ProductInput:
    """Read-only metadata and explicit verified artifact access supplied to Work.run().

    Metadata access does not read model bytes. ``artifact(name)`` verifies precisely that promoted
    file before returning its path. Treat input paths as read-only, like ordinary file inputs.
    Do not guess producer Job paths or use the producer YAML/status as dependency authority.
    """

    product: StudyProduct
    _root: Path
    selected_artifact: str | None = None

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.product.payload

    @property
    def content_id(self) -> str:
        return self.product.content_id

    def artifact(self, name: str | None = None) -> Path:
        selected = self.selected_artifact if name is None else name
        if selected is None:
            raise ValueError("Choose a registered product artifact explicitly.")
        if self.selected_artifact is not None and selected != self.selected_artifact:
            raise ValueError(
                "Requested artifact differs from the pinned product artifact selection."
            )
        return ProductRegistry(self._root).artifact_path(self.content_id, selected)


def resolve_product_input(value: Any, source_dir: Path) -> tuple[ProductInput, Path]:
    """Resolve declared compatibility from the execution project's catalog, without writes."""
    requirement = ProductRequirement.from_mapping(value, source_dir=source_dir)
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
    if (
        requirement.artifact is not None
        and len([row for row in product.artifacts if row.name == requirement.artifact]) != 1
    ):
        raise ValueError("Selected artifact is not uniquely registered in the exact product.")
    manifest = (
        registry.root / "objects" / product.content_id.removeprefix("sha256:") / "manifest.json"
    )
    return ProductInput(product, registry.root, requirement.artifact), manifest


def product_identity(product: ProductInput) -> Mapping[str, Any]:
    """Compatibility depends on sealed science/content, never the full producer YAML hash."""
    return FrozenJsonMapping(
        {
            "product": product.content_id,
            "contract": product.product.contract.identifier,
            "scientific_id": product.product.scientific_id,
        }
        | ({"artifact": product.selected_artifact} if product.selected_artifact is not None else {})
    )


def historical_product_requirement(
    value: Mapping[str, Any],
    source_dir: Path | None = None,
    *,
    results_root: Path | None = None,
) -> ProductRequirement:
    """Resolve a published native output, inferring a contract only from its exact receipt."""
    from lambdaforge.work.ResultStore import ResultStore

    if set(value) - {"from", "contract", "expect", "artifact"}:
        raise ValueError(
            "Historical product input supports only from, contract, expect and artifact."
        )
    origin = value["from"]
    if not isinstance(origin, Mapping) or set(origin) != {"execution", "output"}:
        raise ValueError(
            "Product from requires {execution: NAME_OR_ID, output: PUBLISHED_PRODUCT}."
        )
    if any(not isinstance(origin[key], str) or not origin[key].strip() for key in origin):
        raise ValueError("Historical product execution/output must be non-empty selectors.")
    configured = (
        str(results_root) if results_root else os.environ.get("LAMBDAFORGE_RESULT_INPUT_ROOT")
    )
    root = (
        configured or ProjectContext.discover(source_dir or Path.cwd()).root / ".lambdaforge/runs"
    )
    store = ResultStore(root)
    selected = store.select(origin["execution"])
    if selected.get("already_deleted"):
        raise ValueError(
            "Producer history was deleted; select the independent product by name/contract."
        )
    directory = store._evidence_dir(Path(selected["_manifest_path"]))
    record = ProductRegistry._read(directory / "products.json")
    if (
        record.get("product_publication_version") != 1
        or record.get("execution_id") != selected["execution_id"]
    ):
        raise ValueError("Historical product receipt differs from its Execution.")
    matches = [
        item
        for item in record.get("items", ())
        if item.get("name") == origin["output"] and item.get("status") == "published"
    ]
    if len(matches) != 1:
        raise ValueError(
            "No unique published product for the chosen Execution/output; "
            "finalize publication first."
        )
    published = matches[0]
    contract = value.get("contract", published["contract"])
    if contract != published["contract"]:
        raise ValueError(
            "Historical published product contract differs from the requested contract."
        )
    return ProductRequirement(
        published["content_id"], contract, value.get("expect", {}), value.get("artifact")
    )
