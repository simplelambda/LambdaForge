"""Explicit launch-time typed input selection, preserving the authored YAML and source directory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def pin_bindings(bindings: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Resolve exact metadata identities; never copy or execute producer content."""
    from lambdaforge.products.dependency import ProductRequirement, resolve_product_input
    from lambdaforge.work.ResultInput import resolve_result_input

    pinned = {}
    for name, marker in bindings.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(marker, Mapping):
            raise ValueError(
                "Input bindings require parameter names and typed product/result inputs."
            )
        if set(marker) == {"result"}:
            result, _ = resolve_result_input(marker["result"], source.parent)
            pinned[name] = {"result": result.requirement.to_dict()}
        elif set(marker) == {"product"}:
            requirement = ProductRequirement.from_mapping(
                marker["product"], source_dir=source.parent
            )
            product, _ = resolve_product_input(marker["product"], source.parent)
            pinned[name] = {"product": {**requirement.to_dict(), "name": product.content_id}}
        else:
            raise ValueError("Only explicit historical ResultInput/ProductInput may be bound.")
    return pinned


def bind_document(values: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Selection is explicit, never an edit of authored files or arbitrary execution options."""
    if "steps" in values:
        raise ValueError(
            "Launch input selection requires one Work; declare composed inputs in YAML."
        )
    return {**values, "with": {**values.get("with", {}), **bindings}}


def parse_bindings(values: list[str]) -> dict[str, Any]:
    import json

    bindings = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or name in bindings:
            raise ValueError("Use --input-ref PARAMETER=JSON once per selected input.")
        bindings[name] = json.loads(encoded)
    return bindings
