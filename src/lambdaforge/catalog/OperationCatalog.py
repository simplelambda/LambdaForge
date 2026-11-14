"""Versioned descriptors for native science and installed entry-point plugins."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata as metadata
import inspect
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.plugins import PluginKind, PluginReference, PluginRegistry


class OperationCatalog:
    """Discover packaged JSON, not scientific modules, input assets or network resources.

    Plugin distributions may ship `lambdaforge.operations.json` beside METADATA. Without it,
    discovery still reports the entry point honestly as contract_status='undeclared'. GUI code
    must not invent ports for those plugins. Selected resolution retains PluginRegistry's checks.
    """

    VERSION = "operation-catalog-v1"

    def __init__(self, plugins: PluginRegistry | None = None) -> None:
        self.plugins = plugins or PluginRegistry.default()

    def discover(self, kind: str | None = None) -> dict[str, Any]:
        native = json.loads(files("lambdaforge.catalog").joinpath("native.json").read_text())
        rows = native["operations"]
        for plugin in self.plugins.discover():
            identity = f"plugin:{plugin.kind.value}:{plugin.name}"
            declaration = None
            if plugin.distribution:
                try:
                    raw = metadata.distribution(plugin.distribution).read_text(
                        "lambdaforge.operations.json"
                    )
                except metadata.PackageNotFoundError:
                    raw = None
                if raw:
                    if len(raw.encode()) > 1024 * 1024:
                        raise ContractError(
                            "CATALOG_LIMIT", "Plugin contract metadata exceeds 1 MiB."
                        )
                    try:
                        document = json.loads(raw)
                        json.dumps(document, allow_nan=False)
                    except (ValueError, TypeError) as error:
                        raise ContractError(
                            "INVALID_CONTRACT", "Invalid plugin contract JSON."
                        ) from error
                    if (
                        not isinstance(document, dict)
                        or document.get("schema_version") != self.VERSION
                    ):
                        raise ContractError(
                            "CONTRACT_VERSION", "Unsupported plugin contract version."
                        )
                    from jsonschema import Draft202012Validator
                    from jsonschema.exceptions import ValidationError

                    try:
                        schema = json.loads(
                            files("lambdaforge.schemas")
                            .joinpath("operation-catalog-v1.json")
                            .read_text()
                        )
                        Draft202012Validator(schema).validate(document)
                        declarations = [v for v in document["operations"] if v["id"] == identity]
                        if len(declarations) > 1:
                            raise ValueError("Duplicate plugin contract identity.")
                        declaration = declarations[0] if declarations else None
                    except (ValueError, KeyError, TypeError, ValidationError) as error:
                        raise ContractError(
                            "INVALID_CONTRACT", "Invalid plugin declarations."
                        ) from error
            rows.append(
                {
                    **(declaration or {}),
                    "id": identity,
                    "kind": plugin.kind.value,
                    "label": plugin.name,
                    "provider": plugin.distribution,
                    "version": plugin.version,
                    "implementation": plugin.value,
                    "aliases": [plugin.name],
                    "contract_status": "declared" if declaration else "undeclared",
                    "plugin": plugin.to_dict(),
                    "preview": {
                        "supported": False,
                        "reason": "Plugin code is not a native bounded preview or a sandbox.",
                    },
                }
            )
        if kind:
            rows = [row for row in rows if row["kind"] == kind]
        return {
            "schema_version": self.VERSION,
            "framework_version": native["framework_version"],
            "operations": sorted(rows, key=lambda row: row["id"]),
        }

    def describe(self, selector: str, *, kind: str | None = None) -> dict[str, Any]:
        rows = self.discover(kind)["operations"]
        exact = [row for row in rows if row["id"] == selector]
        key = self._alias(selector)
        matches = exact or [r for r in rows if any(self._alias(a) == key for a in r["aliases"])]
        if not matches:
            raise ContractError("UNKNOWN_OPERATION", f"Unknown operation {selector!r}.")
        if len(matches) != 1:
            raise ContractError(
                "AMBIGUOUS_ALIAS",
                f"Ambiguous operation {selector!r}.",
                candidates=[m["id"] for m in matches],
            )
        return copy.deepcopy(matches[0])

    def audit(self, *, kind: str | None = None) -> dict[str, Any]:
        """Metadata-only inventory of honest composition/effect completeness, not execution."""
        rows = self.discover(kind)["operations"]
        counts: dict[str, int] = {}
        details = []
        for row in rows:
            status = row.get("composition", {}).get("status", "undeclared")
            counts[status] = counts.get(status, 0) + 1
            details.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "composition": status,
                    "preview": row.get("preview", {"supported": False}),
                }
            )
        return {"schema_version": "operation-audit-v1", "counts": counts, "operations": details}

    def resolve(self, selector: str, *, kind: str | None = None) -> type[Any]:
        row = self.describe(selector, kind=kind)
        try:
            if row.get("plugin"):
                return self.plugins.resolve(PluginReference(PluginKind(row["kind"]), row["label"]))
            module, name = row["implementation"].split(":")
            selected = getattr(importlib.import_module(module), name)
            if not isinstance(selected, type):
                raise ContractError("INVALID_CONTRACT", "Selected implementation is not a class.")
            return selected
        except ImportError as error:
            raise ContractError("MISSING_DEPENDENCY", str(error), operation=row["id"]) from error

    def validate(
        self, selector: str, parameters: Mapping[str, Any], *, kind: str | None = None
    ) -> dict[str, Any]:
        """Validate selected constructor binding, without constructing/training a model."""
        descriptor = self.describe(selector, kind=kind)
        selected = self.resolve(descriptor["id"], kind=kind)
        if inspect.isabstract(selected):
            raise ContractError("ABSTRACT_OPERATION", "Select a concrete native operation.")
        try:
            inspect.signature(selected).bind(**parameters)
        except TypeError as error:
            raise ContractError("INVALID_PARAMETERS", str(error), operation=selector) from error
        from jsonschema import Draft202012Validator

        types = {"integer", "number", "boolean", "string"}
        properties: dict[str, dict[str, Any]] = {}
        for p in descriptor.get("parameters", []):
            constraints: dict[str, Any] = {"type": p["type"]} if p["type"] in types else {}
            if "options" in p:
                constraints["enum"] = p["options"]
            if "minimum" in p:
                constraints["minimum"] = p["minimum"]
            if "maximum" in p:
                constraints["maximum"] = p["maximum"]
            resource = p.get("schema_resource")
            if resource:
                if resource not in {"csv-options-v1.json", "table-schema-v1.json"}:
                    raise ContractError("INVALID_CONTRACT", "Unsupported schema resource.")
                nested = json.loads(files("lambdaforge.schemas").joinpath(resource).read_text())
                constraints = (
                    {"anyOf": [nested, {"type": "null"}]} if not p.get("required") else nested
                )
            properties[p["name"]] = constraints
        errors = list(
            Draft202012Validator({"type": "object", "properties": properties}).iter_errors(
                dict(parameters)
            )
        )
        if errors:
            raise ContractError(
                "INVALID_PARAMETERS",
                "Selected constructor types are incompatible.",
                fields=[list(e.path) for e in errors],
            )
        return descriptor

    def validate_descriptor(self, selector: str, *, kind: str | None = None) -> dict[str, Any]:
        """Compare selected installed signatures with the packaged descriptor, without calls."""
        descriptor = self.describe(selector, kind=kind)
        selected = self.resolve(descriptor["id"])
        if bool(descriptor.get("abstract")) != inspect.isabstract(selected):
            raise ContractError(
                "CONTRACT_DRIFT", "Abstract/concrete status differs from installed class."
            )
        if descriptor.get("constructor_status") == "declared":
            actual = {
                k: v
                for k, v in inspect.signature(selected).parameters.items()
                if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
            advertised = {p["name"]: p for p in descriptor["parameters"]}
            if set(actual) != set(advertised) or any(
                (v.default is inspect.Parameter.empty) != advertised[k]["required"]
                for k, v in actual.items()
            ):
                raise ContractError(
                    "CONTRACT_DRIFT", "Constructor signature differs from installed metadata."
                )
            for key, value in actual.items():
                default = advertised[key].get("default")
                if (
                    value.default is inspect.Parameter.empty
                    or isinstance(default, dict)
                    and "python" in default
                ):
                    continue
                try:
                    encoded = json.loads(json.dumps(value.default, allow_nan=False))
                except (TypeError, ValueError):
                    continue  # Symbolic Python defaults are not JSON-equality contracts.
                if encoded != default:
                    raise ContractError(
                        "CONTRACT_DRIFT",
                        "Constructor default differs from installed metadata.",
                        field=key,
                    )
        for method in descriptor.get("methods", []):
            actual_method = getattr(selected, method["name"], None)
            if actual_method is None:
                raise ContractError("CONTRACT_DRIFT", "Advertised method does not exist.")
            signature = inspect.signature(actual_method)
            if any(p["name"] not in signature.parameters for p in method["inputs"]):
                raise ContractError(
                    "CONTRACT_DRIFT", "Advertised port differs from installed method."
                )
        return {
            "schema_version": "operation-validation-v1",
            "id": descriptor["id"],
            "valid": True,
            "scope": "signatures-not-tensor-equations",
            "abstract": inspect.isabstract(selected),
        }

    @staticmethod
    def check_ports(source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        """Reject explicitly incompatible contracts; unknown dimensions remain unresolved."""
        for key in ("kind", "dtype", "units"):
            if source.get(key) and target.get(key) and source[key] != target[key]:
                raise ContractError("INCOMPATIBLE_PORT", f"Port {key} differs.", field=key)
        left, right = source.get("shape"), target.get("shape")
        if left is not None and right is not None and ("..." in left or "..." in right):
            # The leading ellipsis is a variable batch prefix, not an extra fixed axis.
            if left[0] == "..." and right[0] != "...":
                right = right[-(len(left) - 1) :] if len(left) > 1 else []
                left = left[1:]
            elif right[0] == "..." and left[0] != "...":
                left = left[-(len(right) - 1) :] if len(right) > 1 else []
                right = right[1:]
        if (
            left is not None
            and right is not None
            and (
                len(left) != len(right)
                or any(
                    isinstance(a, int) and isinstance(b, int) and a != b
                    for a, b in zip(left, right, strict=False)
                )
            )
        ):
            raise ContractError("INCOMPATIBLE_SHAPE", "Tensor shapes differ.")

    @staticmethod
    def _alias(value: str) -> str:
        return value.strip().casefold().replace("_", "").replace("-", "")
