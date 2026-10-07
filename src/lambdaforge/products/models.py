"""Immutable product contracts and identities, independent of producer configuration/placement.

These models do not certify numerical equivalence. The publisher owns
byte verification; the project explicitly owns the scientific fields in its versioned contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity

_CONTRACT = re.compile(r"[a-z][a-z0-9_-]*(?:/[a-z][a-z0-9_-]*)*:v[1-9][0-9]*\Z")
_KINDS = {
    "StudyDecision",
    "ModelArtifact",
    "ModelSet",
    "Selection",
    "AnalysisResult",
    "Dataset",
    "ScientificReport",
}


def _json_value(value: Any) -> Any:
    """Copy strict JSON without coercing path objects, non-string keys or non-finite evidence."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Product JSON keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise TypeError("Product evidence must be strict finite JSON, not paths or Python objects.")


def _mapping(value: Mapping[str, Any], *, field: str) -> FrozenJsonMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"Product {field} must be a JSON object.")
    return FrozenJsonMapping(_json_value(value))


def _text(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Product {field} must be non-empty text of at most 256 characters.")


@dataclass(frozen=True, slots=True)
class ProductContract:
    """Exact versioned meaning; compatibility is explicit, never inferred from a producer YAML.

    Scientific fields are a declaration, not arbitrary format exclusions. A consumer can inspect
    the contract and require exact expected meanings. Changing this declaration requires a new
    contract version; equal identifiers with different declarations are conflicting contracts.
    """

    identifier: str
    scientific_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identifier, str)
            or len(self.identifier) > 256
            or not _CONTRACT.fullmatch(self.identifier)
        ):
            raise ValueError(
                "Product contract must be a versioned identifier such as project/selection:v1."
            )
        if not isinstance(self.scientific_fields, tuple | list) or not self.scientific_fields:
            raise ValueError("Product contract must declare its scientific fields explicitly.")
        fields = tuple(self.scientific_fields)
        for field in fields:
            _text(field, field="scientific field")
        if len(set(fields)) != len(fields):
            raise ValueError("Product contract scientific fields must be unique.")
        object.__setattr__(self, "scientific_fields", tuple(sorted(fields)))

    def validate(self, meaning: Mapping[str, Any]) -> FrozenJsonMapping:
        """Require every declared scientific field, with no silently ignored extra meanings."""
        frozen = _mapping(meaning, field="scientific meaning")
        required = set(self.scientific_fields)
        supplied = set(frozen)
        if supplied != required:
            raise ValueError(
                f"Product contract {self.identifier} fields differ: "
                f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}."
            )
        return frozen

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_contract_version": 1,
            "identifier": self.identifier,
            "scientific_fields": list(self.scientific_fields),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProductContract:
        if (
            type(value.get("product_contract_version")) is not int
            or value["product_contract_version"] != 1
        ):
            raise ValueError("Unsupported ProductContract schema version.")
        if not isinstance(value.get("scientific_fields"), list):
            raise ValueError("ProductContract scientific_fields must be an array.")
        return cls(value["identifier"], tuple(value["scientific_fields"]))


@dataclass(frozen=True, slots=True)
class ProductArtifact:
    """Exact independently promoted bytes; this descriptor is not itself proof of a file."""

    name: str
    path: str
    sha256: str
    size_bytes: int
    role: str = "artifact"

    def __post_init__(self) -> None:
        _text(self.name, field="artifact name")
        _text(self.role, field="artifact role")
        if (
            not isinstance(self.path, str)
            or not self.path
            or "\\" in self.path
            or ":" in self.path
            or any(ord(character) < 32 for character in self.path)
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
            or PurePosixPath(self.path).is_absolute()
        ):
            raise ValueError(
                "Product artifact path must be a portable relative path without traversal."
            )
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Product artifact requires an exact SHA-256 byte checksum.")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("Product artifact size_bytes must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class StudyProduct:
    """Sealed scientific meaning and exact content, with operational provenance kept separate."""

    name: str
    kind: str
    contract: ProductContract
    payload: Mapping[str, Any]
    scientific_meaning: Mapping[str, Any]
    producer: Mapping[str, Any]
    artifacts: tuple[ProductArtifact, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, field="name")
        if self.kind not in _KINDS:
            raise ValueError(f"Unsupported product kind {self.kind!r}.")
        if not isinstance(self.contract, ProductContract):
            raise TypeError("StudyProduct requires an explicit ProductContract.")
        object.__setattr__(self, "payload", _mapping(self.payload, field="payload"))
        object.__setattr__(
            self, "scientific_meaning", self.contract.validate(self.scientific_meaning)
        )
        producer = _mapping(self.producer, field="producer")
        for field in ("execution_id", "evidence_fingerprint"):
            _text(producer.get(field), field=f"producer.{field}")
        object.__setattr__(self, "producer", producer)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ProductArtifact) for item in artifacts):
            raise TypeError("StudyProduct artifacts must be ProductArtifact descriptors.")
        if len({item.name for item in artifacts}) != len(artifacts) or len(
            {item.path for item in artifacts}
        ) != len(artifacts):
            raise ValueError("Product artifact names and paths must be unique.")
        paths = {item.path for item in artifacts}
        if any(
            path.split("/")[0] in {"manifest.json", "provenance", "consumers"} for path in paths
        ) or any(str(parent) in paths for path in paths for parent in PurePosixPath(path).parents):
            raise ValueError("Product artifact paths overlap or collide with owned metadata.")
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.name)))

    @property
    def scientific_id(self) -> str:
        """Identify explicitly declared meaning, never the complete producer config or placement."""
        return ScientificIdentity.from_payload(
            {
                "study_product_identity_version": 1,
                "kind": self.kind,
                "contract": self.contract.to_dict(),
                "meaning": dict(self.scientific_meaning),
            }
        ).digest

    def content_descriptor(self) -> dict[str, Any]:
        """Canonical publication contents; aliases, producer history and locations are not bytes."""
        return {
            "study_product_version": 1,
            "kind": self.kind,
            "contract": self.contract.to_dict(),
            "scientific_meaning": dict(self.scientific_meaning),
            "payload": dict(self.payload),
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @property
    def content_id(self) -> str:
        encoded = json.dumps(
            self.content_descriptor(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.content_descriptor(),
            "name": self.name,
            "producer": dict(self.producer),
            "scientific_id": self.scientific_id,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StudyProduct:
        if (
            type(value.get("study_product_version")) is not int
            or value["study_product_version"] != 1
        ):
            raise ValueError("Unsupported StudyProduct schema version.")
        product = cls(
            value["name"],
            value["kind"],
            ProductContract.from_dict(value["contract"]),
            value["payload"],
            value["scientific_meaning"],
            value["producer"],
            tuple(ProductArtifact(**item) for item in value.get("artifacts", ())),
        )
        if (
            value.get("content_id") != product.content_id
            or value.get("scientific_id") != product.scientific_id
        ):
            raise ValueError(
                "StudyProduct content or scientific identity does not match its manifest."
            )
        return product
