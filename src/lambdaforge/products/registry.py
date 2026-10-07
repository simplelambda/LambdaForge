"""Preview-first local product publication with immutable independent byte ownership.

Read paths never create directories. The native Work/Study publisher must supply producer evidence
and the operational product root; this store does not launch Studies or infer scientific meaning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.StorageAdmission import StorageAdmission
from lambdaforge.products.models import ProductArtifact, ProductContract, StudyProduct
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import _fsync_directory, atomic_write_bytes

_MAX_METADATA_BYTES = 512 * 1024


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class ProductRegistry:
    """One project product catalog; exact names are immutable bindings, not config selectors."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("LAMBDAFORGE_PRODUCT_ROOT")
        self.root = (
            Path(configured or ProjectContext.discover().root / ".lambdaforge/products")
            .expanduser()
            .absolute()
        )
        if self.root.resolve() != self.root:
            raise ValueError("Product registry requires an absolute non-symlinked root.")

    def _path(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        if path.resolve() != path or not path.is_relative_to(self.root):
            raise ValueError("Product path is symbolic or outside its owned registry.")
        return path

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError(f"Product requires a regular persisted file: {path}")
        if path.stat().st_size > _MAX_METADATA_BYTES:
            raise ValueError(
                f"Product metadata exceeds its {_MAX_METADATA_BYTES} byte contract: {path}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Corrupt product metadata: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Product metadata must be an object: {path}")
        return value

    def _write(self, path: Path, value: Any) -> None:
        """Use the same byte contract for persisted and transferred metadata."""
        encoded = _encoded(value)
        if len(encoded) > _MAX_METADATA_BYTES:
            raise ValueError("Product metadata exceeds the bounded metadata contract.")
        if self._path(*path.relative_to(self.root).parts) != path:
            raise ValueError("Product metadata escaped its registry.")
        atomic_write_bytes(path, encoded)

    def read_manifest(self, path: str | Path) -> StudyProduct:
        """Read an explicitly supplied portable manifest, not a catalog lookup or publication."""
        return StudyProduct.from_dict(self._read(Path(path).expanduser().absolute()))

    @staticmethod
    def _key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _alias(self, name: str) -> Path:
        if not isinstance(name, str) or not name.strip() or len(name) > 256:
            raise ValueError("Product name must be non-empty text of at most 256 characters.")
        return self._path("names", self._key(name) + ".json")

    def _object(self, content_id: str) -> Path:
        if not isinstance(content_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", content_id):
            raise ValueError("Product content ID must be an exact SHA-256 identity.")
        return self._path("objects", content_id.removeprefix("sha256:"))

    def show(self, selector: str) -> StudyProduct:
        """Read one bounded manifest; byte verification is the explicit verify operation."""
        if not isinstance(selector, str):
            raise ValueError("Product selector must be an exact name or SHA-256 identity.")
        alias_name: str | None = None
        if selector.startswith("sha256:"):
            content_id = selector
        else:
            path = self._alias(selector)
            if not path.exists():
                raise KeyError(f"Unknown product {selector!r}.")
            alias = self._read(path)
            if (
                type(alias.get("product_binding_version")) is not int
                or alias["product_binding_version"] != 1
                or alias.get("name") != selector
                or "content_id" not in alias
            ):
                raise ValueError("Product name binding is corrupt.")
            content_id = alias["content_id"]
            alias_name = selector
        manifest = self._object(content_id) / "manifest.json"
        try:
            product = StudyProduct.from_dict(self._read(manifest))
        except (KeyError, TypeError) as error:
            raise ValueError(f"Corrupt product manifest: {manifest}") from error
        if product.content_id != content_id:
            raise ValueError("Product object directory and content identity disagree.")
        return replace(product, name=alias_name) if alias_name else product

    def list(self, *, offset: int = 0, limit: int = 100) -> tuple[StudyProduct, ...]:
        """Page product metadata only; never open checkpoint or artifact contents."""
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Product metadata pages require offset >= 0 and limit 1–1000.")
        names = self._path("names")
        if not names.exists():
            return ()
        paths = sorted(names.glob("*.json"))[offset : offset + limit]
        products = []
        for path in paths:
            alias = self._read(path)
            name = alias.get("name")
            if not isinstance(name, str) or self._alias(name) != path:
                raise ValueError("Product name binding path is corrupt.")
            products.append(self.show(name))
        return tuple(products)

    def resolve(
        self,
        selector: str,
        *,
        contract: ProductContract | str,
        scientific_expectations: Mapping[str, Any] | None = None,
    ) -> StudyProduct:
        """Resolve an exact contract/meaning without depending on producer config or status."""
        product = self.show(selector)
        identifier = contract.identifier if isinstance(contract, ProductContract) else contract
        if product.contract.identifier != identifier or (
            isinstance(contract, ProductContract) and product.contract != contract
        ):
            raise ValueError(
                f"Product contract mismatch: expected {identifier}, "
                f"observed {product.contract.identifier}."
            )
        for field, expected in (scientific_expectations or {}).items():
            if field not in product.scientific_meaning or _encoded(expected) != _encoded(
                product.scientific_meaning[field]
            ):
                raise ValueError(f"Product scientific expectation differs for {field!r}.")
        return product

    @staticmethod
    def _source(path: Path, artifact: ProductArtifact) -> None:
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Product source must be an existing non-symlinked regular file: {path}"
            )
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"Product source size differs for {artifact.name!r}.")

    @staticmethod
    def _verify_file(path: Path, artifact: ProductArtifact) -> None:
        ProductRegistry._source(path, artifact)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(f"Product artifact checksum differs for {artifact.name!r}.")

    def verify(self, selector: str) -> dict[str, Any]:
        """Explicitly verify every independent object byte, not just its manifest claim."""
        product = self.show(selector)
        root = self._object(product.content_id)
        for artifact in product.artifacts:
            path = root / artifact.path
            if not path.is_relative_to(root):
                raise ValueError("Product artifact escaped its object root.")
            self._verify_file(path, artifact)
        return {
            "name": product.name,
            "content_id": product.content_id,
            "verified": True,
            "artifacts": len(product.artifacts),
            "size_bytes": sum(item.size_bytes for item in product.artifacts),
        }

    def artifact_path(self, selector: str, name: str, *, verify: bool = True) -> Path:
        """Resolve one promoted regular file; by default verify bytes before returning it."""
        product = self.show(selector)
        matches = [artifact for artifact in product.artifacts if artifact.name == name]
        if len(matches) != 1:
            raise KeyError(f"Product {product.name!r} has no artifact {name!r}.")
        artifact = matches[0]
        path = self._path("objects", product.content_id.removeprefix("sha256:"), artifact.path)
        if verify:
            self._verify_file(path, artifact)
        return path

    def provenance(
        self, selector: str, *, offset: int = 0, limit: int = 100
    ) -> tuple[dict[str, Any], ...]:
        """Read immutable producer attestations; a later config never replaces historical origin."""
        product = self.show(selector)
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Provenance pages require offset >= 0 and limit 1–1000.")
        root = self._path("objects", product.content_id.removeprefix("sha256:"), "provenance")
        records = []
        for path in sorted(root.glob("*.json"))[offset : offset + limit]:
            record = self._read(path)
            if path.stem != self._key(_encoded(record).decode("utf-8")):
                raise ValueError("Product producer attestation checksum differs.")
            # Validate required origin fields without asserting that a later producer was original.
            replace(product, producer=record)
            records.append(record)
        return tuple(records)

    @staticmethod
    def _consumer_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        fields = {"execution_id", "run_id", "attempt_id", "input", "scientific_fingerprint"}
        if set(value) != fields or any(
            not isinstance(value[field], str) or not value[field].strip() or len(value[field]) > 256
            for field in fields
        ):
            raise ValueError(
                "Product consumer requires exact Execution/Run/Attempt/input identity."
            )
        return {field: value[field] for field in sorted(fields)}

    @classmethod
    def _consumer_key(cls, value: Mapping[str, Any]) -> str:
        identity = cls._consumer_identity(value)
        identity.pop("scientific_fingerprint")
        return cls._key(_encoded(identity).decode("utf-8"))

    def record_consumer(self, selector: str, value: Mapping[str, Any]) -> dict[str, Any]:
        """Idempotent factual consumption audit; called only when a native Attempt starts.

        This does not make a missing dependency available, schedule a Study or update its science.
        The immutable binding uses logical consumer/input identity, not arbitrary display names.
        """
        identity = self._consumer_identity(value)
        product = self.show(selector)
        path = self._path(
            "objects",
            product.content_id.removeprefix("sha256:"),
            "consumers",
            self._consumer_key(identity) + ".json",
        )
        record = {
            "product_consumer_version": 1,
            "content_id": product.content_id,
            "contract": product.contract.identifier,
            "consumer": identity,
        }
        with CrossProcessFileLock(
            self._path(".registry.lock"),
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.02,
        ):
            self.show(product.content_id)
            if path.exists():
                if _encoded(self._read(path)) != _encoded(record):
                    raise ValueError(
                        "Conflicting scientific identity for the same product consumer."
                    )
            else:
                self._write(path, record)
        return record

    def consumers(
        self,
        selector: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        """Read bounded actual consumer audits without opening product/scientific file bytes."""
        product = self.show(selector)
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Consumer pages require offset >= 0 and limit 1–1000.")
        root = self._path("objects", product.content_id.removeprefix("sha256:"), "consumers")
        records = []
        for path in sorted(root.glob("*.json"))[offset : offset + limit]:
            record = self._read(path)
            if (
                set(record) != {"product_consumer_version", "content_id", "contract", "consumer"}
                or type(record["product_consumer_version"]) is not int
                or record["product_consumer_version"] != 1
                or record["content_id"] != product.content_id
                or record["contract"] != product.contract.identifier
                or not isinstance(record["consumer"], Mapping)
                or self._consumer_key(record["consumer"]) != path.stem
            ):
                raise ValueError("Product consumer audit is corrupt or outside its identity.")
            records.append(record)
        return tuple(records)

    def publish(
        self,
        product: StudyProduct,
        *,
        files: Mapping[str, str | Path] | None = None,
        origins: Sequence[Mapping[str, Any]] = (),
        apply: bool = False,
    ) -> dict[str, Any]:
        """Preview or publish an immutable product with independently copied artifact bytes.

        Preview inspects metadata/size only, does not create locks or copy large files, and never
        claims checksum verification. Apply streams and verifies exact bytes before publication.
        """
        if not isinstance(product, StudyProduct):
            raise TypeError("Product publication requires a sealed StudyProduct.")
        if type(apply) is not bool:
            raise TypeError("Product publication apply must be an explicit boolean.")
        encoded = _encoded(product.to_dict())
        if len(encoded) > _MAX_METADATA_BYTES:
            raise ValueError("Product manifest exceeds the bounded metadata contract.")
        attestations = {}
        for origin in (product.producer, *origins):
            record = dict(replace(product, producer=origin).producer)
            origin_bytes = _encoded(record)
            if len(origin_bytes) > _MAX_METADATA_BYTES:
                raise ValueError("Product origin exceeds the bounded metadata contract.")
            attestations[hashlib.sha256(origin_bytes).hexdigest()] = record
        alias = self._alias(product.name)
        destination = self._object(product.content_id)
        contract_path = self._path("contracts", self._key(product.contract.identifier) + ".json")
        if alias.exists():
            existing = self.show(product.name)
            if existing.content_id != product.content_id:
                raise ValueError(
                    "Product name already binds different immutable content; "
                    "choose a new name/version."
                )
        if (
            contract_path.exists()
            and ProductContract.from_dict(self._read(contract_path)) != product.contract
        ):
            raise ValueError(
                "Product contract identifier already has a different declaration; "
                "use a new version."
            )
        exists = destination.exists()
        supplied = {
            key: Path(value).expanduser().absolute() for key, value in (files or {}).items()
        }
        if not exists:
            if set(supplied) != {item.name for item in product.artifacts}:
                raise ValueError(
                    "Product publication requires exactly the declared artifact source files."
                )
            for artifact in product.artifacts:
                if artifact.path == "manifest.json" or artifact.path.split("/")[0] == "provenance":
                    raise ValueError("Product artifact path collides with owned metadata.")
                self._source(supplied[artifact.name], artifact)
        plan: dict[str, Any] = {
            "name": product.name,
            "kind": product.kind,
            "contract": product.contract.identifier,
            "content_id": product.content_id,
            "scientific_id": product.scientific_id,
            "size_bytes": sum(item.size_bytes for item in product.artifacts),
            "reuse": exists,
            "applied": False,
            "verification": "exact bytes verified on apply",
        }
        if not apply:
            return plan
        with CrossProcessFileLock(
            self._path(".registry.lock"),
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.02,
        ):
            # Recheck bindings and contract under the writer lock, never trust the preview.
            self.publish(product, files=files, origins=origins, apply=False)
            if destination.exists():
                existing = self.show(product.content_id)
                if existing.content_descriptor() != product.content_descriptor():
                    raise ValueError("Existing product object is corrupt or incompatible.")
                self.verify(product.content_id)
            else:
                stage = self._path("objects", ".publish-" + uuid4().hex)
                with StorageAdmission.transaction(
                    stage,
                    plan["size_bytes"]
                    + len(encoded)
                    + sum(len(_encoded(record)) for record in attestations.values()),
                    purpose="product publication",
                ):
                    stage.mkdir(parents=True, exist_ok=False)
                    try:
                        for artifact in product.artifacts:
                            source = supplied[artifact.name]
                            self._source(source, artifact)
                            target = stage / artifact.path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with source.open("rb") as reader, target.open("xb") as writer:
                                if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                                    raise ValueError("Product source changed into a special file.")
                                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                                writer.flush()
                                os.fsync(writer.fileno())
                            self._verify_file(target, artifact)
                        self._write(stage / "manifest.json", product.to_dict())
                        os.replace(stage, destination)
                        _fsync_directory(destination.parent)
                    finally:
                        if stage.exists():
                            shutil.rmtree(
                                stage
                            )  # Exact owned unpublished copy; originals stay intact.
            self._write(contract_path, product.contract.to_dict())
            for key, record in attestations.items():
                producer_path = self._path(
                    "objects",
                    product.content_id.removeprefix("sha256:"),
                    "provenance",
                    key + ".json",
                )
                if not producer_path.exists():
                    self._write(producer_path, record)
                elif _encoded(self._read(producer_path)) != _encoded(record):
                    raise ValueError("Product producer attestation is corrupt.")
            if not alias.exists():
                self._write(
                    alias,
                    {
                        "product_binding_version": 1,
                        "name": product.name,
                        "content_id": product.content_id,
                    },
                )
        return {**plan, "applied": True, "verification": "exact bytes verified"}
