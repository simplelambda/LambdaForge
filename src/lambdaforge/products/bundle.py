"""Portable exact product transport; importing registers evidence and never executes a Study."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.StorageAdmission import StorageAdmission
from lambdaforge.products.registry import ProductRegistry, _encoded
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import _fsync_directory, atomic_write_bytes


class ProductBundle:
    """Verified independent product bytes and immutable origin attestations, not live state."""

    @staticmethod
    def _root(path: str | Path) -> Path:
        root = Path(path).expanduser().absolute()
        if root.resolve() != root or root == root.parent:
            raise ValueError("Product bundle requires a non-root, non-symbolic absolute path.")
        return root

    @staticmethod
    def _origins(registry: ProductRegistry, selector: str) -> dict[str, dict[str, Any]]:
        origins: dict[str, dict[str, Any]] = {}
        offset = 0
        while records := registry.provenance(selector, offset=offset, limit=100):
            for record in records:
                origins[hashlib.sha256(_encoded(record)).hexdigest()] = record
            offset += len(records)
        return origins

    @staticmethod
    def _origin_digest(origins: dict[str, dict[str, Any]]) -> str:
        # Fixed-length canonical attestation hashes bind the entire provenance set.
        return hashlib.sha256("".join(sorted(origins)).encode("ascii")).hexdigest()

    @classmethod
    def export(
        cls, registry: ProductRegistry, selector: str, output: str | Path, *, apply: bool = False
    ) -> dict[str, Any]:
        """Preview or atomically copy a product independently of its original Execution."""
        if type(apply) is not bool:
            raise TypeError("Product export apply must be an explicit boolean.")
        destination = cls._root(output)
        if destination.is_relative_to(registry.root) or registry.root.is_relative_to(destination):
            raise ValueError("Product export destination cannot overlap its registry.")
        if destination.exists():
            raise FileExistsError(f"Product export destination already exists: {destination}")
        product = registry.show(selector)
        origins = cls._origins(registry, selector)
        if hashlib.sha256(_encoded(dict(product.producer))).hexdigest() not in origins:
            raise ValueError("Product export is missing its original producer attestation.")
        encoded = _encoded(product.to_dict())
        descriptor = {
            "product_bundle_version": 1,
            "content_id": product.content_id,
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "provenance_count": len(origins),
            "provenance_sha256": cls._origin_digest(origins),
        }
        size = sum(artifact.size_bytes for artifact in product.artifacts)
        plan = {
            "name": product.name,
            "content_id": product.content_id,
            "path": str(destination),
            "size_bytes": size,
            "applied": False,
        }
        if not apply:
            return plan
        # Only this explicit write operation creates destination parents and its coordination lock.
        with CrossProcessFileLock(
            destination.with_name(f".{destination.name}.product-export.lock"),
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.02,
        ):
            if destination.exists():
                raise FileExistsError(f"Product export destination already exists: {destination}")
            if cls._root(destination) != destination:
                raise ValueError("Product export destination changed ownership.")
            stage = destination.with_name(f".{destination.name}.product-export-{uuid4().hex}")
            metadata_size = (
                len(encoded)
                + len(_encoded(descriptor))
                + sum(len(_encoded(record)) for record in origins.values())
            )
            with StorageAdmission.transaction(
                stage, size + metadata_size, purpose="product export"
            ):
                stage.mkdir(parents=True, exist_ok=False)
                try:
                    for artifact in product.artifacts:
                        source = registry.artifact_path(product.content_id, artifact.name)
                        target = stage / "files" / artifact.path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        ProductRegistry._verify_file(target, artifact)
                        with target.open("rb") as handle:
                            os.fsync(handle.fileno())
                    atomic_write_bytes(stage / "product.json", encoded)
                    for key, record in origins.items():
                        atomic_write_bytes(stage / "provenance" / (key + ".json"), _encoded(record))
                    atomic_write_bytes(stage / "bundle.json", _encoded(descriptor))
                    if destination.exists():
                        raise FileExistsError(f"Product export destination appeared: {destination}")
                    os.replace(stage, destination)
                    _fsync_directory(destination.parent)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)  # Exact unpublished transaction, not source content.
        return {**plan, "applied": True}

    @classmethod
    def import_bundle(
        cls, registry: ProductRegistry, source: str | Path, *, apply: bool = False
    ) -> dict[str, Any]:
        """Verify the portable product, preserve origins and idempotently register exact bytes."""
        if type(apply) is not bool:
            raise TypeError("Product import apply must be an explicit boolean.")
        root = cls._root(source)
        if root.is_relative_to(registry.root) or registry.root.is_relative_to(root):
            raise ValueError("Product import source cannot overlap its destination registry.")
        descriptor = registry._read(root / "bundle.json")
        if (
            type(descriptor.get("product_bundle_version")) is not int
            or descriptor["product_bundle_version"] != 1
        ):
            raise ValueError("Unsupported product bundle version.")
        product = registry.read_manifest(root / "product.json")
        if product.content_id != descriptor.get("content_id") or hashlib.sha256(
            (root / "product.json").read_bytes()
        ).hexdigest() != descriptor.get("manifest_sha256"):
            raise ValueError("Product bundle manifest identity/checksum differs.")
        provenance = root / "provenance"
        if provenance.resolve() != provenance:
            raise ValueError("Product bundle provenance cannot be symbolic.")
        origins = {}
        for path in sorted(provenance.glob("*.json")):
            record = registry._read(path)
            if path.stem != hashlib.sha256(_encoded(record)).hexdigest():
                raise ValueError("Product bundle producer attestation checksum differs.")
            # All supplied attestations must satisfy the same explicit required provenance fields.
            replace(product, producer=record)
            origins[path.stem] = record
        if (
            type(descriptor.get("provenance_count")) is not int
            or descriptor["provenance_count"] != len(origins)
            or descriptor.get("provenance_sha256") != cls._origin_digest(origins)
        ):
            raise ValueError("Product bundle provenance inventory differs.")
        if hashlib.sha256(_encoded(product.producer)).hexdigest() not in origins:
            raise ValueError("Product bundle is missing its immutable original producer.")
        files = {artifact.name: root / "files" / artifact.path for artifact in product.artifacts}
        for artifact in product.artifacts:
            # Even an existing destination never excuses corrupt/missing import bytes.
            ProductRegistry._verify_file(files[artifact.name], artifact)
        plan = registry.publish(product, files=files, origins=tuple(origins.values()), apply=False)
        if apply:
            plan = registry.publish(
                product, files=files, origins=tuple(origins.values()), apply=True
            )
        return {
            **plan,
            "source": str(root),
            "provenance_records": len(origins),
            "verification": "exact import bytes verified",
        }
