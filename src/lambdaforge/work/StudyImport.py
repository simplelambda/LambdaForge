"""Verify portable native evidence, not consumer code, before registering an imported Study.

The original package is retained intact below an owned ResultStore entry. Compact presentation
indexes sit beside an import receipt; machine paths are relocated only in observer read models.
Product publication uses the existing exact product transaction, not a second execution engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.StorageAdmission import StorageAdmission
from lambdaforge.products.bundle import ProductBundle
from lambdaforge.products.registry import ProductRegistry, _encoded
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import _fsync_directory
from lambdaforge.work.models import atomic_json


def _mapping(path: Path, *, verified_aggregate: bool = False) -> dict[str, Any]:
    """Explicit import may inspect a large aggregate, never an unbounded individual document."""
    if (
        path.resolve() != path
        or not path.is_file()
        or (not verified_aggregate and path.stat().st_size > 64 * 1024 * 1024)
    ):
        raise ValueError(f"Missing, symbolic or oversized Study import document: {path.name}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate Study metadata field {key!r}.")
            value[key] = item
        return value

    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite Study metadata: {value}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Study import metadata must be an object: {path.name}")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Study inventory requires portable relative file paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError(f"Unsafe Study inventory path: {value!r}")
    return value


def _files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for base, directories, names in os.walk(root, followlinks=False):
        parent = Path(base)
        for name in (*directories, *names):
            path = parent / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)) or path.resolve() != path:
                raise ValueError(f"Study import refuses symbolic/special evidence: {path.name}")
            if stat.S_ISREG(mode):
                if path.stat().st_nlink != 1:
                    raise ValueError("Study import requires independent files, not hard links.")
                files[path.relative_to(root).as_posix()] = path
    return files


def _verify(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Study import requires an existing non-symbolic package directory.")
    manifest = _mapping(root / "manifest.json")
    if (
        type(manifest.get("lambdaforge_export_version")) is not int
        or manifest["lambdaforge_export_version"] != 2
    ):
        raise ValueError("Study import requires a native version-2 export manifest.")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or len(inventory) > 1_000_000:
        raise ValueError("Study import inventory is missing or exceeds the file safety bound.")
    files = _files(root)
    expected: set[str] = set()
    total = 0
    for record in inventory:
        if not isinstance(record, Mapping) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError("Invalid Study inventory record schema.")
        relative = _relative(record["path"])
        if relative == "manifest.json" or relative in expected:
            raise ValueError("Duplicate/self-referential Study inventory entry.")
        expected.add(relative)
        size, checksum = record["size_bytes"], record["sha256"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ValueError("Invalid Study inventory size/checksum.")
        path = files.get(relative)
        if path is None or path.stat().st_size != size:
            raise ValueError(f"Study inventory file missing or size differs: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Study inventory checksum differs: {relative}")
        total += size
    if set(files) != expected | {"manifest.json"}:
        raise ValueError("Study package has missing or unlisted evidence files.")
    if (
        type(manifest.get("file_count")) is not int
        or manifest["file_count"] != len(expected)
        or type(manifest.get("size_bytes")) is not int
        or manifest["size_bytes"] != total
    ):
        raise ValueError("Study inventory count/size differs from its manifest.")
    origin = _mapping(root / "execution" / "execution.json")
    execution_id = manifest.get("execution_id")
    if not isinstance(execution_id, str) or not re.fullmatch(
        r"execution-[A-Za-z0-9_-]+", execution_id
    ):
        raise ValueError("Invalid Study Execution identity.")
    if (
        origin.get("execution_id") != execution_id
        or origin.get("name") != manifest.get("name")
        or origin.get("scientific_fingerprint") != manifest.get("scientific_fingerprint")
        or origin.get("ownership", {}).get("execution_dir") != "owned"
    ):
        raise ValueError("Study provenance identity/ownership differs from its manifest.")
    _mapping(root / "execution" / "configuration.json")
    finalized = manifest.get("finalized")
    if type(finalized) is not bool or manifest.get("export_kind") not in {"snapshot", "final"}:
        raise ValueError("Invalid Study final/snapshot declaration.")
    if finalized:
        # A native aggregate grows with the number of Runs. Its exact size and every byte
        # were verified against the inventory above; the individual-document cap is not a
        # Study-size limit. Only this explicitly requested, verified aggregate is exempt.
        result = _mapping(root / "execution" / "result.json", verified_aggregate=True)
        if (
            type(result.get("execution_result_version")) is not int
            or result["execution_result_version"] != 1
            or result.get("execution_id") != execution_id
            or result.get("scientific_fingerprint") != origin.get("scientific_fingerprint")
            or result.get("status") != manifest.get("execution_status")
            or result.get("name") != manifest.get("name")
            or not isinstance(result.get("runs"), list)
        ):
            raise ValueError("Native Study result schema/provenance differs from its package.")
        for run in result["runs"]:
            if not isinstance(run, Mapping) or not isinstance(run.get("run_id"), str):
                raise ValueError("Invalid native Study Run record.")
            run_path = relative_run_path(run)
            if (
                type(run.get("attempt_number")) is not int
                or run["attempt_number"] != int(run_path.name.removeprefix("attempt-"))
                or run.get("attempt_id") != run_path.name
                or run.get("execution_id") != execution_id
            ):
                raise ValueError("Study Run Attempt identity differs from its owned path.")
    else:
        if (root / "execution" / "result.json").exists():
            raise ValueError("Study snapshot falsely declares a missing final result.")
        result = {
            "execution_result_version": 1,
            "execution_id": execution_id,
            "scientific_fingerprint": origin["scientific_fingerprint"],
            "name": manifest["name"],
            "status": manifest.get("status", "unknown"),
            "runs": [],
            "summary": {},
        }
    if manifest.get("export_kind") == "final" and (
        not finalized or result["status"] != "succeeded"
    ):
        raise ValueError("A final Study export must contain a successful native result.")
    package_id = hashlib.sha256(_encoded(manifest)).hexdigest()
    return manifest, result, package_id


def relative_run_path(run: Mapping[str, Any]) -> Path:
    """Resolve only the native logical Run/Attempt shape, not an imported machine path."""
    run_id = run.get("run_id")
    path = PurePosixPath(str(run.get("run_dir", "")))
    if (
        not isinstance(run_id, str)
        or not re.fullmatch(r"run-[A-Za-z0-9_-]+", run_id)
        or len(path.parts) < 4
        or path.parts[-4:-1] != ("runs", run_id, "attempts")
        or not re.fullmatch(r"attempt-[0-9]{4,}", path.name)
    ):
        raise ValueError("Imported Run has no safe native logical Run/Attempt path.")
    return Path("runs") / run_id / "attempts" / path.name


class StudyImport:
    """Native ResultStore registration with immutable portable evidence and explicit ownership."""

    @classmethod
    def inspect(
        cls, source: str | Path, root: Path, registry: ProductRegistry, *, apply: bool = False
    ) -> dict[str, Any]:
        if type(apply) is not bool:
            raise TypeError("Study import apply must be an explicit boolean.")
        package = Path(source).expanduser().absolute()
        if package.is_relative_to(root) or root.is_relative_to(package):
            raise ValueError("Study package cannot overlap its destination result store.")
        manifest, result, package_id = _verify(package)
        name = manifest["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
            raise ValueError("Study name requires a safe native ResultStore component.")
        destination = root / name / manifest["execution_id"]
        if destination.resolve() != destination:
            raise ValueError("Study destination cannot be symbolic.")
        products = manifest.get("products", [])
        if not isinstance(products, list) or len(products) > 128:
            raise ValueError("Invalid Study product bundle inventory.")
        plans: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in products:
            if not isinstance(item, Mapping) or set(item) != {"path", "name", "content_id"}:
                raise ValueError("Invalid Study product reference schema.")
            relative = _relative(item["path"])
            if relative in seen or not relative.startswith("products/"):
                raise ValueError("Duplicate/unsafe Study product bundle reference.")
            seen.add(relative)
            plan = ProductBundle.import_bundle(registry, package / relative)
            if plan["name"] != item["name"] or plan["content_id"] != item["content_id"]:
                raise ValueError("Study product bundle differs from its manifest reference.")
            plans.append(plan)
        bundled = {
            p.parent.relative_to(package).as_posix()
            for p in (package / "products").glob("*/bundle.json")
        }
        if seen != bundled:
            raise ValueError("Study product inventory has unlisted or missing bundles.")
        publication_path = package / "execution/products.json"
        if publication_path.exists():
            publication = ProductRegistry._read(publication_path)
            if publication.get("product_publication_version") != 1:
                raise ValueError("Unsupported native product publication schema.")
            published = {
                (item["name"], item["content_id"])
                for item in publication.get("items", ())
                if item.get("status") == "published"
            }
            if published != {(item["name"], item["content_id"]) for item in products}:
                raise ValueError(
                    "Study import omits a published product or fabricates a publication."
                )
        payload = {
            "status": "verified",
            "applied": False,
            "reused": False,
            "package_id": package_id,
            "execution_id": manifest["execution_id"],
            "name": name,
            "captured_state": manifest["status"],
            "finalized": manifest["finalized"],
            "path": str(destination),
            "products": plans,
            "will_execute": False,
            "size_bytes": manifest["size_bytes"],
        }
        if destination.exists():
            cls._existing(destination, package_id)
            payload["reused"] = True
        if not apply:
            return payload
        with CrossProcessFileLock(
            root / ".study-import.lock",
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.02,
        ):
            if destination.exists():
                cls._existing(destination, package_id)
                # Older valid imports can acquire local presentation indexes on explicit
                # re-apply. Never migrate them implicitly while refreshing a root screen.
                if not (destination / "import-study.json").exists():
                    from lambdaforge.work.ImportedStudy import import_index, write_import_views

                    stage = destination.with_name(f".{destination.name}-view-{uuid4().hex}")
                    prior = stage / "previous-view"
                    view = destination / "import-view"
                    try:
                        write_import_views(stage, result, package=destination / "portable")
                        if view.is_symlink() or view.resolve() != view:
                            raise ValueError("Imported presentation root cannot be symbolic.")
                        if view.exists():
                            os.replace(view, prior)
                        os.replace(stage / "import-view", view)
                        atomic_json(
                            destination / "import-study.json",
                            import_index(manifest, result, package=destination / "portable"),
                        )
                    except Exception:
                        if prior.exists():
                            if view.exists():
                                os.replace(view, stage / "discarded-view")
                            os.replace(prior, view)
                        raise
                    finally:
                        if stage.exists():
                            shutil.rmtree(stage)  # Exact unpublished presentation only.
            else:
                if destination.resolve() != destination:
                    raise ValueError("Study import destination ownership changed.")
                stage = destination.with_name(f".{destination.name}-import-{uuid4().hex}")
                with StorageAdmission.transaction(
                    stage, manifest["size_bytes"], purpose="Study import"
                ):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copytree(package, stage / "portable")
                        # Reverify copies before publication; mutation during transfer fails closed.
                        _, _, copied_id = _verify(stage / "portable")
                        if copied_id != package_id:
                            raise ValueError("Study package changed during import.")
                        receipt = {
                            "study_import_version": 1,
                            "package_id": package_id,
                            "execution_id": manifest["execution_id"],
                            "scientific_fingerprint": manifest["scientific_fingerprint"],
                            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                            "source": str(package),
                            "evidence_root": "portable/execution",
                            "will_execute": False,
                        }
                        atomic_json(stage / "result.json", result)
                        atomic_json(stage / "import.json", receipt)
                        from lambdaforge.work.ImportedStudy import import_index, write_import_views

                        atomic_json(
                            stage / "import-study.json",
                            import_index(manifest, result, package=stage / "portable"),
                        )
                        write_import_views(stage, result)
                        os.replace(stage, destination)
                        _fsync_directory(destination.parent)
                    finally:
                        if stage.exists():
                            shutil.rmtree(stage)  # Exact unpublished transaction only.
            # Registration is durable first. Product transactions are independently idempotent:
            # a later conflict/failure leaves a verified archive and can safely be retried.
            for item in products:
                ProductBundle.import_bundle(
                    registry, destination / "portable" / item["path"], apply=True
                )
        return {**payload, "status": "imported", "applied": True}

    @staticmethod
    def _existing(destination: Path, package_id: str) -> None:
        receipt = _mapping(destination / "import.json")
        if (
            receipt.get("study_import_version") != 1
            or receipt.get("package_id") != package_id
            or receipt.get("execution_id") != destination.name
        ):
            raise ValueError(
                "Existing Execution is not this verified Study import; refusing overwrite."
            )
        _, result, existing_id = _verify(destination / "portable")
        if (
            existing_id != package_id
            or _mapping(destination / "result.json", verified_aggregate=True) != result
        ):
            raise ValueError("Previously imported Study evidence/index is corrupt.")
