"""Managed, sealed presentation bundles; arrays remain consumer-authored opaque bytes."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.work.managed import ManagedOutput, fingerprint, owned_path

MANIFEST_NAME = "lambdaforge-report.json"
MANIFEST_LIMIT = 8 * 1024 * 1024
ENTRY_LIMIT = 16 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class ReportBundleError(ValueError):
    """A presentation/asset problem, never scientific equivalence or evidence of model failure."""


def relative_name(value: Any) -> str:
    """One canonical portable relative filename; reject instead of silently normalizing."""
    if not isinstance(value, str) or not value or any(c in value for c in ("\\", ":", "\0")):
        raise ReportBundleError("Report asset names must be portable relative paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ReportBundleError("Report asset traversal/absolute paths are forbidden.")
    if unicodedata.normalize("NFC", value) != value:
        raise ReportBundleError("Report asset names must use canonical NFC Unicode.")
    return value


def asset_id(value: Any) -> str:
    """Stable logical ID, not a path or a URL; labels/selection remain consumer metadata."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ReportBundleError("Report asset ID must contain 1–160 ASCII letters/digits/._-.")
    return value


def file_record(root: Path, name: str, identifier: str, **annotations: Any) -> dict[str, Any]:
    path = owned_path(root, relative_name(name), must_exist=True)
    if not path.is_file():
        raise ReportBundleError("Report assets must be regular files, not directories or links.")
    before = path.stat()
    digest, size = fingerprint(path)
    raw = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            raw.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReportBundleError("Report asset changed during finalization.")
    return {
        "id": asset_id(identifier),
        "name": name,
        "size_bytes": size,
        "sha256": digest,
        "content_sha256": raw.hexdigest(),
        **annotations,
    }


class ManagedReportBundle:
    """One ordinary managed directory with a bounded entry and declared lazy files.

    ``entry.write_text`` writes the HTML. ``asset(id, filename=...).build(...)`` accepts
    existing serializers without copying arrays into a report string. No inference is performed.
    """

    def __init__(self, output: ManagedOutput, *, entry: str = "index.html") -> None:
        self.output = output
        name = relative_name(entry)
        if name == MANIFEST_NAME:
            raise ReportBundleError("Entry filename collides with the framework manifest.")
        self._files: dict[str, dict[str, Any]] = {}
        self._names = {name.casefold(), MANIFEST_NAME.casefold()}
        self.entry = self._file(name)
        self._entry_name = name

    def _file(self, name: str) -> ManagedOutput:
        path = owned_path(self.output.path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return ManagedOutput(path, kind="file")

    def asset(
        self,
        identifier: str,
        *,
        filename: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedOutput:
        """Declare an owned opaque file. Metadata/IDs never choose scientific winners."""
        identifier, filename = asset_id(identifier), relative_name(filename)
        if (
            identifier.casefold() in {key.casefold() for key in self._files}
            or identifier == "entry"
        ):
            raise ReportBundleError("Duplicate/ambiguous report asset ID.")
        if filename.casefold() in self._names:
            raise ReportBundleError("Duplicate/ambiguous report filename.")
        if not isinstance(media_type, str) or not 1 <= len(media_type) <= 160 or "\n" in media_type:
            raise ReportBundleError("Invalid report asset media type.")
        annotations = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
        self._files[identifier] = {
            "name": filename,
            "media_type": media_type,
            "metadata": annotations,
        }
        self._names.add(filename.casefold())
        return self._file(filename)

    def seal(self) -> dict[str, Any]:
        """Finalize normal file integrity and attest the manifest without embedding any arrays."""
        root = self.output.path
        declared = {self._entry_name, MANIFEST_NAME} | {r["name"] for r in self._files.values()}
        for path in root.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(path.stat().st_mode)):
                raise ReportBundleError("Report bundle contains a symbolic/special path.")
            if path.is_file() and path.relative_to(root).as_posix() not in declared:
                raise ReportBundleError("Report bundle contains an undeclared file.")
        entry = file_record(root, self._entry_name, "entry", media_type="text/html", metadata={})
        if entry["size_bytes"] > ENTRY_LIMIT:
            raise ReportBundleError(
                "Report entry exceeds the 16 MiB HTML limit; move data into assets."
            )
        self.entry.path.read_text(encoding="utf-8")
        assets = [
            file_record(
                root,
                row["name"],
                identifier,
                media_type=row["media_type"],
                metadata=row["metadata"],
            )
            for identifier, row in self._files.items()
        ]
        value = {"report_bundle_version": 1, "entry": entry, "assets": assets}
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > MANIFEST_LIMIT:
            raise ReportBundleError(
                "Report manifest exceeds 8 MiB; metadata must not embed arrays."
            )
        ManagedOutput(root / MANIFEST_NAME, kind="file").write_bytes(encoded)
        digest, size = fingerprint(root / MANIFEST_NAME)
        return {"version": 1, "manifest": MANIFEST_NAME, "sha256": digest, "size_bytes": size}


def read_manifest(root: Path, attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the attested manifest. Never hash/walk all retained assets during viewing."""
    if attestation.get("version") != 1 or attestation.get("manifest") != MANIFEST_NAME:
        raise ReportBundleError("Missing/unsupported finalized report bundle attestation.")
    path = owned_path(root, MANIFEST_NAME, must_exist=True)
    if not path.is_file() or path.stat().st_size > MANIFEST_LIMIT:
        raise ReportBundleError("Missing/unsafe/oversized report manifest.")
    if fingerprint(path) != (attestation.get("sha256"), attestation.get("size_bytes")):
        raise ReportBundleError("Report manifest differs from its finalized checksum.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_bundle_version") != 1:
        raise ReportBundleError("Invalid report manifest schema.")
    assets = value.get("assets")
    if not isinstance(assets, list) or not isinstance(value.get("entry"), dict):
        raise ReportBundleError("Report manifest needs an entry and explicit assets.")
    identifiers, names = set(), {MANIFEST_NAME.casefold()}
    for row in [value["entry"], *assets]:
        if not isinstance(row, dict):
            raise ReportBundleError("Invalid report asset declaration.")
        identifier, name = asset_id(row.get("id")), relative_name(row.get("name"))
        if identifier.casefold() in identifiers or name.casefold() in names:
            raise ReportBundleError("Duplicate/ambiguous report asset declaration.")
        identifiers.add(identifier.casefold())
        names.add(name.casefold())
        if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
            raise ReportBundleError("Invalid report asset size.")
        if any(
            not isinstance(row.get(key), str) or not _HASH.fullmatch(row[key])
            for key in ("sha256", "content_sha256")
        ):
            raise ReportBundleError("Invalid report asset checksum.")
        if not isinstance(row.get("metadata", {}), dict):
            raise ReportBundleError("Invalid report asset metadata.")
    if value["entry"]["id"] != "entry" or value["entry"]["size_bytes"] > ENTRY_LIMIT:
        raise ReportBundleError("Invalid/oversized report entry.")
    return value


def verified_asset(root: Path, row: Mapping[str, Any]) -> Path:
    """Explicit selected-asset read; one changed/missing file never authorizes substitutes."""
    path = owned_path(root, relative_name(row.get("name")), must_exist=True)
    if not path.is_file() or fingerprint(path) != (row.get("sha256"), row.get("size_bytes")):
        raise ReportBundleError(
            f"Report asset {row.get('id')!r} differs from its finalized checksum."
        )
    return path
