"""Exact Markdown snapshots and a metadata-only, cursor-paged local edit journal.

No scientific records, plugins, indexes or arbitrary vault files are exported. The journal
records intent before mutation and completion afterwards; an interrupted intent is not success.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lambdaforge.agents.scope import ProjectScope

if TYPE_CHECKING:
    from lambdaforge.agents.knowledge import KnowledgeService

_JOURNAL = ".lambdaforge/agents/knowledge-audit.jsonl"
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_RECORD = 4096


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validate_event(value: Any, scope: ProjectScope) -> None:
    common = {"schema_version", "project_id", "timestamp_utc", "event_id", "action", "state"}
    export = isinstance(value, dict) and value.get("action") == "export"
    fields = common | (
        {"plan_hash"}
        if export
        else {"note_id", "note_author_type", "previous_revision", "revision"}
    )
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("project_id") != scope.project.project_id
        or value.get("action") not in {"create", "update", "link", "export"}
        or value.get("state") not in {"intent", "committed"}
        or not isinstance(value.get("event_id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["event_id"])
    ):
        raise ValueError("Knowledge audit record is corrupt or belongs to another project.")
    stamp = value["timestamp_utc"]
    if not isinstance(stamp, str) or datetime.fromisoformat(stamp).tzinfo is None:
        raise ValueError("Knowledge audit timestamp requires a timezone.")
    if not export:
        if (
            not isinstance(value["note_id"], str)
            or not re.fullmatch(r"[a-z][a-z0-9-]{0,79}", value["note_id"])
            or value["note_author_type"] not in {"human", "ai"}
        ):
            raise ValueError("Knowledge audit note identity/provenance is corrupt.")
    for field in ("plan_hash",) if export else ("revision", "previous_revision"):
        digest = value[field]
        if field == "previous_revision" and digest is None:
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Knowledge audit requires exact revision/plan hashes.")


def append_event(scope: ProjectScope, event: dict[str, Any]) -> None:
    """Append a regular-file event while the caller holds the native knowledge lock."""
    scope.assert_current()
    path = scope.path(_JOURNAL, internal=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "project_id": scope.project.project_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    _validate_event(record, scope)
    raw = _encode(record) + b"\n"
    if len(raw) > _MAX_RECORD:
        raise ValueError("Knowledge audit event exceeds its metadata bound.")
    fd = os.open(
        path,
        os.O_RDWR
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )
    with os.fdopen(fd, "r+b") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError("Knowledge audit must be a regular local file.")
        if os.name != "nt" and info.st_mode & 0o077:
            raise PermissionError("Knowledge audit must be private to the local user (0600).")
        if info.st_size:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise ValueError("Knowledge audit has an incomplete tail; inspect before writing.")
            stream.seek(max(0, info.st_size - _MAX_RECORD - 1))
            last = stream.read(_MAX_RECORD + 1)[:-1].rsplit(b"\n", 1)[-1]
            if len(last) >= _MAX_RECORD:
                raise ValueError("Knowledge audit tail exceeds its metadata bound.")
            _validate_event(json.loads(last), scope)
        stream.write(raw)
        stream.flush()
        os.fsync(fd)


def read_audit(scope: ProjectScope, *, cursor: int = 0, limit: int = 20) -> dict[str, Any]:
    """Read one bounded byte-cursor page, not the whole journal or any note bodies."""
    if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Audit requires a non-negative byte cursor and limit 1–100.")
    scope.assert_current()
    path = scope.path(_JOURNAL, internal=True)
    if not path.exists():
        if cursor:
            raise ValueError("Audit cursor is beyond the current journal.")
        return {"items": [], "next_cursor": None, "cursor_unit": "bytes"}
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError("Knowledge audit must be a regular local file.")
        if cursor > info.st_size:
            raise ValueError("Audit cursor is beyond the current journal.")
        if cursor:
            stream.seek(cursor - 1)
            if stream.read(1) != b"\n":
                raise ValueError("Use the exact next_cursor returned by the previous page.")
        items = []
        for _ in range(limit):
            raw = stream.readline(_MAX_RECORD + 1)
            if not raw:
                break
            if len(raw) > _MAX_RECORD or not raw.endswith(b"\n"):
                raise ValueError("Knowledge audit record is oversized or incomplete.")
            record = json.loads(raw)
            _validate_event(record, scope)
            items.append(record)
        following = stream.tell()
    return {
        "items": items,
        "next_cursor": following if following < info.st_size else None,
        "cursor_unit": "bytes",
        "trust": "local operational journal, not scientific verification",
    }


class KnowledgeArchive:
    """ZIP export of exactly validated canonical notes; never silently skip corrupt evidence."""

    def __init__(self, vault: KnowledgeService) -> None:
        self.vault = vault

    def _destination(self, destination: str) -> Path:
        path = self.vault.scope.path(destination)
        if path.suffix.lower() != ".zip":
            raise ValueError("Knowledge export destination must be a project-relative .zip file.")
        if path.exists():
            raise FileExistsError("Export destination exists; choose a new filename.")
        if not path.parent.is_dir():
            raise ValueError("Create the export parent directory before previewing.")
        if path.is_relative_to(self.vault.root):
            raise PermissionError("Place exports outside the canonical knowledge vault.")
        return path

    def _snapshot(self) -> tuple[list[dict[str, Any]], int]:
        from lambdaforge.agents.knowledge import KINDS

        scope = self.vault.scope
        scope.assert_current()
        scope.path(self.vault.directory)
        if not self.vault.root.is_dir():
            raise FileNotFoundError("Initialize knowledge before exporting.")
        inventory: list[dict[str, Any]] = []
        identifiers: set[str] = set()
        total = 0
        for kind in KINDS:
            directory = scope.path(Path(self.vault.directory) / kind)
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                if len(inventory) >= 10000:
                    raise ValueError("Export exceeds 10000 notes; split the vault explicitly.")
                relative = path.relative_to(scope.root).as_posix()
                note = self.vault._read(relative)
                identifier = note["metadata"]["id"]
                if identifier in identifiers:
                    raise ValueError("Duplicate note identity; resolve the vault before exporting.")
                identifiers.add(identifier)
                # Preserve exact original bytes, including external CRLF/BOM edits.
                raw = scope.file(relative, limit=64 * 1024)
                if hashlib.sha256(raw).hexdigest() != note["revision"]:
                    raise ValueError("A note changed during export inspection; retry the preview.")
                total += len(raw)
                if total > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Knowledge snapshot exceeds 64 MiB; split it explicitly.")
                inventory.append(
                    {
                        "path": path.relative_to(self.vault.root).as_posix(),
                        "id": identifier,
                        "sha256": note["revision"],
                        "size": len(raw),
                    }
                )
        return inventory, total

    def preview(self, destination: str) -> dict[str, Any]:
        """Pure inventory/content-hash plan; no indexes, audit, archive or approval is created."""
        self._destination(destination)
        inventory, total = self._snapshot()
        value = {
            "schema_version": 1,
            "project_id": self.vault.scope.project.project_id,
            "destination": destination,
            "notes": len(inventory),
            "bytes": total,
            "inventory_hash": hashlib.sha256(_encode(inventory)).hexdigest(),
            "format": "zip",
            "exact_note_bytes": True,
            "scientific_records_included": False,
            "excluded": ["indexes", "plugins", "desktop configuration", "unmanaged vault files"],
        }
        return {**value, "plan_hash": hashlib.sha256(_encode(value)).hexdigest()}

    def export(self, destination: str, *, plan_hash: str) -> dict[str, Any]:
        """Revalidate and atomically publish a new ZIP; no overwrites or silent partial export."""
        vault = self.vault
        with vault._lock():
            plan = self.preview(destination)
            if plan["plan_hash"] != plan_hash:
                raise ValueError("Knowledge export plan changed; preview and approve again.")
            target = self._destination(destination)
            inventory, _ = self._snapshot()
            if hashlib.sha256(_encode(inventory)).hexdigest() != plan["inventory_hash"]:
                raise ValueError("Knowledge changed during export; preview again.")
            event = {"event_id": uuid4().hex, "action": "export", "plan_hash": plan_hash}
            append_event(vault.scope, {**event, "state": "intent"})
            descriptor, temporary = tempfile.mkstemp(prefix=".lf-knowledge-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "w+b") as stream:
                    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for item in inventory:
                            raw = vault.scope.file(
                                Path(vault.directory) / item["path"], limit=64 * 1024
                            )
                            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                                raise ValueError("A note changed during export; no ZIP published.")
                            archive.writestr("notes/" + item["path"], raw)
                        archive.writestr("manifest.json", _encode({**plan, "files": inventory}))
                        archive.writestr(
                            "README.md",
                            "# Knowledge snapshot\n\nExact Markdown interpretations.\n"
                            "Not a scientific result store or executable recovery package.\n"
                            "Use manifest.json for project identity and exact file checksums.\n",
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                # Recheck containment and publish with exclusive creation (unlike os.replace).
                self._destination(destination)
                os.link(temporary, target)
                append_event(vault.scope, {**event, "state": "committed"})
            finally:
                Path(temporary).unlink(missing_ok=True)
        return {**plan, "status": "exported", "overwrite": False}
