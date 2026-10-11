"""Portable Markdown knowledge, with a disposable incremental SQLite search index.

Interpretations are never scientific records. Writes are explicit Python/CLI operations or
separately approved MCP actions. No embeddings, desktop application or network is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from lambdaforge.agents.scope import ProjectScope
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_bytes, atomic_write_text

KINDS = (
    "decision",
    "hypothesis",
    "experiment",
    "finding",
    "concept",
    "architecture",
    "issue",
    "research-question",
    "procedure",
    "literature",
    "task",
)
TEMPLATES = {
    "decision": (
        "Context",
        "Alternatives",
        "Decision",
        "Evidence",
        "Consequences",
        "Open questions",
    ),
    "hypothesis": (
        "Motivation",
        "Testable predictions",
        "Evidence",
        "Interpretation",
        "Limitations",
    ),
    "experiment": (
        "Objective",
        "Configuration",
        "Hypotheses",
        "Observations",
        "Limitations",
        "Next steps",
    ),
    "finding": ("Observation", "Evidence", "Interpretation", "Limitations"),
    "concept": ("Definition", "References", "Project relevance"),
    "architecture": ("Context", "Contract", "Alternatives", "Evidence", "Consequences"),
    "issue": ("Symptoms", "Evidence", "Suspected cause", "Resolution", "Verification"),
    "research-question": (
        "Question",
        "Existing evidence",
        "Missing evidence",
        "Proposed experiment",
    ),
    "procedure": ("Purpose", "Prerequisites", "Steps", "Safety", "Verification"),
    "literature": ("Source", "Claims", "Evidence", "Limitations", "Project relevance"),
    "task": ("Goal", "Scope", "Acceptance", "Progress", "Evidence"),
}
_ID = re.compile(r"[a-z][a-z0-9-]{0,79}\Z")
_MAX_NOTE = 64 * 1024
_FIELDS = {
    "schema_version",
    "id",
    "kind",
    "title",
    "status",
    "created_at",
    "updated_at",
    "author_type",
    "project_id",
    "tags",
    "references",
    "evidence_refs",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page(offset: int, limit: int) -> None:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Knowledge pages require offset >= 0 and limit 1–100.")


class KnowledgeService:
    """One owned Markdown vault. The index can always be deleted/rebuilt without losing notes."""

    def __init__(self, scope: ProjectScope, directory: str = "knowledge") -> None:
        self.scope = scope
        self.directory = directory
        self.root = scope.path(directory)
        self.cache = scope.path(".lambdaforge/agent-knowledge", internal=True)
        self.warnings: tuple[str, ...] = ()
        self._indexed_at = 0.0

    def _lock(self) -> CrossProcessFileLock:
        self.scope.assert_current()
        self.scope.path(self.directory)
        self.scope.path(self.cache.relative_to(self.scope.root), internal=True)
        lock = self.scope.path(
            (self.cache / "knowledge.lock").relative_to(self.scope.root), internal=True
        )
        return CrossProcessFileLock(
            lock,
            shared=False,
            timeout_seconds=10,
            poll_interval_seconds=0.05,
        )

    def initialize(self, *, obsidian: bool = False) -> dict[str, Any]:
        """Explicitly prepare Markdown documentation without replacing existing notes/config."""
        with self._lock():
            self.root.mkdir(parents=True, exist_ok=True)
            readme = self.scope.path(self.root.relative_to(self.scope.root) / "README.md")
            if not readme.exists():
                atomic_write_text(
                    readme,
                    (
                        "# Project knowledge\n\nMarkdown is canonical; indexes are disposable.\n"
                        "Notes are interpretations, not verified results. Link exact evidence.\n"
                        "Use `lf ai knowledge` or edit Markdown. Reindex after external edits.\n"
                        "Do not put secrets, checkpoints or models here.\n"
                    ),
                )
            if obsidian:
                # A folder of Markdown is already an Obsidian vault; no config/plugins required.
                marker = self.root / ".obsidian"
                if marker.is_symlink() or marker.resolve() != marker:
                    raise PermissionError("Obsidian configuration directory cannot be symbolic.")
                marker.mkdir(exist_ok=True)
        return self.status()

    def status(self) -> dict[str, Any]:
        self.scope.assert_current()
        self.scope.path(self.directory)
        return {
            "schema_version": 1,
            "project_id": self.scope.project.project_id,
            "configured": self.root.is_dir(),
            "directory": self.directory,
            "storage": "markdown",
            "semantic_search": False,
            "obsidian_vault": (self.root / ".obsidian").is_dir(),
            "authoritative_science": False,
        }

    @staticmethod
    def template(kind: str) -> str:
        if kind not in TEMPLATES:
            raise ValueError(f"Unknown note kind; choose one of {', '.join(KINDS)}.")
        return "\n\n".join(f"## {heading}\n" for heading in TEMPLATES[kind]) + "\n"

    def _read(self, relative: str) -> dict[str, Any]:
        self.scope.assert_current()
        raw = self.scope.file(relative, limit=_MAX_NOTE)
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if not text.startswith("---\n"):
            raise ValueError("Knowledge note requires YAML frontmatter.")
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            raise ValueError("Knowledge frontmatter is not terminated.")
        metadata = yaml.safe_load(parts[0][4:])
        self._validate(metadata)
        path = self.scope.path(relative)
        if path.stem != metadata["id"] or path.parent != self.root / metadata["kind"]:
            raise ValueError("Note ID/kind must match its owned kind/ID.md location.")
        return {
            "metadata": metadata,
            "body": parts[1],
            "path": relative,
            "revision": hashlib.sha256(raw).hexdigest(),
            "trust": "untrusted_project_text",
        }

    def _validate(self, value: Any) -> None:
        if not isinstance(value, dict) or set(value) != _FIELDS:
            raise ValueError(
                "Knowledge frontmatter must contain exactly the versioned note fields."
            )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported knowledge schema version.")
        if value["project_id"] != self.scope.project.project_id:
            raise PermissionError("Knowledge belongs to a different project.")
        if not isinstance(value["id"], str) or not _ID.fullmatch(value["id"]):
            raise ValueError(
                "Note ID must use lowercase letters/digits/hyphens, at most 80 characters."
            )
        if (
            not isinstance(value["kind"], str)
            or value["kind"] not in KINDS
            or not isinstance(value["author_type"], str)
            or value["author_type"] not in {"human", "ai"}
        ):
            raise ValueError("Invalid note kind or author type.")
        for key in ("title", "status", "created_at", "updated_at"):
            if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 256:
                raise ValueError(f"Note {key} must be non-empty bounded text.")
        for key in ("created_at", "updated_at"):
            if datetime.fromisoformat(value[key]).tzinfo is None:
                raise ValueError("Note timestamps require a timezone.")
        for key in ("tags", "references"):
            items = value[key]
            if (
                not isinstance(items, list)
                or len(items) > 100
                or any(not isinstance(item, str) or not item or len(item) > 256 for item in items)
                or len(set(items)) != len(items)
            ):
                raise ValueError(f"Note {key} must be a bounded list of unique strings.")
        if any(not _ID.fullmatch(item) for item in value["references"]):
            raise ValueError("Knowledge links must be exact note IDs, not filesystem paths.")
        refs = value["evidence_refs"]
        if not isinstance(refs, list) or len(refs) > 100:
            raise ValueError("Evidence references must be a bounded list.")
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref) - {"kind", "id", "run", "attempt"}
                or not isinstance(ref.get("kind"), str)
                or ref.get("kind") not in {"execution", "run", "attempt", "product", "dataset"}
                or not isinstance(ref.get("id"), str)
                or len(ref["id"]) > 256
            ):
                raise ValueError(
                    "Evidence references need an exact kind/id and optional Run/Attempt."
                )
            kind, identifier = ref["kind"], ref["id"]
            valid = (
                bool(re.fullmatch(r"sha256:[0-9a-f]{64}", identifier))
                if kind in {"product", "dataset"}
                else bool(re.fullmatch(r"execution-[a-z0-9-]+", identifier))
            )
            if not valid:
                raise ValueError(
                    "Evidence must use exact Execution/content IDs, never names or paths."
                )
            if "run" in ref and not re.fullmatch(r"run-[a-z0-9-]+", str(ref["run"])):
                raise ValueError("Evidence Run requires an exact logical Run ID.")
            if "attempt" in ref and (type(ref["attempt"]) is not int or ref["attempt"] < 1):
                raise ValueError("Evidence Attempt must be a positive integer.")
            if kind in {"run", "attempt"} and "run" not in ref:
                raise ValueError("Run/Attempt evidence requires its exact Run coordinate.")
            if kind == "attempt" and "attempt" not in ref:
                raise ValueError("Attempt evidence requires its Attempt number.")

    @contextmanager
    def _database(self, *, rebuild: bool = False) -> Iterator[sqlite3.Connection]:
        self.scope.assert_current()
        self.scope.path(self.cache.relative_to(self.scope.root), internal=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        database = self.scope.path(
            (self.cache / "index.sqlite3").relative_to(self.scope.root),
            internal=True,
        )
        if database.exists() and not database.is_file():
            raise PermissionError("Knowledge index must be a regular owned file.")
        if rebuild and database.exists():
            # Preserve a corrupt derived index for diagnosis; never remove canonical Markdown.
            backup = self.scope.path(
                (self.cache / f"index-{uuid4().hex}.previous.sqlite3").relative_to(self.scope.root),
                internal=True,
            )
            sidecars = []
            for suffix in ("-journal", "-wal", "-shm"):
                sidecar = self.scope.path(
                    Path(str(database) + suffix).relative_to(self.scope.root),
                    internal=True,
                )
                if sidecar.exists():
                    if not sidecar.is_file():
                        raise PermissionError("Knowledge index sidecars must be regular files.")
                    sidecars.append((sidecar, Path(str(backup) + suffix)))
            database.rename(backup)
            for sidecar, destination in sidecars:
                sidecar.rename(destination)
        connection = sqlite3.connect(database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, stamp TEXT NOT NULL,
                    metadata TEXT NOT NULL, body TEXT NOT NULL, revision TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, title, body);
            """)
            with connection:
                yield connection
        finally:
            connection.close()

    def reindex(self, *, force: bool = False, rebuild: bool = False) -> dict[str, Any]:
        """Rebuild derived search entries after external edits; corrupt Markdown stays untouched."""
        if not self.root.is_dir():
            raise FileNotFoundError("Knowledge is not configured; run lf ai knowledge init.")
        seen: set[str] = set()
        warnings: list[str] = []
        changed = 0
        paths: dict[str, list[Path]] = {}
        for kind in KINDS:
            folder = self.scope.path(Path(self.directory) / kind)
            for path in sorted(folder.glob("*.md")):
                paths.setdefault(path.stem, []).append(path)
        ambiguous = {identifier for identifier, entries in paths.items() if len(entries) != 1}
        warnings.extend(
            f"Ambiguous note ID {identifier}; excluded from index."
            for identifier in sorted(ambiguous)
        )
        with self._lock(), self._database(rebuild=rebuild) as db:
            for kind in KINDS:
                folder = self.scope.path(Path(self.directory) / kind)
                if not folder.exists():
                    continue
                for path in sorted(folder.glob("*.md")):
                    if path.stem in ambiguous:
                        continue
                    if len(seen) >= 10000:
                        raise ValueError("Knowledge index exceeds 10000 notes; split the vault.")
                    relative = path.relative_to(self.scope.root).as_posix()
                    try:
                        self.scope.path(relative)
                        metadata = path.stat()
                        stamp = f"{metadata.st_mtime_ns}:{metadata.st_ctime_ns}:{metadata.st_size}"
                        old = db.execute(
                            "SELECT stamp,id FROM notes WHERE path=?", (relative,)
                        ).fetchone()
                        if old is not None and old["stamp"] == stamp and not force:
                            seen.add(relative)
                            continue
                        note = self._read(relative)
                        identifier = note["metadata"]["id"]
                        db.execute("DELETE FROM search WHERE id=?", (identifier,))
                        db.execute(
                            "INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?,?)",
                            (
                                identifier,
                                relative,
                                stamp,
                                json.dumps(note["metadata"]),
                                note["body"],
                                note["revision"],
                            ),
                        )
                        db.execute(
                            "INSERT INTO search VALUES (?,?,?)",
                            (
                                identifier,
                                note["metadata"]["title"],
                                note["body"],
                            ),
                        )
                        seen.add(relative)
                        changed += 1
                    except (
                        ValueError,
                        PermissionError,
                        OSError,
                        UnicodeError,
                        yaml.YAMLError,
                    ) as error:
                        warnings.append(f"{relative}: {type(error).__name__}: {error}")
            for row in db.execute("SELECT id,path FROM notes").fetchall():
                if row["path"] not in seen:
                    db.execute("DELETE FROM search WHERE id=?", (row["id"],))
                    db.execute("DELETE FROM notes WHERE id=?", (row["id"],))
        self.warnings = tuple(warnings)
        self._indexed_at = time.monotonic()
        return {"changed": changed, "indexed": len(seen), "warnings": warnings, "rebuilt": rebuild}

    def _refresh(self) -> None:
        if time.monotonic() - self._indexed_at >= 5:
            self.reindex()

    def search(
        self,
        query: str = "",
        *,
        kind: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        statuses: list[str] | None = None,
        evidence_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Text/title FTS plus exact metadata filters; snippets are explicitly untrusted data."""
        _page(offset, limit)
        if len(query) > 1000 or (kind is not None and kind not in KINDS):
            raise ValueError("Invalid knowledge query/kind.")
        self._refresh()
        conditions, arguments = [], []
        if query.strip():
            # Treat query as words, not untrusted SQL or arbitrary FTS expressions.
            terms = re.findall(r"\w+", query, flags=re.UNICODE)
            if not terms:
                return {"items": [], "offset": offset, "next_offset": None, "warnings": []}
            conditions.append("n.id IN (SELECT id FROM search WHERE search MATCH ?)")
            arguments.append(" AND ".join('"' + term + '"' for term in terms))
        for field, value in (("kind", kind), ("status", status)):
            if value is not None:
                conditions.append(f"json_extract(n.metadata,'$.{field}')=?")
                arguments.append(value)
        if statuses is not None:
            if (
                status is not None
                or len(statuses) > 20
                or any(not isinstance(s, str) for s in statuses)
            ):
                raise ValueError("Use either one status or a bounded list of statuses.")
            conditions.append(
                "json_extract(n.metadata,'$.status') IN (SELECT value FROM json_each(?))"
            )
            arguments.append(json.dumps(statuses))
        for field, value in (("tags", tag), ("evidence_refs", evidence_id)):
            if value is not None:
                selection = "value" if field == "tags" else "json_extract(value,'$.id')"
                conditions.append(
                    f"EXISTS (SELECT 1 FROM json_each(n.metadata,'$.{field}') WHERE {selection}=?)"
                )
                arguments.append(value)
        for operator, value in ((">=", after), ("<=", before)):
            if value is not None:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    raise ValueError("Knowledge date filters require a timezone.")
                conditions.append(
                    f"julianday(json_extract(n.metadata,'$.updated_at')){operator}julianday(?)"
                )
                arguments.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database() as db:
            rows = db.execute(
                "SELECT n.* FROM notes n"
                + where
                + " ORDER BY json_extract(n.metadata,'$.updated_at') DESC,n.id "
                "LIMIT ? OFFSET ?",
                (*arguments, limit + 1, offset),
            ).fetchall()
        return {
            "items": [
                {
                    "metadata": json.loads(row["metadata"]),
                    "revision": row["revision"],
                    "snippet": row["body"][:500],
                    "trust": "untrusted_project_text",
                }
                for row in rows[:limit]
            ],
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
            "warnings": list(self.warnings),
        }

    def read(self, note_id: str) -> dict[str, Any]:
        if not isinstance(note_id, str) or not _ID.fullmatch(note_id):
            raise ValueError("Invalid note ID.")
        paths = [Path(self.directory) / kind / f"{note_id}.md" for kind in KINDS]
        matches = [path for path in paths if self.scope.path(path).exists()]
        if len(matches) != 1:
            raise ValueError("Note ID is missing or ambiguous; inspect/reindex the Markdown vault.")
        return self._read(matches[0].as_posix())

    def create(
        self,
        kind: str,
        title: str,
        *,
        body: str | None = None,
        author_type: str = "human",
        tags: list[str] | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Explicit write, never automatic conversation capture or a scientific verification."""
        prepared = self.prepare_create(
            kind, title, body=body, author_type=author_type, tags=tags, evidence_refs=evidence_refs
        )
        metadata = prepared["metadata"]
        note_id = "note-" + uuid4().hex
        metadata["id"] = note_id
        content = prepared["body"]
        with self._lock():
            path = self.scope.path(Path(self.directory) / kind / f"{note_id}.md")
            self._write(path, metadata, content, action="create")
        self._indexed_at = 0
        return self.read(note_id)

    def prepare_create(
        self,
        kind: str,
        title: str,
        *,
        body: str | None = None,
        author_type: str = "human",
        tags: list[str] | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate a prospective note without creating IDs, files or index entries."""
        if not self.status()["configured"]:
            raise FileNotFoundError("Initialize knowledge before creating notes.")
        metadata = {
            "schema_version": 1,
            "id": "note-" + "0" * 32,
            "kind": kind,
            "title": title,
            "status": "proposed" if kind == "hypothesis" else "draft",
            "created_at": _now(),
            "updated_at": _now(),
            "author_type": author_type,
            "project_id": self.scope.project.project_id,
            "tags": tags or [],
            "references": [],
            "evidence_refs": evidence_refs or [],
        }
        self._validate(metadata)
        content = self.template(kind) if body is None else body
        self._encode(metadata, content)
        self.scope.path(Path(self.directory) / kind)
        return {"metadata": metadata, "body": content, "prospective_id_not_allocated": True}

    def update(self, note_id: str, *, revision: str, body: str) -> dict[str, Any]:
        """Compare-and-swap Markdown; preserve provenance/status and reject stale revisions."""
        with self._lock():
            prepared = self.prepare_update(note_id, revision=revision, body=body)
            self._write(
                self.scope.path(prepared["path"]),
                prepared["metadata"],
                body,
                action="update",
                expected_revision=revision,
            )
        self._indexed_at = 0
        return self.read(note_id)

    def prepare_update(self, note_id: str, *, revision: str, body: str) -> dict[str, Any]:
        """Pure same-revision native edit validation shared by CLI, approvals and apply."""
        note = self.read(note_id)
        if note["revision"] != revision:
            raise ValueError("Knowledge write conflict: reload the current note before updating.")
        metadata = {**note["metadata"], "updated_at": _now()}
        self._encode(metadata, body)
        return {**note, "metadata": metadata, "body": body}

    def link(self, note_id: str, target_id: str, *, revision: str) -> dict[str, Any]:
        """Add an exact existing note reference with the same optimistic write contract."""
        with self._lock():
            prepared = self.prepare_link(note_id, target_id, revision=revision)
            self._write(
                self.scope.path(prepared["path"]),
                prepared["metadata"],
                prepared["body"],
                action="link",
                expected_revision=revision,
            )
        self._indexed_at = 0
        return self.read(note_id)

    def prepare_link(self, note_id: str, target_id: str, *, revision: str) -> dict[str, Any]:
        """Pure exact link preview; validates both identities and the source revision."""
        target = self.read(target_id)
        note = self.read(note_id)
        prepared = self.prepare_update(note_id, revision=revision, body=note["body"])
        metadata = prepared["metadata"]
        metadata["references"] = list(metadata["references"])
        if target_id not in metadata["references"]:
            metadata["references"].append(target_id)
        self._validate(metadata)
        self._encode(metadata, note["body"])
        return {**prepared, "target_revision": target["revision"]}

    def _write(
        self,
        path: Path,
        metadata: Mapping[str, Any],
        body: str,
        *,
        action: str,
        expected_revision: str | None = None,
    ) -> None:
        from lambdaforge.agents.knowledge_archive import append_event

        raw = self._encode(metadata, body)
        previous = (
            hashlib.sha256(
                self.scope.file(path.relative_to(self.scope.root), limit=_MAX_NOTE)
            ).hexdigest()
            if path.exists()
            else None
        )
        if previous != expected_revision:
            raise ValueError("Knowledge write conflict: file changed after validation; reload it.")
        event = {
            "event_id": uuid4().hex,
            "action": action,
            "note_id": metadata["id"],
            "note_author_type": metadata["author_type"],
            "previous_revision": previous,
            "revision": hashlib.sha256(raw).hexdigest(),
        }
        append_event(self.scope, {**event, "state": "intent"})
        if path.exists():
            current = hashlib.sha256(
                self.scope.file(path.relative_to(self.scope.root), limit=_MAX_NOTE)
            ).hexdigest()
            if current != expected_revision:
                raise ValueError(
                    "Knowledge write conflict: external edit before commit; reload it."
                )
        elif expected_revision is not None:
            raise ValueError("Knowledge write conflict: note disappeared before commit.")
        atomic_write_bytes(path, raw)
        append_event(self.scope, {**event, "state": "committed"})

    def audit(self, *, cursor: int = 0, limit: int = 20) -> dict[str, Any]:
        """Bounded journal of explicit native edits; external Markdown edits are not attributed."""
        from lambdaforge.agents.knowledge_archive import read_audit

        return read_audit(self.scope, cursor=cursor, limit=limit)

    def preview_export(self, destination: str) -> dict[str, Any]:
        """Inspect an exact notes-only compressed snapshot without changing the vault."""
        from lambdaforge.agents.knowledge_archive import KnowledgeArchive

        return KnowledgeArchive(self).preview(destination)

    def export(self, destination: str, *, plan_hash: str) -> dict[str, Any]:
        """Explicit new snapshot export; immutable scientific evidence is never bundled here."""
        from lambdaforge.agents.knowledge_archive import KnowledgeArchive

        return KnowledgeArchive(self).export(destination, plan_hash=plan_hash)

    @staticmethod
    def _encode(metadata: Mapping[str, Any], body: str) -> bytes:
        if not isinstance(body, str):
            raise ValueError("Knowledge body must be text.")
        text = "---\n" + yaml.safe_dump(dict(metadata), sort_keys=False) + "---\n" + body
        if len(text.encode("utf-8")) > _MAX_NOTE:
            raise ValueError("Knowledge note exceeds its 64 KiB limit.")
        return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")

    def related(self, note_id: str, *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Exact links/shared tags/evidence, not invented semantic similarity or truth scores."""
        note = self.read(note_id)
        _page(offset, limit)
        self._refresh()
        metadata = note["metadata"]
        links = metadata["references"]
        tags = metadata["tags"]
        evidence_ids = [ref["id"] for ref in metadata["evidence_refs"]]
        with self._database() as db:
            rows = db.execute(
                "SELECT n.id,n.metadata,n.revision FROM notes n WHERE n.id != ? AND ("
                "n.id IN (SELECT value FROM json_each(?)) OR "
                "EXISTS (SELECT 1 FROM json_each(n.metadata,'$.references') WHERE value=?) OR "
                "EXISTS (SELECT 1 FROM json_each(n.metadata,'$.tags') WHERE value IN "
                "(SELECT value FROM json_each(?))) OR "
                "EXISTS (SELECT 1 FROM json_each(n.metadata,'$.evidence_refs') WHERE "
                "json_extract(value,'$.id') IN (SELECT value FROM json_each(?)))) "
                "ORDER BY n.id LIMIT ? OFFSET ?",
                (
                    note_id,
                    json.dumps(links),
                    note_id,
                    json.dumps(tags),
                    json.dumps(evidence_ids),
                    limit + 1,
                    offset,
                ),
            ).fetchall()
        items = []
        for row in rows[:limit]:
            other = json.loads(row["metadata"])
            reasons = []
            if row["id"] in links or note_id in other["references"]:
                reasons.append("explicit_note_link")
            if set(tags).intersection(other["tags"]):
                reasons.append("shared_tags")
            if set(evidence_ids).intersection(ref["id"] for ref in other["evidence_refs"]):
                reasons.append("shared_exact_evidence")
            items.append(
                {
                    "metadata": other,
                    "revision": row["revision"],
                    "reasons": reasons,
                    "trust": "untrusted_project_text",
                }
            )
        return {
            "items": items,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
            "method": "explicit_links_tags_exact_evidence",
            "warnings": list(self.warnings),
        }

    def backlinks(self, note_id: str, *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Exact frontmatter links and [[note-id]] links, not execution of Markdown contents."""
        self.read(note_id)
        _page(offset, limit)
        self._refresh()
        with self._database() as db:
            rows = db.execute(
                """
                SELECT id,metadata FROM notes
                WHERE EXISTS (SELECT 1 FROM json_each(metadata,'$.references') WHERE value=?)
                    OR instr(body,?)>0
                ORDER BY id LIMIT ? OFFSET ?
            """,
                (note_id, f"[[{note_id}]]", limit + 1, offset),
            ).fetchall()
        return {
            "items": [json.loads(row["metadata"]) for row in rows[:limit]],
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    def verify_references(
        self,
        note_id: str,
        resolver: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Resolve native evidence references, never verify an interpretation."""
        note = self.read(note_id)
        references = []
        for ref in note["metadata"]["evidence_refs"]:
            try:
                value = resolver(ref)
            except (FileNotFoundError, KeyError):
                value = {"status": "unavailable"}
            except ValueError:
                value = {"status": "incompatible_or_corrupt"}
            references.append({"reference": ref, **value})
        return {
            "note_id": note_id,
            "references": references,
            "interpretation": "requires_review",
            "scientifically_verified": False,
        }


__all__ = ["KINDS", "KnowledgeService"]
