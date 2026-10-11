"""Local human approval for a closed set of project actions, never agent-issued approval.

The MCP surface may preview/request/apply, but cannot approve. A human reviews the exact
request in a separate local CLI/Console channel. This protects the protocol boundary, not
against another program with independent unrestricted access to the same OS account.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TextIO
from uuid import uuid4

from lambdaforge.agents.contracts import Permission, page, redact
from lambdaforge.agents.knowledge import KnowledgeService
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_bytes

_ID = re.compile(r"approval-[0-9a-f]{32}\Z")
_MAX_REQUEST = 128 * 1024
_TTL = 900


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ActionSpec:
    """One explicit action, its authority, native owner and permitted argument names."""

    permission: Permission
    service: str
    arguments: frozenset[str]
    required: frozenset[str]


ACTIONS = {
    "knowledge.create": ActionSpec(
        "WRITE_PROJECT",
        "KnowledgeService",
        frozenset({"kind", "title", "body", "tags", "evidence_refs"}),
        frozenset({"kind", "title", "body"}),
    ),
    "knowledge.update": ActionSpec(
        "WRITE_PROJECT",
        "KnowledgeService",
        frozenset({"note_id", "revision", "body"}),
        frozenset({"note_id", "revision", "body"}),
    ),
    "knowledge.link": ActionSpec(
        "WRITE_PROJECT",
        "KnowledgeService",
        frozenset({"note_id", "revision", "target_id"}),
        frozenset({"note_id", "revision", "target_id"}),
    ),
    "knowledge.export": ActionSpec(
        "WRITE_PROJECT",
        "KnowledgeService",
        frozenset({"destination"}),
        frozenset({"destination"}),
    ),
}


class ApprovalService:
    """Signed expiring local plans; one locked claim precedes any native mutation."""

    def __init__(self, scope: ProjectScope) -> None:
        self.scope = scope
        self.directory = ".lambdaforge/agents/approvals"

    def _path(self, name: str) -> Any:
        self.scope.assert_current()
        return self.scope.path(f"{self.directory}/{name}", internal=True)

    def _lock(self) -> CrossProcessFileLock:
        return CrossProcessFileLock(
            self._path("approval.lock"),
            shared=False,
            timeout_seconds=10,
            poll_interval_seconds=0.05,
        )

    def _key(self, *, create: bool = False) -> bytes:
        path = self._path("key")
        if not path.exists():
            if not create:
                raise FileNotFoundError(
                    "Approval key is missing; existing requests cannot be trusted or revived."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise PermissionError("Approval key must be private to the local user (mode 0600).")
        key = self.scope.file(f"{self.directory}/key", limit=32, internal=True)
        if len(key) != 32:
            raise ValueError(
                "Approval key is corrupt; do not replace it to revive existing requests."
            )
        return key

    def _write(self, record: dict[str, Any]) -> None:
        value = {
            **record,
            "seal": hmac.new(
                self._key(create=True), _canonical(record), hashlib.sha256
            ).hexdigest(),
        }
        raw = _canonical(value)
        if len(raw) > _MAX_REQUEST:
            raise ValueError("Approval request exceeds its metadata limit.")
        path = self._path(record["request_id"] + ".json")
        atomic_write_bytes(path, raw)
        if os.name != "nt":
            path.chmod(0o600)

    def _read(self, request_id: str) -> dict[str, Any]:
        if not _ID.fullmatch(request_id):
            raise ValueError("Invalid approval request ID.")
        value = json.loads(
            self.scope.file(
                f"{self.directory}/{request_id}.json",
                limit=_MAX_REQUEST,
                internal=True,
            )
        )
        if not isinstance(value, dict) or "seal" not in value:
            raise ValueError("Approval request is corrupt.")
        seal = value.pop("seal")
        expected = hmac.new(self._key(), _canonical(value), hashlib.sha256).hexdigest()
        if not isinstance(seal, str) or not hmac.compare_digest(seal, expected):
            raise PermissionError(
                "Approval request was modified outside the trusted approval channel."
            )
        if (
            value.get("project_id") != self.scope.project.project_id
            or value.get("request_id") != request_id
        ):
            raise PermissionError("Approval belongs to another project or request.")
        return value

    def preview(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pure scoped preview. Never create a key, request, index or scientific record."""
        self.scope.assert_current()
        spec = ACTIONS.get(action)
        if spec is None:
            raise ValueError(
                "Unsupported action. Discover agent_actions before requesting a mutation."
            )
        if (
            not isinstance(arguments, dict)
            or set(arguments) - spec.arguments
            or not spec.required <= set(arguments)
        ):
            raise ValueError(
                "Action arguments are missing or unknown; approval booleans are not accepted."
            )
        if len(_canonical(arguments)) > 64 * 1024 or redact(arguments) != arguments:
            raise ValueError(
                "Action is too large or contains detectable secrets; do not store secrets in notes."
            )
        vault = KnowledgeService(self.scope)
        if not vault.status()["configured"]:
            raise FileNotFoundError(
                "Initialize the project knowledge vault through lf ai knowledge init --apply."
            )
        # Validate all prospective metadata without allocating a note or changing a revision.
        if action == "knowledge.create":
            vault.prepare_create(**arguments, author_type="ai")
            target = self.scope.path(f"knowledge/{arguments['kind']}")
            evidence = {
                "vault": "knowledge",
                "kind": arguments["kind"],
                "kind_exists": target.exists(),
            }
        elif action == "knowledge.export":
            if not isinstance(arguments["destination"], str):
                raise ValueError("Export destination must be project-relative text.")
            evidence = vault.preview_export(arguments["destination"])
        else:
            for name in ("note_id", "revision", "target_id"):
                if name in arguments and not isinstance(arguments[name], str):
                    raise ValueError("Note identity and revision must be text.")
            current = (
                vault.prepare_link(**arguments)
                if action == "knowledge.link"
                else vault.prepare_update(**arguments)
            )
            evidence = {"path": current["path"], "revision": current["revision"]}
            if action == "knowledge.link":
                evidence["target_revision"] = current["target_revision"]
        value = {
            "schema_version": 1,
            "project_id": self.scope.project.project_id,
            "action": action,
            "permission": spec.permission,
            "service": spec.service,
            "arguments_hash": _digest(arguments),
            "evidence": evidence,
            "effect": (
                "Create an exact notes-only ZIP; scientific evidence is unchanged."
                if action == "knowledge.export"
                else "Edit interpretation Markdown only; scientific evidence is unchanged."
            ),
            "human_approval_required": True,
        }
        return {**value, "plan_hash": _digest(value)}

    def request(
        self, action: str, arguments: dict[str, Any], *, ttl_seconds: int = _TTL
    ) -> dict[str, Any]:
        """Store a proposed action, never approved by this method or by its arguments."""
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= _TTL:
            raise ValueError("Approval expiry must be 1–900 seconds.")
        with self._lock():
            plan = self.preview(action, arguments)
            existing = self._records()
            for record in existing:
                if (
                    record["plan_hash"] == plan["plan_hash"]
                    and record["state"] in {"pending", "approved"}
                    and record["expires_at"] > time.time()
                ):
                    return self._public(record)
            if (
                sum(
                    r["state"] in {"pending", "approved", "applying"}
                    and r["expires_at"] > time.time()
                    for r in existing
                )
                >= 128
            ):
                raise ValueError(
                    "Too many pending requests; reject old requests before adding more."
                )
            record = {
                **plan,
                "request_id": "approval-" + uuid4().hex,
                "arguments": arguments,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds,
                "state": "pending",
                "approval": None,
            }
            self._audit(record, "requested")
            self._write(record)
            return self._public(record)

    def _record_paths(self) -> list[Any]:
        directory = self._path("approval.lock").parent
        if not directory.exists():
            return []
        paths = sorted(directory.glob("approval-*.json"))
        if len(paths) > 2000:
            raise ValueError(
                "Approval history exceeds the bounded inventory; archive it locally after review."
            )
        return paths

    def _records(self) -> list[dict[str, Any]]:
        return [self._read(path.stem) for path in self._record_paths()]

    def list(self, *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Validate only the selected request page; no canonical note files are read."""
        page([], offset, limit)
        selected = page(self._record_paths(), offset, limit)
        return {
            **selected,
            "items": [self._public(self._read(path.stem)) for path in selected["items"]],
        }

    def inspect(self, request_id: str, *, include_arguments: bool = False) -> dict[str, Any]:
        record = self._read(request_id)
        return {
            **self._public(record),
            **({"arguments": record["arguments"]} if include_arguments else {}),
        }

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in record.items() if key not in {"arguments", "result"}
        } | {
            "expired": record["expires_at"] <= time.time(),
            "result_available": "result" in record,
            "trust": "untrusted_action_proposal_not_authorization",
        }

    def _audit(self, record: dict[str, Any], event: str) -> None:
        path = self._path("audit.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            path,
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
        with os.fdopen(fd, "ab") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PermissionError("Approval audit must be a regular local file.")
            stream.write(
                _canonical(
                    {
                        "schema_version": 1,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "request_id": record["request_id"],
                        "project_id": record["project_id"],
                        "action": record["action"],
                        "plan_hash": record["plan_hash"],
                        "event": event,
                        "approval": record.get("approval"),
                    }
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def approve_local(
        self,
        request_id: str,
        *,
        plan_hash: str,
        channel: Literal["cli", "console"],
        confirmation: str,
    ) -> dict[str, Any]:
        """Trusted local UI boundary ONLY. Never register this method as an MCP tool.

        The caller must be the local interactive CLI or a human Console event handler which
        rendered the full proposal. Same-account external shell/Python access is not sandboxed.
        """
        if channel not in {"cli", "console"} or confirmation != request_id:
            raise PermissionError("Exact human request-ID confirmation is required.")
        with self._lock():
            record = self._read(request_id)
            self._validate_pending(record, plan_hash, {"pending"})
            record["approval"] = {
                "channel": channel,
                "approved_at": time.time(),
                "request_id": request_id,
                "plan_hash": plan_hash,
            }
            record["state"] = "approved"
            self._audit(record, "approved")
            self._write(record)
            return self._public(record)

    def approve_interactive(
        self, request_id: str, *, plan_hash: str, stdin: TextIO, stderr: TextIO
    ) -> dict[str, Any]:
        """TTY-only reviewer; redirected stdin cannot become an agent approval shortcut."""
        if not stdin.isatty() or not stderr.isatty():
            raise PermissionError(
                "Approval requires a local interactive terminal or the Research Console."
            )
        proposal = self.inspect(request_id, include_arguments=True)
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console(file=stderr)
        table = Table("Field", "Exact value", show_lines=True)
        for key in ("project_id", "request_id", "action", "plan_hash", "expires_at", "effect"):
            table.add_row(key, Text(str(proposal[key])))
        console.print(table)
        for key, value in proposal["arguments"].items():
            console.print(Panel(Text(str(value)), title=key))
        console.print(
            "Review all values. Type the full request ID to approve, or Enter to decline:"
        )
        return self.approve_local(
            request_id, plan_hash=plan_hash, channel="cli", confirmation=stdin.readline().strip()
        )

    def reject_local(self, request_id: str) -> dict[str, Any]:
        """Local operator refusal does not execute or silently replace any plan."""
        with self._lock():
            record = self._read(request_id)
            if record["state"] not in {"pending", "approved"}:
                raise ValueError("Only pending/approved requests may be rejected.")
            record["state"] = "rejected"
            self._audit(record, "rejected")
            self._write(record)
            return self._public(record)

    def _validate_pending(self, record: dict[str, Any], plan_hash: str, states: set[str]) -> None:
        if record["state"] not in states:
            raise PermissionError(
                "Request is already claimed, consumed or rejected; replay is refused."
            )
        if record["expires_at"] <= time.time():
            raise PermissionError("Approval has expired; propose and review a fresh plan.")
        plan = self.preview(record["action"], record["arguments"])
        if record["plan_hash"] != plan_hash or plan["plan_hash"] != plan_hash:
            raise PermissionError(
                "Approved arguments/evidence have changed; propose and review again."
            )

    def apply(
        self, request_id: str, *, plan_hash: str, permissions: tuple[Permission, ...]
    ) -> dict[str, Any]:
        """Consume once, revalidate and invoke the exact native action. No automatic replay.

        A crash after the durable claim is deliberately ambiguous, never permission to repeat a
        possibly completed write. Inspect canonical Markdown and audit before a new request.
        """
        with self._lock():
            record = self._read(request_id)
            spec = ACTIONS[record["action"]]
            if spec.permission not in permissions:
                raise PermissionError(
                    "Action permission is not enabled in this operator-configured session."
                )
            self._validate_pending(record, plan_hash, {"approved"})
            if record["approval"] is None or record["approval"]["plan_hash"] != plan_hash:
                raise PermissionError("A trusted local approval is required.")
            record["state"] = "applying"
            self._audit(record, "claimed")
            self._write(record)
            try:
                vault = KnowledgeService(self.scope)
                arguments = record["arguments"]
                if record["action"] == "knowledge.create":
                    result = vault.create(**arguments, author_type="ai")
                elif record["action"] == "knowledge.update":
                    result = vault.update(**arguments)
                elif record["action"] == "knowledge.export":
                    result = vault.export(**arguments, plan_hash=record["evidence"]["plan_hash"])
                else:
                    result = vault.link(**arguments)
            except Exception:
                record["state"] = "failed"
                self._audit(record, "failed_no_automatic_replay")
                self._write(record)
                raise
            record["state"] = "applied"
            record["result"] = (
                {
                    "destination": result["destination"],
                    "notes": result["notes"],
                    "inventory_hash": result["inventory_hash"],
                }
                if record["action"] == "knowledge.export"
                else {"note_id": result["metadata"]["id"], "revision": result["revision"]}
            )
            self._write(record)
            self._audit(record, "applied")
            return {**self._public(record), "result": record["result"]}


__all__ = ["ACTIONS", "ApprovalService"]
