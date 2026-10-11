"""Exact bounded note transport and operational edit history, never scientific authority."""

import hashlib
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from lambdaforge.agents.approvals import ApprovalService
from lambdaforge.agents.knowledge import KnowledgeService
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.agents.services import AgentServices


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="archive"\nversion="0.1"\n')
    result = KnowledgeService(ProjectScope(tmp_path))
    result.initialize(obsidian=True)
    return result


def test_exact_compressed_export_pure_preview_and_no_operational_files(vault):
    note = vault.create("finding", "Exact evidence", body="An interpretation.\n")
    path = vault.scope.root / note["path"]
    # Canonical bytes include external BOM/line endings; exporter must not normalize them.
    raw = b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(raw)
    (vault.root / ".obsidian" / "settings.json").write_text('{"private":true}')
    (vault.root / "unmanaged.md").write_text("Unrelated personal notes")
    before = sorted(vault.scope.root.rglob("*"))
    preview = vault.preview_export("snapshot.zip")
    assert sorted(vault.scope.root.rglob("*")) == before
    assert preview["notes"] == 1 and preview["bytes"] == len(raw)
    assert not (vault.cache / "index.sqlite3").exists()
    result = vault.export("snapshot.zip", plan_hash=preview["plan_hash"])
    assert result["status"] == "exported"
    with zipfile.ZipFile(vault.scope.root / "snapshot.zip") as archive:
        member = "notes/finding/" + note["metadata"]["id"] + ".md"
        assert set(archive.namelist()) == {member, "manifest.json", "README.md"}
        assert archive.read(member) == raw
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["files"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert manifest["scientific_records_included"] is False
        assert archive.getinfo(member).compress_type == zipfile.ZIP_DEFLATED
    assert not list(vault.scope.root.glob(".lf-knowledge-*"))
    with pytest.raises(FileExistsError):
        vault.export("snapshot.zip", plan_hash=preview["plan_hash"])


def test_export_rejects_changed_corrupt_duplicate_and_unsafe_sources(vault, tmp_path):
    note = vault.create("finding", "First", body="Before")
    plan = vault.preview_export("changed.zip")
    vault.update(note["metadata"]["id"], revision=note["revision"], body="After")
    with pytest.raises(ValueError, match="changed"):
        vault.export("changed.zip", plan_hash=plan["plan_hash"])
    assert not (tmp_path / "changed.zip").exists()
    for destination in ("../outside.zip", ".lambdaforge/private.zip", "knowledge/archive.zip"):
        with pytest.raises((ValueError, PermissionError)):
            vault.preview_export(destination)
    bad = vault.root / "finding/bad.md"
    bad.write_text("corrupt")
    with pytest.raises(ValueError, match="frontmatter"):
        vault.preview_export("corrupt.zip")
    bad.unlink()
    source = vault.scope.root / note["path"]
    duplicate = vault.root / "decision" / source.name
    duplicate.parent.mkdir()
    duplicate.write_text(source.read_text().replace("kind: finding", "kind: decision"))
    with pytest.raises(ValueError, match="Duplicate"):
        vault.preview_export("duplicate.zip")
    duplicate.unlink()
    alias = vault.root / "finding/alias.md"
    alias.symlink_to(source)
    with pytest.raises(PermissionError, match="Symbolic"):
        vault.preview_export("alias.zip")


def test_journal_paging_revisions_no_bodies_and_truncated_tail_protection(vault):
    first = vault.create("finding", "Private title", body="private note body")
    second = vault.create("decision", "Second", body="second body")
    changed = vault.update(first["metadata"]["id"], revision=first["revision"], body="updated body")
    vault.link(first["metadata"]["id"], second["metadata"]["id"], revision=changed["revision"])
    cursor, events = 0, []
    while True:
        result = vault.audit(cursor=cursor, limit=1)
        events.extend(result["items"])
        if result["next_cursor"] is None:
            break
        cursor = result["next_cursor"]
    assert len(events) == 8
    assert [e["state"] for e in events] == ["intent", "committed"] * 4
    assert events[4]["previous_revision"] == first["revision"]
    assert events[4]["revision"] == changed["revision"]
    assert not any(word in json.dumps(events) for word in ("Private title", "private note body"))
    with pytest.raises(ValueError, match="exact next_cursor"):
        vault.audit(cursor=1)
    journal = vault.scope.root / ".lambdaforge/agents/knowledge-audit.jsonl"
    journal.write_bytes(journal.read_bytes() + b'{"incomplete":')
    current = vault.read(first["metadata"]["id"])
    with pytest.raises(ValueError, match="incomplete tail"):
        vault.update(first["metadata"]["id"], revision=current["revision"], body="must not apply")
    assert vault.read(first["metadata"]["id"])["revision"] == current["revision"]
    with pytest.raises(ValueError, match="incomplete"):
        vault.audit(limit=100)


def test_audit_rejects_non_metadata_records_before_any_further_mutation(vault):
    note = vault.create("finding", "Original", body="Preserved")
    path = vault.scope.root / ".lambdaforge/agents/knowledge-audit.jsonl"
    events = vault.audit()["items"]
    # A newline does not make a corrupt/user-modified operational record authoritative.
    events[-1]["body"] = "must not appear through an audit read"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        vault.audit()
    with pytest.raises(ValueError, match="corrupt"):
        vault.update(note["metadata"]["id"], revision=note["revision"], body="Must not apply")
    assert vault.read(note["metadata"]["id"])["revision"] == note["revision"]


def test_external_edit_after_native_validation_is_not_overwritten_or_journaled(vault, monkeypatch):
    note = vault.create("finding", "External editor", body="Before")
    native = vault.prepare_update
    path = vault.scope.root / note["path"]
    previous_events = vault.audit()["items"]

    def edit_after_validation(*args, **kwargs):
        result = native(*args, **kwargs)
        path.write_text(path.read_text().replace("Before", "External edit"))
        return result

    monkeypatch.setattr(vault, "prepare_update", edit_after_validation)
    with pytest.raises(ValueError, match="changed after validation"):
        vault.update(note["metadata"]["id"], revision=note["revision"], body="Must not overwrite")
    assert vault.read(note["metadata"]["id"])["body"] == "External edit"
    assert vault.audit()["items"] == previous_events


def test_native_and_approved_export_share_stale_plan_and_single_publish_guarantees(vault):
    vault.create("finding", "One", body="Observation")
    scope = vault.scope
    read = AgentServices(scope.root)
    preview = read.knowledge_export("approved.zip")
    assert preview.status == "ok" and not (scope.root / "approved.zip").exists()
    assert (
        read.request_agent_action("knowledge.export", {"destination": "approved.zip"}).status
        == "error"
    )
    approvals = ApprovalService(scope)
    request = approvals.request("knowledge.export", {"destination": "approved.zip"})
    approvals.approve_local(
        request["request_id"],
        plan_hash=request["plan_hash"],
        channel="console",
        confirmation=request["request_id"],
    )

    def apply():
        try:
            return approvals.apply(
                request["request_id"],
                plan_hash=request["plan_hash"],
                permissions=("READ", "PLAN", "WRITE_PROJECT"),
            )["state"]
        except PermissionError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(lambda _: apply(), range(2))) == ["applied", "refused"]
    assert zipfile.is_zipfile(scope.root / "approved.zip")
    stale = approvals.request("knowledge.export", {"destination": "stale.zip"})
    approvals.approve_local(
        stale["request_id"],
        plan_hash=stale["plan_hash"],
        channel="cli",
        confirmation=stale["request_id"],
    )
    vault.create("finding", "Two", body="New observation")
    with pytest.raises(PermissionError, match="changed"):
        approvals.apply(
            stale["request_id"], plan_hash=stale["plan_hash"], permissions=("WRITE_PROJECT",)
        )
    assert not (scope.root / "stale.zip").exists()


def test_failed_snapshot_cleans_only_owned_temporary_and_preserves_notes(vault, monkeypatch):
    note = vault.create("finding", "Original", body="preserved")
    preview = vault.preview_export("failed.zip")

    def refuse_link(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", refuse_link)
    with pytest.raises(OSError, match="publication failure"):
        vault.export("failed.zip", plan_hash=preview["plan_hash"])
    assert vault.read(note["metadata"]["id"])["revision"] == note["revision"]
    assert not (vault.scope.root / "failed.zip").exists()
    assert not list(vault.scope.root.glob(".lf-knowledge-*"))
    assert vault.audit()["items"][-1]["state"] == "intent"
