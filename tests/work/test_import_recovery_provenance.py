"""Recovered portable evidence keeps origin and audited current revision distinct."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambdaforge.work.ResultStore import ResultStore, _inventory

ORIGIN = "sha256:" + "1" * 64
CURRENT = "sha256:" + "2" * 64
CODE = {
    "provider": "git",
    "revision": "a" * 40,
    "dirty": False,
    "changes_digest": None,
    "source_digest": None,
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(root: Path) -> None:
    path = root / "manifest.json"
    value = json.loads(path.read_text())
    inventory = [r for r in _inventory(root) if r["path"] != "manifest.json"]
    value.update(
        inventory=inventory,
        file_count=len(inventory),
        size_bytes=sum(r["size_bytes"] for r in inventory),
    )
    write(path, value)


@pytest.fixture
def recovered(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    origin = {
        "execution_id": "execution-original",
        "name": "recovered",
        "scientific_fingerprint": ORIGIN,
        "ownership": {"execution_dir": "owned"},
    }
    write(root / "execution/execution.json", origin)
    write(root / "execution/configuration.json", {"name": "recovered", "run": "unused.Work"})
    write(
        root / "execution/current-code.json",
        {"scientific_fingerprint": CURRENT, "code_identity": CODE},
    )
    record = {
        "timestamp_utc": "2026-10-13T09:58:56+00:00",
        "compatible_evidence_reuse": True,
        "scientific_fingerprint": CURRENT,
        "code_identity": CODE,
        "job_id": "job-recovery",
    }
    (root / "execution/recovery-history.jsonl").write_text(json.dumps(record) + "\n")
    write(
        root / "execution/result.json",
        {
            "execution_result_version": 1,
            "execution_id": "execution-original",
            "name": "recovered",
            "scientific_fingerprint": CURRENT,
            "status": "succeeded",
            "runs": [],
            "summary": {},
        },
    )
    write(
        root / "manifest.json",
        {
            "lambdaforge_export_version": 2,
            "execution_id": "execution-original",
            "name": "recovered",
            "scientific_fingerprint": CURRENT,
            "finalized": True,
            "export_kind": "final",
            "execution_status": "succeeded",
            "status": "succeeded",
            "products": [],
        },
    )
    seal(root)
    return root


def test_recovered_import_preserves_origin_and_read_only_result(
    recovered: Path, tmp_path: Path
) -> None:
    store = ResultStore(tmp_path / "runs")
    before = {
        str(p.relative_to(recovered)): p.read_bytes() for p in recovered.rglob("*") if p.is_file()
    }
    preview = store.import_export(recovered, product_root=tmp_path / "products")
    assert preview["will_execute"] is False and not store.root.exists()
    receipt = store.import_export(recovered, product_root=tmp_path / "products", apply=True)
    assert receipt["applied"]
    target = Path(receipt["path"])
    assert (
        json.loads((target / "portable/execution/execution.json").read_text())[
            "scientific_fingerprint"
        ]
        == ORIGIN
    )
    assert json.loads((target / "result.json").read_text())["scientific_fingerprint"] == CURRENT
    assert all((target / "portable" / p).read_bytes() == body for p, body in before.items())
    assert all((recovered / p).read_bytes() == body for p, body in before.items())
    assert store.import_export(recovered, product_root=tmp_path / "products", apply=True)["reused"]


@pytest.mark.parametrize(
    "damage",
    [
        "missing-history",
        "missing-revision",
        "unattested",
        "stale-history",
        "wrong-code",
        "wrong-result",
        "wrong-execution",
        "wrong-owner",
        "malformed-history",
        "duplicate-fields",
        "stale-manifest",
    ],
)
def test_recovered_import_rejects_inconsistent_provenance_without_writes(
    recovered: Path, tmp_path: Path, damage: str
) -> None:
    history = recovered / "execution/recovery-history.jsonl"
    revision = recovered / "execution/current-code.json"
    if damage == "missing-history":
        history.unlink()
    elif damage == "missing-revision":
        revision.unlink()
    elif damage in {"unattested", "stale-history", "wrong-code"}:
        r = json.loads(history.read_text())
        if damage == "unattested":
            r["compatible_evidence_reuse"] = False
        elif damage == "stale-history":
            r["scientific_fingerprint"] = ORIGIN
        else:
            r["code_identity"] = {**CODE, "revision": "b" * 40}
        history.write_text(json.dumps(r) + "\n")
    elif damage == "malformed-history":
        history.write_text("not JSON\n")
    elif damage == "duplicate-fields":
        history.write_text('{"compatible_evidence_reuse":false,"compatible_evidence_reuse":true}\n')
    else:
        path = recovered / (
            "manifest.json"
            if damage == "stale-manifest"
            else "execution/result.json"
            if damage == "wrong-result"
            else "execution/execution.json"
        )
        r = json.loads(path.read_text())
        if damage == "wrong-execution":
            r["execution_id"] = "execution-other"
        elif damage == "wrong-owner":
            r["ownership"]["execution_dir"] = "foreign"
        else:
            r["scientific_fingerprint"] = "sha256:" + "3" * 64
        write(path, r)
    seal(recovered)
    with pytest.raises(ValueError):
        ResultStore(tmp_path / "runs").import_export(
            recovered, product_root=tmp_path / "products", apply=True
        )
    assert not (tmp_path / "runs").exists() and not (tmp_path / "products").exists()


def test_recovered_snapshot_keeps_active_revision(recovered: Path, tmp_path: Path) -> None:
    (recovered / "execution/result.json").unlink()
    path = recovered / "manifest.json"
    value = json.loads(path.read_text())
    value.update(finalized=False, export_kind="snapshot", status="running", execution_status=None)
    write(path, value)
    seal(recovered)
    receipt = ResultStore(tmp_path / "runs").import_export(
        recovered, product_root=tmp_path / "products", apply=True
    )
    assert not receipt["finalized"] and receipt["will_execute"] is False
    result = json.loads((Path(receipt["path"]) / "result.json").read_text())
    assert result["scientific_fingerprint"] == CURRENT and result["runs"] == []
