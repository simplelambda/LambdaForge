"""Exact compressed placement transfer, without real clusters or scientific data."""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.controlplane.BinaryCommand import subprocess_stream
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.DatasetTransfer import pack, receive


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"deterministic scientific bytes\n" * 10000)
    registry = DatasetRegistry(tmp_path / "controller" / "datasets.json")
    record = DatasetPublisher(registry).publish_members(
        "example",
        "7",
        [{"id": "sample", "assets": {"data": "payload.bin"}}],
        source_root=source,
        publication_root=tmp_path / "published",
        cluster="local",
        build_provenance={"work": "prepare"},
    )
    profiles = {}
    for name in ("local", "gpu12", "gpu16"):
        base = tmp_path / name
        profiles[name] = ClusterProfile(
            name,
            workspace=str(base),
            python=sys.executable,
            storage=ClusterStoragePolicy(
                str(base / "state"),
                str(base / "cache"),
                str(base / "jobs"),
                str(base / "datasets"),
                safety_min_free_percent=0,
            ),
        )
    service = DatasetService(registry, ClusterCatalog(profiles))
    monkeypatch.setattr(service, "_python", lambda *_args: sys.executable)
    # restore environment if helper is exercised directly in this process.
    monkeypatch.setenv(
        "LAMBDAFORGE_STORAGE_POLICY", json.dumps(profiles["gpu16"].storage.to_dict())
    )
    return service, record, tmp_path


def options(service, record, name="gpu16"):
    storage = service.clusters.get(name).storage
    return {
        "record": record.to_dict(),
        "root": storage.dataset_root,
        "registry": str(Path(storage.state_root) / "datasets.json"),
        "cluster": name,
        "storage": storage.to_dict(),
    }


def wire(record):
    output = io.BytesIO()
    pack(Path(record.placements[0].root), record.dataset_id, output)
    return output.getvalue()


def test_preview_then_apply_exact_compressed_and_idempotent(transfer):
    service, record, tmp = transfer
    before = set(tmp.rglob("*"))
    plan = service.replicate(record.key, source="local", destination="gpu16")
    assert plan.action == "REPLICATE" and plan.compression == "tar-gzip-3"
    assert set(tmp.rglob("*")) == before
    progress = []
    service.replicate(
        record.key, source="local", destination="gpu16", apply=True, progress=progress.append
    )
    actual = service.registry.get(record.key)
    assert {item.cluster for item in actual.placements} == {"local", "gpu16"}
    remote = DatasetRegistry(options(service, record)["registry"]).get(record.key)
    assert remote.dataset_id == record.dataset_id
    root = Path(remote.placements[0].root)
    assert DatasetOperations.verify(root, record.dataset_id)["valid"]
    original = Path(record.placements[0].root)
    assert {
        p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()
    } == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert len(wire(record)) < record.placements[0].size_bytes / 5
    assert progress[-1]["phase"] == "completed"
    assert any(p.get("compressed_bytes", 0) > 0 for p in progress)
    assert (
        service.replicate(record.key, source="local", destination="gpu16", apply=True).action
        == "NOOP"
    )
    assert not list(tmp.rglob(".replication-*.tmp"))
    assert not list(tmp.rglob("*.gz"))


def test_remote_source_to_second_placement_without_local_archive(transfer):
    service, record, _ = transfer
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    service.replicate(record.key, source="gpu12", destination="gpu16", apply=True)
    assert {p.cluster for p in service.registry.get(record.key).placements} == {
        "local",
        "gpu12",
        "gpu16",
    }
    assert service.resolve_placement(record.key, "gpu16").physically_available
    assert len(service.list(all_clusters=True)) == 1


def test_direct_route_runs_sender_to_receiver_with_no_controller_stream(transfer, monkeypatch):
    service, record, _ = transfer
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    for name in ("gpu12", "gpu16"):
        service.clusters._profiles[name] = replace(
            service.clusters.get(name),
            transport="ssh",
            host=f"{name}.invalid",
        )
    transport = LocalTransport()
    monkeypatch.setattr(service.factory, "transport", lambda _p: transport)
    # Loopback replaces only SSH routing; the real helper's direct Popen/pack/receive runs.
    monkeypatch.setattr(service, "_site_ssh", lambda _p, command: command)
    monkeypatch.setattr(
        transport, "stream", lambda _c: pytest.fail("No archive on controller route")
    )
    plan = service.replicate(record.key, source="gpu12", destination="gpu16", apply=True)
    assert plan.transfer_route == "direct-ssh"
    assert service.resolve_placement(record.key, "gpu16").physically_available


def test_remote_only_publication_can_be_discovered_then_copied(transfer):
    service, record, _ = transfer
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    service.registry.discard(record.key)
    service.replicate(record.key, source="gpu12", destination="gpu16", apply=True)
    assert {p.cluster for p in service.registry.get(record.key).placements} == {"gpu12", "gpu16"}


def test_receiver_preserves_empty_directories_and_rejects_changed_source(transfer):
    service, record, _ = transfer
    original = Path(record.placements[0].root)
    (original / "empty" / "nested").mkdir(parents=True)
    response = receive(options(service, record), io.BytesIO(wire(record)))
    assert (Path(response["root"]) / "empty" / "nested").is_dir()
    asset = next(path for path in (original / "assets").rglob("*") if path.is_file())
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Source Dataset verification failed"):
        wire(record)


def test_destination_registration_failure_rolls_back_only_new_bytes(transfer, monkeypatch):
    service, record, tmp = transfer
    data = wire(record)

    def refuse(_self, _record):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(DatasetRegistry, "register", refuse)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        receive(options(service, record), io.BytesIO(data))
    destination = Path(options(service, record)["root"]) / record.name / record.version
    assert not list(destination.glob(record.dataset_id.removeprefix("sha256:")[:16]))
    assert not list(tmp.rglob(".replication-*.tmp"))


def test_existing_same_destination_is_not_deleted_on_registration_failure(transfer, monkeypatch):
    service, record, _ = transfer
    data = wire(record)
    root = receive(options(service, record), io.BytesIO(data))["root"]

    def refuse(_self, _record):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(DatasetRegistry, "register", refuse)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        receive(options(service, record), io.BytesIO(data))
    assert DatasetOperations.verify(root, record.dataset_id)["valid"]


def test_interrupted_binary_pipeline_cleans_receiver_staging(transfer, monkeypatch):
    service, record, tmp = transfer
    sender, receiver = LocalTransport(), LocalTransport()
    data = wire(record)[:-30]
    monkeypatch.setattr(
        sender,
        "stream",
        lambda _c: subprocess_stream(
            (
                sys.executable,
                "-c",
                f"import sys;sys.stdout.buffer.write({data!r});sys.exit(7)",
            )
        ),
    )
    monkeypatch.setattr(
        service.factory,
        "transport",
        lambda profile: sender if profile.name == "local" else receiver,
    )
    with pytest.raises(RuntimeError, match="code 7"):
        service.replicate(record.key, source="local", destination="gpu16", apply=True)
    assert not list(tmp.rglob(".replication-*.tmp"))
    assert not DatasetRegistry(options(service, record)["registry"]).records()


def test_receiver_disk_admission_failure_preserves_source_and_indexes(transfer):
    service, record, tmp = transfer
    oversized = b'{"bytes":1180591620717411303424,"entries":1}\n'
    with pytest.raises(OSError, match="destination filesystem"):
        receive(options(service, record), io.BytesIO(oversized))
    assert DatasetOperations.verify(record.placements[0].root, record.dataset_id)["valid"]
    assert not DatasetRegistry(options(service, record)["registry"]).records()
    assert not list(tmp.rglob(".replication-*.tmp"))


def test_receiver_inode_shortage_is_rejected_before_copy(transfer, monkeypatch):
    from lambdaforge.controlplane.StorageAdmission import StorageAdmission

    service, record, tmp = transfer
    monkeypatch.setattr(
        StorageAdmission,
        "filesystem",
        lambda *_a: {
            "inode_accounting": True,
            "free_inodes": 0,
        },
    )
    with pytest.raises(OSError, match="insufficient inodes"):
        receive(options(service, record), io.BytesIO(wire(record)))
    assert not list(tmp.rglob(".replication-*.tmp"))
    assert not DatasetRegistry(options(service, record)["registry"]).records()


def test_reference_cannot_bypass_controller_identity(transfer):
    service, record, _ = transfer
    with pytest.raises(Exception, match="controller Dataset identity"):
        service.resolve_placement(
            record.key,
            "gpu16",
            reference=replace(
                record,
                dataset_id="sha256:" + "f" * 64,
            ),
        )


def test_cli_json_preview_and_apply_are_clean(transfer, monkeypatch, capsys):
    from types import SimpleNamespace

    from lambdaforge.cli import DatasetCommands as module

    service, record, _ = transfer
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _path: service.clusters)
    monkeypatch.setattr(module, "DatasetService", lambda **_k: service)
    arguments = SimpleNamespace(
        dataset_command="replicate",
        clusters=None,
        dataset=record.key,
        source="local",
        destination="gpu16",
        route="auto",
        apply=False,
        json=True,
    )
    assert module.DatasetCommands.run(arguments) == 0
    captured = capsys.readouterr()
    assert not captured.err
    preview = json.loads(captured.out)
    assert preview["content_id"] == record.dataset_id and not preview["applied"]
    arguments.apply = True
    assert module.DatasetCommands.run(arguments) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["applied"] and not captured.err


@pytest.mark.parametrize("damage", ["truncated", "checksum", "gzip-crc"])
def test_bad_stream_does_not_publish_and_cleans_owned_temporary(transfer, damage):
    service, record, tmp = transfer
    data = wire(record)
    if damage == "truncated":
        data = data[:-40]
    elif damage == "gzip-crc":
        data = data[:-8] + b"bad!" + data[-4:]
    else:
        header, compressed = data.split(b"\n", 1)
        archive = gzip.decompress(compressed)
        archive = archive.replace(b"deterministic", b"nondeterminis", 1)
        data = header + b"\n" + gzip.compress(archive)
    with pytest.raises((ValueError, EOFError, OSError, tarfile.TarError)):
        receive(options(service, record), io.BytesIO(data))
    assert not DatasetRegistry(options(service, record)["registry"]).records()
    assert not list(tmp.rglob(".replication-*.tmp"))
    assert service.registry.get(record.key).placements == record.placements


@pytest.mark.parametrize("entry", ["../escape", "/escape", "symlink", "hardlink", "device"])
def test_hostile_archive_rejected(transfer, entry):
    service, record, tmp = transfer
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo(entry)
        if entry == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        elif entry == "hardlink":
            info.type, info.linkname = tarfile.LNKTYPE, "/etc/passwd"
        elif entry == "device":
            info.type = tarfile.CHRTYPE
        archive.addfile(info)
    data = b'{"bytes":0,"entries":1}\n' + gzip.compress(stream.getvalue())
    with pytest.raises(ValueError, match="Unsafe"):
        receive(options(service, record), io.BytesIO(data))
    assert not list(tmp.rglob(".replication-*.tmp"))
    assert not DatasetRegistry(options(service, record)["registry"]).records()


def test_registry_conflict_and_changed_selection_fail_closed(transfer):
    service, record, _ = transfer
    remote = DatasetRegistry(options(service, record)["registry"])
    remote.register(replace(record, dataset_id="sha256:" + "b" * 64, placements=()))
    with pytest.raises(Exception, match="conflict"):
        service.replicate(record.key, source="local", destination="gpu16", apply=True)
    assert remote.get(record.key).dataset_id != record.dataset_id
    with pytest.raises(Exception, match="identity changed"):
        service.replicate(
            record.key,
            source="local",
            destination="gpu12",
            expected_content_id="sha256:" + "c" * 64,
        )


def test_concurrent_duplicate_transfers_converge_on_one_identity(transfer):
    service, record, tmp = transfer

    def copy():
        return service.replicate(record.key, source="local", destination="gpu16", apply=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(copy) for _ in range(2)]
        for future in futures:
            future.result(timeout=45)
    assert len(DatasetRegistry(options(service, record)["registry"]).records()) == 1
    assert len(service.registry.get(record.key).placements) == 2
    assert not list(tmp.rglob(".replication-*.tmp"))


def test_direct_probe_is_read_only_secure_and_falls_back(transfer, monkeypatch):
    service, _, _ = transfer
    for name in ("gpu12", "gpu16"):
        service.clusters._profiles[name] = replace(
            service.clusters.get(name),
            transport="ssh",
            host=f"{name}.invalid",
            user="scientist",
        )
    transport = LocalTransport()
    commands = []

    def probe(command, **_kwargs):
        commands.append(command)
        return CommandResult(255, "", "no site key")

    monkeypatch.setattr(transport, "run", probe)
    monkeypatch.setattr(service.factory, "transport", lambda _profile: transport)
    assert service._transfer_route("gpu12", "gpu16", "auto") == "controller-stream"
    assert "StrictHostKeyChecking=yes" in commands[0] and "ForwardAgent=no" in commands[0]
    assert "IdentityAgent=none" in commands[0]
    with pytest.raises(Exception, match="site SSH is unavailable"):
        service._transfer_route("gpu12", "gpu16", "direct")
    monkeypatch.setattr(
        transport, "run", lambda *_a, **_k: CommandResult(0, "LF_DATA_TRANSFER_READY\n", "")
    )
    assert service._transfer_route("gpu12", "gpu16", "auto") == "direct-ssh"
