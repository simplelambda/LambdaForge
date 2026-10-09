"""Host-side compressed Dataset transfer. Invoked only by the native placement service.

The controller sends this helper to an existing managed Python, so a destination need not
rebuild its immutable scientific environment merely to receive data. No consumer code runs.
"""

from __future__ import annotations

import errno
import gzip
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast
from uuid import uuid4


def pack(root: Path, identity: str, output: BinaryIO) -> None:
    """Verify first, then stream regular files and empty directories as tar/gzip."""
    from lambdaforge.data.DatasetOperations import DatasetOperations
    from lambdaforge.work.snapshot import validate_path

    validate_path(root)
    verified = DatasetOperations.verify(root, identity)
    if not verified.get("valid"):
        raise ValueError("Source Dataset verification failed: " + str(verified.get("errors")))
    entries = sorted(root.rglob("*"))
    size = 0
    for path in entries:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"Dataset transfer rejects symbolic/special entries: {path}")
        if stat.S_ISREG(mode):
            size += path.stat().st_size
    output.write(json.dumps({"bytes": size, "entries": len(entries)}).encode() + b"\n")
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=3, mtime=0) as encoded:
        with tarfile.open(fileobj=encoded, mode="w|") as archive:
            for path in entries:
                info = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                if not (info.isfile() or info.isdir()):
                    raise ValueError("Dataset source changed to a symbolic/special entry.")
                if info.isfile():
                    validate_path(path)
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(descriptor, "rb") as content:
                        if not stat.S_ISREG(os.fstat(content.fileno()).st_mode):
                            raise ValueError("Dataset source is no longer a regular file.")
                        archive.addfile(info, content)
                else:
                    archive.addfile(info)
    output.flush()


def receive(options: dict[str, Any], source: BinaryIO) -> dict[str, Any]:
    """Extract only into owned staging, verify, then lock/commit/register exact bytes."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from lambdaforge.controlplane.StorageAdmission import StorageAdmission
    from lambdaforge.data.DatasetOperations import DatasetOperations
    from lambdaforge.data.DatasetPlacement import DatasetPlacement
    from lambdaforge.data.DatasetRecord import DatasetRecord
    from lambdaforge.data.DatasetRegistry import DatasetRegistry
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
    from lambdaforge.work.snapshot import validate_path

    record = DatasetRecord.from_mapping(options["record"])
    base = Path(options["root"]).expanduser().absolute()
    destination = (
        base / record.name / record.version / record.dataset_id.removeprefix("sha256:")[:16]
    )
    if destination.resolve() != destination or not destination.is_relative_to(base):
        raise ValueError("Unsafe Dataset destination.")
    validate_path(destination)
    from lambdaforge.data.DatasetPublisher import DatasetPublisher

    DatasetPublisher._publication_label(record.name, field="name")
    DatasetPublisher._publication_label(record.version, field="version")
    registry = DatasetRegistry(options["registry"])
    # Registry.register below also validates the complete scientific declaration.
    try:
        previous = registry.get(record.key)
    except KeyError:
        previous = None
    if previous is not None and (
        previous.dataset_id != record.dataset_id
        or previous.metadata.get("lambdaforge_science")
        != record.metadata.get("lambdaforge_science")
    ):
        raise ValueError("Destination already registers a different immutable Dataset identity.")
    raw = source.readline(65537)
    if len(raw) > 65536 or not raw.endswith(b"\n"):
        raise ValueError("Invalid Dataset transfer header.")
    header = json.loads(raw)
    total, count = header["bytes"], header["entries"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (total, count)
    ):
        raise ValueError("Invalid Dataset transfer inventory size.")
    os.environ["LAMBDAFORGE_STORAGE_POLICY"] = json.dumps(options["storage"])
    staging = destination.parent / f".replication-{uuid4().hex}.tmp"
    capacity = StorageAdmission.filesystem(staging, options["storage"])
    if capacity["inode_accounting"] and capacity["free_inodes"] <= count:
        raise OSError(
            errno.ENOSPC,
            f"Dataset destination has insufficient inodes: need={count + 1}, "
            f"available={capacity['free_inodes']}.",
            str(staging),
        )
    created = False
    lock = destination.parent.parent / f".{record.version}.publication.lock"
    with StorageAdmission.transaction(staging, total, purpose="dataset-replication"):
        staging.mkdir(parents=True, exist_ok=False)
        try:
            seen: set[str] = set()
            received = 0
            with gzip.GzipFile(fileobj=source, mode="rb") as decoded:
                with tarfile.open(fileobj=decoded, mode="r|") as archive:
                    for member in archive:
                        name = PurePosixPath(member.name)
                        if (
                            name.is_absolute()
                            or ".." in name.parts
                            or "\\" in member.name
                            or member.name in seen
                            or not name.parts
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ValueError("Unsafe or duplicate Dataset archive entry.")
                        seen.add(member.name)
                        if len(seen) > count or member.size < 0 or received + member.size > total:
                            raise ValueError("Dataset archive exceeds its declared inventory.")
                        path = staging.joinpath(*name.parts)
                        if member.isdir():
                            path.mkdir(parents=True, exist_ok=True)
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            content = archive.extractfile(member)
                            if content is None:
                                raise ValueError("Dataset archive file is missing.")
                            with content, path.open("xb") as sink:
                                shutil.copyfileobj(content, sink, length=1024 * 1024)
                            received += member.size
                # Force gzip CRC/footer verification even after tar's end marker.
                trailing = 0
                while chunk := decoded.read(16384):
                    trailing += len(chunk)
                    if trailing > 65536 or any(chunk):
                        raise ValueError("Unexpected data after Dataset archive.")
            if received != total or len(seen) != count:
                raise ValueError("Incomplete Dataset transfer inventory.")
            verified = DatasetOperations.verify(staging, record.dataset_id)
            if not verified.get("valid"):
                raise ValueError(
                    "Received Dataset checksum verification failed: " + str(verified.get("errors"))
                )
            from lambdaforge.data.DatasetArtifact import DatasetArtifact

            artifact = DatasetArtifact.read_json(staging / "dataset-artifact.json")
            if (artifact.name, artifact.version) != (record.name, record.version):
                raise ValueError("Received Dataset logical version differs.")
            placement = DatasetPlacement(
                options["cluster"],
                str(destination),
                datetime.now(timezone.utc).isoformat(),
                total,
                sum(path.is_file() for path in staging.rglob("*")),
                True,
            )
            actual = registry.artifact_record(
                staging / "dataset-artifact.json",
                cluster=options["cluster"],
                root=staging,
                producer=record.producer,
            )
            if actual.metadata.get("lambdaforge_science") != record.metadata.get(
                "lambdaforge_science"
            ):
                raise ValueError("Received Dataset scientific declaration differs.")
            updated = replace(actual, placements=(placement,))
            with CrossProcessFileLock(
                lock,
                shared=False,
                timeout_seconds=30,
                poll_interval_seconds=0.05,
            ):
                if destination.exists():
                    if not DatasetOperations.verify(destination, record.dataset_id).get("valid"):
                        raise ValueError(
                            "Refusing to overwrite an invalid/different Dataset destination."
                        )
                    existing = DatasetArtifact.read_json(destination / "dataset-artifact.json")
                    if (existing.name, existing.version, existing.scientific_identity) != (
                        artifact.name,
                        artifact.version,
                        artifact.scientific_identity,
                    ):
                        raise ValueError("Existing Dataset declaration differs.")
                else:
                    os.replace(staging, destination)
                    created = True
                try:
                    registry.register(updated)
                except Exception:
                    if created:
                        shutil.rmtree(destination)
                    raise
            return {"root": str(destination), "placement": placement.to_dict(), "verified": True}
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main() -> None:
    """Private wire endpoint used by DatasetService, not a second public command."""
    operation = sys.argv[1]
    if operation == "pack":
        pack(Path(sys.argv[2]), sys.argv[3], sys.stdout.buffer)
    elif operation == "receive":
        print(json.dumps(receive(json.loads(sys.argv[2]), sys.stdin.buffer)))
    elif operation == "direct":
        # Only previously probed, host-key-verifying site SSH argv is accepted from the service.
        with subprocess.Popen(json.loads(sys.argv[4]), stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            try:
                pack(Path(sys.argv[2]), sys.argv[3], cast(BinaryIO, process.stdin))
            finally:
                process.stdin.close()
            if process.wait():
                raise RuntimeError("Direct Dataset transfer receiver failed.")
    else:
        raise ValueError("Unknown Dataset transfer endpoint.")


if __name__ == "__main__":
    main()
