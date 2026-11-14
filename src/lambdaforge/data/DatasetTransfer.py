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
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast
from uuid import uuid4


def read_password() -> str:
    """Read a bounded secret from an encrypted channel or anonymous native Job pipe."""
    raw = sys.stdin.buffer.read(4097)
    if not raw or len(raw) > 4096:
        raise ValueError("A nonempty transient password of at most 4096 bytes is required.")
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        raise ValueError("Transient password encoding is invalid.") from None


def password_transport(peer: dict[str, Any], password: str) -> Any:
    """Native optional password transport; trust belongs to the initiating host."""
    from lambdaforge.controlplane.PasswordSshTransport import PasswordSshTransport

    return PasswordSshTransport(
        peer["host"],
        user=peer.get("user"),
        port=int(peer["port"]),
        password_provider=lambda: password,
        timeout=10,
    )


@contextmanager
def peer_stream(command: list[str], peer: dict[str, Any] | None, password: str | None) -> Any:
    """Use native binary streams and the same archive engine with either authentication."""
    from lambdaforge.controlplane.BinaryCommand import subprocess_stream

    if password is None:
        with subprocess_stream(command) as stream:
            yield stream
        return
    if not peer:
        raise ValueError("Password SSH requires the explicitly approved peer.")
    transport = password_transport(peer, password)
    try:
        with transport.stream(command) as stream:
            yield stream
    finally:
        transport.close()


def probe(options: dict[str, Any]) -> None:
    """Check effective prerequisites without modifying files or opening Dataset asset bytes."""
    from lambdaforge.controlplane.StorageAdmission import StorageAdmission  # noqa: F401
    from lambdaforge.data.DatasetRecord import DatasetRecord
    from lambdaforge.data.DatasetRegistry import DatasetRegistry  # noqa: F401
    from lambdaforge.work.snapshot import validate_path

    if options.get("password_support"):
        import importlib.util
        import inspect

        from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor

        if importlib.util.find_spec("paramiko") is None:
            raise ValueError(
                "Initiating runtime needs optional lambdaforge[cluster-password]; no password sent."
            )
        if "stdin_data" not in inspect.signature(ProcessSupervisor.launch).parameters:
            raise ValueError(
                "Update/bootstrap the initiating LF runtime for ephemeral Job input "
                "before password transfers."
            )

    role = options.get("role")
    raw = options.get("root")
    if raw:
        root = Path(raw)
        validate_path(root)
        if role == "source":
            if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
                raise PermissionError("Source Dataset directory is not readable.")
            manifest = root / "dataset-artifact.json"
            if manifest.is_symlink() or not manifest.is_file() or not os.access(manifest, os.R_OK):
                raise PermissionError("Source Dataset declaration is not readable.")
        else:
            record = options.get("record")
            if record:
                value = DatasetRecord.from_mapping(record)
                candidate = (
                    root
                    / value.name
                    / value.version
                    / value.dataset_id.removeprefix("sha256:")[:16]
                )
                validate_path(candidate)
                if not candidate.is_relative_to(root):
                    raise ValueError("Unsafe destination publication path.")
            parent = root
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                raise PermissionError("Destination storage is not writable.")
    print("LF_DATA_TRANSFER_READY", flush=True)


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


def receive(options: dict[str, Any], source: BinaryIO, progress: Any = None) -> dict[str, Any]:
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
            if progress:
                progress("verify-destination", "Verifying exact destination checksums.")
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
                    if progress:
                        progress("publication", "Publishing verified immutable Dataset content.")
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


def pull_copy(
    options: dict[str, Any],
    sender_command: list[str],
    progress: Any = None,
    *,
    peer: dict[str, Any] | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Receive on the destination host using the same archive verification/publication path."""
    import time

    with peer_stream(sender_command, peer, password) as sender:

        class SourceStream:
            def __init__(self) -> None:
                self.count = 0
                self.last_update = 0.0

            def readline(self, size: int = -1) -> bytes:
                header = sender.stdout.readline(size)
                if progress:
                    progress("transfer", "Receiving compressed bytes directly from source.")
                return header

            def read(self, size: int = -1) -> bytes:
                chunk = sender.stdout.read(size)
                self.count += len(chunk)
                now = time.monotonic()
                if chunk and progress and now - self.last_update >= 1.0:
                    progress(
                        "transfer",
                        "Receiving compressed bytes directly from source.",
                        compressed_bytes=self.count,
                    )
                    self.last_update = now
                return chunk

        def boundary(phase: str, message: str) -> None:
            if phase == "verify-destination":
                sender.wait()  # Sender failure must precede any final publication.
            if progress:
                progress(phase, message)

        try:
            return receive(options, cast(BinaryIO, SourceStream()), boundary)
        except (EOFError, OSError, tarfile.TarError, ValueError) as error:
            diagnostic = getattr(sender, "diagnostics", None)
            peer = (
                diagnostic()
                if callable(diagnostic)
                else {"exit_code": None, "stderr": "Peer diagnostic unavailable in host runtime."}
            )
            raise RuntimeError(
                f"Direct sender receive failed: {type(error).__name__}: {error}; peer={peer}"
            ) from error


def push_copy(
    root: Path,
    identity: str,
    receiver_command: list[str],
    progress: Any = None,
    *,
    peer: dict[str, Any] | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Direct sender with bounded peer diagnostics and the same exact archive engine."""
    import time

    with peer_stream(receiver_command, peer, password) as receiver:

        class ProgressStream:
            def __init__(self) -> None:
                self.count = 0
                self.last_update = 0.0
                self.header_written = False

            def write(self, data: bytes) -> int:
                count = receiver.stdin.write(data)
                if not self.header_written:
                    self.header_written = True
                    return len(data) if count is None else count
                self.count += len(data) if count is None else count
                now = time.monotonic()
                if progress and now - self.last_update >= 1.0:
                    progress(
                        "transfer",
                        "Streaming compressed Dataset bytes between clusters.",
                        compressed_bytes=self.count,
                    )
                    self.last_update = now
                return len(data) if count is None else count

            def flush(self) -> None:
                receiver.stdin.flush()

        try:
            pack(root, identity, cast(BinaryIO, ProgressStream()))
            receiver.finish_input()
        except (EOFError, OSError) as error:
            diagnostic = getattr(receiver, "diagnostics", None)
            peer = (
                diagnostic()
                if callable(diagnostic)
                else {
                    "exit_code": None,
                    "stderr": "Peer diagnostic unavailable in this host runtime; inspect Job logs.",
                }
            )
            raise RuntimeError(
                f"Direct receiver stream failed: {type(error).__name__}; peer={peer}"
            ) from error
        if progress:
            progress("verify-destination", "Verifying and registering the remote destination.")
        # A bounded control response follows the binary input: destination verification and
        # publication are real receiver observations, not phases guessed by the sender.
        for _ in range(8):
            raw = receiver.stdout.readline(65537)
            if len(raw) > 65536:
                raise ValueError("Dataset receiver acknowledgement exceeds its metadata bound.")
            response = json.loads(raw)
            phase = response.get("transfer_phase")
            if phase in {"verify-destination", "publication"}:
                if progress:
                    progress(phase, str(response.get("message", ""))[:1024])
            else:
                break
        else:
            raise ValueError("Dataset receiver phase responses exceed their metadata bound.")
        receiver.wait()
        if not response.get("verified"):
            raise ValueError("Destination did not acknowledge verified Dataset content.")
        return response


def direct_job(path: Path) -> None:
    """Native supervised host-to-host copy; an explicitly supplied password is memory-only."""
    from lambdaforge.work.atomic import atomic_write_json
    from lambdaforge.work.snapshot import validate_path

    validate_path(path)
    request = json.loads(path.read_text(encoding="utf-8"))
    password = read_password() if request.get("peer_authentication") == "password" else None
    progress_path = Path(os.environ["LAMBDAFORGE_PROGRESS_PATH"])

    def progress(phase: str, message: str, **values: Any) -> None:
        from datetime import datetime, timezone

        old = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        now = datetime.now(timezone.utc).isoformat()
        advanced = values.get("compressed_bytes", 0) > old.get("compressed_bytes", 0)
        atomic_write_json(
            progress_path,
            {
                **old,
                "phase": phase,
                "message": message,
                "updated_at_utc": now,
                **values,
                "last_byte_progress_at_utc": now
                if advanced
                else old.get("last_byte_progress_at_utc"),
            },
        )

    if request.get("direction") == "pull":
        progress("verify-source", "Source host is verifying exact checksums before sending.")
        response = pull_copy(request["receive_options"], request["sender"], progress)
        response.update(
            {
                "dataset": request["dataset"],
                "content_id": request["content_id"],
                "destination": request["destination"],
            }
        )
        atomic_write_json(Path(os.environ["LAMBDAFORGE_JOB_RESULT_PATH"]), response)
        progress("completed", "Dataset verified and registered on the destination.")
        return

    progress("verify-source", "Verifying exact source checksums.")
    response = push_copy(
        Path(request["source_root"]),
        request["content_id"],
        request["receiver"],
        progress,
        peer=request.get("peer"),
        password=password,
    )
    result = {
        "dataset": request["dataset"],
        "content_id": request["content_id"],
        "destination": request["destination"],
        **response,
    }
    atomic_write_json(Path(os.environ["LAMBDAFORGE_JOB_RESULT_PATH"]), result)
    progress("completed", "Dataset verified and registered on the destination.")
    print("Verified Dataset replication completed.", flush=True)


def main() -> None:
    """Private wire endpoint used by DatasetService, not a second public command."""
    operation = sys.argv[1]
    if operation == "direct-job":
        direct_job(Path(sys.argv[2]))
    elif operation == "password-probe":
        options = json.loads(sys.argv[2])
        transport = password_transport(options["peer"], read_password())
        try:
            result = transport.run(options["command"], timeout=15)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            if result.returncode:
                raise SystemExit(result.returncode)
        finally:
            transport.close()
    elif operation == "probe":
        probe(json.loads(sys.argv[2]))
    elif operation == "pack":
        pack(Path(sys.argv[2]), sys.argv[3], sys.stdout.buffer)
    elif operation == "receive":
        options = json.loads(sys.argv[2])

        def report(phase: str, message: str) -> None:
            print(json.dumps({"transfer_phase": phase, "message": message}), flush=True)

        print(
            json.dumps(
                receive(options, sys.stdin.buffer, report if options.get("report_phases") else None)
            )
        )
    elif operation == "pull":
        print(json.dumps(pull_copy(json.loads(sys.argv[2]), json.loads(sys.argv[3]))))
    elif operation == "direct":
        print(json.dumps(push_copy(Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4]))))
    else:
        raise ValueError("Unknown Dataset transfer endpoint.")


if __name__ == "__main__":
    main()
