"""Safe path-like files owned and integrity-checked by the Work runtime."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from typing import Any

from lambdaforge.runtime import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_build_file, atomic_write_bytes, atomic_write_text
from lambdaforge.work.models import atomic_json

CANONICAL_FINGERPRINT_ALGORITHM = "lambdaforge-content-v2"
LEGACY_FINGERPRINT_ALGORITHM = "lambdaforge-content-v1"


def _canonical_relative_path(root: Path, item: Path) -> tuple[str, tuple[bytes, ...]]:
    relative = item.relative_to(root)
    normalized = tuple(unicodedata.normalize("NFC", part) for part in relative.parts)
    encoded = tuple(part.encode("utf-8", errors="strict") for part in normalized)
    return "/".join(normalized), encoded


def _canonical_entries(
    path: Path,
) -> tuple[tuple[str, str, str, Path, tuple[bytes, ...], tuple[bytes, ...]], ...]:
    """Return a safe tree inventory in locale- and platform-independent order."""
    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
        raise ValueError(f"Managed content must be a regular file or directory: {path}")
    if path.is_file():
        raw = path.name
        return (("file", "", raw, path, (), (raw.encode("utf-8", errors="strict"),)),)
    entries: list[
        tuple[str, str, str, Path, tuple[bytes, ...], tuple[bytes, ...]]
    ] = []
    identities: dict[str, Path] = {}
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Managed content cannot contain symbolic links: {item}")
        mode = item.stat(follow_symlinks=False).st_mode
        if S_ISDIR(mode):
            kind = "directory"
        elif S_ISREG(mode):
            kind = "file"
        else:
            raise ValueError(f"Managed content cannot contain special filesystem entries: {item}")
        relative, key = _canonical_relative_path(path, item)
        raw_relative_path = item.relative_to(path)
        raw_relative = raw_relative_path.as_posix()
        raw_key = tuple(part.encode("utf-8", errors="strict") for part in raw_relative_path.parts)
        previous = identities.setdefault(relative, item)
        if previous != item:
            raise ValueError(
                "Managed content contains paths that collide after canonical Unicode "
                f"normalization: {previous} and {item}"
            )
        entries.append((kind, relative, raw_relative, item, key, raw_key))
    entries.sort(key=lambda entry: entry[4])
    return tuple(entries)


def canonical_fingerprint(path: Path) -> tuple[str, int]:
    """Hash one path using LambdaForge's versioned portable content format.

    The identity is independent of filesystem enumeration, locale, host path
    separators, timestamps and permissions. Logical paths use NFC Unicode and
    length-delimited records, so copied trees have the same identity on
    supported operating systems without ambiguous record boundaries.
    """
    root = Path(path)
    entries = _canonical_entries(root)
    digest = hashlib.sha256(b"LambdaForge content identity\0v2\0")
    digest.update(b"F" if root.is_file() else b"D")
    size = 0
    for kind, relative, _raw_relative, item, _key, _raw_key in entries:
        encoded = relative.encode("utf-8", errors="strict")
        digest.update(b"F" if kind == "file" else b"D")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        if kind == "directory":
            continue
        before = item.stat(follow_symlinks=False)
        digest.update(before.st_size.to_bytes(8, byteorder="big", signed=False))
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = item.stat(follow_symlinks=False)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Managed content changed while it was being fingerprinted: {item}")
    return digest.hexdigest(), size


def owned_path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path below ``root`` without permitting links or traversal."""
    owner = root.resolve()
    raw = Path(value)
    unresolved = raw if raw.is_absolute() else owner / raw
    lexical = Path(os.path.abspath(unresolved))
    if not lexical.is_relative_to(owner):
        raise ValueError(f"Managed path escapes its owning root: {value}")
    cursor = owner
    for part in lexical.relative_to(owner).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Managed paths cannot traverse symbolic links: {value}")
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(owner):
        raise ValueError(f"Managed path escapes its owning root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Managed path does not exist: {resolved}")
    return resolved


def fingerprint(path: Path) -> tuple[str, int]:
    """Return the historical SHA-256 and byte count for compatible persisted data.

    Ordering is explicitly based on UTF-8 path components rather than host
    ``Path`` ordering. This preserves existing POSIX identities while removing
    locale, directory-enumeration and operating-system ordering differences.
    New cross-machine input contracts use :func:`canonical_fingerprint`.
    """
    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
        raise ValueError(f"Managed content must be a regular file or directory: {path}")
    digest = hashlib.sha256()
    size = 0
    entries: tuple[tuple[str, Path], ...]
    if path.is_file():
        entries = ((path.name, path),)
    else:
        entries = tuple(
            (raw_relative, item)
            for kind, _relative, raw_relative, item, _key, _raw_key in sorted(
                _canonical_entries(path), key=lambda entry: entry[5]
            )
            if kind == "file"
        )
    for relative, item in entries:
        if item.is_symlink():
            raise ValueError(f"Managed content cannot contain symbolic links: {item}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class ManagedFile(os.PathLike[str]):
    """A logical, integrity-checked file physically owned by LambdaForge.

    The physical path is an interoperability escape hatch. Persist ``key`` and
    content evidence—not ``path``—when a file must survive process or machine
    boundaries.
    """

    key: str
    _path: Path
    sha256: str
    size_bytes: int
    scope: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Keep process-worker transport logical and free of mutable proxy internals."""
        return (
            type(self),
            (self.key, self._path, self.sha256, self.size_bytes, self.scope, dict(self.metadata)),
        )

    @property
    def path(self) -> Path:
        """Return the framework-owned physical path for library interoperability."""
        return self._path

    @property
    def exists(self) -> bool:
        """Return whether the physical regular file still exists."""
        return self._path.is_file() and not self._path.is_symlink()

    def open(self, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        """Open the managed file in a read-only mode."""
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise ValueError("Managed cache/checkpoint files are read-only after publication.")
        return self._path.open(mode, *args, **kwargs)

    def read_text(self, encoding: str = "utf-8", errors: str | None = None) -> str:
        """Read text without exposing storage layout."""
        return self._path.read_text(encoding=encoding, errors=errors)

    def read_bytes(self) -> bytes:
        """Read bytes without exposing storage layout."""
        return self._path.read_bytes()

    def reference(self) -> dict[str, Any]:
        """Return the safe logical checkpoint representation."""
        return {
            "scope": self.scope,
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class ManagedOutput(os.PathLike[str]):
    """A declared file or directory automatically finalized as a Work artifact."""

    def __init__(self, path: Path, *, kind: str) -> None:
        self._path = path
        self.kind = kind

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __truediv__(self, value: str | Path) -> Path:
        """Return a safe child of a managed directory for ordinary library writes."""
        if self.kind != "directory":
            raise TypeError("Only managed directory outputs have child paths.")
        return owned_path(self._path, value)

    @property
    def path(self) -> Path:
        """Return the attempt-owned physical path for advanced interoperability."""
        return self._path

    @property
    def exists(self) -> bool:
        """Return whether the declared output currently exists."""
        return self._path.exists() and not self._path.is_symlink()

    def write_bytes(self, value: bytes) -> ManagedOutput:
        """Atomically replace a managed file output with bytes."""
        self._require_file()
        atomic_write_bytes(self._path, value)
        return self

    def write_text(self, value: str, *, encoding: str = "utf-8") -> ManagedOutput:
        """Atomically replace a managed file output with text."""
        self._require_file()
        atomic_write_text(self._path, value, encoding=encoding)
        return self

    def write_json(self, value: Any) -> ManagedOutput:
        """Atomically replace a managed file output with strict JSON."""
        self._require_file()
        atomic_json(self._path, value)
        return self

    def build(self, producer: Callable[[Path], Any]) -> ManagedOutput:
        """Build a file at a temporary path and atomically promote it."""
        self._require_file()
        atomic_build_file(self._path, producer)
        return self

    def _require_file(self) -> None:
        if self.kind != "file":
            raise TypeError("Atomic write helpers apply to managed file outputs, not directories.")


class ManagedFileStore:
    """Internal reusable implementation for cache and checkpoint file services."""

    RECORD_VERSION = 1

    def __init__(self, root: Path, *, scope: str, data_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.scope = scope
        self.data_root = (data_root or self.root).resolve()
        self.records_root = self.root / ".managed" / "records"
        self.locks_root = self.root / ".managed" / "locks"
        for directory in (self.root, self.data_root, self.records_root, self.locks_root):
            directory.mkdir(parents=True, exist_ok=True)

    def file(
        self,
        key: str,
        *,
        build: Callable[[Path], Any] | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
    ) -> ManagedFile:
        """Restore or atomically build one key under an exclusive per-key lock."""
        selected = self._key(key)
        restored = self._restore(selected, validate=validate)
        if restored is not None:
            return restored
        if build is None:
            raise FileNotFoundError(f"No valid managed {self.scope} file exists for {selected!r}.")
        lock_key = hashlib.sha256(selected.encode()).hexdigest()
        with CrossProcessFileLock(
            self.locks_root / f"{lock_key}.lock",
            shared=False,
            timeout_seconds=300.0,
            poll_interval_seconds=0.05,
        ):
            restored = self._restore(selected, validate=validate)
            if restored is not None:
                return restored
            destination = owned_path(self.data_root, selected)

            def guarded_build(temporary: Path) -> None:
                build(temporary)
                candidate_path = temporary
                if not candidate_path.is_file() or candidate_path.is_symlink():
                    raise FileNotFoundError(
                        f"Producer for managed {self.scope} file {selected!r} created no safe file."
                    )
                digest, size = fingerprint(candidate_path)
                candidate = ManagedFile(selected, candidate_path, digest, size, self.scope, {})
                if validate is not None and not bool(validate(candidate)):
                    raise ValueError(
                        f"Semantic validation rejected managed {self.scope} file {selected!r}."
                    )

            atomic_build_file(destination, guarded_build)
            digest, size = fingerprint(destination)
            metadata = {
                "managed_file_version": self.RECORD_VERSION,
                "scope": self.scope,
                "key": selected,
                "sha256": digest,
                "size_bytes": size,
            }
            atomic_json(self._record_path(selected), metadata)
            return ManagedFile(selected, destination, digest, size, self.scope, metadata)

    def put_bytes(
        self,
        key: str,
        value: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedFile:
        """Atomically set one key to exact bytes and return its managed representation."""
        selected = self._key(key)
        if not isinstance(value, bytes):
            raise TypeError("Managed file bytes must be a bytes object.")
        annotations = dict(metadata or {})
        reserved = {
            "managed_file_version",
            "scope",
            "key",
            "sha256",
            "size_bytes",
        }
        overlap = reserved.intersection(annotations)
        if overlap:
            raise ValueError(f"Managed metadata uses reserved fields: {sorted(overlap)}.")
        destination = owned_path(self.data_root, selected)
        digest = hashlib.sha256()
        digest.update(destination.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value)
        expected_digest = digest.hexdigest()
        lock_key = hashlib.sha256(selected.encode()).hexdigest()
        with CrossProcessFileLock(
            self.locks_root / f"{lock_key}.lock",
            shared=False,
            timeout_seconds=300.0,
            poll_interval_seconds=0.05,
        ):
            restored = self._restore(selected)
            if (
                restored is not None
                and restored.sha256 == expected_digest
                and all(restored.metadata.get(name) == item for name, item in annotations.items())
            ):
                return restored
            atomic_write_bytes(destination, value)
            actual_digest, size = fingerprint(destination)
            record = {
                "managed_file_version": self.RECORD_VERSION,
                "scope": self.scope,
                "key": selected,
                "sha256": actual_digest,
                "size_bytes": size,
                **annotations,
            }
            atomic_json(self._record_path(selected), record)
            return ManagedFile(selected, destination, actual_digest, size, self.scope, record)

    def restore(
        self,
        key: str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> ManagedFile | None:
        """Restore a key only when current bytes match its record and expected evidence."""
        restored = self._restore(self._key(key))
        if restored is None:
            return None
        if sha256 is not None and restored.sha256 != sha256:
            return None
        if size_bytes is not None and restored.size_bytes != size_bytes:
            return None
        return restored

    def _restore(
        self,
        key: str,
        *,
        validate: Callable[[ManagedFile], bool] | None = None,
    ) -> ManagedFile | None:
        destination = owned_path(self.data_root, key)
        record_path = self._record_path(key)
        if not destination.is_file() or destination.is_symlink() or not record_path.is_file():
            return None
        try:
            metadata = json.loads(record_path.read_text(encoding="utf-8"))
            digest, size = fingerprint(destination)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if (
            metadata.get("managed_file_version") != self.RECORD_VERSION
            or metadata.get("scope") != self.scope
            or metadata.get("key") != key
            or metadata.get("sha256") != digest
            or metadata.get("size_bytes") != size
        ):
            return None
        managed = ManagedFile(key, destination, digest, size, self.scope, metadata)
        if validate is not None and not bool(validate(managed)):
            return None
        return managed

    def _record_path(self, key: str) -> Path:
        return self.records_root / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    @staticmethod
    def _key(value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Managed file keys must be strings.")
        selected = value.strip().replace("\\", "/")
        path = Path(selected)
        if (
            not selected
            or path.is_absolute()
            or ".." in path.parts
            or path == Path(".")
            or path.parts[0] == ".managed"
        ):
            raise ValueError(f"Managed file key must be a non-empty relative path: {value!r}.")
        return path.as_posix()
