"""Authenticated, quota-coordinated filesystem cache backend."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from lambdaforge.data.cache.CacheBackend import CacheBackend
from lambdaforge.data.cache.CacheFileLock import CacheFileLock
from lambdaforge.data.cache.CacheIntegrityError import CacheIntegrityError
from lambdaforge.data.cache.CacheNamespaceManifest import CacheNamespaceManifest
from lambdaforge.data.cache.CacheRecord import CacheRecord
from lambdaforge.data.cache.CacheRecordCodec import CacheRecordCodec
from lambdaforge.data.cache.CacheUsage import CacheUsage


class DiskCacheBackend(CacheBackend):
    """Store verified records under one transactional local-filesystem quota.

    Every cooperating process uses the same OS file lock and immutable
    namespace manifest. Writers pre-evict before atomic replacement, so the
    completed `.lfcache` files satisfy both quotas even if a process exits
    immediately after `os.replace`.
    """

    FILE_SUFFIX = ".lfcache"
    MANIFEST_NAME = ".namespace.json"
    LOCK_NAME = ".quota.lock"
    MANIFEST_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        max_bytes: int,
        max_entries: int = 100_000,
        record_codec: CacheRecordCodec | None = None,
        lock_timeout_seconds: float = 60.0,
        lock_poll_interval_seconds: float = 0.01,
        remove_invalid_records: bool = True,
    ) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty dataset/version identifier.")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer.")
        if (
            isinstance(lock_timeout_seconds, bool)
            or not isinstance(lock_timeout_seconds, (int, float))
            or lock_timeout_seconds <= 0
        ):
            raise ValueError("lock_timeout_seconds must be a positive number.")
        if (
            isinstance(lock_poll_interval_seconds, bool)
            or not isinstance(lock_poll_interval_seconds, (int, float))
            or lock_poll_interval_seconds <= 0
        ):
            raise ValueError("lock_poll_interval_seconds must be a positive number.")
        if not isinstance(remove_invalid_records, bool):
            raise TypeError("remove_invalid_records must be a bool.")
        namespace_id = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:20]
        self.root = Path(root)
        self.namespace = namespace
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.record_codec = record_codec or CacheRecordCodec()
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_interval_seconds = float(lock_poll_interval_seconds)
        self.remove_invalid_records = remove_invalid_records
        self.directory = self.root / namespace_id
        self.manifest_path = self.directory / self.MANIFEST_NAME
        self.lock_path = self.directory / self.LOCK_NAME
        self._process_id = os.getpid()
        self._lock = threading.RLock()
        self.recover()

    def read(self, key: str) -> CacheRecord | None:
        """Verify one record under a shared lease and refresh its LRU time."""
        self._ensure_process_local_state()
        path = self._path_for(key)
        invalid_token: str | None = None
        try:
            with self._lock, self._file_lock(shared=True):
                try:
                    encoded = path.read_bytes()
                except FileNotFoundError:
                    return None
                invalid_token = hashlib.sha256(encoded).hexdigest()
                payload = self.record_codec.decode(
                    encoded,
                    associated_data=self._associated_data(key),
                )
                try:
                    os.utime(path, None)
                except OSError:
                    pass
                return CacheRecord(payload, token=invalid_token)
        except CacheIntegrityError:
            if self.remove_invalid_records:
                self.remove_if_unchanged(key, invalid_token)
            raise

    def write(self, key: str, payload: bytes) -> bool:
        """Reserve quota, then atomically publish one complete verified record."""
        self._ensure_process_local_state()
        self._path_for(key)
        encoded = self.record_codec.encode(
            payload,
            associated_data=self._associated_data(key),
        )
        if len(encoded) > self.max_bytes:
            return False
        with self._lock, self._file_lock(shared=False):
            self._prepare_unlocked()
            target = self._path_for(key)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.directory,
                    prefix=f".{key}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary_path = Path(handle.name)
                if not self._reserve_unlocked(target, len(encoded)):
                    return False
                os.replace(temporary_path, target)
                temporary_path = None
                self._fsync_directory()
                usage = self._usage_unlocked()
                if usage.bytes > self.max_bytes or usage.entries > self.max_entries:
                    target.unlink(missing_ok=True)
                    raise RuntimeError("Persistent cache quota postcondition was violated.")
                return True
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def remove(self, key: str) -> None:
        """Remove one namespaced generation under the exclusive quota lock."""
        self._ensure_process_local_state()
        with self._lock, self._file_lock(shared=False):
            self._prepare_unlocked()
            self._path_for(key).unlink(missing_ok=True)

    def remove_if_unchanged(self, key: str, token: str | None) -> bool:
        """Remove only the generation that a caller actually inspected."""
        self._ensure_process_local_state()
        if token is None:
            self.remove(key)
            return True
        with self._lock, self._file_lock(shared=False):
            self._prepare_unlocked()
            path = self._path_for(key)
            try:
                current = path.read_bytes()
            except FileNotFoundError:
                return False
            if hashlib.sha256(current).hexdigest() != token:
                return False
            path.unlink(missing_ok=True)
            return True

    def clear(self) -> None:
        """Remove records and orphan temporaries while preserving coordination metadata."""
        self._ensure_process_local_state()
        with self._lock, self._file_lock(shared=False):
            self._cleanup_temporaries_unlocked()
            self._ensure_manifest_unlocked()
            for path in self._records():
                path.unlink(missing_ok=True)

    def recover(self) -> CacheUsage:
        """Reconcile an interrupted namespace and return its valid bounded usage."""
        self._ensure_process_local_state()
        with self._lock, self._file_lock(shared=False):
            self._prepare_unlocked()
            return self._usage_unlocked()

    def usage(self) -> CacheUsage:
        """Return one coherent usage snapshot after crash reconciliation."""
        return self.recover()

    @property
    def current_bytes(self) -> int:
        """Return bytes occupied by complete envelope plus payload records."""
        return self.usage().bytes

    @property
    def entry_count(self) -> int:
        """Return complete record count from the coordinated namespace."""
        return self.usage().entries

    def __getstate__(self) -> dict[str, Any]:
        """Drop the non-pickleable thread lock for DataLoader spawn."""
        state = dict(self.__dict__)
        state["_process_id"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore process-local synchronization after DataLoader spawn."""
        self.__dict__.update(state)
        self._process_id = os.getpid()
        self._lock = threading.RLock()

    def _path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache backend keys must be lowercase SHA-256 hex digests.")
        return self.directory / f"{key}{self.FILE_SUFFIX}"

    def _records(self) -> list[Path]:
        if not self.directory.exists():
            return []
        records: list[Path] = []
        for path in self.directory.glob(f"*{self.FILE_SUFFIX}"):
            stem = path.name[: -len(self.FILE_SUFFIX)]
            if len(stem) == 64 and all(character in "0123456789abcdef" for character in stem):
                records.append(path)
        return records

    def _file_lock(self, *, shared: bool) -> CacheFileLock:
        return CacheFileLock(
            self.lock_path,
            shared=shared,
            timeout_seconds=self.lock_timeout_seconds,
            poll_interval_seconds=self.lock_poll_interval_seconds,
        )

    def _expected_manifest(self) -> CacheNamespaceManifest:
        return CacheNamespaceManifest(
            format_version=self.MANIFEST_VERSION,
            namespace=self.namespace,
            max_bytes=self.max_bytes,
            max_entries=self.max_entries,
            record_codec_fingerprint=self.record_codec.format_fingerprint,
        )

    def _prepare_unlocked(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cleanup_temporaries_unlocked()
        self._ensure_manifest_unlocked()
        self._enforce_existing_quota_unlocked()

    def _ensure_manifest_unlocked(self) -> None:
        expected = self._expected_manifest()
        if self.manifest_path.exists():
            expected.assert_compatible(CacheNamespaceManifest.read(self.manifest_path))
            return
        if self._records():
            raise ValueError(
                "Legacy cache records have no namespace manifest. Use a new namespace or "
                "clear the old cache explicitly before adopting the hardened format."
            )
        expected.write_atomic(self.manifest_path)
        self._fsync_directory()

    def _cleanup_temporaries_unlocked(self) -> None:
        if not self.directory.exists():
            return
        for path in self.directory.glob("*.tmp"):
            path.unlink(missing_ok=True)

    def _reserve_unlocked(self, target: Path, encoded_size: int) -> bool:
        records = self._records()
        total = sum(self._safe_size(path) for path in records)
        target_present = target in records
        previous_size = self._safe_size(target) if target_present else 0
        projected_bytes = total - previous_size + encoded_size
        projected_entries = len(records) if target_present else len(records) + 1
        candidates = sorted(
            (path for path in records if path != target),
            key=lambda path: (self._safe_mtime_ns(path), path.name),
        )
        for path in candidates:
            if projected_bytes <= self.max_bytes and projected_entries <= self.max_entries:
                break
            size = self._safe_size(path)
            path.unlink()
            projected_bytes -= size
            projected_entries -= 1
        return projected_bytes <= self.max_bytes and projected_entries <= self.max_entries

    def _enforce_existing_quota_unlocked(self) -> None:
        records = sorted(
            self._records(),
            key=lambda path: (self._safe_mtime_ns(path), path.name),
        )
        total = sum(self._safe_size(path) for path in records)
        remaining = len(records)
        for path in records:
            if total <= self.max_bytes and remaining <= self.max_entries:
                break
            size = self._safe_size(path)
            path.unlink()
            total -= size
            remaining -= 1
        if total > self.max_bytes or remaining > self.max_entries:
            raise RuntimeError("Persistent cache could not reconcile its shared quota.")

    def _usage_unlocked(self) -> CacheUsage:
        records = self._records()
        return CacheUsage(
            entries=len(records),
            bytes=sum(self._safe_size(path) for path in records),
        )

    def _associated_data(self, key: str) -> bytes:
        return (f"lambdaforge-cache-namespace:{self.namespace}\0record-key:{key}").encode()

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _ensure_process_local_state(self) -> None:
        """Replace a thread lock inherited from another process."""
        process_id = os.getpid()
        if process_id != self._process_id:
            self._process_id = process_id
            self._lock = threading.RLock()

    @staticmethod
    def _safe_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @staticmethod
    def _safe_mtime_ns(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return 0
