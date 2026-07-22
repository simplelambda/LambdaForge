"""Memory-safe LRU wrapper for map-style PyTorch datasets."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence, Set, Sized
from dataclasses import fields, is_dataclass
from typing import Any, cast

import torch
from torch.utils.data import Dataset, get_worker_info

from lambdaforge.data.cache.CacheBackend import CacheBackend
from lambdaforge.data.cache.CacheStats import CacheStats
from lambdaforge.data.cache.DatasetFingerprint import DatasetFingerprint
from lambdaforge.data.cache.DatasetSerializer import DatasetSerializer
from lambdaforge.data.cache.PickleDatasetSerializer import PickleDatasetSerializer


class DatasetCache(Dataset[Any]):
    """Cache map-style dataset samples with explicit per-process budgets.

    The RAM limit counts immutable serialized payload bytes exactly; it does
    not claim to cap transient deserialization memory, the wrapped dataset, or
    Python allocator overhead. RAM caching is disabled inside DataLoader
    workers by default because every worker owns a separate dataset replica.
    A bounded disk backend may still be shared by those workers.
    """

    KEY_VERSION = "lambdaforge-dataset-cache-v1"
    FINGERPRINT_KEY_VERSION = "lambdaforge-dataset-cache-v2"
    _DESERIALIZATION_FAILED = object()

    def __init__(
        self,
        dataset: Dataset[Any],
        max_memory_bytes_per_process: int,
        max_memory_entries: int = 10_000,
        backend: CacheBackend | None = None,
        serializer: DatasetSerializer | None = None,
        key_fn: Callable[[Any], Hashable] | None = None,
        cache_in_workers: bool = False,
        strict: bool = False,
        fingerprint: DatasetFingerprint | None = None,
    ) -> None:
        if (
            not isinstance(max_memory_bytes_per_process, int)
            or isinstance(max_memory_bytes_per_process, bool)
            or max_memory_bytes_per_process < 0
        ):
            raise ValueError("max_memory_bytes_per_process must be a non-negative integer.")
        if (
            not isinstance(max_memory_entries, int)
            or isinstance(max_memory_entries, bool)
            or max_memory_entries < 1
        ):
            raise ValueError("max_memory_entries must be a positive integer.")
        if not isinstance(cache_in_workers, bool):
            raise TypeError("cache_in_workers must be a bool.")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool.")
        if fingerprint is not None and not isinstance(fingerprint, DatasetFingerprint):
            raise TypeError("fingerprint must be a DatasetFingerprint or None.")
        self.dataset = dataset
        self.max_memory_bytes_per_process = int(max_memory_bytes_per_process)
        self.max_memory_entries = int(max_memory_entries)
        self.backend = backend
        self.serializer = serializer or PickleDatasetSerializer()
        self.key_fn = key_fn
        self.cache_in_workers = cache_in_workers
        self.strict = strict
        self.fingerprint = fingerprint
        self._memory: OrderedDict[str, bytes] = OrderedDict()
        self._memory_bytes = 0
        self._memory_hits = 0
        self._backend_hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0
        self._skipped_oversize = 0
        self._serialization_errors = 0
        self._backend_errors = 0
        self._process_id = os.getpid()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Return the wrapped map-style dataset length."""
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: Any) -> Any:
        """Return an isolated cached sample or fetch and cache a miss."""
        self._ensure_process_local_state()
        memory_enabled = self._memory_enabled()
        if not memory_enabled and self.backend is None:
            return self.dataset[index]
        digest = self._digest(index)
        if digest is None:
            return self.dataset[index]

        if memory_enabled:
            payload = self._memory_payload(digest)
            if payload is not None:
                value = self._deserialize(payload, digest)
                if value is not self._DESERIALIZATION_FAILED:
                    with self._lock:
                        self._memory_hits += 1
                    return value

        if self.backend is not None:
            record = self._read_backend(digest)
            if record is not None:
                try:
                    with record:
                        value = self._deserialize(record.payload, digest)
                        if value is not self._DESERIALIZATION_FAILED:
                            if memory_enabled:
                                self._remember(digest, bytes(record.payload))
                            with self._lock:
                                self._backend_hits += 1
                            return value
                except RuntimeError:
                    self._remove_backend(digest, record.token)
                    raise
                self._remove_backend(digest, record.token)

        with self._lock:
            self._misses += 1
        value = self.dataset[index]
        self._store_value(digest, value, memory_enabled)
        return value

    def stats(self) -> CacheStats:
        """Return an immutable process-local cache statistics snapshot."""
        self._ensure_process_local_state()
        with self._lock:
            backend_entries, backend_bytes = self._backend_usage()
            return CacheStats(
                memory_hits=self._memory_hits,
                backend_hits=self._backend_hits,
                misses=self._misses,
                writes=self._writes,
                evictions=self._evictions,
                skipped_oversize=self._skipped_oversize,
                serialization_errors=self._serialization_errors,
                backend_errors=self._backend_errors,
                memory_entries=len(self._memory),
                memory_bytes=self._memory_bytes,
                max_memory_bytes_per_process=self.max_memory_bytes_per_process,
                max_memory_entries=self.max_memory_entries,
                backend_entries=backend_entries,
                backend_bytes=backend_bytes,
                process_id=self._process_id,
            )

    def clear(self, include_backend: bool = False) -> None:
        """Clear process-local RAM and optionally the shared disk namespace."""
        self._ensure_process_local_state()
        with self._lock:
            self._memory.clear()
            self._memory_bytes = 0
        if include_backend and self.backend is not None:
            try:
                self.backend.clear()
            except Exception as error:
                self._handle_backend_error(error)

    def invalidate(self, index: Any, include_backend: bool = True) -> None:
        """Remove one indexed sample from RAM and, by default, the disk backend."""
        self._ensure_process_local_state()
        digest = self._digest(index)
        if digest is None:
            return
        with self._lock:
            payload = self._memory.pop(digest, None)
            if payload is not None:
                self._memory_bytes -= len(payload)
        if include_backend:
            self._remove_backend(digest)

    def __getstate__(self) -> dict[str, Any]:
        """Avoid copying cached RAM and locks into spawned DataLoader workers."""
        state = dict(self.__dict__)
        state["_memory"] = OrderedDict()
        state["_memory_bytes"] = 0
        state["_memory_hits"] = 0
        state["_backend_hits"] = 0
        state["_misses"] = 0
        state["_writes"] = 0
        state["_evictions"] = 0
        state["_skipped_oversize"] = 0
        state["_serialization_errors"] = 0
        state["_backend_errors"] = 0
        state["_process_id"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a process-local lock after DataLoader spawn."""
        self.__dict__.update(state)
        self._process_id = os.getpid()
        self._lock = threading.RLock()

    def _digest(self, index: Any) -> str | None:
        key = self.key_fn(index) if self.key_fn is not None else index
        material: tuple[Any, ...]
        if self.fingerprint is None:
            material = (self.KEY_VERSION, key)
        else:
            material = (
                self.FINGERPRINT_KEY_VERSION,
                self.fingerprint.digest,
                self.serializer.format_fingerprint,
                key,
            )
        try:
            payload = self.serializer.dumps(material)
        except Exception as error:
            self._handle_serialization_error(error)
            return None
        return hashlib.sha256(payload).hexdigest()

    def _memory_enabled(self) -> bool:
        return self.max_memory_bytes_per_process > 0 and (
            self.cache_in_workers or get_worker_info() is None
        )

    def _memory_payload(self, digest: str) -> bytes | None:
        with self._lock:
            payload = self._memory.pop(digest, None)
            if payload is not None:
                self._memory[digest] = payload
            return payload

    def _remember(self, digest: str, payload: bytes) -> bool:
        if len(payload) > self.max_memory_bytes_per_process:
            return False
        with self._lock:
            previous = self._memory.pop(digest, None)
            if previous is not None:
                self._memory_bytes -= len(previous)
            while self._memory and (
                self._memory_bytes + len(payload) > self.max_memory_bytes_per_process
                or len(self._memory) >= self.max_memory_entries
            ):
                _, evicted = self._memory.popitem(last=False)
                self._memory_bytes -= len(evicted)
                self._evictions += 1
            self._memory[digest] = payload
            self._memory_bytes += len(payload)
            return True

    def _store_value(self, digest: str, value: Any, memory_enabled: bool) -> None:
        if self._contains_cuda_tensor(value):
            error = ValueError("DatasetCache does not cache CUDA tensors; return CPU samples.")
            self._handle_serialization_error(error)
            return
        try:
            payload = self.serializer.dumps(value)
        except Exception as error:
            self._handle_serialization_error(error)
            return

        stored = False
        if self.backend is not None:
            try:
                stored = self.backend.write(digest, payload) or stored
            except Exception as error:
                self._handle_backend_error(error)
        if memory_enabled:
            stored = self._remember(digest, payload) or stored
        if stored:
            with self._lock:
                self._writes += 1
        elif self.backend is not None or memory_enabled:
            with self._lock:
                self._skipped_oversize += 1

    def _deserialize(self, payload: Any, digest: str) -> Any:
        try:
            return self.serializer.loads(payload)
        except Exception as error:
            with self._lock:
                stale = self._memory.pop(digest, None)
                if stale is not None:
                    self._memory_bytes -= len(stale)
            self._handle_serialization_error(error)
            return self._DESERIALIZATION_FAILED

    def _handle_serialization_error(self, error: Exception) -> None:
        with self._lock:
            self._serialization_errors += 1
        if self.strict:
            raise RuntimeError("Dataset cache serialization failed.") from error

    def _ensure_process_local_state(self) -> None:
        """Discard memory inherited through ``fork`` before it can be reused."""
        process_id = os.getpid()
        if process_id == self._process_id:
            return
        self._lock = threading.RLock()
        self._memory = OrderedDict()
        self._memory_bytes = 0
        self._memory_hits = 0
        self._backend_hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0
        self._skipped_oversize = 0
        self._serialization_errors = 0
        self._backend_errors = 0
        self._process_id = process_id

    def _read_backend(self, digest: str) -> Any:
        try:
            return self.backend.read(digest) if self.backend is not None else None
        except Exception as error:
            self._handle_backend_error(error)
            return None

    def _remove_backend(self, digest: str, token: str | None = None) -> None:
        if self.backend is None:
            return
        try:
            self.backend.remove_if_unchanged(digest, token)
        except Exception as error:
            self._handle_backend_error(error)

    def _backend_usage(self) -> tuple[int, int]:
        if self.backend is None:
            return 0, 0
        try:
            usage = self.backend.usage()
            return usage.entries, usage.bytes
        except Exception as error:
            self._handle_backend_error(error)
            return 0, 0

    def _handle_backend_error(self, error: Exception) -> None:
        with self._lock:
            self._backend_errors += 1
        if self.strict:
            raise RuntimeError("Dataset cache backend operation failed.") from error

    @classmethod
    def _contains_cuda_tensor(cls, value: Any, seen: set[int] | None = None) -> bool:
        """Detect CUDA tensors in nested containers without following cycles."""
        if torch.is_tensor(value):
            return bool(value.is_cuda)
        seen = seen if seen is not None else set()
        identity = id(value)
        if identity in seen:
            return False
        if isinstance(value, Mapping):
            seen.add(identity)
            return any(
                cls._contains_cuda_tensor(item, seen) for pair in value.items() for item in pair
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seen.add(identity)
            return any(cls._contains_cuda_tensor(item, seen) for item in value)
        if isinstance(value, Set):
            seen.add(identity)
            return any(cls._contains_cuda_tensor(item, seen) for item in value)
        if is_dataclass(value) and not isinstance(value, type):
            seen.add(identity)
            return any(
                cls._contains_cuda_tensor(getattr(value, field.name), seen)
                for field in fields(value)
            )
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            seen.add(identity)
            return any(cls._contains_cuda_tensor(item, seen) for item in attributes.values())
        return False
