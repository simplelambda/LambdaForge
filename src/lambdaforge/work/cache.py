"""High-level reconstructible file caching for Work implementations."""

from __future__ import annotations

import gzip
import json
import math
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from threading import Lock
from typing import Any

from lambdaforge.runtime import CrossProcessFileLock
from lambdaforge.work.managed import ManagedFile, ManagedFileStore, owned_path


class RateLimit:
    """Thread-safe, process-local pacing for requests to one logical service."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(requests_per_second, bool)
            or not isinstance(requests_per_second, int | float)
            or not math.isfinite(float(requests_per_second))
            or requests_per_second <= 0
        ):
            raise ValueError("requests_per_second must be a positive finite number.")
        self.requests_per_second = float(requests_per_second)
        self._interval = 1.0 / self.requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = Lock()

    def acquire(self) -> None:
        """Wait without busy-spinning until this limiter grants one request slot."""
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next - now)
            if wait:
                self._sleep(wait)
                now = self._clock()
            self._next = max(now, self._next) + self._interval


class WorkCache:
    """Identity-scoped reconstructible key/value and managed-file storage."""

    VALUE_VERSION = 1

    def __init__(self, root: Path, *, hold_gc_lease: bool = False) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = ManagedFileStore(
            self._root,
            scope="cache",
            data_root=self._root / "files",
        )
        self._rate_limits: dict[str, RateLimit] = {}
        self._rate_lock = Lock()
        self._dependency_collectors: ContextVar[tuple[list[ManagedFile], ...]] = ContextVar(
            f"lambdaforge_work_cache_dependencies_{id(self)}", default=()
        )
        self._gc_lease: CrossProcessFileLock | None = None
        if hold_gc_lease:
            self._gc_lease = CrossProcessFileLock(
                self._root.parent.parent / ".gc.lock",
                shared=True,
                timeout_seconds=300.0,
                poll_interval_seconds=0.05,
            )
            self._gc_lease.acquire()

    def file(
        self,
        key: str,
        *,
        build: Callable[[Path], Any] | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
    ) -> ManagedFile:
        """Return a valid cached file or atomically construct it once per key."""
        managed = self._store.file(key, build=build, validate=validate)
        self._track(managed)
        return managed

    def put(self, key: str, content: Any) -> None:
        """Store bytes, text or strict JSON content under one logical key.

        Reusing a key replaces its prior value atomically. Cache values are
        reconstructible and remain scoped to this Work's scientific identity.
        """
        encoding, payload = self._encode_value(content)
        managed = self._store.put_bytes(
            key,
            payload,
            metadata={
                "cache_value_version": self.VALUE_VERSION,
                "content_encoding": encoding,
            },
        )
        self._track(managed)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value stored by :meth:`put`, or ``default`` on a cache miss."""
        managed = self._store.restore(key)
        if managed is None:
            return default
        if managed.metadata.get("cache_value_version") != self.VALUE_VERSION:
            raise TypeError(
                f"Cache key {key!r} contains a managed file, not a value stored by cache.put()."
            )
        encoding = managed.metadata.get("content_encoding")
        payload = managed.read_bytes()
        if encoding == "bytes":
            value: Any = payload
        elif encoding == "text":
            value = payload.decode("utf-8")
        elif encoding == "json":
            value = json.loads(payload.decode("utf-8"))
        else:
            raise ValueError(f"Cache key {key!r} has unsupported value encoding {encoding!r}.")
        self._track(managed)
        return value

    def fetch(
        self,
        url: str,
        *,
        key: str,
        retries: int = 3,
        retry_backoff: float = 0.5,
        timeout: float = 30.0,
        decompress: str | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
        rate_limit: RateLimit | None = None,
    ) -> ManagedFile:
        """Download and atomically cache one bounded HTTP(S) resource.

        ``retries`` counts retries after the initial attempt. ``decompress`` may
        be ``"gzip"``; the cached fingerprint describes decompressed bytes.
        """
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer.")
        if (
            isinstance(retry_backoff, bool)
            or not isinstance(retry_backoff, int | float)
            or retry_backoff < 0
        ):
            raise ValueError("retry_backoff must be a non-negative number.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number.")
        if decompress not in {None, "gzip"}:
            raise ValueError("decompress must be null or 'gzip'.")
        scheme = urllib.parse.urlsplit(str(url)).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("cache.fetch accepts only explicit HTTP or HTTPS URLs.")
        if rate_limit is not None and not isinstance(rate_limit, RateLimit):
            raise TypeError("rate_limit must come from Work.cache.rate_limit().")

        def download(target: Path) -> None:
            last_error: BaseException | None = None
            for attempt in range(retries + 1):
                if rate_limit is not None:
                    rate_limit.acquire()
                try:
                    request = urllib.request.Request(
                        str(url),
                        headers={"User-Agent": "LambdaForge scientific cache"},
                    )
                    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                        if decompress == "gzip":
                            with (
                                gzip.GzipFile(fileobj=response) as decoded,
                                target.open("wb") as sink,
                            ):
                                shutil.copyfileobj(decoded, sink, length=1024 * 1024)
                        else:
                            with target.open("wb") as sink:
                                shutil.copyfileobj(response, sink, length=1024 * 1024)
                    return
                except (OSError, EOFError, urllib.error.URLError) as error:
                    last_error = error
                    target.unlink(missing_ok=True)
                    if attempt == retries:
                        break
                    time.sleep(float(retry_backoff) * (2**attempt))
            assert last_error is not None
            raise RuntimeError(
                f"Could not fetch {url!r} after {retries + 1} attempt(s): {last_error}"
            ) from last_error

        return self.file(key, build=download, validate=validate)

    def rate_limit(self, name: str, *, requests_per_second: float) -> RateLimit:
        """Return one named limiter shared by threads using this Work cache instance."""
        selected = str(name).strip()
        if not selected:
            raise ValueError("Rate-limit names cannot be empty.")
        with self._rate_lock:
            existing = self._rate_limits.get(selected)
            if existing is not None:
                if existing.requests_per_second != float(requests_per_second):
                    raise ValueError(
                        f"Rate limit {selected!r} already uses "
                        f"{existing.requests_per_second} requests/second."
                    )
                return existing
            limiter = RateLimit(requests_per_second)
            self._rate_limits[selected] = limiter
            return limiter

    def path(self, name: str, *, create: bool = True) -> Path:
        """Return a raw contained cache path for advanced compatibility code."""
        selected = owned_path(self._root, name)
        if create:
            selected.parent.mkdir(parents=True, exist_ok=True)
        return selected

    def restore_reference(
        self,
        key: str,
        *,
        sha256: str,
        size_bytes: int,
    ) -> ManagedFile | None:
        """Restore a logical map dependency only when its bytes still match."""
        return self._store.restore(key, sha256=sha256, size_bytes=size_bytes)

    def _track(self, managed: ManagedFile) -> None:
        for collector in self._dependency_collectors.get():
            collector.append(managed)

    @classmethod
    def _encode_value(cls, content: Any) -> tuple[str, bytes]:
        if isinstance(content, bytes):
            return "bytes", content
        if isinstance(content, str):
            return "text", content.encode("utf-8")
        cls._validate_json_value(content)
        try:
            encoded = json.dumps(
                content,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise TypeError(f"Cache content must be bytes, text or strict JSON: {error}") from error
        return "json", encoded.encode("utf-8")

    @classmethod
    def _validate_json_value(cls, value: Any) -> None:
        if value is None or isinstance(value, str | bool | int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("Cache JSON numbers must be finite.")
            return
        if isinstance(value, list):
            for item in value:
                cls._validate_json_value(item)
            return
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Cache JSON object keys must be strings.")
            for item in value.values():
                cls._validate_json_value(item)
            return
        raise TypeError(
            f"Cache content must be bytes, text or strict JSON, not {type(value).__name__}."
        )

    def close(self) -> None:
        """Release the optional run-long lease that excludes cache collection."""
        if self._gc_lease is not None:
            self._gc_lease.release()
            self._gc_lease = None

    @contextmanager
    def track_dependencies(self) -> Iterator[list[ManagedFile]]:
        """Collect cache files touched by one map callback in its execution context."""
        collector: list[ManagedFile] = []
        token: Token[tuple[list[ManagedFile], ...]] = self._dependency_collectors.set(
            (*self._dependency_collectors.get(), collector)
        )
        try:
            yield collector
        finally:
            self._dependency_collectors.reset(token)


__all__ = ["RateLimit", "WorkCache"]
