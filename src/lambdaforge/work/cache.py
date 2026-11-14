"""High-level reconstructible file caching for Work implementations."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from lambdaforge.runtime import CrossProcessFileLock
from lambdaforge.work.HttpCooldownPolicy import HttpCooldownPolicy
from lambdaforge.work.managed import ManagedFile, ManagedFileStore, owned_path
from lambdaforge.work.models import atomic_json

_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, *range(500, 600)})
_RETRYABLE_HTTP_PROTOCOL_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    http.client.ResponseNotReady,
    http.client.BadStatusLine,
)


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a bearer token to another host, port or transport."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        def origin(url: str) -> tuple[str, str | None, int]:
            try:
                parsed = urllib.parse.urlsplit(url)
                if parsed.username or parsed.password:
                    raise ValueError("Credential-bearing redirect.")
                return parsed.scheme, parsed.hostname, parsed.port or 443
            except ValueError:
                raise ValueError("Malformed authenticated redirect origin.") from None

        if origin(req.full_url) != origin(newurl):
            raise ValueError("Authenticated cross-origin redirect refused.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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

    def __init__(
        self,
        root: Path,
        *,
        hold_gc_lease: bool = False,
        recovery_roots: tuple[Path, ...] = (),
        recovering: bool = False,
        dependency_root: Path | None = None,
    ) -> None:
        self._root = root.resolve()
        self.recovering = recovering
        self._dependency_root = dependency_root
        self._recovery_roots = recovery_roots
        if any(
            p.parent != self._root.parent or p.resolve() != p or p.is_symlink()
            for p in recovery_roots
        ):
            raise ValueError("Recovery cache namespaces must be owned sibling roots.")
        self._gc_lease: CrossProcessFileLock | None = None
        if hold_gc_lease:
            self._gc_lease = CrossProcessFileLock(
                self._root.parent / f".{self._root.name}.lease",
                shared=True,
                timeout_seconds=300.0,
                poll_interval_seconds=0.05,
            )
            self._gc_lease.acquire()
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

    def file(
        self,
        key: str,
        *,
        build: Callable[[Path], Any] | None = None,
        validate: Callable[[ManagedFile], bool] | None = None,
    ) -> ManagedFile:
        """Return a valid cached file or atomically construct it once per key."""
        managed = self._store.file(key, build=build, validate=validate)
        managed = replace(
            managed, metadata={**managed.metadata, "cache_namespace": self._root.name}
        )
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
        max_bytes: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        credential: str | None = None,
        cooldown_policy: HttpCooldownPolicy | None = None,
        cooldown_scope: str | None = None,
    ) -> ManagedFile:
        """Download and atomically cache one bounded HTTP(S) resource.

        ``retries`` counts retries after the initial attempt. ``decompress`` may
        be ``"gzip"``; the cached fingerprint describes decompressed bytes.
        ``max_bytes`` bounds decoded bytes; cancellation is checked between reads/retries.
        ``credential`` is an env:/keyring: bearer reference, resolved only for an HTTPS
        request. Authenticated redirects are restricted to the exact origin. References
        and secret values are never written into the cache record.
        """
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer.")
        if (
            isinstance(retry_backoff, bool)
            or not isinstance(retry_backoff, int | float)
            or not math.isfinite(float(retry_backoff))
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
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max_bytes must be a positive integer.")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be a callable.")
        parsed = urllib.parse.urlsplit(str(url))
        if cooldown_policy is not None and not isinstance(cooldown_policy, HttpCooldownPolicy):
            raise TypeError("cooldown_policy must be an HttpCooldownPolicy.")
        policy = cooldown_policy or HttpCooldownPolicy()
        port = parsed.port or (443 if scheme == "https" else 80)
        origin = f"{parsed.scheme}://{parsed.hostname}:{port}"
        scope = hashlib.sha256((cooldown_scope or origin).encode()).hexdigest()
        cooldown = self._root.parent / ".http-cooldowns" / f"{scope}.json"

        def cooldown_delay(seconds: float = 0.0) -> float:
            with CrossProcessFileLock(
                cooldown.with_suffix(".lock"),
                shared=False,
                timeout_seconds=30,
                poll_interval_seconds=0.01,
            ):
                state = json.loads(cooldown.read_text()) if cooldown.exists() else {}
                deadline = max(
                    float(state.get("until", 0)), time.time() + seconds if seconds else 0
                )
                if seconds:
                    atomic_json(cooldown, {"until": deadline})
                return max(0.0, deadline - time.time())

        if credential is not None and (
            scheme != "https" or parsed.username or parsed.password or parsed.query
        ):
            raise ValueError("Authenticated fetch requires HTTPS without URL credentials/query.")
        if credential is not None and not credential.startswith(("env:", "keyring:")):
            raise ValueError("Credentials require an env: or keyring: reference.")

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise RuntimeError("Cache download cancelled; no partial content published.")

        def pause(seconds: float) -> None:
            if cancelled is None:
                time.sleep(seconds)
                return
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                check_cancelled()
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))

        def copy_response(source: Any, sink: Any) -> None:
            count = 0
            while True:
                check_cancelled()
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if max_bytes is not None and count > max_bytes:
                    raise ValueError("Cache download exceeds max_bytes; no content published.")
                sink.write(chunk)
            check_cancelled()

        def download(target: Path) -> None:
            last_error: BaseException | None = None
            for attempt in range(retries + 1):
                check_cancelled()
                # ``ManagedFileStore`` gives this builder a private unpublished path.
                # Remove any bytes left by the preceding attempt before rate limiting and
                # opening a fresh response.
                target.unlink(missing_ok=True)
                if rate_limit is not None:
                    rate_limit.acquire()
                while remaining := cooldown_delay():
                    pause(min(remaining, 0.1))
                check_cancelled()
                try:
                    headers = {"User-Agent": "LambdaForge scientific cache"}
                    opener: Callable[..., Any] = urllib.request.urlopen
                    if credential is not None:
                        from lambdaforge.controlplane.CredentialService import CredentialService

                        secret = CredentialService().resolve_reference(credential)
                        if not secret or "\r" in secret or "\n" in secret:
                            raise ValueError("Invalid bearer credential value.")
                        headers["Authorization"] = "Bearer " + secret
                        opener = urllib.request.build_opener(_SameOriginRedirect()).open
                    request = urllib.request.Request(str(url), headers=headers)
                    with opener(request, timeout=float(timeout)) as response:
                        if decompress == "gzip":
                            with (
                                gzip.GzipFile(fileobj=response) as decoded,
                                target.open("wb") as sink,
                            ):
                                copy_response(decoded, sink)
                        else:
                            with target.open("wb") as sink:
                                copy_response(response, sink)
                    return
                except urllib.error.HTTPError as error:
                    last_error = error
                    target.unlink(missing_ok=True)
                    if error.code in _RETRYABLE_HTTP_STATUS:
                        body = error.read(65536) if policy.json_field else b""
                        delay = policy.delay(error.headers, body)
                        cooldown_delay(max(delay, float(retry_backoff) * (2**attempt)))
                    error.close()
                    if error.code not in _RETRYABLE_HTTP_STATUS or attempt == retries:
                        break
                except (
                    *_RETRYABLE_HTTP_PROTOCOL_ERRORS,
                    OSError,
                    EOFError,
                    urllib.error.URLError,
                ) as error:
                    last_error = error
                    target.unlink(missing_ok=True)
                    if attempt == retries:
                        break
                    pause(float(retry_backoff) * (2**attempt))
            assert last_error is not None
            if credential is not None:
                # Servers may echo authorization in an error reason. Do not retain that
                # exception as a chained traceback or diagnostic payload.
                raise RuntimeError(
                    f"Authenticated fetch failed after {attempt + 1} attempt(s): "
                    f"{type(last_error).__name__}; check endpoint/access."
                ) from None
            raise RuntimeError(
                f"Could not fetch {url!r} after {attempt + 1} attempt(s); final cause: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error

        check_cancelled()
        receipt = (
            self._dependency_root / "cache-dependencies.json" if self._dependency_root else None
        )
        request_identity = hashlib.sha256(json.dumps([str(url), decompress]).encode()).hexdigest()
        if receipt is not None and receipt.is_file():
            reference = json.loads(receipt.read_text()).get("observations", {}).get(key)
            if reference is not None:
                if reference.get("request_identity", request_identity) != request_identity:
                    raise ValueError(
                        "Observed input key was reused for a different source contract."
                    )
                prior = self.restore_reference(
                    key,
                    sha256=reference["sha256"],
                    size_bytes=reference["size_bytes"],
                    namespace=reference.get("namespace"),
                )
                if prior is None:
                    raise RuntimeError(
                        "Original observed input is unavailable; refusing URL replay. "
                        "Inspect recovery dependencies or start a new acquisition."
                    )
                if validate is not None and not validate(prior):
                    raise ValueError("Original observed input failed application validation.")
                if max_bytes is not None and prior.size_bytes > max_bytes:
                    raise ValueError("Original observed input exceeds max_bytes.")
                prior = replace(prior, metadata={**prior.metadata, "external_observation": True})
                self._track(prior)
                return prior
        result = self.file(key, build=download, validate=validate)
        result = replace(result, metadata={**result.metadata, "external_observation": True})
        self._track(result)
        if self._dependency_root is not None:
            receipt = self._dependency_root / "cache-dependencies.json"
            with CrossProcessFileLock(
                receipt.with_suffix(".lock"),
                shared=False,
                timeout_seconds=30,
                poll_interval_seconds=0.01,
            ):
                document = (
                    json.loads(receipt.read_text()) if receipt.exists() else {"namespaces": []}
                )
                document["namespaces"] = sorted(set(document["namespaces"]) | {self._root.name})
                document.setdefault("observations", {})[key] = {
                    **result.reference(),
                    "request_identity": request_identity,
                }
                atomic_json(receipt, document)
        check_cancelled()
        if max_bytes is not None and result.path.stat().st_size > max_bytes:
            raise ValueError("Cached resource exceeds max_bytes.")
        return result

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
        namespace: str | None = None,
    ) -> ManagedFile | None:
        """Restore a logical map dependency only when its bytes still match."""
        roots = (self._root, *self._recovery_roots)
        for root in roots:
            if namespace is not None and namespace != root.name:
                continue
            store = ManagedFileStore(root, scope="cache", data_root=root / "files")
            managed = store.restore(key, sha256=sha256, size_bytes=size_bytes)
            if managed is not None:
                return replace(managed, metadata={**managed.metadata, "cache_namespace": root.name})
        return None

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


__all__ = ["HttpCooldownPolicy", "RateLimit", "WorkCache"]
