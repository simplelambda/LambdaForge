"""Real HTTP regressions for atomic, retrying Work cache downloads."""

from __future__ import annotations

import gzip
import socket
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.cache import RateLimit, WorkCache


class _DownloadServer(ThreadingHTTPServer):
    truncate_always: bool
    counts: defaultdict[str, int]
    counts_lock: threading.Lock


class _ChunkedGzipHandler(BaseHTTPRequestHandler):
    server: _DownloadServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback contract
        with self.server.counts_lock:
            self.server.counts[self.path] += 1
            attempt = self.server.counts[self.path]
        if self.path == "/missing.gz":
            self.send_error(404)
            return
        payload = gzip.compress(f"payload:{self.path}".encode())
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if self.server.truncate_always or attempt == 1:
            # Advertise one complete chunk, send only part of it and close. urllib then
            # raises http.client.IncompleteRead while gzip is consuming the response.
            partial = payload[: max(1, len(payload) // 2)]
            self.wfile.write(f"{len(payload):X}\r\n".encode() + partial)
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextmanager
def _server(*, truncate_always: bool = False) -> Iterator[_DownloadServer]:
    server = _DownloadServer(("127.0.0.1", 0), _ChunkedGzipHandler)
    server.truncate_always = truncate_always
    server.counts = defaultdict(int)
    server.counts_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _url(server: _DownloadServer, path: str) -> str:
    return f"http://127.0.0.1:{server.server_port}/{path}"


class _CountingRateLimit(RateLimit):
    def __init__(self) -> None:
        super().__init__(1_000_000)
        self.calls = 0

    def acquire(self) -> None:
        self.calls += 1


def test_incomplete_chunked_gzip_is_retried_and_atomically_published(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")
    limiter = _CountingRateLimit()
    with _server() as server:
        result = cache.fetch(
            _url(server, "evidence.gz"),
            key="downloads/evidence.txt",
            retries=2,
            retry_backoff=0,
            decompress="gzip",
            rate_limit=limiter,
        )

    assert result.read_bytes() == b"payload:/evidence.gz"
    assert server.counts["/evidence.gz"] == 2
    assert limiter.calls == 2
    assert not tuple((tmp_path / "cache").rglob("*.tmp*"))


def test_exhausted_incomplete_download_leaves_no_cache_entry_or_temporary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    cache = WorkCache(root)
    with _server(truncate_always=True) as server:
        with pytest.raises(
            RuntimeError,
            match=r"after 3 attempt\(s\); final cause: IncompleteRead",
        ):
            cache.fetch(
                _url(server, "broken.gz"),
                key="downloads/broken.txt",
                retries=2,
                retry_backoff=0,
                decompress="gzip",
            )

    assert server.counts["/broken.gz"] == 3
    assert cache._store.restore("downloads/broken.txt") is None
    assert not tuple((root / ".managed" / "records").glob("*.json"))
    assert not tuple(root.rglob("*.tmp*"))


def test_permanent_http_error_is_not_retried(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")
    with _server() as server:
        with pytest.raises(RuntimeError, match=r"after 1 attempt\(s\).*HTTPError: HTTP Error 404"):
            cache.fetch(
                _url(server, "missing.gz"),
                key="downloads/missing.txt",
                retries=5,
                retry_backoff=0,
                decompress="gzip",
            )

    assert server.counts["/missing.gz"] == 1


def test_incomplete_fetches_complete_inside_concurrent_resume_map(tmp_path: Path) -> None:
    with _server() as server:
        config = WorkConfig.from_mapping(
            {
                "name": "concurrent-downloads",
                "run": "tests.work_cases.FetchMapWork",
                "with": {
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                    "count": 4,
                },
            },
            source=tmp_path / "work.yaml",
        )
        result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert result.runs[0].primary_result == {
        "items": [{"id": str(index), "value": f"payload:/{index}.gz"} for index in range(4)]
    }
    assert set(server.counts.values()) == {2}
