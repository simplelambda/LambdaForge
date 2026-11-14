"""Synthetic HTTP failures exercise shared finite cooldowns without external requests."""

import io
import multiprocessing
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lambdaforge.work import HttpCooldownPolicy, WorkCache


def _spawn_fetch(root, url, key):
    try:
        WorkCache(Path(root)).fetch(url, key=key, retries=0)
    except RuntimeError:
        pass


def test_retry_after_seconds_date_and_provider_json():
    policy = HttpCooldownPolicy(json_field="retryAfter")
    assert policy.delay({"Retry-After": "60"}, b'{"retryAfter": 90}', now=10) == 90
    assert policy.delay({"Retry-After": formatdate(100, usegmt=True)}, now=40) == 60
    assert policy.delay({"Retry-After": "invalid"}, b"bad", now=40) == 0
    assert HttpCooldownPolicy().delay({}, b'{"retryAfter": 90}') == 0


@pytest.mark.parametrize("status", [429, 503])
def test_cooldown_is_shared_across_cache_instances(tmp_path, monkeypatch, status):
    calls = []
    failed = threading.Event()

    def request(*args, **kwargs):
        calls.append(time.time())
        if len(calls) == 1:
            failed.set()
            raise urllib.error.HTTPError(
                "https://fixture.invalid/source",
                status,
                "wait",
                {"Retry-After": "0.2"},
                io.BytesIO(b"{}"),
            )
        return io.BytesIO(b"exact")

    monkeypatch.setattr(urllib.request, "urlopen", request)
    a, b = WorkCache(tmp_path / "a"), WorkCache(tmp_path / "b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            a.fetch, "https://fixture.invalid/source", key="x", retries=1, retry_backoff=0
        )
        assert failed.wait(1)
        deadline = time.monotonic() + 2
        while not list((tmp_path / ".http-cooldowns").glob("*.json")):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        second = pool.submit(b.fetch, "https://fixture.invalid/other", key="y", retries=0)
        assert first.result().read_bytes() == second.result().read_bytes() == b"exact"
    assert len(calls) == 3
    assert min(calls[1:]) - calls[0] >= 0.19


def test_permanent_failure_is_once_and_shared_cooldown_is_cancellable(tmp_path, monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://fixture.invalid/x", 400, "bad", {}, io.BytesIO())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    cache = WorkCache(tmp_path / "a")
    with pytest.raises(RuntimeError, match="1 attempt"):
        cache.fetch("https://fixture.invalid/x", key="bad", retries=10)
    assert len(calls) == 1

    def throttle(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://fixture.invalid/x", 429, "wait", {"Retry-After": "60"}, io.BytesIO()
        )

    monkeypatch.setattr(urllib.request, "urlopen", throttle)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        cache.fetch(
            "https://fixture.invalid/x",
            key="cancel",
            retries=5,
            cancelled=lambda: time.monotonic() - started > 0.1,
        )
    assert time.monotonic() - started < 1


def test_spawned_clients_share_service_cooldown(tmp_path):
    arrivals = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            arrivals.append(time.monotonic())
            self.send_response(429 if len(arrivals) == 1 else 200)
            if len(arrivals) == 1:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(b"data")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = multiprocessing.get_context("spawn")
    url = f"http://127.0.0.1:{server.server_port}/data"
    first = context.Process(target=_spawn_fetch, args=(str(tmp_path / "a"), url, "a"))
    second = context.Process(target=_spawn_fetch, args=(str(tmp_path / "b"), url, "b"))
    try:
        first.start()
        first.join(10)
        assert first.exitcode == 0
        second.start()
        second.join(10)
        assert second.exitcode == 0
        assert len(arrivals) == 2 and arrivals[1] - arrivals[0] >= 0.95
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join()
        server.shutdown()
        server.server_close()


def test_provider_json_hint_delays_retry_without_header(tmp_path, monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "https://fixture.invalid/json", 429, "wait", {}, io.BytesIO(b'{"retryAfter":0.1}')
            )
        return io.BytesIO(b"exact")

    monkeypatch.setattr(urllib.request, "urlopen", request)
    result = WorkCache(tmp_path / "a").fetch(
        "https://fixture.invalid/json",
        key="x",
        retry_backoff=0,
        cooldown_policy=HttpCooldownPolicy(json_field="retryAfter"),
    )
    assert result.read_bytes() == b"exact" and calls[1] - calls[0] >= 0.095
