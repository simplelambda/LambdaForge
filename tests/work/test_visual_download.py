"""Bounded native downloads, cancellation and credential containment with disposable files."""

import hashlib
import io
import json
import traceback
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from lambdaforge.etl import Download
from lambdaforge.work.cache import WorkCache, _SameOriginRedirect


def test_download_native_checksum_and_atomic_cache_reuse(tmp_path, monkeypatch):
    payload = b"synthetic rows"
    calls = []

    def open_fixture(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", open_fixture)
    work = SimpleNamespace(cache=WorkCache(tmp_path / "cache"), stop_requested=False)
    operation = Download(max_bytes=100)
    sha = hashlib.sha256(payload).hexdigest()
    first = operation(work, "https://example.invalid/data", key="input", sha256=sha)
    second = operation(work, "https://example.invalid/data", key="input", sha256=sha)
    assert first.read_bytes() == payload and first.sha256 == second.sha256
    assert calls == [1]


def test_limits_and_midstream_cancellation_leave_no_published_entry(tmp_path, monkeypatch):
    work = SimpleNamespace(cache=WorkCache(tmp_path / "cache"), stop_requested=False)

    class CancelResponse(io.BytesIO):
        def read(self, *args):
            work.stop_requested = True
            return super().read(*args)

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: CancelResponse(b"data"))
    with pytest.raises(RuntimeError, match="cancelled"):
        work.cache.fetch(
            "https://example.invalid/data", key="cancel", cancelled=lambda: work.stop_requested
        )
    assert work.cache._store.restore("cancel") is None
    work.stop_requested = False
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"too much"))
    with pytest.raises(ValueError, match="max_bytes"):
        work.cache.fetch("https://example.invalid/data", key="limit", max_bytes=2)
    assert work.cache._store.restore("limit") is None


def test_https_bearer_reference_not_stored_and_not_echoed(tmp_path, monkeypatch):
    secret = "synthetic-credential-unique"
    monkeypatch.setenv("VISUAL_TEST_TOKEN", secret)
    seen = []

    def open_fixture(request, **kwargs):
        seen.append(request.get_header("Authorization"))
        return io.BytesIO(b"rows")

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_fixture)
    )
    cache = WorkCache(tmp_path / "cache")
    cache.fetch("https://example.invalid/data", key="auth", credential="env:VISUAL_TEST_TOKEN")
    assert seen == ["Bearer " + secret]
    assert all(secret not in p.read_text() for p in (tmp_path / "cache").rglob("*.json"))

    def fail(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 403, secret, {}, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: SimpleNamespace(open=fail))
    with pytest.raises(RuntimeError) as caught:
        cache.fetch(
            "https://example.invalid/data", key="failed", credential="env:VISUAL_TEST_TOKEN"
        )
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert cache._store.restore("failed") is None
    with pytest.raises(ValueError, match="HTTPS"):
        cache.fetch("http://example.invalid/data", key="bad", credential="env:VISUAL_TEST_TOKEN")


def test_authenticated_redirect_cannot_forward_to_another_origin():
    handler = _SameOriginRedirect()
    request = urllib.request.Request(
        "https://example.invalid/data", headers={"Authorization": "Bearer x"}
    )
    for destination in (
        "http://example.invalid/data",
        "https://other.invalid/data",
        "https://example.invalid:8443/data",
    ):
        with pytest.raises(ValueError, match="cross-origin"):
            handler.redirect_request(request, None, 302, "redirect", {}, destination)
    redirected = handler.redirect_request(
        request, None, 302, "redirect", {}, "https://example.invalid/new"
    )
    assert redirected.get_header("Authorization") == "Bearer x"
    assert json.dumps({"reference": "env:VISUAL_TEST_TOKEN"})
