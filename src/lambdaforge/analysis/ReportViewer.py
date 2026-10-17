"""Native read-only loopback viewer for scoped managed report assets, not a scientific server."""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlsplit

from lambdaforge.analysis.report_assets import local_asset, remote_asset
from lambdaforge.work.report_bundle import ReportBundleError, asset_id


class _BoundedServer(ThreadingHTTPServer):
    """Bound reader connection threads as well as active binary transfers."""

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._connections = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class ReportViewer:
    """Serve one generated native report and its declared assets only.

    A private URL capability, exact Host and a same-origin asset header prevent ambient
    cross-site reads. Project code remains opaque-origin with connect-src none; it gets
    a scoped MessagePort, never Python callbacks, filesystem paths or provider credentials.
    """

    def __init__(
        self,
        report: str | Path,
        *,
        jobs: Any = None,
        max_asset_bytes: int = 256 * 1024 * 1024,
        idle_seconds: float = 1800,
    ) -> None:
        path = Path(report).expanduser().absolute()
        if path.resolve() != path or not path.is_file() or path.suffix != ".html":
            raise ValueError("Report viewer requires a regular non-symbolic generated .html file.")
        if type(max_asset_bytes) is not int or max_asset_bytes <= 0 or idle_seconds <= 0:
            raise ValueError("Report asset/idle limits must be positive.")
        self.report, self.jobs = path, jobs
        self.max_asset_bytes, self.idle_seconds = max_asset_bytes, float(idle_seconds)
        self.bundles: list[dict[str, Any]] = []
        sidecar = path.with_suffix(path.suffix + ".assets.json")
        if sidecar.exists():
            if (
                sidecar.resolve() != sidecar
                or not sidecar.is_file()
                or sidecar.stat().st_size > 64 * 1024 * 1024
            ):
                raise ValueError("Unsafe/oversized report asset deployment metadata.")
            value = json.loads(sidecar.read_text())
            if value.get("report_assets_version") != 1 or not isinstance(
                value.get("bundles"), list
            ):
                raise ValueError("Invalid report asset deployment metadata.")
            self.bundles = value["bundles"]
            for bundle in self.bundles:
                if not isinstance(bundle, dict) or not isinstance(bundle.get("source"), dict):
                    raise ValueError("Invalid report bundle deployment metadata.")
                source = bundle["source"]
                if source.pop("relative", False):
                    for key in ("owner", "run_dir"):
                        target = path.parent / source[key]
                        if any(parent.is_symlink() for parent in (target, *target.parents)):
                            raise ValueError("Symlinked report deployment source.")
                        source[key] = str(target.resolve())
        self.token = secrets.token_urlsafe(32)
        self._slots = threading.BoundedSemaphore(2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_request = time.monotonic()
        self._report_identity = self._identity(path)
        self.server = _BoundedServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.server.timeout = 0.5
        self.host = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.host}/{self.token}/report.html"

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int]:
        info = path.stat()
        return info.st_ino, info.st_size, info.st_mtime_ns

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(10)

            def log_message(self, format: str, *args: Any) -> None:
                # Never log capabilities, provider paths or scientific data.
                pass

            def error(self, code: str, message: str, status: int = 400) -> None:
                raw = json.dumps({"code": code, "message": message}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def peer_closed(self) -> bool:
                import select

                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable and not self.connection.recv(1, socket.MSG_PEEK))
                except OSError:
                    return True

            def do_GET(self) -> None:
                target = urlsplit(self.path)
                prefix = f"/{viewer.token}/"
                origin = self.headers.get("Origin")
                if (
                    len(self.headers.get_all("Host", [])) != 1
                    or len(self.headers.get_all("Origin", [])) > 1
                    or self.headers.get("Host") != viewer.host
                    or target.query
                    or target.fragment
                    or not target.path.startswith(prefix)
                    or (origin is not None and origin != f"http://{viewer.host}")
                    or self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}
                ):
                    self.error("FORBIDDEN", "Report origin/capability is invalid.", 403)
                    return
                viewer._last_request = time.monotonic()
                route = target.path[len(prefix) :]
                if route == "report.html":
                    if (
                        viewer.report.resolve() != viewer.report
                        or viewer._identity(viewer.report) != viewer._report_identity
                    ):
                        self.error(
                            "STALE_REPORT", "Report changed; reopen the generated report.", 409
                        )
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header(
                        "Content-Security-Policy",
                        "frame-src 'none'; connect-src 'self'; object-src 'none'; base-uri 'none'",
                    )
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(viewer.report.stat().st_size))
                    self.end_headers()
                    with viewer.report.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                    return
                if self.headers.get("X-LambdaForge-Report") != viewer.token:
                    self.error(
                        "FORBIDDEN", "Assets require the native same-origin report loader.", 403
                    )
                    return
                parts = route.split("/")
                if len(parts) != 3 or parts[0] != ".assets" or not parts[1].isdigit():
                    self.error(
                        "UNDECLARED_ASSET", "Only declared report asset routes are supported.", 404
                    )
                    return
                if not viewer._slots.acquire(blocking=False):
                    self.error("VIEW_BUSY", "Two assets are loading; await/cancel one first.", 429)
                    return
                try:
                    identifier = asset_id(unquote(parts[2]))
                    index = int(parts[1])
                    with viewer._asset(
                        index,
                        identifier,
                        cancel=lambda: viewer._stop.is_set() or self.peer_closed(),
                    ) as (stream, row):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header("Content-Length", str(row["size_bytes"]))
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except FileNotFoundError:
                    self.error(
                        "MISSING_ASSET",
                        "Presentation asset is missing; restore exact retained content "
                        "or regenerate presentation.",
                        404,
                    )
                except (ValueError, OSError, RuntimeError, IndexError, KeyError, EOFError) as error:
                    # Framework errors only; never expose provider stderr/credentials to project JS.
                    message = (
                        str(error)
                        if isinstance(error, ReportBundleError)
                        else "Asset unavailable, corrupt or outside this bundle. "
                        "Inspect native report diagnostics; science is unchanged."
                    )
                    self.error("ASSET_UNAVAILABLE", message, 409)
                finally:
                    viewer._slots.release()

        return Handler

    @contextmanager
    def _asset(
        self, index: int, identifier: str, *, cancel: Any
    ) -> Iterator[tuple[BinaryIO, dict[str, Any]]]:
        if not 0 <= index < len(self.bundles):
            raise ReportBundleError("Unknown report bundle.")
        bundle = self.bundles[index]
        rows = [r for r in bundle["manifest"]["assets"] if r["id"] == identifier]
        if len(rows) != 1:
            raise ReportBundleError("Asset is not declared by the selected bundle.")
        row = rows[0]
        if row["size_bytes"] > self.max_asset_bytes:
            raise ReportBundleError(
                "Asset exceeds the active-view limit. Reopen with lf reports serve "
                "--max-asset-mib SIZE; no points/proteins were omitted."
            )
        source = bundle["source"]
        if source.get("remote"):
            if self.jobs is None:
                from lambdaforge.controlplane.JobService import JobService

                self.jobs = JobService()
            if shutil.disk_usage(self.report.parent).free < row["size_bytes"]:
                raise ReportBundleError(
                    "Insufficient space beside this report for one verified asset; "
                    "choose a report location with more free space."
                )
            with tempfile.TemporaryDirectory(
                prefix=".lf-report-asset-", dir=self.report.parent
            ) as directory:
                path = Path(directory) / "asset"
                remote_asset(
                    self.jobs,
                    source["job_id"],
                    source,
                    identifier,
                    path,
                    cancel=cancel,
                    expected_size=row["size_bytes"],
                )
                if path.stat().st_size != row["size_bytes"]:
                    raise ReportBundleError("Remote asset size differs from finalized metadata.")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != row["content_sha256"]:
                    raise ReportBundleError(
                        "Remote asset differs from its finalized content checksum."
                    )
                with path.open("rb") as stream:
                    yield stream, row
        else:
            with local_asset(source, identifier) as stream:
                yield stream, row

    def serve(self) -> None:
        """Foreground native viewer; Ctrl-C or explicit close ends only report serving."""
        try:
            while (
                not self._stop.is_set()
                and time.monotonic() - self._last_request < self.idle_seconds
            ):
                self.server.handle_request()
        finally:
            self.server.server_close()

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self.serve, daemon=True, name="lf-report-viewer")
            self._thread.start()
        return self.url

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        else:
            self.server.server_close()


_VIEWERS: list[ReportViewer] = []


def open_report(report: Path, *, jobs: Any = None) -> str:
    """Open requested presentation; old self-contained files remain file:// compatible."""
    if not report.with_suffix(report.suffix + ".assets.json").is_file():
        return report.as_uri()
    viewer = ReportViewer(report, jobs=jobs)
    _VIEWERS.append(viewer)
    while len(_VIEWERS) > 8:
        _VIEWERS.pop(0).close()
    return viewer.start()
