"""Opt-in loopback MCP for explicitly paired local applications, never a public endpoint.

This local capability-token transport does not pretend to be an OAuth issuer. Public/network
deployment requires an external authenticated gateway and is deliberately unsupported here.
The official SDK still owns HTTP protocol/session handling; no REST scientific backend exists.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from typing import Any
from urllib.parse import urlsplit

from lambdaforge.agents.scope import ProjectScope

TOKEN_PATH = ".lambdaforge/agents/http/token"
_TOKEN = re.compile(rb"[A-Za-z0-9_-]{43}\Z")


def prepare_token(scope: ProjectScope, *, apply: bool = False) -> dict[str, Any]:
    """Operator-only pairing file. Never return token bytes or overwrite an existing token."""
    scope.assert_current()
    path = scope.path(TOKEN_PATH, internal=True)
    if path.exists():
        _read_token(scope)
    elif apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(secrets.token_urlsafe(32).encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "configured": path.exists(),
        "token_file": str(path),
        "secret_returned": False,
        "transport": "loopback capability-token HTTP",
        "network_deployment_supported": False,
    }


def _read_token(scope: ProjectScope) -> bytes:
    scope.assert_current()
    path = scope.path(TOKEN_PATH, internal=True)
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise PermissionError("Local HTTP pairing token must have mode 0600.")
    raw = scope.file(TOKEN_PATH, limit=64, internal=True)
    if not _TOKEN.fullmatch(raw):
        raise ValueError("Local HTTP pairing token is corrupt; prepare a fresh pairing explicitly.")
    return raw


class LocalAccess:
    """ASGI boundary checks every request before any SDK resource/tool handler."""

    def __init__(
        self, app: Any, *, project: ProjectScope, port: int, origins: tuple[str, ...]
    ) -> None:
        self.app = app
        self.project = project
        self.host = f"127.0.0.1:{port}"
        self.origins = origins

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.responses import JSONResponse, Response

        headers: dict[str, list[str]] = {}
        for name, value in scope.get("headers", []):
            headers.setdefault(name.decode("latin1").lower(), []).append(value.decode("latin1"))
        # Never trust Forwarded/X-Forwarded-For or permit DNS-rebinding host names.
        peer = scope.get("client")
        origin = headers.get("origin", [])
        if (
            not peer
            or peer[0] not in {"127.0.0.1", "::1"}
            or headers.get("host") != [self.host]
            or len(origin) > 1
            or (origin and origin[0] not in self.origins)
        ):
            await JSONResponse(
                {"error": "Local MCP host/origin/network policy refused this request."},
                status_code=403,
            )(scope, receive, send)
            return
        cors = (
            {
                "Access-Control-Allow-Origin": origin[0],
                "Access-Control-Expose-Headers": "Mcp-Session-Id",
                "Vary": "Origin",
            }
            if origin
            else {}
        )
        if scope.get("method") == "OPTIONS" and origin:
            # Preflight exposes no metadata. It never authorizes the following real request.
            await Response(
                status_code=204,
                headers={
                    **cors,
                    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization,Content-Type,"
                    "Mcp-Protocol-Version,Mcp-Session-Id,Mcp-Method,Mcp-Name,Last-Event-ID",
                },
            )(scope, receive, send)
            return
        try:
            token = _read_token(self.project)
        except (ValueError, OSError):
            await JSONResponse(
                {"error": "Local pairing/project is no longer valid."}, status_code=403
            )(scope, receive, send)
            return
        authorization = headers.get("authorization", [])
        expected = b"Bearer " + token
        valid = len(authorization) == 1 and hmac.compare_digest(
            hashlib.sha256(authorization[0].encode("latin1")).digest(),
            hashlib.sha256(expected).digest(),
        )
        if not valid or scope.get("query_string"):
            await JSONResponse(
                {
                    "error": "A paired Authorization bearer token is required; "
                    "query tokens are forbidden."
                },
                status_code=401,
                headers={
                    **cors,
                    "WWW-Authenticate": 'Bearer realm="LambdaForge local pairing"',
                },
            )(scope, receive, send)
            return

        async def cors_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        *((key.lower().encode(), value.encode()) for key, value in cors.items()),
                    ],
                }
            await send(message)

        await self.app(scope, receive, cors_send)


def create_http_app(
    project: str,
    *,
    port: int = 8765,
    origins: tuple[str, ...] = (),
    toolsets: tuple[str, ...] = ("core", "work", "knowledge"),
    allow_project_code: bool = False,
) -> Any:
    """Shared SDK/domain backend; loopback only, paired READ/PLAN, bounded sessions/requests."""
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Choose a local HTTP port between 1024 and 65535.")
    if len(origins) > 8 or any(
        urlsplit(origin).scheme not in {"http", "https"}
        or not urlsplit(origin).netloc
        or urlsplit(origin).path
        or urlsplit(origin).query
        or urlsplit(origin).fragment
        or urlsplit(origin).username
        or "*" in origin
        or origin == "null"
        for origin in origins
    ):
        raise ValueError(
            "HTTP origins require up to eight exact http(s) scheme/host/port origins; no wildcards."
        )
    scope = ProjectScope(project)
    _read_token(scope)  # Fail before binding a network socket.
    from lambdaforge.agents.mcp_server import create_server

    server = create_server(project, toolsets=toolsets, allow_project_code=allow_project_code)
    from mcp.server.transport_security import TransportSecuritySettings

    app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        max_request_body_size=512 * 1024,
        max_sessions=16,
        session_idle_timeout=120,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[f"127.0.0.1:{port}"],
            allowed_origins=list(origins),
        ),
    )
    return LocalAccess(app, project=scope, port=port, origins=origins)


def serve_http(
    project: str,
    *,
    port: int = 8765,
    origins: tuple[str, ...] = (),
    toolsets: tuple[str, ...] = ("core", "work", "knowledge"),
    allow_project_code: bool = False,
) -> None:
    """Explicit local transport. No wildcard bind, reverse-proxy trust or public deployment."""
    app = create_http_app(
        project,
        port=port,
        origins=origins,
        toolsets=toolsets,
        allow_project_code=allow_project_code,
    )
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        proxy_headers=False,
        access_log=False,
        limit_concurrency=32,
        timeout_keep_alive=10,
    )


__all__ = ["create_http_app", "prepare_token", "serve_http"]
