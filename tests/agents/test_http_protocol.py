"""Paired loopback HTTP: real SDK requests, denial before handlers, rotation and limits."""

import asyncio
import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="Optional official SDK required for HTTP protocol tests.")
import httpx2
import uvicorn
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from lambdaforge.agents.http import TOKEN_PATH, create_http_app, prepare_token
from lambdaforge.agents.scope import ProjectScope


@pytest.fixture
def pairing(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="http"\nversion="0.1"\n')
    scope = ProjectScope(tmp_path)
    preview = prepare_token(scope)
    assert preview["configured"] is False
    assert not (tmp_path / ".lambdaforge").exists()
    created = prepare_token(scope, apply=True)
    assert created["secret_returned"] is False
    return scope, (tmp_path / TOKEN_PATH).read_text()


def test_http_rejects_unpaired_origin_host_network_query_and_large_request(pairing):
    scope, token = pairing

    async def scenario():
        app = create_http_app(str(scope.root), port=8765, origins=("http://127.0.0.1:3000",))
        headers = {"Authorization": "Bearer " + token}
        async with app.app.router.lifespan_context(app.app):
            transport = httpx2.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://127.0.0.1:8765"
            ) as client:
                denied = await client.post("/mcp", json={})
                assert denied.status_code == 401
                assert token not in denied.text
                assert (
                    await client.post("/mcp", headers={"Authorization": "Bearer forged"}, json={})
                ).status_code == 401
                assert (
                    await client.post("/mcp?token=" + token, headers=headers, json={})
                ).status_code == 401
                assert (
                    await client.post(
                        "/mcp", headers={**headers, "Origin": "https://evil.example"}, json={}
                    )
                ).status_code == 403
                assert (
                    await client.post(
                        "/mcp", headers={**headers, "Host": "evil.example:8765"}, json={}
                    )
                ).status_code == 403
                preflight = await client.options(
                    "/mcp", headers={"Origin": "http://127.0.0.1:3000"}
                )
                assert preflight.status_code == 204
                assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
                large = await client.post(
                    "/mcp",
                    headers={
                        **headers,
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                    content=b"x" * (512 * 1024 + 1),
                )
                assert large.status_code == 413
            external = httpx2.ASGITransport(app=app, client=("192.0.2.1", 12345))
            async with httpx2.AsyncClient(
                transport=external, base_url="http://127.0.0.1:8765"
            ) as client:
                assert (await client.post("/mcp", headers=headers, json={})).status_code == 403

    asyncio.run(scenario())


def test_loopback_network_sdk_client_gets_only_read_plan_and_revocation_takes_effect(pairing):
    scope, token = pairing

    async def scenario():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            app = create_http_app(str(scope.root), port=port, toolsets=("core", "knowledge"))
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="error", access_log=False, proxy_headers=False)
            )
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                for _ in range(200):
                    if server.started:
                        break
                    if task.done():
                        await task
                        pytest.fail("HTTP server exited before initialization")
                    await asyncio.sleep(0.01)
                assert server.started
                async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}) as http:
                    transport = streamable_http_client(
                        f"http://127.0.0.1:{port}/mcp", http_client=http
                    )
                    async with Client(transport, read_timeout_seconds=10, cache=None) as client:
                        reply = (await client.call_tool("project_info")).structured_content
                        assert reply["status"] == "ok" and reply["data"]["root"] == str(scope.root)
                        names = {t.name for t in (await client.list_tools()).tools}
                        assert "apply_agent_action" not in names and "submit_work" not in names
                        resource = await client.read_resource("lambdaforge://project/info")
                        assert (
                            json.loads(resource.contents[0].text)["project_id"]
                            == scope.project.project_id
                        )
                # A private token replacement/revocation is checked on every real request.
                (scope.root / TOKEN_PATH).write_text("A" * 43)
                async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}) as old:
                    assert (
                        await old.post(f"http://127.0.0.1:{port}/mcp", json={})
                    ).status_code == 401
            finally:
                server.should_exit = True
                await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())


def test_pairing_never_overwrites_or_reveals_secret_and_wildcard_origin_refused(pairing):
    scope, token = pairing
    assert (scope.root / TOKEN_PATH).read_text() == token
    summary = prepare_token(scope, apply=True)
    assert token not in json.dumps(summary)
    assert (scope.root / TOKEN_PATH).read_text() == token
    for origin in (
        "*",
        "null",
        "https://*.example",
        "https://user:secret@example",
        "https://example/path",
    ):
        with pytest.raises(ValueError, match="origins"):
            create_http_app(str(scope.root), origins=(origin,))
    with pytest.raises(ValueError, match="port"):
        create_http_app(str(scope.root), port=80)
    (scope.root / TOKEN_PATH).write_text("not-a-token")
    with pytest.raises(ValueError, match="corrupt"):
        create_http_app(str(scope.root))
