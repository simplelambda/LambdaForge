"""Optional real Codex MCP client acceptance, with no inference or user configuration edits.

Run with LAMBDAFORGE_TEST_CODEX=/absolute/codex and the optional MCP dependencies installed.
This checks actual discovery and calls, not an LLM conversation or scientific authorization.
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest

CODEX = os.environ.get("LAMBDAFORGE_TEST_CODEX")
pytestmark = pytest.mark.skipif(not CODEX, reason="Explicit opt-in installed Codex binary required")


def test_real_codex_client_discovers_calls_and_keeps_foreign_cwd_pinned(tmp_path: Path):
    pytest.importorskip("mcp")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="codex-fixture"\nversion="0.1"\n')
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    codex_state = tmp_path / "isolated-codex"
    codex_state.mkdir()
    source = str(Path(__file__).resolve().parents[2] / "src")
    config = (
        "[analytics]\nenabled=false\n[mcp_servers.lambdaforge]\ncommand="
        + json.dumps(sys.executable)
        + "\nargs="
        + json.dumps(["-m", "lambdaforge", "ai", "--project", str(project), "mcp", "serve"])
        + "\nstartup_timeout_sec=20\n[mcp_servers.lambdaforge.env]\nPYTHONPATH="
        + json.dumps(source + os.pathsep + os.environ.get("PYTHONPATH", ""))
        + "\n"
    )
    (codex_state / "config.toml").write_text(config)

    async def scenario():
        # CODEX_HOME is the documented client configuration location, not a cleanup target.
        env = {**os.environ, "CODEX_HOME": str(codex_state)}
        for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "ACCESS_TOKEN"):
            env.pop(key, None)
        process = await asyncio.create_subprocess_exec(
            CODEX,
            "app-server",
            "--stdio",
            cwd=foreign,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            # Codex's full discovery reply contains many tool schemas, not a single view.
            # Keep framing bounded but above asyncio's unrelated 64 KiB default line limit.
            limit=2 * 1024 * 1024,
            start_new_session=os.name != "nt",
        )
        serial = 0

        async def request(method, params):
            nonlocal serial
            serial += 1
            identity = serial
            process.stdin.write(
                (json.dumps({"id": identity, "method": method, "params": params}) + "\n").encode()
            )
            await process.stdin.drain()
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), 25)
                assert line, "Codex exited before returning its protocol response"
                message = json.loads(line)
                if message.get("id") == identity:
                    assert "error" not in message, message.get("error")
                    return message["result"]

        try:
            await request(
                "initialize",
                {
                    "clientInfo": {"name": "lambdaforge-acceptance", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            process.stdin.write(b'{"method":"initialized","params":{}}\n')
            await process.stdin.drain()
            thread = await request(
                "thread/start",
                {
                    "cwd": str(foreign),
                    "ephemeral": True,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                },
            )
            thread_id = thread["thread"]["id"]
            inventory = await request(
                "mcpServerStatus/list",
                {"threadId": thread_id, "serverName": "lambdaforge", "detail": "full"},
            )
            rows = inventory["data"]
            assert len(rows) == 1 and rows[0]["name"] == "lambdaforge"
            assert any(t["name"] == "project_info" for t in rows[0]["tools"].values())
            reply = await request(
                "mcpServer/tool/call",
                {
                    "threadId": thread_id,
                    "server": "lambdaforge",
                    "tool": "project_info",
                    "arguments": {},
                },
            )
            evidence = reply["structuredContent"]
            assert evidence["status"] == "ok"
            assert evidence["data"]["root"] == str(project)
            assert evidence["data"]["permissions"] == ["READ", "PLAN"]
            assert not list(project.rglob("execution.json"))
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                await asyncio.wait_for(process.wait(), 5)

    asyncio.run(scenario())
    assert (codex_state / "config.toml").read_text() == config
