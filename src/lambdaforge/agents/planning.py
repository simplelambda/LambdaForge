"""Isolate explicitly permitted consumer imports from a stdio protocol process.

This is process isolation for protocol/liveness, NOT a sandbox against trusted project code.
Only existing facade PLAN methods are dispatched; there is no arbitrary callable or shell API.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

from lambdaforge.agents.contracts import AgentResponse


def _worker(connection: Any, root: str, name: str, arguments: dict[str, Any]) -> None:
    # Project print()/logging must never emit bytes on the parent's MCP stdout.
    sys.stdout = sys.stderr
    try:
        from lambdaforge.agents.catalog import tool_operations
        from lambdaforge.agents.services import AgentServices

        service = AgentServices(root, allow_project_code=True)
        operations = tool_operations(service)
        if name not in operations or not operations[name].project_code:
            raise PermissionError("Only native trusted-project PLAN operations can be isolated.")
        connection.send(getattr(service, name)(**arguments))
    finally:
        connection.close()


async def invoke_planning(root: Path, name: str, arguments: dict[str, Any]) -> AgentResponse:
    """Bound imports/planning in time and reap the owned child on protocol cancellation."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, str(root), name, arguments))
    process.start()
    child.close()
    deadline = time.monotonic() + 60
    try:
        while not parent.poll():
            if not process.is_alive():
                raise RuntimeError(
                    "Native planning worker exited without a result; inspect stderr."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("Native planning exceeded 60 seconds; no Work was submitted.")
            await asyncio.sleep(0.05)
        try:
            result = parent.recv()
        except EOFError as error:
            raise RuntimeError("Native planning worker lost its result; inspect stderr.") from error
        if not isinstance(result, AgentResponse):
            raise RuntimeError("Native planning returned an invalid response contract.")
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
