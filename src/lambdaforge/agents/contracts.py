"""Small versioned interchange contracts shared by Python and optional MCP clients."""

from __future__ import annotations

import inspect
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Concatenate, Literal, ParamSpec

import yaml

from lambdaforge.controlplane.SecretRedactor import SecretRedactor

Permission = Literal["READ", "PLAN", "WRITE_PROJECT", "SUBMIT", "OPERATE", "DESTRUCTIVE"]
_MAX_REPLY = 512 * 1024
_Arguments = ParamSpec("_Arguments")


@dataclass(frozen=True)
class AgentError:
    """Stable bounded operational error; no unredacted native traceback crosses the adapter."""

    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class AgentResponse:
    """Data is domain-owned JSON; text from consumers is untrusted, not instructions."""

    schema_version: int
    project_id: str
    source: str
    retrieved_at_utc: str
    status: Literal["ok", "error"]
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: AgentError | None = None
    trust: str = "domain_metadata_with_untrusted_consumer_text"


@dataclass(frozen=True)
class Operation:
    """Discoverable permission/service contract for one implemented facade method."""

    identifier: str
    toolset: str
    permission: Permission
    service: str
    documentation: str
    external: bool = False
    project_code: bool = False
    side_effects: tuple[str, ...] = ()
    idempotent: bool = True


def redact(value: Any) -> Any:
    """Redact recursive structured secrets and common text credentials before transfer."""
    if isinstance(value, Mapping):
        return {
            str(key): "***" if SecretRedactor.is_secret_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SecretRedactor.redact(value)
    return value


def operation(
    identifier: str,
    toolset: str,
    permission: Permission,
    service: str,
    documentation: str,
    *,
    external: bool = False,
    project_code: bool = False,
    side_effects: tuple[str, ...] = (),
    idempotent: bool = True,
) -> Callable[
    [Callable[Concatenate[Any, _Arguments], dict[str, Any]]],
    Callable[Concatenate[Any, _Arguments], AgentResponse],
]:
    """One discoverable contract around an existing domain service, not a second implementation."""
    specification = Operation(
        identifier,
        toolset,
        permission,
        service,
        documentation,
        external,
        project_code,
        side_effects,
        idempotent,
    )

    def decorate(
        function: Callable[Concatenate[Any, _Arguments], dict[str, Any]],
    ) -> Callable[Concatenate[Any, _Arguments], AgentResponse]:
        @wraps(function)
        def call(self: Any, *args: _Arguments.args, **kwargs: _Arguments.kwargs) -> AgentResponse:
            now = datetime.now(timezone.utc).isoformat()
            try:
                self.check_operation(specification)
                data = redact(function(self, *args, **kwargs))
                if not isinstance(data, dict):
                    raise TypeError("An agent operation must return a structured object.")
                encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if len(encoded) > _MAX_REPLY:
                    raise OverflowError("Response exceeds 512 KiB; request a smaller page/detail.")
                return AgentResponse(
                    1, self.scope.project.project_id, specification.service, now, "ok", data
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                RuntimeError,
                OverflowError,
                yaml.YAMLError,
                sqlite3.DatabaseError,
            ) as error:
                code = (
                    "PERMISSION_DENIED"
                    if isinstance(error, PermissionError)
                    else "NOT_FOUND"
                    if isinstance(error, FileNotFoundError | KeyError)
                    else "RESPONSE_TOO_LARGE"
                    if isinstance(error, OverflowError)
                    else "INVALID_ARGUMENT"
                    if isinstance(error, ValueError | TypeError | yaml.YAMLError)
                    else "EVIDENCE_UNAVAILABLE"
                )
                return AgentResponse(
                    1,
                    self.scope.project.project_id,
                    specification.service,
                    now,
                    "error",
                    error=AgentError(
                        code,
                        (
                            "Knowledge index is unreadable; use lf ai knowledge reindex "
                            "--rebuild. Canonical Markdown remains unchanged."
                            if isinstance(error, sqlite3.DatabaseError)
                            and specification.toolset == "knowledge"
                            else SecretRedactor.redact(str(error))[-1800:]
                        ),
                    ),
                )

        call.__annotations__ = {**function.__annotations__, "return": AgentResponse}
        call.__signature__ = inspect.signature(function).replace(  # type: ignore[attr-defined]
            return_annotation=AgentResponse,
        )
        call.agent_operation = specification  # type: ignore[attr-defined]
        return call

    return decorate


def page(items: list[Any], offset: int, limit: int) -> dict[str, Any]:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Pages require offset >= 0 and limit 1–100.")
    return {
        "items": items[offset : offset + limit],
        "offset": offset,
        "next_offset": offset + limit if len(items) > offset + limit else None,
    }
