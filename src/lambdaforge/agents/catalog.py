"""Versioned capability discovery derived from implemented facade operations."""

from __future__ import annotations

import inspect
import types
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from lambdaforge._version import VERSION
from lambdaforge.agents.contracts import AgentResponse, Operation


def type_schema(annotation: Any) -> dict[str, Any]:
    """Interchange schemas for the small Python facade contracts, independent of MCP imports."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return {}
    if origin in {Union, types.UnionType}:
        return {"anyOf": [type_schema(item) for item in arguments]}
    if origin is Literal:
        return {"enum": list(arguments)}
    if origin is list:
        return {"type": "array", "items": type_schema(arguments[0])}
    if origin is dict:
        return {"type": "object", "additionalProperties": type_schema(arguments[1])}
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return {
            "type": "object",
            "properties": {item.name: type_schema(hints[item.name]) for item in fields(annotation)},
            "required": [item.name for item in fields(annotation)],
            "additionalProperties": False,
        }
    return {
        "type": {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }[annotation]
    }


def capabilities(services: Any) -> list[dict[str, Any]]:
    """Describe implementations, with unsupported operations explicitly labelled."""
    rows = []
    for name, function in inspect.getmembers(type(services), predicate=inspect.isfunction):
        spec: Operation | None = getattr(function, "agent_operation", None)
        if spec is None:
            continue
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        parameters = {}
        required = []
        for key, parameter in signature.parameters.items():
            if key == "self":
                continue
            parameters[key] = type_schema(hints[key])
            if parameter.default is inspect.Parameter.empty:
                required.append(key)
            else:
                parameters[key]["default"] = parameter.default
        rows.append(
            {
                "schema_version": 1,
                "identifier": spec.identifier,
                "name": name,
                "category": spec.toolset,
                "description": inspect.getdoc(function) or "",
                "introduced": VERSION,
                "installed_version": VERSION,
                "status": "available",
                "session_enabled": (
                    spec.permission in services.permissions
                    and (not spec.project_code or services.allow_project_code)
                    and (
                        services.toolsets is None
                        or spec.toolset == "core"
                        or spec.toolset in services.toolsets
                    )
                ),
                "permission": spec.permission,
                "service": spec.service,
                "documentation": spec.documentation,
                "optional_dependencies": [],
                "requirements": ["explicit trusted project-code permission"]
                if spec.project_code
                else [],
                "cost": "explicit provider read" if spec.external else "bounded local metadata",
                "side_effects": list(spec.side_effects)
                or (
                    ["rebuild disposable knowledge index"]
                    if spec.identifier
                    in {
                        "knowledge.search",
                        "knowledge.backlinks",
                        "knowledge.related",
                        "knowledge.hypotheses",
                        "knowledge.decisions",
                        "knowledge.questions",
                    }
                    else []
                ),
                "input_schema": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                    "additionalProperties": False,
                },
                "output_schema": type_schema(AgentResponse),
                "risk_annotations": {
                    "read_only": not spec.side_effects and spec.permission in {"READ", "PLAN"},
                    "destructive": False,
                    "idempotent": spec.idempotent,
                    "external": spec.external,
                },
                "errors": [
                    "NOT_FOUND",
                    "INVALID_ARGUMENT",
                    "PERMISSION_DENIED",
                    "EVIDENCE_UNAVAILABLE",
                    "RESPONSE_TOO_LARGE",
                ],
            }
        )
    for identifier, description in (
        (
            "fleet.recovery",
            "Native coordinated Study recovery is not supported; never replay it on one host.",
        ),
        (
            "agent.scientific_mutation",
            "Submission, operation and destructive agent actions remain disabled; "
            "approved knowledge edits do not authorize science.",
        ),
        (
            "agent.http.network",
            "Public/network MCP deployment and OAuth gateway are not supported; "
            "local paired HTTP does not authorize network exposure.",
        ),
        (
            "knowledge.semantic",
            "Semantic retrieval is not enabled; textual search remains usable offline.",
        ),
    ):
        rows.append(
            {
                "schema_version": 1,
                "identifier": identifier,
                "name": identifier,
                "category": "admin",
                "description": description,
                "status": "unsupported",
                "session_enabled": False,
                "installed_version": VERSION,
                "introduced": None,
                "permission": "READ",
                "service": None,
                "documentation": "docs/ai/IMPLEMENTATION.md",
                "optional_dependencies": [],
                "requirements": ["not implemented; no session permission can enable it"],
                "cost": "not available",
                "side_effects": [],
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
                "output_schema": type_schema(AgentResponse),
                "risk_annotations": {
                    "read_only": True,
                    "destructive": False,
                    "idempotent": True,
                    "external": False,
                },
                "errors": ["UNSUPPORTED"],
            }
        )
    return sorted(rows, key=lambda row: row["identifier"])


def tool_operations(services: Any) -> dict[str, Operation]:
    return {
        name: function.agent_operation
        for name, function in inspect.getmembers(type(services), predicate=inspect.isfunction)
        if hasattr(function, "agent_operation")
    }


__all__ = ["capabilities", "tool_operations"]
