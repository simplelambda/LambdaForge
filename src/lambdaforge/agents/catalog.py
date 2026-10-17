"""Versioned capability discovery derived from implemented facade operations."""

from __future__ import annotations

import inspect
import types
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

from lambdaforge._version import VERSION
from lambdaforge.agents.contracts import AgentResponse, Operation
from lambdaforge.agents.discovery import operation_semantics
from lambdaforge.agents.response_models import response_type


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
    if is_typeddict(annotation):
        return {
            "type": "object",
            "properties": {
                name: type_schema(value) for name, value in get_type_hints(annotation).items()
            },
            "required": sorted(annotation.__required_keys__),
            "additionalProperties": True,
        }
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


def response_schema(identifier: str) -> dict[str, Any]:
    """The common v1 envelope remains compatible; constrain data only for successful replies."""
    schema = type_schema(AgentResponse)
    schema["allOf"] = [
        {
            "if": {"properties": {"status": {"const": "ok"}}, "required": ["status"]},
            "then": {"properties": {"data": type_schema(response_type(identifier))}},
        }
    ]
    return schema


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
        session_enabled = (
            spec.permission in services.permissions
            and (not spec.project_code or services.allow_project_code)
            and (
                services.toolsets is None
                or spec.toolset == "core"
                or spec.toolset in services.toolsets
            )
        )
        rows.append(
            {
                **operation_semantics(spec, name, inspect.getdoc(function) or ""),
                "schema_version": 1,
                "identifier": spec.identifier,
                "name": name,
                "category": spec.toolset,
                "description": inspect.getdoc(function) or "",
                "introduced": spec.introduced_version,
                "installed_version": VERSION,
                "status": "available",
                "session_enabled": session_enabled,
                "availability": {
                    "installed": True,
                    "session_enabled": session_enabled,
                    "requires_project_code_consent": spec.project_code
                    and not services.allow_project_code,
                    "requires_operator_permission": spec.permission not in services.permissions,
                    "requires_provider_access": spec.external,
                    "configuration_verified": False,
                },
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
                "output_schema": response_schema(spec.identifier),
                "risk_annotations": {
                    "read_only": not spec.side_effects and spec.permission in {"READ", "PLAN"},
                    "destructive": spec.destructive,
                    "idempotent": spec.idempotent,
                    "external": spec.external,
                },
            }
        )
    for identifier, description in (
        (
            "fleet.recovery",
            "Native coordinated Study recovery is not supported; never replay it on one host.",
        ),
        (
            "agent.scientific_mutation",
            "Submission, recovery and destructive agent actions remain disabled; "
            "exact Job cancellation alone supports OPERATE plus local human approval. "
            "Approved knowledge edits do not authorize science.",
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
                "availability": {"installed": False, "session_enabled": False},
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
