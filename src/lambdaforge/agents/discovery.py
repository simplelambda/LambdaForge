"""Deterministic semantic discovery metadata; no embeddings, execution or science inference."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from lambdaforge.agents.contracts import Operation

_TERMS = {
    "validate": {"validate", "validation", "validar", "validacion"},
    "connect": {"connect", "connection", "conectar", "conexion"},
    "inspect": {"inspect", "inspection", "inspeccionar", "mostrar", "ver", "details", "detalle"},
    "watch": {"watch", "follow", "seguir", "seguimiento", "observar"},
    "publish": {"publish", "publication", "publicar", "publicacion", "promote", "promocion"},
    "product": {"product", "products", "producto", "productos", "weights", "pesos", "trained"},
    "model": {"model", "models", "modelo", "modelos", "architecture", "arquitectura"},
    "recover": {"recover", "recovery", "resume", "retry", "recuperar", "reanudar"},
    "study": {
        "study",
        "studies",
        "estudio",
        "estudios",
        "experiment",
        "experimento",
        "training",
        "entrenamiento",
        "entrenamientos",
    },
    "knowledge": {"knowledge", "memory", "memoria", "note", "notes", "nota", "notas"},
    "dataset": {"dataset", "datasets", "datos", "data"},
    "cluster": {"cluster", "clusters", "hpc", "remote", "remoto"},
    "failure": {"failure", "failed", "error", "fallo", "fallido", "incident", "incidente"},
    "approval": {"approval", "approve", "aprobacion", "aprobar", "authorization", "autorizar"},
    "capability": {
        "capability",
        "capabilities",
        "capacidades",
        "capacidad",
        "tools",
        "herramientas",
    },
}
_STOP = set(
    (
        "i want to a an the my in on of for and how can do que quiero para "
        "un una el la los las mi en de con como puedo por y"
    ).split()
)
_NEXT_TOOLS = {
    "framework.capabilities": ["capability_describe"],
    "project.configs": ["inspect_config", "validate_work", "explain_work"],
    "project.work_classes": ["api_reference", "list_project_configs"],
    "work.config.validate": ["explain_work", "preview_execution"],
    "results.list": ["get_result", "get_historical_reference"],
    "jobs.get": ["get_execution_health", "get_work_logs", "prepare_recovery"],
    "agent.action.preview": ["request_agent_action"],
    "agent.action.request": ["agent_action_status"],
    "jobs.cancel.preview": ["request_job_cancellation"],
    "jobs.cancel.request": ["job_cancellation_status"],
    "jobs.cancel.status": ["get_job"],
    "knowledge.search": ["knowledge_read", "knowledge_verify_references"],
}


def _tokens(text: str) -> set[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    words = set(re.findall(r"\w+", normalized)) - _STOP
    for canonical, aliases in _TERMS.items():
        if words & aliases:
            words.add(canonical)
    return words


def operation_semantics(spec: Operation, name: str, description: str) -> dict[str, Any]:
    """Specific native ownership, permission, cost and freshness without guessed guarantees."""
    errors = [
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "PERMISSION_DENIED",
        "RESPONSE_TOO_LARGE",
        "INTERNAL_ERROR",
    ]
    if spec.external:
        errors += ["PROVIDER_UNREACHABLE", "PROVIDER_UNCERTAIN", "TIMEOUT"]
    if spec.toolset in {"results", "products", "datasets", "study", "knowledge", "operations"}:
        errors += ["EVIDENCE_UNAVAILABLE", "EVIDENCE_CORRUPT"]
    if spec.identifier.startswith(("agent.action.", "jobs.cancel.")):
        errors += [
            "UNSUPPORTED",
            "APPROVAL_REQUIRED",
            "APPROVAL_EXPIRED",
            "PLAN_STALE",
            "OPERATION_CONFLICT",
        ]
    return {
        "title": name.replace("_", " ").capitalize(),
        "summary": description.splitlines()[0] if description else spec.identifier,
        "toolset": spec.toolset,
        "domain_service": spec.service,
        "introduced_version": spec.introduced_version,
        "permissions": [spec.permission],
        "preconditions": [
            "unchanged pinned project identity",
            f"operator-enabled {spec.permission} permission",
        ]
        + (["explicit trusted project-code import permission"] if spec.project_code else []),
        "postconditions": [
            "project-bound bounded native metadata",
            "no scientific identity inferred from display names",
        ],
        "cost_class": "provider_metadata" if spec.external else "local_metadata",
        "idempotency": "idempotent" if spec.idempotent else "single_use_no_automatic_replay",
        "retry_behavior": (
            "Refresh reads deliberately; never repeat an apply from retryability alone."
        ),
        "freshness_semantics": "explicit current provider observation"
        if spec.external
        else "persisted or installed snapshot; retrieval time is not scientific progress",
        "identity_requirements": [
            "pinned project",
            "exact native identities for historical science",
        ],
        "suggested_next_tools": _NEXT_TOOLS.get(spec.identifier, []),
        "examples": [{"tool": name, "arguments": {}}]
        if spec.identifier in {"framework.version", "framework.api.work", "framework.api.study"}
        else [],
        "errors": errors,
    }


def search_capabilities(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rank technical synonyms deterministically; relevance is not scientific confidence."""
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("Capability query must be text of at most 500 characters.")
    terms = _tokens(query)
    if not terms:
        return rows
    ranked = []
    for row in rows:
        text = " ".join(
            str(row.get(key, "")) for key in ("identifier", "category", "description", "service")
        )
        matches = terms & _tokens(text)
        if matches:
            ranked.append((len(matches) / len(terms), row["identifier"], row))
    return [row for _, _, row in sorted(ranked, key=lambda entry: (-entry[0], entry[1]))]
