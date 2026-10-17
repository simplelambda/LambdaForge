"""Precise JSON payloads for implemented discovery/read operations, independent of MCP."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class FrameworkVersionPayload(TypedDict):
    """Installed release and common agent-envelope version, not protocol-version guesses."""

    version: str
    agent_schema_version: int


class PathPagePayload(TypedDict):
    """Owned path inventory only; a listed YAML is not yet a validated Work."""

    items: list[str]
    offset: int
    next_offset: int | None


class ObjectPagePayload(TypedDict):
    """Bounded metadata object page; domain rows retain their native extensible fields."""

    items: list[dict[str, Any]]
    offset: int
    next_offset: int | None


class JobPayload(TypedDict):
    """Exact saved Job observation, including independent operational/recovery facts."""

    job_id: str
    cluster: str
    scheduler: str
    scheduler_id: str | None
    state: str
    resources: dict[str, Any]
    created_at_utc: str
    updated_at_utc: str
    retry_of: str | None
    operational: dict[str, Any]
    recovery: dict[str, Any] | None


class ApiMember(TypedDict):
    """An actual installed public member signature/documentation, with absent contracts explicit."""

    name: str
    signature: str | None
    documentation: str | None


class WorkApiPayload(TypedDict):
    """Installed Work API discovery without importing a consumer or an accelerator backend."""

    api: str
    version: str
    members: list[ApiMember]


class CancellationBinding(TypedDict):
    """Stable exact native owner; no raw argv, epoch data or artifact payloads."""

    job_id: str
    project_id: str
    cluster: str
    scheduler: str
    scheduler_id: str | None
    created_at_utc: str
    work_dir: str
    bundle_id: str | None
    scientific_identity: str | None
    command_hash: str
    storage_hash: str
    provider: dict[str, Any]
    scope: str


class CancellationPlan(TypedDict):
    """Side-effect-free scoped plan; a plan is never human authorization."""

    schema_version: int
    project_id: str
    action: Literal["jobs.cancel"]
    permission: Literal["OPERATE"]
    service: str
    arguments_hash: str
    evidence: CancellationBinding
    effect: str
    human_approval_required: bool
    plan_hash: str


class CancellationRequest(CancellationPlan):
    """Signed expiring request state; failed applies never grant replay rights."""

    request_id: str
    created_at: float
    expires_at: float
    state: Literal["pending", "approved", "applying", "applied", "rejected", "failed"]
    approval: dict[str, Any] | None
    expired: bool
    result_available: bool
    trust: str


RESPONSE_TYPES: dict[str, Any] = {
    "framework.version": FrameworkVersionPayload,
    "framework.api.work": WorkApiPayload,
    "project.structure": PathPagePayload,
    "project.configs": PathPagePayload,
    "jobs.get": JobPayload,
    "jobs.cancel.preview": CancellationPlan,
    "jobs.cancel.request": CancellationRequest,
    "jobs.cancel.status": CancellationRequest,
    "jobs.cancel.apply": CancellationRequest,
    **dict.fromkeys(
        (
            "framework.skills.list",
            "framework.capabilities",
            "framework.capability.search",
            "project.work_classes",
            "jobs.list",
            "works.list",
            "results.list",
            "products.list",
            "datasets.list",
            "agent.approvals",
        ),
        ObjectPagePayload,
    ),
}


def response_type(identifier: str) -> Any:
    """Native heterogeneous views remain extensible until their exact payload is declared."""
    return RESPONSE_TYPES.get(identifier, dict[str, Any])
