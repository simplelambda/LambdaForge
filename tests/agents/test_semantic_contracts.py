"""Contract semantics, schema/response compatibility and native adapter boundaries."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from lambdaforge.agents.approvals import ApprovalService
from lambdaforge.agents.catalog import capabilities, response_schema
from lambdaforge.agents.contracts import operation
from lambdaforge.agents.errors import AgentApprovalError, classify_error
from lambdaforge.agents.services import AgentServices
from lambdaforge.diagnostics.models import (
    ErrorCategory,
    LambdaForgeError,
    RetryDisposition,
    diagnostic,
)


@pytest.fixture
def service(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="contracts"\nversion="0.1"\n')
    return AgentServices(tmp_path)


def test_data_contracts_validate_success_errors_and_reject_wrong_native_payload(service):
    reply = service.framework_version()
    schema = response_schema("framework.version")
    Draft202012Validator(schema).validate(asdict(reply))
    data_schema = schema["allOf"][0]["then"]["properties"]["data"]
    assert set(data_schema["required"]) == {"version", "agent_schema_version"}
    assert data_schema["properties"]["version"]["type"] == "string"
    assert service.framework_version("unexpected").error.code == "INVALID_ARGUMENT"

    class WrongPayload:
        scope = service.scope

        def check_operation(self, specification):
            pass

        @operation("framework.version", "core", "READ", "fixture", "fixture")
        def wrong(self):
            return {"version": 12, "agent_schema_version": 1}

    rejected = WrongPayload().wrong()
    assert rejected.status == "error" and rejected.error.code == "INTERNAL_ERROR"
    Draft202012Validator(schema).validate(asdict(rejected))


def test_unexpected_internal_exception_has_no_private_traceback_or_secret(service):
    class Broken:
        scope = service.scope

        def check_operation(self, specification):
            pass

        @operation("fixture.broken", "core", "READ", "fixture", "fixture")
        def broken(self):
            raise TypeError("private arbitrary consumer text /secret/server/path")

    response = Broken().broken()
    assert response.error.code == "INTERNAL_ERROR"
    assert "/secret" not in response.error.message
    assert "traceback" not in asdict(response)


def test_resource_report_is_saved_metadata_only_without_enabling_external_calls(
    service, monkeypatch
):
    store = SimpleNamespace(
        bounded_metadata=lambda **kw: {"records": (), "skipped": 0, "truncated": 0}
    )
    monkeypatch.setattr(service, "_job_service", lambda: SimpleNamespace(store=store))
    response = service.get_resource_report(
        since="2026-10-13T10:00:00Z", until="2026-10-13T12:00:00Z"
    )
    assert response.status == "ok" and response.data["job_count"] == 0
    assert response.data["coverage"]["complete_timeline"] is False
    capability = next(
        row for row in capabilities(service) if row["identifier"] == "clusters.resource_report"
    )
    assert capability["permission"] == "READ"


@pytest.mark.parametrize(
    "code", ["APPROVAL_REQUIRED", "APPROVAL_EXPIRED", "PLAN_STALE", "OPERATION_CONFLICT"]
)
def test_approval_refusals_keep_distinct_codes_without_automatic_retry(code):
    reply = classify_error(AgentApprovalError(code, "password=do-not-leak"))
    assert reply.code == code and not reply.retryable
    assert "do-not-leak" not in reply.message


@pytest.mark.parametrize(
    "error,code",
    [(TimeoutError("timeout"), "TIMEOUT"), (ConnectionError("offline"), "PROVIDER_UNREACHABLE")],
)
def test_transient_provider_failures_are_not_invalid_arguments(error, code):
    reply = classify_error(error)
    assert reply.code == code and reply.retryable


def test_native_diagnostic_supplies_safe_reference_fix_and_retry_semantics():
    native = diagnostic(
        ErrorCategory.CONNECTION,
        "Unavailable",
        "provider offline",
        reason="transport failed",
        fixes=("Reconnect after inspecting credentials.",),
        retryable=RetryDisposition.IMMEDIATE,
    ).with_diagnostic_path("errors/exact.json")
    response = classify_error(LambdaForgeError(native))
    assert response.code == "PROVIDER_UNREACHABLE"
    assert response.retryable and response.diagnostic_reference == "errors/exact.json"
    assert response.recommended_action == "Reconnect after inspecting credentials."


def test_capability_intent_search_is_offline_deterministic_and_keeps_negative_capabilities(service):
    first = service.capability_search("Quiero recuperar un estudio Fleet", limit=100)
    second = service.capability_search("Quiero recuperar un estudio Fleet", limit=100)
    assert first.status == "ok" and first.data == second.data
    rows = {row["identifier"]: row for row in first.data["items"]}
    assert rows["fleet.recovery"]["status"] == "unsupported"
    assert rows["fleet.recovery"]["session_enabled"] is False
    publishing = service.capability_search("Quiero publicar un modelo entrenado", limit=100)
    assert any(row["category"] == "products" for row in publishing.data["items"])
    described = service.capability_describe("jobs.get").data
    assert described["introduced_version"] == "0.19.0"
    assert "snapshot" in described["freshness_semantics"]
    for row in capabilities(service):
        Draft202012Validator.check_schema(row["input_schema"])
        Draft202012Validator.check_schema(row["output_schema"])
        assert all(hasattr(service, tool) for tool in row.get("suggested_next_tools", []))


def test_results_adapter_keeps_facade_signatures_native_ownership_and_lazy_reads(
    service, monkeypatch
):
    calls = []
    reader = SimpleNamespace(
        catalog=lambda: [{"execution_id": "execution-exact", "_manifest_path": "private"}],
        view=lambda *args, **kwargs: calls.append((args, kwargs)) or {"state": "failed"},
    )
    monkeypatch.setattr(service, "_results", lambda: reader)
    assert service.list_results().data["items"] == [{"execution_id": "execution-exact"}]
    assert not calls
    reply = service.get_result("execution-exact", run_id="run-exact", attempt=2, points=20)
    assert reply.status == "ok"
    assert calls == [
        (
            ("execution-exact",),
            {
                "view": "overview",
                "run_id": "run-exact",
                "attempt": 2,
                "points": 20,
                "tail": 100,
            },
        )
    ]
    assert service.get_result("display-name").error.code == "INVALID_ARGUMENT"
    assert len(calls) == 1


def test_changed_canonical_note_is_a_stale_approval_not_an_invalid_argument(service):
    operator = AgentServices(service.scope.root, permissions=("READ", "PLAN", "WRITE_PROJECT"))
    operator.knowledge.initialize()
    note = operator.knowledge.create("finding", "Initial", body="original")
    approvals = ApprovalService(operator.scope)
    plan = approvals.request(
        "knowledge.update",
        {
            "note_id": note["metadata"]["id"],
            "revision": note["revision"],
            "body": "proposed",
        },
    )
    approvals.approve_local(
        plan["request_id"],
        plan_hash=plan["plan_hash"],
        channel="console",
        confirmation=plan["request_id"],
    )
    operator.knowledge.update(note["metadata"]["id"], revision=note["revision"], body="external")
    response = operator.apply_agent_action(plan["request_id"], plan["plan_hash"])
    assert response.error.code == "PLAN_STALE"
    assert operator.knowledge.read(note["metadata"]["id"])["body"] == "external"
    assert approvals.inspect(plan["request_id"])["state"] == "approved"
