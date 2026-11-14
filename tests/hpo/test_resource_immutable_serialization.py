"""Nested immutable Work parameters survive the actual resource telemetry persistence path."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    AdmissionDecision,
    CandidateResourceAction,
    GPUResourceState,
    ResourceHistoryStore,
    ResourceTrajectorySample,
    ScientificActionValue,
    resource_identity,
)
from lambdaforge.ImmutableJson import FrozenJsonList, FrozenJsonMapping
from tests.hpo.test_adaptive_resources import observation, prediction

PLAIN = {
    "atomic_representation": {"sources": ["atom_name", {"levels": [0, 1, None, True]}]},
    "scale": 0.015,
    "pooling": ["mean", "sum"],
}


@pytest.fixture(params=["frozen", "mixed", "proxy"])
def parameters(request):
    if request.param == "frozen":
        return FrozenJsonMapping(PLAIN)
    if request.param == "proxy":
        return MappingProxyType(
            {
                "atomic_representation": MappingProxyType(
                    {"sources": FrozenJsonList(PLAIN["atomic_representation"]["sources"])}
                ),
                "scale": 0.015,
                "pooling": FrozenJsonList(["mean", "sum"]),
            }
        )
    return {
        "atomic_representation": FrozenJsonMapping(PLAIN["atomic_representation"]),
        "scale": 0.015,
        "pooling": FrozenJsonList(["mean", "sum"]),
    }


def active(parameters):
    return ActiveResourceEvidence(
        candidate_key="candidate",
        compatibility_key="compatible",
        parameters=parameters,
        hardware="H200",
        total_bytes=140,
        current_bytes=30,
        running_peak_bytes=35,
        future_peak_samples=(35, 140),
        future_peak_weights=(0.99, 0.01),
        resource_state="PROVISIONALLY_STABLE",
        elapsed_seconds=89,
        phase="evaluation",
        step=89,
        trajectory=(ResourceTrajectorySample(89, 30, step=89, phase="evaluation"),),
        trajectory_statistics=FrozenJsonMapping({"phase_counts": {"evaluation": 1}}),
    )


def assert_builtin(value):
    if isinstance(value, Mapping):
        assert type(value) is dict
        for item in value.values():
            assert_builtin(item)
    elif isinstance(value, (list, tuple)):
        assert type(value) is list
        for item in value:
            assert_builtin(item)


def test_persist_active_round_trip_preserves_parameters_and_identity(tmp_path, parameters):
    evidence = active(parameters)
    specification = {
        "parameters": parameters,
        "trial_parameters": parameters,
        "definition": {"work_class": "project.Train"},
    }
    identity_args = dict(code_fingerprint="code", environment_fingerprint="env", hardware="H200")
    before = resource_identity(specification, **identity_args)
    store = ResourceHistoryStore(tmp_path)
    store.persist_active((evidence,))
    payload = json.loads((tmp_path / "active-evidence.json").read_text())
    assert payload["active_evidence_version"] == 2
    assert payload["items"][0]["parameters"] == PLAIN
    assert payload["items"][0]["step"] == 89
    assert payload["items"][0]["phase"] == "evaluation"
    assert_builtin(evidence.to_dict())
    assert json.dumps(payload, allow_nan=False)
    restored = store.load_active()[0]
    assert restored.parameters == PLAIN
    assert restored.to_dict() == evidence.to_dict()
    after = {
        **specification,
        "parameters": restored.parameters,
        "trial_parameters": restored.parameters,
    }
    assert resource_identity(after, **identity_args) == before
    output = evidence.to_dict()
    output["parameters"]["atomic_representation"]["sources"].append("changed-copy")
    assert evidence.parameters == PLAIN
    with pytest.raises(TypeError):
        evidence.parameters["atomic_representation"]["sources"].append("mutate-original")
    assert not list(tmp_path.glob("*.tmp"))


def test_terminal_observation_decisions_and_nested_ledger_use_portable_boundary(
    tmp_path, parameters
):
    store = ResourceHistoryStore(tmp_path)
    terminal = replace(
        observation("candidate", 35),
        parameters=parameters,
        fixed_arguments=FrozenJsonMapping({"configuration": parameters}),
        trajectory=(FrozenJsonMapping({"step": 89, "details": parameters}),),
    )
    store.append(terminal)
    restored = store.load()[0]
    assert restored.parameters == PLAIN
    assert restored.fixed_arguments == {"configuration": PLAIN}
    assert restored.to_dict() == terminal.to_dict()
    assert_builtin(terminal.to_dict())

    pred = prediction("candidate", 35)
    resident = ActiveResourceCommitment(
        "resident", 40, phase_hazards=FrozenJsonMapping({"evaluation": 0.1})
    )
    device = GPUResourceState(0, "granted", "H200", 140, 100, 0, (resident,), 4)
    decision = AdmissionDecision(
        "candidate",
        1,
        "ADMITTED",
        0,
        "safe",
        1.0,
        FrozenJsonMapping(pred.to_dict()),
        (FrozenJsonMapping({"details": parameters}),),
        0.99,
        10.0,
    )
    assert_builtin(decision.to_dict())
    store.record_decisions((decision,))
    store.persist_ledger((device,))
    store.record_trace(
        elapsed_seconds=89,
        devices=(device,),
        actions=(
            CandidateResourceAction(
                "candidate", {}, 1.0, pred, value_contract=ScientificActionValue(rank=0)
            ),
        ),
        decisions=(decision,),
    )
    for path in tmp_path.iterdir():
        if path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                json.loads(line)


def test_plain_resource_json_wire_format_is_unchanged():
    evidence = active(PLAIN)
    assert json.loads(json.dumps(asdict(evidence))) == evidence.to_dict()
    terminal = observation("candidate", 35, parameters=PLAIN)
    assert json.loads(json.dumps(asdict(terminal))) == terminal.to_dict()


def test_invalid_telemetry_is_not_silently_ignored(tmp_path: Path):
    store = ResourceHistoryStore(tmp_path)
    store.persist_active((active(PLAIN),))
    previous = (tmp_path / "active-evidence.json").read_bytes()
    invalid = active({"unsupported": object()})
    with pytest.raises(TypeError, match="JSON serializable"):
        store.persist_active((invalid,))
    assert (tmp_path / "active-evidence.json").read_bytes() == previous
