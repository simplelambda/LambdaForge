"""One scientific design, independent of dependency order and inactive branches."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from lambdaforge.analysis.SearchSpace import build_space, condition_active, valid_point
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.TargetCapacity import check_target_capacity, validate_gpu_capacity
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.hpo.ActivationCondition import ActivationCondition
from lambdaforge.hpo.CandidateGenerator import DeterministicCandidateGenerator
from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.hpo.RandomSearch import RandomSearch
from lambdaforge.work.config import WorkConfig, _exhaustive_variants, _validate_sweep_reference
from lambdaforge.work.runner import WorkRunner


def joint_space() -> dict[str, Any]:
    # Children deliberately precede their parent. No consumer-domain code is imported.
    return {
        "area_mode": {
            "values": ["point", "area"],
            "when": {"pooling_type": {"in": ["mean", "attention", "topk", "log_sum_exp"]}},
        },
        "attention_hidden": {"values": [8, 16, 32, 64, 128], "when": {"pooling_type": "attention"}},
        "topk_fraction": {
            "values": [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4],
            "when": {"pooling_type": "topk"},
        },
        "local_scale": {
            "values": [0, 0.5, 1, 1.5, 2.5, 4, 6, 8, 12],
            "when": {"pooling_type": "local_mean_max"},
        },
        "beta": {
            "values": [0.5, 1, 2, 5, 10, 20, 40, 80, 160],
            "when": {"pooling_type": "log_sum_exp"},
        },
        "pooling_type": {
            "values": ["max", "mean", "attention", "topk", "local_mean_max", "log_sum_exp"]
        },
    }


def test_exact_conditional_56_candidates_and_224_required_runs(tmp_path: Path) -> None:
    space = joint_space()
    points = _exhaustive_variants(space)
    assert len(points) == 56
    assert len({json.dumps(p, sort_keys=True) for p in points}) == 56
    assert [p for p in points if p["pooling_type"] == "max"] == [{"pooling_type": "max"}]
    counts = {
        name: sum(p["pooling_type"] == name for p in points)
        for name in space["pooling_type"]["values"]
    }
    assert list(counts.values()) == [1, 2, 10, 16, 9, 18]
    # Equality partitioning describes exactly the same intended scientific set.
    partitioned = []
    for parent in space["pooling_type"]["values"]:
        branch = copy.deepcopy(space)
        branch["pooling_type"]["values"] = [parent]
        branch = {
            name: rule
            for name, rule in branch.items()
            if name == "pooling_type"
            or ActivationCondition.parse(rule.get("when")).matches({"pooling_type": parent})
        }
        for rule in branch.values():
            rule.pop("when", None)
        partitioned.extend(_exhaustive_variants(branch))
    assert {json.dumps(p, sort_keys=True) for p in points} == {
        json.dumps(p, sort_keys=True) for p in partitioned
    }
    config = WorkConfig.from_mapping(
        {
            "name": "joint",
            "run": "tests.work_cases.MembershipSweepWork",
            "seeds": [4, 7, 32, 54],
            "sweep": {"space": space, "reference": {"pooling_type": "max"}},
            "resources": {"gpu": 2, "time": "168h"},
            "execution": {"max_parallel": 10, "max_time": "24h"},
        },
        source=tmp_path / "joint.yaml",
    )
    assert config.planned_runs == 224
    assert len(config.levels) == 1
    facts = config.preflight()["studies"][0]
    assert facts["required_runs"] == 224
    assert facts["study_time_budget_seconds"] == 86400
    assert facts["scheduler_wall_time_seconds"] == 604800
    assert [item["candidates"] for item in facts["conditional_branches"]["counts"]] == [
        1,
        2,
        10,
        16,
        9,
        18,
    ]
    assert config.levels[0].runs[0].study_design is not None
    assert config.levels[0].runs[0].study_design.evidence.to_dict()["required_run_count"] == 224
    assert WorkRunner().plan(config).preflight["required_runs"] == 224


def test_equality_compatibility_and_canonical_membership_identity(tmp_path: Path) -> None:
    scalar = ActivationCondition.parse({"parent": "a"})
    assert scalar == ActivationCondition.parse({"parent": {"eq": "a"}})
    assert scalar.to_mapping() == {"parent": "a"}
    first = ActivationCondition.parse({"parent": {"in": ["b", "a"]}, "flag": True})
    assert first == ActivationCondition.parse({"flag": True, "parent": {"in": ["a", "b"]}})
    assert first.matches({"parent": "a", "flag": True})
    assert not first.matches({"parent": "a"})
    assert pickle.loads(pickle.dumps(first)) == first
    base = {
        "name": "stable",
        "run": "tests.work_cases.ConditionalSweepWork",
        "seeds": [4],
        "sweep": {
            "space": {"model": ["a", "b"], "depth": {"values": [1, 2], "when": {"model": "a"}}}
        },
    }
    before = WorkConfig.from_mapping(base, source=tmp_path / "stable.yaml")
    base["sweep"]["space"]["depth"]["when"]["model"] = {"eq": "a"}
    after = WorkConfig.from_mapping(base, source=tmp_path / "stable.yaml")
    assert WorkRunner._study_identity(before) == WorkRunner._study_identity(after)


@pytest.mark.parametrize(
    "condition",
    [
        {"parent": []},
        {"parent": {"in": []}},
        {"parent": {"in": ["a", "a"]}},
        {"parent": {"not": "a"}},
        {"parent": {"in": ["a"], "eq": "a"}},
    ],
)
def test_invalid_grammar(condition: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ActivationCondition.parse(condition)


@pytest.mark.parametrize(
    "space,match",
    [
        ({"child": {"values": [1], "when": {"absent": "a"}}}, "missing parent"),
        ({"parent": ["a"], "child": {"values": [1], "when": {"parent": {"in": ["b"]}}}}, "outside"),
        (
            {"a": {"values": [1], "when": {"b": 1}}, "b": {"values": [1], "when": {"a": 1}}},
            "Cyclic",
        ),
    ],
)
def test_invalid_dependencies_and_domains(space: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ParameterSpace.from_schema(space)


def test_common_geometry_generation_encoding_and_analysis() -> None:
    authored = joint_space()
    geometry = ParameterSpace.from_schema(authored)
    points = DeterministicCandidateGenerator(authored).prefix(20)
    assert all(geometry.valid(p) for p in points)
    for point in points:
        assert geometry.decode(geometry.encode(point)) == point
    scientific = build_space([{"parameters": p} for p in points], authored)
    assert all(valid_point(p, scientific) for p in points)
    assert not condition_active(scientific["area_mode"], {"pooling_type": "max"})
    assert condition_active(scientific["area_mode"], {"pooling_type": "mean"})
    random = RandomSearch(
        {name: {**rule, "type": "choice"} for name, rule in authored.items()}
    ).trials(10)
    assert all(geometry.valid(p.parameters) for p in random)
    restored = ParameterSpace.from_schema(json.loads(json.dumps(geometry.to_schema())))
    assert restored.to_schema() == geometry.to_schema()
    with pytest.raises(TypeError):
        geometry.descriptor("area_mode").when["pooling_type"]["in"].append("max")  # type: ignore[index]


def test_reference_and_composition_preflight(tmp_path: Path) -> None:
    points = _exhaustive_variants(joint_space())
    _validate_sweep_reference({"pooling_type": "max"}, points)
    for reference in ({"pooling_type": "missing"}, {"pooling_type": "mean"}):
        with pytest.raises(ValueError):
            _validate_sweep_reference(reference, points)
    steps = [{"name": f"s{i}", "run": "tests.work_cases.SeedWork"} for i in range(6)]
    source = tmp_path / "steps.yaml"
    sequential = WorkConfig.from_mapping(
        {"name": "sequence", "steps": steps, "resources": {"time": "168h"}}, source=source
    )
    assert sequential.resources.runtime_seconds == 1008 * 3600
    parallel = WorkConfig.from_mapping(
        {
            "name": "parallel",
            "steps": [{"parallel": steps[:2]}, *steps[2:4]],
            "resources": {"time": "24h"},
        },
        source=source,
    )
    assert parallel.resources.runtime_seconds == 72 * 3600
    for field, value in {
        "with": {},
        "seeds": [1],
        "replicates": 2,
        "search": {},
        "sweep": {},
        "execution": {},
        "objective": "score",
        "analysis": {},
    }.items():
        with pytest.raises(ValueError, match="only root resources"):
            WorkConfig.from_mapping(
                {"name": "invalid", "steps": steps, field: value}, source=source
            )


def test_capacity_is_not_occupancy_or_an_unknown_guess() -> None:
    with pytest.raises(ValueError, match="Requested 3 GPUs"):
        validate_gpu_capacity(3, {"reliable": True, "allocatable_gpus": 2})
    validate_gpu_capacity(2, {"reliable": True, "allocatable_gpus": 2})
    validate_gpu_capacity(3, {"reliable": False, "allocatable_gpus": 2})
    validate_gpu_capacity(3, {})


def test_capacity_rejects_before_bundle_or_scientific_submission(tmp_path: Path) -> None:
    import yaml

    source = tmp_path / "request.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "capacity",
                "run": "tests.work_cases.SeedWork",
                "resources": {"gpu": 3},
            }
        )
    )
    profile = ClusterProfile(
        "synthetic",
        transport="ssh",
        host="never-contact.invalid",
        workspace="/synthetic",
    )
    transport = Mock(spec=Transport)
    transport.run.return_value = CommandResult(
        0, json.dumps({"reliable": True, "allocatable_gpus": 2})
    )
    factory = Mock(spec=ControlPlaneFactory)
    factory.transport.return_value = transport
    jobs = Mock(spec=JobService)
    bundles = Mock(spec=ExecutionBundleBuilder)
    plane = ControlPlane(ClusterCatalog({profile.name: profile}), jobs, bundles, factory)
    with pytest.raises(ValueError, match="Requested 3 GPUs"):
        plane.submit(source, cluster=profile.name, dry_run=True)
    jobs.submit.assert_not_called()
    bundles.build.assert_not_called()
    assert transport.run.call_count == 1
    command = transport.run.call_args.args[0]
    assert command[:2] == (profile.python, "-c")
    assert "CUDA_VISIBLE_DEVICES')" in command[2]
    assert "claim" not in command[2]
    assert "import torch" not in command[2]


def test_unknown_and_site_capacity_are_not_invented() -> None:
    transport = Mock(spec=Transport)
    transport.run.side_effect = TimeoutError("Unavailable")
    assert check_target_capacity(ClusterProfile("direct"), transport, 3)["reliable"] is False
    transport.run.reset_mock()
    for profile in (
        ClusterProfile("site", gpu_access=GpuAccessPolicy("command", ("gpu", "exec"))),
        ClusterProfile("slurm", scheduler="slurm"),
    ):
        assert check_target_capacity(profile, transport, 3)["allocatable_gpus"] is None
    transport.run.assert_not_called()


def test_membership_identity_and_schema_round_trip(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    base = {
        "name": "stable",
        "run": "tests.work_cases.ConditionalSweepWork",
        "seeds": [4],
        "sweep": {
            "space": {
                "model": ["a", "b"],
                "depth": {"values": [1, 2], "when": {"model": {"in": ["b", "a"]}}},
            }
        },
    }
    schema = json.loads(
        (Path(__file__).parents[2] / "src/lambdaforge/schemas/work.schema.json").read_text()
    )
    jsonschema.validate(base, schema)
    before = WorkConfig.from_mapping(base, source=tmp_path / "stable.yaml")
    base["sweep"]["space"]["depth"]["when"]["model"]["in"].reverse()
    after = WorkConfig.from_mapping(base, source=tmp_path / "stable.yaml")
    assert WorkRunner._study_identity(before) == WorkRunner._study_identity(after)
    jsonschema.validate(after.to_dict(), schema)
    assert after.levels[0].runs[0].variants == before.levels[0].runs[0].variants


def test_observations_do_not_redefine_an_authored_numeric_parent_domain() -> None:
    authored = {
        "parent": {"range": [0, 1]},
        "child": {"values": [1, 2], "when": {"parent": 0.5}},
    }
    geometry = ParameterSpace.from_schema(authored, [{"parent": 0.25}, {"parent": 0.75}])
    assert geometry.descriptor("parent").values == ()
    assert geometry.descriptor("child").active({"parent": 0.5})
    assert geometry.descriptor("parent").sample(0.4) == 0.4
    authored["child"]["when"]["parent"] = {"in": [0.25]}  # type: ignore[index,assignment]
    with pytest.raises(ValueError, match="explicit finite domain"):
        ParameterSpace.from_schema(authored, [{"parent": 0.25}, {"parent": 0.75}])
