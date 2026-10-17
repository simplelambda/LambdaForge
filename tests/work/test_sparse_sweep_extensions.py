"""Exact sparse/conditional design authority; no consumer project or real GPU work."""

import copy
import json

import pytest

from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.extension_design import candidate_id, compare_fixed_designs


def document():
    return {
        "name": "conditional-extension",
        "run": "tests.work_cases.MembershipSweepWork",
        "seeds": [4, 7, 32, 54],
        "objective": {"metric": "score", "mode": "max"},
        "sweep": {
            "reference": {"pooling_type": "max"},
            "space": {
                "pooling_type": ["max", "mean", "log_sum_exp"],
                "area_mode": {
                    "values": ["point", "area"],
                    "when": {"pooling_type": {"in": ["mean", "log_sum_exp"]}},
                },
                "beta": {"values": [1, 2], "when": {"pooling_type": "log_sum_exp"}},
            },
        },
    }


def parse(raw, tmp_path):
    return WorkConfig.from_mapping(raw, source=tmp_path / "original.yaml")


def test_conditional_value_extension_does_not_repeat_inactive_controls(tmp_path):
    old = document()
    new = copy.deepcopy(old)
    new["sweep"]["space"]["beta"]["values"].append(3)
    result = compare_fixed_designs(parse(old, tmp_path), parse(new, tmp_path)).to_dict()
    assert result["counts"] == {"reused": 28, "new": 8, "total": 36}
    delta = parse(result["delta_configuration"], tmp_path)
    points = list(delta.levels[0].runs[0].variants)
    assert points == [
        {"pooling_type": "log_sum_exp", "area_mode": mode, "beta": 3}
        for mode in ("point", "area")
    ]
    assert delta.planned_runs == 8
    assert delta.levels[0].runs[0].study_design.reference is None
    assert all("beta" not in row["parameters"] for row in result["reused_cells"]
               if row["parameters"]["pooling_type"] == "max")


def test_selective_include_is_union_not_cartesian_and_preserves_geometry(tmp_path):
    old = document()
    new = copy.deepcopy(old)
    selected = {"pooling_type": "log_sum_exp", "area_mode": "area", "beta": 3}
    new["sweep"]["include"] = [{"pooling_type": "max"}, selected]
    parsed = parse(new, tmp_path)
    definition = parsed.levels[0].runs[0]
    assert len(definition.variants) == 8  # No beta=3 for point, no duplicate max.
    geometry = ParameterSpace.from_schema(definition.study_design.space)
    assert all(geometry.valid(point) for point in definition.variants)
    assert geometry.descriptor("beta").conditional
    result = compare_fixed_designs(parse(old, tmp_path), parsed).to_dict()
    assert result["counts"] == {"reused": 28, "new": 4, "total": 32}
    assert parse(result["delta_configuration"], tmp_path).planned_runs == 4
    assert result["delta_configuration"]["sweep"]["candidates"] == [selected]
    # Identity and generated plan survive native frozen-configuration serialization.
    restored = parse(json.loads(json.dumps(new)), tmp_path)
    assert [candidate_id(p) for p in restored.levels[0].runs[0].variants] == [
        candidate_id(p) for p in definition.variants
    ]


@pytest.mark.parametrize("base,added", [(15, 8), (12, 7), (12, 8), (24, 7)])
def test_reported_count_shapes_are_exact_without_application_specific_rules(tmp_path, base, added):
    old = document()
    old["sweep"]["space"] = {
        "pooling_type": ["max", "mean"],
        "area_mode": {"values": [str(i) for i in range(base - 1)],
                      "when": {"pooling_type": "mean"}},
    }
    new = copy.deepcopy(old)
    new["sweep"]["include"] = [
        {"pooling_type": "mean", "area_mode": f"extra-{i}"} for i in range(added)
    ]
    result = compare_fixed_designs(parse(old, tmp_path), parse(new, tmp_path)).to_dict()
    assert result["counts"] == {
        "reused": base * 4, "new": added * 4, "total": (base + added) * 4
    }


@pytest.mark.parametrize("point,match", [
    ({"pooling_type": "max", "beta": 3}, "inactive"),
    ({"pooling_type": "log_sum_exp", "beta": 3}, "active"),
    ({"pooling_type": "max", "typo": 3}, "unknown"),
    ({"pooling_type": "max", "beta": [1, 2]}, "scalars"),
])
def test_include_never_guesses_missing_or_inactive_parameters(tmp_path, point, match):
    raw = document()
    raw["sweep"]["include"] = [point]
    with pytest.raises(ValueError, match=match):
        parse(raw, tmp_path)


def test_candidate_selection_validates_exact_types_domains_duplicates_and_modes(tmp_path):
    for points, match in [
        ([{"pooling_type": "mean", "area_mode": "absent"}], "domain"),
        ([{"pooling_type": "log_sum_exp", "area_mode": "area", "beta": 1.0}], "domain"),
        ([{"pooling_type": "max"}] * 2, "duplicate"),
        ([], "non-empty"),
    ]:
        raw = document()
        raw["sweep"]["candidates"] = points
        with pytest.raises(ValueError, match=match):
            parse(raw, tmp_path)
    raw = document()
    raw["sweep"].update(include=[{"pooling_type": "max"}], candidates=[{"pooling_type": "max"}])
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse(raw, tmp_path)


def test_new_joint_candidate_with_no_new_marginal_value_still_executes(tmp_path):
    old = document()
    old["sweep"]["candidates"] = [
        {"pooling_type": "max"},
        {"pooling_type": "log_sum_exp", "area_mode": "area", "beta": 1},
        {"pooling_type": "log_sum_exp", "area_mode": "point", "beta": 2},
    ]
    new = copy.deepcopy(old)
    new["sweep"]["candidates"].append(
        {"pooling_type": "log_sum_exp", "area_mode": "area", "beta": 2}
    )
    result = compare_fixed_designs(parse(old, tmp_path), parse(new, tmp_path)).to_dict()
    assert result["added_values"] == {} and result["kind"] == "candidates"
    assert result["counts"] == {"reused": 12, "new": 4, "total": 16}


def test_condition_changes_or_candidate_removal_remain_incompatible(tmp_path):
    old = document()
    new = copy.deepcopy(old)
    new["sweep"]["space"]["beta"]["when"] = {"pooling_type": "mean"}
    with pytest.raises(ValueError, match="when contract"):
        compare_fixed_designs(parse(old, tmp_path), parse(new, tmp_path))
    new = copy.deepcopy(old)
    new["sweep"]["candidates"] = [{"pooling_type": "max"}]
    with pytest.raises(ValueError, match="Old normalized candidates"):
        compare_fixed_designs(parse(old, tmp_path), parse(new, tmp_path))


def test_sparse_native_mode_does_not_enumerate_a_cartesian_pool(tmp_path, monkeypatch):
    import importlib

    config = importlib.import_module("lambdaforge.work.config")

    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse selection must not enumerate a Cartesian pool")

    monkeypatch.setattr(config, "_exhaustive_variants", forbidden)
    raw = document()
    raw["sweep"]["candidates"] = [{"pooling_type": "max"}]
    assert parse(raw, tmp_path).planned_runs == 4


def test_schema_accepts_both_native_modes_but_not_their_combination():
    from pathlib import Path

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).parents[2]
                         / "src/lambdaforge/schemas/work.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for mode in ("include", "candidates"):
        raw = document()
        raw["sweep"][mode] = [{"pooling_type": "max"}]
        validator.validate(raw)
    raw["sweep"]["include"] = [{"pooling_type": "max"}]
    assert list(validator.iter_errors(raw))
