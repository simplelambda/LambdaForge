"""Exact native design comparison, not counts-only reuse or tolerance-based identities."""

import json
from pathlib import Path

import pytest

from lambdaforge.work import WorkConfig
from lambdaforge.work.extension_design import candidate_id, compare_fixed_designs


def config(root: Path, values, seeds=(11, 23, 47, 59, 71), **updates):
    raw = {
        "name": "extension-design",
        "run": "tests.work_cases.FixedRecoveryWork",
        "seeds": list(seeds),
        "objective": {"metric": "score", "mode": "max"},
        "sweep": {"space": {"choice": values}},
        **updates,
    }
    return WorkConfig.from_mapping(raw, source=root / "study.yaml")


def test_four_by_five_to_seven_by_five_preserves_exact_twenty_cells(tmp_path):
    old = config(tmp_path, [0, 1, 2, 3])
    new = config(tmp_path, [0, 1, 2, 3, 5, 7, 10])
    value = compare_fixed_designs(old, new).to_dict()
    assert value["counts"] == {"reused": 20, "new": 15, "total": 35}
    assert {cell["parameters"]["choice"] for cell in value["new_cells"]} == {5, 7, 10}
    assert not {cell["cell_id"] for cell in value["new_cells"]} & {
        cell["cell_id"] for cell in value["reused_cells"]
    }
    delta = WorkConfig.from_mapping(value["delta_configuration"], source=new.source)
    assert delta.planned_runs == 15
    assert [dict(variant) for variant in delta.levels[0].runs[0].variants] == [
        {"choice": 5},
        {"choice": 7},
        {"choice": 10},
    ]
    assert value["proposed_configuration"] == json.loads(json.dumps(dict(new.raw)))


def test_seed_extension_is_only_new_seeds_and_exact_candidate_ids(tmp_path):
    value = compare_fixed_designs(
        config(tmp_path, [0, 1, 2, 3]),
        config(tmp_path, [0, 1, 2, 3], seeds=(11, 23, 47, 59, 71, 73, 79, 83)),
    ).to_dict()
    assert value["counts"] == {"reused": 20, "new": 12, "total": 32}
    assert value["delta_configuration"]["seeds"] == [73, 79, 83]
    assert {cell["seed"] for cell in value["new_cells"]} == {73, 79, 83}


def test_reorder_and_insert_preserves_original_trial_positions(tmp_path):
    value = compare_fixed_designs(
        config(tmp_path, [0, 2, 4]), config(tmp_path, [4, 1, 0, 2])
    ).to_dict()
    moved = next(cell for cell in value["reused_cells"] if cell["parameters"]["choice"] == 4)
    assert moved["original_trial"] == 3 and moved["proposed_trial"] == 1
    assert candidate_id({"a": 1, "b": 2}) == candidate_id({"b": 2, "a": 1})


def test_one_dimension_multidimensional_rectangle_has_no_old_cells(tmp_path):
    old = config(tmp_path, [0, 1], sweep={"space": {"choice": [0, 1], "failures": [1, 2]}})
    new = config(tmp_path, [0, 1, 2], sweep={"space": {"choice": [0, 1, 2], "failures": [1, 2]}})
    value = compare_fixed_designs(old, new).to_dict()
    assert value["counts"] == {"reused": 20, "new": 10, "total": 30}
    assert {cell["parameters"]["choice"] for cell in value["new_cells"]} == {2}


def test_reorder_only_is_noop(tmp_path):
    value = compare_fixed_designs(
        config(tmp_path, [0, 1, 2]), config(tmp_path, [2, 0, 1])
    ).to_dict()
    assert value["kind"] == "no-op" and value["delta_configuration"] is None
    assert value["new_cells"] == []


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"seeds": [11, 23, 47, 59, 71, 73]}, "one dimension OR"),
        ({"seeds": [23, 11, 47, 59, 71]}, "unchanged prefix"),
        ({"with": {"failures": 2}}, "fixed parameters"),
        ({"objective": {"metric": "other", "mode": "max"}}, "objective"),
    ],
)
def test_incompatible_changes_are_rejected(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        compare_fixed_designs(config(tmp_path, [0, 1]), config(tmp_path, [0, 1, 2], **updates))


def test_two_changed_dimensions_rejected(tmp_path):
    old = config(tmp_path, [], sweep={"space": {"choice": [0], "failures": [1]}})
    new = config(tmp_path, [], sweep={"space": {"choice": [0, 1], "failures": [1, 2]}})
    with pytest.raises(ValueError, match="successive revisions"):
        compare_fixed_designs(old, new)


def test_native_json_values_not_merged_by_numeric_tolerance(tmp_path):
    assert len({candidate_id({"x": value}) for value in (True, 1, 1.0)}) == 3
    old = config(tmp_path, [], run="tests.work_cases.SeedWork", sweep={"space": {"scale": [1]}})
    new = config(
        tmp_path, [], run="tests.work_cases.SeedWork", sweep={"space": {"scale": [1.0, 2.0]}}
    )
    with pytest.raises(ValueError, match="Numeric/type differences"):
        compare_fixed_designs(old, new)


@pytest.mark.parametrize(
    "raw",
    [
        {"search": {"trials": 2, "choice": [0, 1]}, "sweep": None},
        {"replicates": 2, "seeds": None},
    ],
)
def test_unsupported_designs_do_not_become_recovery(tmp_path, raw):
    old = config(tmp_path, [0, 1])
    values = dict(old.raw)
    for key, value in raw.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    unsupported = WorkConfig.from_mapping(values, source=old.source)
    with pytest.raises(ValueError, match="fixed sweeps|explicit seeds"):
        compare_fixed_designs(old, unsupported)


def test_preview_serialized_result_cannot_mutate_immutable_comparison(tmp_path):
    result = compare_fixed_designs(config(tmp_path, [0]), config(tmp_path, [0, 1]))
    result.to_dict()["new_cells"].clear()
    assert len(result.to_dict()["new_cells"]) == 5
