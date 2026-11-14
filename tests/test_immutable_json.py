"""Immutable JSON construction obeys dict/list reconstruction protocols."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pytest

from lambdaforge.ImmutableJson import FrozenJsonList, FrozenJsonMapping


@dataclass
class Evidence:
    parameters: dict


@pytest.mark.parametrize("kind", ["mapping", "pairs", "generator"])
def test_nested_frozen_containers_reconstruct_without_weakening_immutability(kind):
    plain = {"sources": ["atom_name", {"channels": [1, 2, None, False]}]}
    values = (
        plain
        if kind == "mapping"
        else (list(plain.items()) if kind == "pairs" else iter(plain.items()))
    )
    frozen = FrozenJsonMapping(values)
    converted = asdict(Evidence({"atomic_representation": frozen}))
    assert json.loads(json.dumps(converted, allow_nan=False)) == {
        "parameters": {"atomic_representation": plain}
    }
    rebuilt = converted["parameters"]["atomic_representation"]
    assert rebuilt is not frozen
    assert isinstance(rebuilt, FrozenJsonMapping)
    assert isinstance(rebuilt["sources"], FrozenJsonList)
    assert isinstance(rebuilt["sources"][1], FrozenJsonMapping)
    for mapping in (frozen, rebuilt):
        with pytest.raises(TypeError, match="immutable"):
            mapping["new"] = 1
        with pytest.raises(TypeError, match="immutable"):
            mapping["sources"].append("new")
        with pytest.raises(TypeError, match="immutable"):
            mapping["sources"][1]["channels"][0] = 9
    assert frozen == plain


def test_empty_and_duplicate_pair_protocol_matches_dict():
    assert FrozenJsonMapping() == FrozenJsonMapping(None) == FrozenJsonMapping(iter(())) == {}
    value = FrozenJsonMapping(iter((("x", [1]), ("x", [2]))))
    assert value == {"x": [2]}
    with pytest.raises((TypeError, ValueError)):
        FrozenJsonMapping(iter((1,)))
