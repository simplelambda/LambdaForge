"""Cross-subsystem contracts for the canonical authored parameter geometry."""

from __future__ import annotations

import math

import pytest

from lambdaforge.hpo.AdaptiveResources import _mixed_distance
from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler
from lambdaforge.hpo.BayesianSampler import BayesianSampler
from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.hpo.SobolSearch import SobolSearch


@pytest.mark.parametrize(
    "values",
    [
        [[1.5, 3.0, 6.0], [1.5, 3.0, 6.0], [1.5, 6.0, 3.0]],
        [{"a": [1, {"b": True}]}, {"a": [1, {"b": True}]}, {"a": [1, {"b": False}]}],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2, "b": 1}],
        [{"flag": True}, {"flag": True}, {"flag": 1}],
    ],
)
def test_structured_observed_values_keep_json_meaning(values) -> None:
    space = ParameterSpace.from_schema(
        {"initialization_profile": {"values": ["baseline", "same_diffusion"]}},
        [{"structured": value} for value in values],
    )
    assert len(space.descriptor("structured").values) == 2
    assert space.distance({"structured": values[0]}, {"structured": values[1]}) == 0
    assert space.distance({"structured": values[0]}, {"structured": values[2]}) > 0
    restored = ParameterSpace.from_schema(space.to_schema())
    assert restored.decode(restored.encode({"structured": values[0]}))["structured"] == values[0]


def test_log_and_linear_distances_follow_the_authored_topology() -> None:
    logarithmic = ParameterSpace.from_schema(
        {"learning_rate": {"range": [1e-5, 1e-3], "scale": "log"}}
    )
    assert math.isclose(
        logarithmic.distance({"learning_rate": 1e-5}, {"learning_rate": 1e-4}),
        logarithmic.distance({"learning_rate": 1e-4}, {"learning_rate": 1e-3}),
    )

    linear = ParameterSpace.from_schema({"x": {"range": [0.0, 10.0]}})
    assert linear.distance({"x": 0.0}, {"x": 1.0}) < linear.distance({"x": 1.0}, {"x": 10.0})


def test_integer_categorical_and_conditional_semantics_are_explicit() -> None:
    space = ParameterSpace.from_schema(
        {
            "family": {"values": ["plain", "attention"]},
            "layers": {"values": [1, 2, 3, 4], "type": "int"},
            "heads": {"values": [2, 8], "type": "int", "when": {"family": "attention"}},
        }
    )
    plain = {"family": "plain", "layers": 1}
    same_plain = {"family": "plain", "layers": 1}
    attention_two = {"family": "attention", "layers": 1, "heads": 2}
    attention_eight = {"family": "attention", "layers": 1, "heads": 8}

    assert space.active_mask(plain) == (True, True, False)
    assert space.distance(plain, same_plain) == 0.0
    assert space.distance(plain, attention_two) > 0.0
    assert space.distance(attention_two, attention_eight) > 0.0
    assert space.normalize("layers", 2) < space.normalize("layers", 4)
    assert space.decode(space.encode(plain)) == plain
    assert space.decode(space.encode(attention_eight)) == attention_eight
    restored = ParameterSpace.from_schema(space.to_schema())
    assert restored.encode(attention_eight) == space.encode(attention_eight)


def test_integer_sampling_uses_equal_discrete_cells() -> None:
    descriptor = ParameterSpace.from_schema(
        {"layers": {"range": [1, 4], "type": "int"}}
    ).descriptor("layers")

    assert [descriptor.sample(value) for value in (0.0, 0.249, 0.25, 0.5, 0.999)] == [
        1,
        1,
        2,
        3,
        4,
    ]


def test_all_sampler_and_resource_paths_share_one_geometry() -> None:
    schema = {
        "learning_rate": {"range": [1e-5, 1e-3], "scale": "log"},
        "family": {"values": ["a", "b"]},
    }
    candidates = {
        1: {"learning_rate": 1e-5, "family": "a"},
        2: {"learning_rate": 1e-4, "family": "a"},
        3: {"learning_rate": 1e-3, "family": "b"},
    }
    space = ParameterSpace.from_schema(schema, tuple(candidates.values()))
    adaptive = AdaptiveSampler(candidates, mode="max", parameter_space=space)
    bayesian = BayesianSampler(candidates, mode="max", parameter_space=space)

    expected = space.distance(candidates[1], candidates[2])
    assert adaptive._distance(candidates[1], candidates[2]) == expected
    assert _mixed_distance(candidates[1], candidates[2], space) == expected
    assert bayesian._vectors[2] == space.encode(candidates[2])

    generated = tuple(trial.parameters for trial in SobolSearch(schema, seed=7).trials(4))
    assert all(1e-5 <= point["learning_rate"] <= 1e-3 for point in generated)
    assert {point["family"] for point in generated} <= {"a", "b"}
