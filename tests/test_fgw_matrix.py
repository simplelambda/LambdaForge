"""Actual POT matrix blocks, explicit partial failure, identity and bounded allocations."""

import numpy as np
import pytest

from lambdaforge.distances import (
    FGWMatrixBuilder,
    FusedGromovWassersteinDistance,
    StructuredMeasure,
)

pytest.importorskip("ot")


@pytest.fixture
def objects():
    return tuple(
        StructuredMeasure.from_points(
            np.arange(n)[:, None], features=np.arange(n)[:, None] / n, weights=np.arange(1, n + 1)
        )
        for n in (2, 3, 4)
    )


def test_blocks_square_rectangular_and_sequential_equivalence(objects):
    distance = FusedGromovWassersteinDistance(alpha=0.4)
    builder = FGWMatrixBuilder(distance, block_size=2)
    result = builder.build(objects, symmetric=True)
    assert result.complete and result.symmetric and len(result.blocks) == 3
    assert result.values == pytest.approx(distance.pairwise(objects, symmetric=True))
    assert result.values == pytest.approx(result.values.T)
    assert not result.values.flags.writeable
    blocks = tuple(builder.blocks(objects, symmetric=True))
    assert [pair for block in blocks for pair in block.pairs] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    assert all(block.complete for block in blocks)
    rectangular = builder.build(objects[:1], objects[1:])
    assert rectangular.values.shape == (1, 2) and not rectangular.symmetric
    assert rectangular.values[0, 1] == pytest.approx(distance(objects[0], objects[2]))
    directed = builder.build(objects)
    assert directed.values == pytest.approx(distance.pairwise(objects))
    with pytest.raises(ValueError, match="square"):
        builder.build(objects, objects, symmetric=True)


def test_partial_errors_remain_nan_and_are_not_success(objects, monkeypatch):
    distance = FusedGromovWassersteinDistance()
    original = distance.solve

    def broken(left, right):
        if left is objects[0] and right is objects[1]:
            raise RuntimeError("deliberate failure")
        return original(left, right)

    monkeypatch.setattr(distance, "solve", broken)
    with pytest.raises(RuntimeError, match="deliberate"):
        FGWMatrixBuilder(distance).build(objects)
    partial = FGWMatrixBuilder(distance, error_policy="record", block_size=2).build(objects)
    assert not partial.complete and np.isnan(partial.values[0, 1])
    assert np.isfinite(partial.values[1, 0])
    assert partial.blocks[0]["failed"] == 1
    assert partial.blocks[0]["errors"][0]["pair"] == [0, 1]


def test_identity_changes_with_objects_parameters_and_symmetry(objects):
    builder = FGWMatrixBuilder(FusedGromovWassersteinDistance(), block_size=1)
    identity = next(builder.blocks(objects)).identity
    assert identity == next(builder.blocks(objects)).identity
    assert identity != next(builder.blocks(objects[::-1])).identity
    assert identity != next(builder.blocks(objects, symmetric=True)).identity
    assert (
        identity
        != next(
            FGWMatrixBuilder(FusedGromovWassersteinDistance(alpha=0.7)).blocks(objects)
        ).identity
    )
    changed = (
        StructuredMeasure(objects[0].structure, features=objects[0].features + 1),
        *objects[1:],
    )
    assert identity != next(builder.blocks(changed)).identity


def test_guards_and_opaque_feature_identity(objects):
    with pytest.raises(MemoryError, match="workspace"):
        FGWMatrixBuilder(FusedGromovWassersteinDistance(), max_memory_bytes=1).build(objects)
    distance = FusedGromovWassersteinDistance(feature_metric=lambda a, b: abs(a["x"] - b["x"]))
    obj = StructuredMeasure([[0.0, 1.0], [1.0, 0.0]], [{"x": 0}, {"x": 2}])
    with pytest.raises(ValueError, match="metric_identity"):
        next(FGWMatrixBuilder(distance).blocks([obj]))
    with pytest.raises(ValueError, match="feature_encoder"):
        next(FGWMatrixBuilder(distance, metric_identity="absolute-dict-x/v1").blocks([obj]))
    import json

    result = FGWMatrixBuilder(
        distance,
        metric_identity="absolute-dict-x/v1",
        feature_encoder=lambda value: json.dumps(value, sort_keys=True).encode(),
    ).build([obj])
    assert result.complete and result.values[0, 0] == 0
