"""ETL and classical estimators are ordinary consumer Python, not node control engines."""

import json

import numpy as np
import pytest

from lambdaforge.catalog import ContractError
from lambdaforge.estimators import SklearnEstimator
from lambdaforge.etl import JoinRecords, PartitionRecords, SelectFields, TrainStandardizer


def test_join_cardinality_partitions_and_selection():
    left = [{"id": "x", "value": 1, "partitions": {"split": "train"}}]
    right = [{"id": "x", "context": 3, "partitions": {"split": "train"}}]
    result = list(JoinRecords()(left, right))
    assert list(SelectFields({"feature": "value"})(result))[0]["id"] == "x"
    with pytest.raises(ContractError) as error:
        list(JoinRecords()(left, [*right, *right]))
    assert error.value.code == "JOIN_CARDINALITY"
    right[0]["partitions"]["split"] = "test"
    with pytest.raises(ContractError) as error:
        list(JoinRecords()(left, right))
    assert error.value.code == "SPLIT_LEAKAGE"


def test_grouped_split_stable_under_input_reordering():
    records = [{"id": str(i), "group": i // 2} for i in range(20)]
    op = PartitionRecords({"train": 0.6, "validation": 0.4}, seed=7, group="group")
    a, b = list(op(records)), list(op(reversed(records)))
    assert {v["id"]: v["partitions"] for v in a} == {v["id"]: v["partitions"] for v in b}
    assert all(a[i]["partitions"] == a[i + 1]["partitions"] for i in range(0, 20, 2))


def test_train_only_fold_state_no_refit_or_test_leakage():
    scaler = TrainStandardizer().fit(
        [[1.0, 2.0], [3.0, 6.0]], ids=["a", "b"], splits=["train"] * 2, fold="cv-1"
    )
    state = scaler.state_dict()
    assert scaler.transform([[101.0, 102.0]], fold="cv-1")[0, 0] == 99
    assert scaler.state_dict() == state
    restored = TrainStandardizer.from_state_dict(json.loads(json.dumps(state)))
    np.testing.assert_equal(restored.transform([[2.0, 4.0]], fold="cv-1"), [[0.0, 0.0]])
    with pytest.raises(ContractError):
        restored.transform([[2.0, 4.0]], fold="cv-2")
    with pytest.raises(ContractError):
        TrainStandardizer().fit([[1.0]], ids=["test"], splits=["test"])
    with pytest.raises(ContractError):
        scaler.fit([[1.0]], ids=["a"], splits=["train"])


@pytest.mark.parametrize("algorithm", list(SklearnEstimator.ALGORITHMS))
def test_classical_estimators_capabilities_and_fit_predict(algorithm):
    pytest.importorskip("sklearn")
    x = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 3)
    y = np.asarray([0, 0, 1, 1] * 3)
    model = SklearnEstimator(algorithm, seed=7).fit(x, y)
    assert model.predict(x).shape == y.shape
    if model.capabilities["predict_proba"]:
        assert model.predict_proba(x).shape == (len(y), 2)
    else:
        with pytest.raises(ContractError):
            model.predict_proba(x)


def test_estimator_missing_persistence_is_explicit(tmp_path):
    import importlib.util

    if importlib.util.find_spec("skops"):
        pytest.skip("This check targets environments without optional skops.")
    with pytest.raises(ContractError) as error:
        SklearnEstimator.inspect_file(tmp_path / "model.skops")
    assert error.value.code == "MISSING_DEPENDENCY"


@pytest.mark.parametrize(
    "format,value",
    [
        ("json", [{"id": "a", "x": 1}]),
        ("csv", [{"id": "a", "x": "1"}]),
        ("npy", np.asarray([1.0, 2.0])),
        ("npz", {"x": np.asarray([1.0, 2.0])}),
    ],
)
def test_format_roundtrip_owned_writers(tmp_path, format, value):
    from lambdaforge.etl import ReadRecords, WriteRecords

    file = tmp_path / ("sample." + format)
    WriteRecords(format)(file, value)
    result = ReadRecords(format, path=("x",) if format == "npz" else ())(file)
    if format in {"npz", "npy"}:
        np.testing.assert_array_equal(result, value["x"] if format == "npz" else value)
    else:
        assert result == value
    with pytest.raises(FileExistsError):
        WriteRecords(format)(file, value)


def test_safe_estimator_persistence_roundtrip_when_extra_is_installed(tmp_path):
    pytest.importorskip("skops.io")
    model = SklearnEstimator("decision_tree_classifier", seed=7).fit([[0], [1]], [0, 1])
    file = tmp_path / "model.skops"
    model.save(file)
    loaded = SklearnEstimator.load(file, trusted_types=SklearnEstimator.inspect_file(file))
    np.testing.assert_array_equal(loaded.predict([[0], [1]]), [0, 1])
    with pytest.raises(ContractError):
        SklearnEstimator.load(file, trusted_types=["unreviewed.Type"])


def test_streaming_record_validation_uses_sample_contract_without_imputation():
    from lambdaforge.data.samples import FieldSpec, SampleSchema
    from lambdaforge.etl import ValidateRecords

    schema = SampleSchema(
        [
            FieldSpec("value", source="metadata", shape=(2,), dtype="float64"),
            FieldSpec("optional", source="metadata", required=False),
        ]
    )
    rows = [{"id": "a", "value": [1.0, 2.0], "partitions": {"split": "train"}}]
    assert list(ValidateRecords(schema.to_dict())(iter(rows))) == rows
    with pytest.raises(ContractError, match="shape"):
        list(ValidateRecords(schema.to_dict())([{"id": "b", "value": [1.0]}]))
    with pytest.raises(ContractError) as caught:
        list(ValidateRecords(schema.to_dict())([{"id": "b"}]))
    assert caught.value.code == "MISSING_FIELD"
