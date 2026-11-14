"""Synthetic multi-asset samples, explicit masks/graphs and bounded lazy reads."""

import json
from pathlib import Path

import numpy as np
import pytest

from lambdaforge.catalog import ContractError
from lambdaforge.data import DatasetPublisher, DatasetRegistry
from lambdaforge.data.samples import (
    FieldSpec,
    SampleDataset,
    SampleSchema,
    collate_graphs,
    collate_samples,
)
from lambdaforge.etl import WriteSampleAssets


@pytest.fixture
def samples(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    members = []
    for i in range(3):
        row = WriteSampleAssets()(
            directory,
            member_id=f"id-{i}",
            arrays={"x": np.full((i + 1, 2), i, dtype=np.float32)},
            sidecar={"context": [float(i), 1.0]},
            targets={"y": i % 2},
            partitions={"split": "train" if i < 2 else "validation"},
        )
        members.append(row)
    registry = DatasetRegistry(tmp_path / "registry.json")
    record = DatasetPublisher(registry).publish_members(
        "multi",
        "1",
        members,
        source_root=directory,
        publication_root=tmp_path / "published",
        build_provenance={"recipe": 1},
    )
    schema = SampleSchema(
        [
            FieldSpec(
                "nodes",
                asset="sample",
                format="npz",
                path=("x",),
                dtype="float32",
                shape=(None, 2),
                axes=("node", "feature"),
                collate="pad",
            ),
            FieldSpec("context", asset="sidecar", format="json", path=("context",), shape=(2,)),
            FieldSpec(
                "optional",
                asset="optional",
                format="npy",
                shape=(1,),
                required=False,
                missing="default",
                default=[0.0],
            ),
            FieldSpec("label", source="targets", path=("y",), role="target", shape=()),
        ],
        {"x": "nodes", "context": "context"},
    )
    return Path(record.placements[0].root), schema


def test_lazy_preflight_preview_and_provenance(samples, monkeypatch):
    root, schema = samples
    original = np.load
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("No arrays during construction/preflight")
    )
    dataset = SampleDataset(root, schema)
    assert dataset.preflight()["compatible"] and not dataset.preflight()["decoded_assets"]
    monkeypatch.setattr(np, "load", original)
    value = dataset[0]
    assert value["id"] == "id-0" and value["present"]["optional"] is False
    assert value["provenance"]["content_id"].startswith("sha256:")
    assert set(schema.model_inputs(value["fields"])) == {"x", "context"}
    assert len(dataset.preview(samples=2)["samples"]) == 2
    assert len(SampleDataset(root, schema, partitions={"split": "train"})) == 2
    assert (
        SampleSchema.from_mapping(json.loads(json.dumps(schema.to_dict()))).to_dict()
        == schema.to_dict()
    )


def test_padding_masks_and_no_implicit_ragged_stack(samples):
    root, schema = samples
    data = SampleDataset(root, schema)
    batch = collate_samples([data[0], data[1]], schema)
    assert batch["fields"]["nodes"].shape == (2, 2, 2)
    assert batch["masks"]["nodes:padding"].tolist() == [[True, False], [True, True]]
    assert batch["masks"]["optional"].tolist() == [False, False]
    plain = SampleSchema([FieldSpec("nodes", asset="sample", format="npz", path=("x",))])
    with pytest.raises(ContractError) as error:
        collate_samples([data[0], data[1]], plain)
    assert error.value.code == "RAGGED_BATCH"


def test_schema_errors_and_corruption_are_not_optional_missing(samples):
    root, schema = samples
    data = SampleDataset(root, schema)
    asset = root / data.members[0].assets["sample"].path
    asset.write_bytes(b"corrupt")
    with pytest.raises(ContractError) as error:
        data[0]
    assert error.value.code == "CONTENT_MISMATCH"
    with pytest.raises(ContractError, match="Targets"):
        SampleSchema([FieldSpec("target", source="targets", role="target")], {"x": "target"})
    with pytest.raises(ContractError):
        FieldSpec("x", asset="sample", format="npz", path=())


def test_byte_budget_rejects_before_decode(samples, monkeypatch):
    root, schema = samples
    dataset = SampleDataset(root, schema)
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("Budget must reject before decode")
    )
    with pytest.raises(ContractError) as error:
        dataset.preview(max_bytes=4)
    assert error.value.code == "READ_LIMIT"


def test_graph_batch_offsets_preserve_polar_vector_axis():
    graphs = [
        {
            "x": np.zeros((n, 2), np.float32),
            "v": np.zeros((n, 4, 3), np.float32),
            "edge_index": np.asarray([[0], [n - 1]], np.int64),
        }
        for n in (2, 3)
    ]
    batch = collate_graphs(graphs, node_fields=("x", "v"))
    assert batch["v"].shape == (5, 4, 3)
    assert batch["edge_index"].tolist() == [[0, 2], [1, 4]]
    assert batch["ptr"].tolist() == [0, 2, 5]
    graphs[0]["edge_index"][1, 0] = 7
    with pytest.raises(ContractError) as error:
        collate_graphs(graphs, node_fields=("x",))
    assert error.value.code == "GRAPH_INDEX"


def test_packaged_schema_and_example_are_valid():
    from importlib.resources import files

    from jsonschema import Draft202012Validator

    document = json.loads((Path(__file__).parents[2] / "examples/sample-schema.json").read_text())
    schema = json.loads(files("lambdaforge.schemas").joinpath("sample-schema-v1.json").read_text())
    Draft202012Validator(schema).validate(document)
    assert SampleSchema.from_mapping(document).bindings["x"] == "features"
    for value in (
        {"schema_version": "sample-schema-v1"},
        {"schema_version": "sample-schema-v1", "fields": [42]},
    ):
        with pytest.raises(ContractError):
            SampleSchema.from_mapping(value)


def test_optional_declared_bytes_missing_is_corruption_not_default(samples):
    root, schema = samples
    data = SampleDataset(root, schema)
    field = FieldSpec(
        "optional",
        asset="sidecar",
        format="json",
        path=("context",),
        required=False,
        missing="default",
        default=[],
    )
    optional = SampleDataset(root, SampleSchema([field]))
    (root / data.members[0].assets["sidecar"].path).unlink()
    with pytest.raises(ContractError) as error:
        optional[0]
    assert error.value.code == "CONTENT_MISMATCH"


def test_all_member_metadata_limits_and_symlink_safety(samples, tmp_path):
    root, schema = samples
    with pytest.raises(ContractError, match="inventory"):
        SampleDataset(root, schema, max_members=1, partitions={"split": "absent"})
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        SampleDataset(linked, schema)


def test_readers_never_load_object_arrays_and_bound_decompressed_npz(tmp_path):
    from lambdaforge.etl import ReadRecords

    np.save(tmp_path / "object.npy", np.asarray([{}], dtype=object))
    with pytest.raises(ContractError) as error:
        ReadRecords("npy")(tmp_path / "object.npy")
    assert error.value.code == "UNSAFE_DTYPE"
    np.savez_compressed(tmp_path / "large.npz", x=np.zeros(1024**2, dtype=np.float32))
    with pytest.raises(ContractError) as error:
        ReadRecords("npz", path=("x",), max_bytes=10_000)(tmp_path / "large.npz")
    assert error.value.code == "READ_LIMIT"
