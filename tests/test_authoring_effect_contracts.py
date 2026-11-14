"""Installed metadata matches selected actual signatures and native provider boundaries."""

import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from lambdaforge.catalog import OperationCatalog
from lambdaforge.configuration import ConfigurationSchemas
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy


def test_effect_metadata_is_conservative_and_discovery_has_no_backend_imports():
    code = """import sys
from lambdaforge.catalog import OperationCatalog
c=OperationCatalog()
for name in ('ReadRecords','ReadSQL','WriteSQL','PreviewArray'):
 d=c.describe(name,kind='etl')
 assert d['effects']['schema_version']=='operation-effects-v1'
assert not any(n in sys.modules for n in ('torch','numpy','sqlite3','psycopg','pyarrow'))
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)
    c = OperationCatalog()
    assert c.describe("ReadRecords")["preview"]["supported"]
    assert not c.describe("WriteSQL")["preview"]["supported"]
    assert "modify-external-source" in c.describe("WriteSQL")["effects"]["capabilities"]
    assert "learn-state" in c.describe("TrainStandardizer")["effects"]["capabilities"]
    effects_schema = json.loads(
        files("lambdaforge.schemas").joinpath("operation-effects-v1.json").read_text()
    )
    for row in c.discover()["operations"]:
        if not row.get("plugin"):
            Draft202012Validator(effects_schema).validate(row["effects"])
    for invalid in ({"delimiter": "::"}, {"header_rows": 0}, {"unknown_option": 1}):
        with pytest.raises(ValueError):
            c.validate("ReadRecords", {"format": "csv", "csv": invalid}, kind="etl")


def test_descriptors_match_all_native_installed_signatures_without_instantiation():
    catalog = OperationCatalog()
    concrete_models = 0
    for row in catalog.discover()["operations"]:
        if row.get("plugin"):
            continue
        checked = catalog.validate_descriptor(row["id"])
        if row["kind"] == "model" and not checked["abstract"]:
            concrete_models += 1
        assert checked["valid"]
    assert concrete_models >= 80


def test_structured_models_and_calibration_state_contracts_match_real_calls():
    import torch
    from torch import nn

    from lambdaforge.nn.models import EnsembleModel, MixtureOfExperts, MultiTaskModel, NeuralODE
    from lambdaforge.nn.uncertainty import ConformalPredictionInterval, TemperatureScaler

    catalog = OperationCatalog()
    multi = catalog.describe("MultiTaskModel")
    forward = next(m for m in multi["methods"] if m["name"] == "forward")
    assert forward["outputs"][0]["kind"] == "named-tensor-mapping"
    model = MultiTaskModel(nn.Linear(2, 3), {"score": nn.Linear(3, 1), "other": nn.Linear(3, 2)})
    assert model(torch.ones(4, 2))["other"].shape == (4, 2)
    assert (
        len(EnsembleModel([nn.Linear(2, 1), nn.Linear(2, 1)]).forward_members(torch.ones(4, 2)))
        == 2
    )
    assert "ALL experts" in catalog.describe("MixtureOfExperts")["methods"][0]["constraints"]
    assert MixtureOfExperts([nn.Linear(2, 1), nn.Linear(2, 1)], nn.Linear(2, 2), top_k=1)(
        torch.ones(4, 2)
    ).shape == (4, 1)

    class Dynamics(nn.Module):
        def forward(self, t, y):
            return y

    ode = NeuralODE(Dynamics(), return_trajectory=True)
    assert ode(torch.ones(2, 3), torch.tensor([0.0, 1.0])).shape == (2, 2, 3)
    interval = ConformalPredictionInterval().fit(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
    assert set(interval(torch.tensor([0.0, 1.0]))) == {"prediction", "lower", "upper"}
    assert "held-out" in catalog.describe("TemperatureScaler")["state_policy"]
    assert TemperatureScaler()(torch.ones(2, 3)).shape == (2, 3)


def test_cluster_schema_defaults_conditions_variants_and_cli_do_not_claim_grants():
    static = ConfigurationSchemas.schema("cluster")
    Draft202012Validator.check_schema(static)
    for prefix in (["gpu"], ["gpu", "exec"], ["site-launcher", "--"]):
        config = {
            "transport": "ssh",
            "host": "synthetic.invalid",
            "workspace": "/scratch/test",
            "gpu_access": {
                "mode": "command",
                "command_prefix": prefix,
                "visibility_command": ["site-grant", "--json"],
            },
            "pytorch": {"require_cuda": True},
            "python": {"strategy": "existing", "executable": "python3"},
            "resource_safety": "throughput",
        }
        effective = ConfigurationSchemas.resolve_cluster("fixture", config)
        native = ClusterProfile.from_mapping("fixture", config)
        assert effective == native.to_dict()
        assert native.gpu_access.wrap(["python3", "work.py"]) == tuple(
            [*prefix, "python3", "work.py"]
        )
        assert native.gpu_access.visibility_probe() == ("site-grant", "--json")
        native.gpu_access.require_owned_reservation_contract()
    with pytest.raises(ValidationError):
        ConfigurationSchemas.resolve_cluster("bad", {"scheduler": "invented"})
    with pytest.raises(ValidationError):
        ConfigurationSchemas.resolve_cluster("bad", {"transport": "ssh"})
    with pytest.raises(ValidationError):
        ConfigurationSchemas.resolve_cluster("bad", {"auth": {"credential": "password-raw"}})
    unsafe = GpuAccessPolicy(
        "command", ("gpu",), claim_command=("gpu", "claim"), release_command=("gpu", "release")
    )
    with pytest.raises(ValueError, match="ownership"):
        unsafe.require_owned_reservation_contract()
    result = subprocess.run(
        [sys.executable, "-m", "lambdaforge", "config", "schema", "--kind", "cluster", "--json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout) == static
    assert ConfigurationSchemas.schema("work")["x-contract-version"] == "work-config-v1"


def test_real_table_schema_example_and_preview_schema(tmp_path):
    from lambdaforge.etl import ReadRecords
    from lambdaforge.etl.tables import TableSchema

    example = json.loads((Path(__file__).parents[1] / "examples/table-schema.json").read_text())
    schema = json.loads(files("lambdaforge.schemas").joinpath("table-schema-v1.json").read_text())
    Draft202012Validator(schema).validate(example)
    TableSchema.from_mapping(example)
    file = tmp_path / "table.csv"
    file.write_text("id,value\na,1\n")
    report = ReadRecords("csv").preview(file)
    Draft202012Validator(
        json.loads(files("lambdaforge.schemas").joinpath("table-preview-v1.json").read_text())
    ).validate(report)


def test_published_tabular_python_snippets_are_syntactically_valid():
    import ast
    import re

    root = Path(__file__).parents[1]
    for name in ("MANUAL.md", "MANUAL.es.md"):
        chapter = (
            (root / "docs" / name).read_text().split('<a id="guide-tabular-connectors"></a>', 1)[1]
        )
        for code in re.findall(r"```python\n(.*?)```", chapter, re.DOTALL):
            ast.parse(code)
