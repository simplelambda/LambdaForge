"""Scientific catalog/factories remain metadata-only until explicit resolution."""

import json
import subprocess
import sys

import pytest

from lambdaforge.catalog import ComponentFactory, ContractError, OperationCatalog
from lambdaforge.plugins import PluginDescriptor, PluginKind, PluginRegistry


def test_discovery_does_not_import_science_or_read_sample_bytes():
    code = """import sys
from lambdaforge.catalog import OperationCatalog
document=OperationCatalog().discover()
assert document['schema_version']=='operation-catalog-v1'
names={v['label'] for v in document['operations']}
assert {'MLP','PaiNN','GVPGNN','NNConv','GINE','SchNet','DimeNet','ReadRecords'}<=names
assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'sklearn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_selected_resolution_does_not_import_unrelated_model_families():
    code = """import sys
from lambdaforge.catalog import OperationCatalog
selected = OperationCatalog().resolve('MLP', kind='model')
assert selected.__name__ == 'MLP'
assert 'lambdaforge.nn.models.graph.equivariant.PaiNN' not in sys.modules
assert 'sklearn' not in sys.modules and 'ot' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_cli_sdk_and_packaged_contract_agree():
    result = subprocess.run(
        [sys.executable, "-m", "lambdaforge", "catalog", "PaiNN", "--kind", "model", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    descriptor = json.loads(result.stdout)
    assert descriptor == OperationCatalog().describe("PaiNN", kind="model")
    methods = {v["name"]: v for v in descriptor["methods"]}
    assert methods["forward_with_vectors"]["outputs"][1]["kind"] == "vector"
    assert methods["forward_with_vectors"]["outputs"][1]["shape"][-1] == 3


def test_factories_keep_independent_parameters_and_receiver_dimensions():
    import torch

    from lambdaforge.nn import MLP

    activation = ComponentFactory("prelu", "activation", {"num_parameters": 1})
    first, second = activation.build(), activation.build()
    assert first.weight.data_ptr() != second.weight.data_ptr()
    norm = ComponentFactory("layernorm", "normalization", {"eps": 1e-6})
    a, b = norm.build(features=4), norm.build(features=8)
    assert a(torch.randn(3, 4)).shape == (3, 4)
    assert b(torch.randn(3, 8)).shape == (3, 8)
    network = MLP(
        2,
        1,
        hidden=[4, 8],
        activation=activation.component_type(),
        activation_kwargs=activation.kwargs,
        normalization=norm.component_type(),
        normalization_kwargs=norm.kwargs,
    )
    assert network(torch.randn(3, 2)).shape == (3, 1)
    assert len(norm.provenance()["component"]["parameters"]) > 0
    with pytest.raises(ContractError, match="dimensions"):
        ComponentFactory("layernorm", "normalization", {"normalized_shape": 99})


def test_structured_errors_and_ambiguous_aliases():
    catalog = OperationCatalog()
    with pytest.raises(ContractError) as caught:
        catalog.describe("identity")
    assert caught.value.code == "AMBIGUOUS_ALIAS"
    with pytest.raises(ContractError) as caught:
        catalog.validate("MLP", {"unknown": 1}, kind="model")
    assert caught.value.to_dict()["code"] == "INVALID_PARAMETERS"
    with pytest.raises(ContractError):
        catalog.check_ports({"kind": "vector", "shape": [3, 4, 3]}, {"kind": "tensor"})


def test_plugin_descriptor_discovery_does_not_load_undeclared_plugins(monkeypatch):
    plugins = PluginRegistry()
    row = PluginDescriptor(
        PluginKind.ACTIVATION, "MiXeD-Name", "unimportable:Class", "provider", "1"
    )
    monkeypatch.setattr(plugins, "discover", lambda *a: (row,))
    catalog = OperationCatalog(plugins)
    descriptor = catalog.describe("mixed_name", kind="activation")
    assert descriptor["contract_status"] == "undeclared" and descriptor["label"] == "MiXeD-Name"
    calls = []
    monkeypatch.setattr(plugins, "resolve", lambda reference: calls.append(reference.name) or str)
    assert catalog.resolve(descriptor["id"]) is str and calls == ["MiXeD-Name"]


def test_native_component_registry_normalizes_but_resolves_real_plugin_name(monkeypatch):
    from lambdaforge.nn import ComponentRegistry, ReLU

    registry = PluginRegistry()
    row = PluginDescriptor(PluginKind.ACTIVATION, "MiXeD-Name", "unused:Class")
    monkeypatch.setattr(PluginRegistry, "_default", registry)
    monkeypatch.setattr(registry, "discover", lambda *a: (row,))
    monkeypatch.setattr(
        registry, "resolve", lambda ref: ReLU if ref.name == "MiXeD-Name" else pytest.fail()
    )
    assert ComponentRegistry.resolve_activation("mixed_name") is ReLU


def test_catalog_schema_determinism_and_shapes():
    from importlib.resources import files

    from jsonschema import Draft202012Validator

    catalog = OperationCatalog()
    data = catalog.discover()
    schema = json.loads(
        files("lambdaforge.schemas").joinpath("operation-catalog-v1.json").read_text()
    )
    Draft202012Validator(schema).validate(data)
    assert data == catalog.discover()
    assert len({r["id"] for r in data["operations"]}) == len(data["operations"])
    descriptor = catalog.describe("GVPGNN", kind="model")
    state = next(m for m in descriptor["methods"] if m["name"] == "forward_with_vectors")
    assert state["outputs"][0]["shape"] == ["N", "scalar_channels"]
    assert state["outputs"][1]["shape"] == ["N", "vector_channels", 3]
    catalog.check_ports({"shape": [2, 3, 8]}, {"shape": ["...", "in_features"]})
    with pytest.raises(ContractError):
        catalog.validate("MLP", {"in_features": "two", "out_features": 1}, kind="model")
    with pytest.raises(ContractError):
        catalog.validate("ReadRecords", {"format": "pickle"}, kind="etl")
    with pytest.raises(ContractError):
        catalog.validate("PaiNN", {"in_channels": 0, "out_channels": 2}, kind="model")


def test_factory_for_unregistered_native_activation_and_dimension_errors():
    import torch

    factory = ComponentFactory("GEGLU", "activation")
    assert factory.build()(torch.ones(2, 4)).shape == (2, 2)
    assert factory.provenance()["component"]["label"] == "GEGLU"
    with pytest.raises(ContractError):
        ComponentFactory("layernorm", "normalization").build(features=1.5)


def test_plugin_declared_metadata_validation_without_import(monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace

    plugins = PluginRegistry()
    row = PluginDescriptor(PluginKind.ACTIVATION, "Private", "absent:Class", "provider", "1")
    monkeypatch.setattr(plugins, "discover", lambda *a: (row,))
    data = {
        "schema_version": "operation-catalog-v1",
        "operations": [
            {
                "id": "plugin:activation:Private",
                "kind": "activation",
                "label": "Private",
                "implementation": "absent:Class",
                "aliases": ["Private"],
                "parameters": [],
                "methods": [],
            }
        ],
    }
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: json.dumps(data)),
    )
    assert OperationCatalog(plugins).describe("Private")["contract_status"] == "declared"
    data["operations"][0]["methods"] = "wrong"
    with pytest.raises(ContractError) as error:
        OperationCatalog(plugins).discover()
    assert error.value.code == "INVALID_CONTRACT"


def test_cli_structured_error_without_traceback():
    result = subprocess.run(
        [sys.executable, "-m", "lambdaforge", "catalog", "missing-operation", "--json"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "UNKNOWN_OPERATION"
