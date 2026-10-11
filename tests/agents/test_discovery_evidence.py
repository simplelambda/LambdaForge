"""Static discovery and lazy domain parity, with no cluster/scientific byte downloads."""

import sys
from pathlib import Path

import pytest

from lambdaforge.agents.integration import example_assets
from lambdaforge.agents.services import AgentServices


@pytest.fixture
def service(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="evidence"\nversion="0.1"\n')
    return AgentServices(tmp_path)


def test_all_public_model_contracts_resolve_without_optional_backend_import(service):
    prior = set(sys.modules)
    catalog = service.list_model_catalog(limit=100)
    assert catalog.status == "ok"
    assert catalog.data["next_offset"] is None
    for row in catalog.data["items"]:
        response = service.get_model_contract(row["name"])
        assert response.status == "ok", (row["name"], response.error)
        assert response.data["backend_imported"] is False
        assert response.data["public_api"] == row["public_api"]
    assert "torch" not in set(sys.modules) - prior
    model = service.get_model_contract("MLP").data
    assert model["declared_methods"] and model["documentation"]
    assert service.get_model_contract("../../private").status == "error"
    assert service.get_model_contract("DefinitelyNotAModel").status == "error"


def test_packaged_recipes_are_exact_and_never_arbitrary_readable_files(service):
    rows = service.list_recipes().data["items"]
    assert {row["name"] for row in rows} == set(example_assets())
    recipe = service.get_recipe("work.yaml")
    assert recipe.data["content"] == example_assets()["work.yaml"]
    assert recipe.data["executed"] is False
    assert service.get_recipe("../pyproject.toml").status == "error"


def test_selected_study_run_and_resources_preserve_native_limits(service, monkeypatch):
    calls = []

    class Store:
        def get(self, job_id):
            calls.append(("owner", job_id))

    class Jobs:
        store = Store()

        def study_run(self, job_id, run_key, **kwargs):
            calls.append(("run", job_id, run_key, kwargs))
            return {"run": {"key": run_key}, "metrics": []}

        def study_panel(self, job_id, view):
            return {"view": view}

    monkeypatch.setattr(service, "_job_service", lambda: Jobs())
    result = service.get_study_run("job-exact", "trial-00001-seed-4", tail=20, points=40)
    assert result.status == "ok"
    assert calls[-1] == ("run", "job-exact", "trial-00001-seed-4", {"tail": 20, "curve_points": 40})
    assert service.get_study_run("job-exact", "key", tail=501).status == "error"
    assert service.get_study_panel("job-exact", "report-sections").data["view"] == "report-sections"
    assert service.get_study_panel("job-exact", "unknown").status == "error"


def test_exact_product_evidence_uses_native_registry_without_verification_scan(
    service, monkeypatch
):
    class Contract:
        def to_dict(self):
            return {"name": "fixture", "version": 1}

    class Member:
        def to_dict(self):
            return {"name": "weights", "sha256": "exact", "size": 9000000000}

    class Product:
        content_id = "sha256:" + "a" * 64
        contract = Contract()
        artifacts = (Member(),)

    class Registry:
        def show(self, key):
            assert key == Product.content_id
            return Product()

        def provenance(self, key, *, offset, limit):
            assert offset == 0 and limit == 2
            return ({"execution": "execution-1"}, {"execution": "execution-2"})

        def verify(self, *args):
            pytest.fail("Metadata discovery must not open model bytes")

    monkeypatch.setattr(service, "_products", lambda: Registry())
    assert service.get_product_evidence(Product.content_id).data["contract"]["name"] == "fixture"
    assert (
        service.get_product_evidence(Product.content_id, "members").data["items"][0]["size"]
        == 9000000000
    )
    assert (
        service.get_product_evidence(Product.content_id, "provenance", limit=1).data["next_offset"]
        == 1
    )
    assert service.get_product_evidence("ambiguous-name").status == "error"


def test_resource_probe_is_explicit_native_and_observation_cache_is_project_owned(
    service, monkeypatch
):
    from lambdaforge.controlplane.ResourceService import ResourceService

    calls = []

    class Catalog:
        def definition(self, name):
            calls.append(("profile", name))

    class Snapshot:
        def to_dict(self):
            return {"online": False, "observed_at_utc": "old", "error": "offline"}

    def initialize(self, catalog, *, cache_path):
        assert cache_path.is_relative_to(service.scope.root)
        assert cache_path.name == "resource-observations.json"
        calls.append(("initialize", catalog))

    monkeypatch.setattr(service, "_clusters", lambda: Catalog())
    monkeypatch.setattr(ResourceService, "__init__", initialize)
    monkeypatch.setattr(ResourceService, "get", lambda self, name: Snapshot())
    assert not calls
    response = service.get_cluster_resources("cluster-exact")
    assert response.status == "ok" and response.data["online"] is False
    assert response.data["observed_at_utc"] == "old"
    assert calls[0] == ("profile", "cluster-exact")
    assert not list(service.scope.root.rglob("execution.json"))


def test_dataset_queries_pin_content_and_member_pages_do_not_scan_all_placements(
    service, monkeypatch
):
    from lambdaforge.data.DatasetRecord import DatasetRecord

    record = DatasetRecord(
        "fixture", "1", "sha256:" + "b" * 64, 2, {"train": 1, "test": 1}, "2026-01-01T00:00:00Z"
    )
    calls = []

    class Registry:
        def records(self):
            return (record,)

    class Datasets:
        registry = Registry()

        def members(self, key, **kwargs):
            calls.append((key, kwargs))
            return {"items": [{"id": "exact-member"}]}

    monkeypatch.setattr(service, "_datasets", lambda: Datasets())
    assert service.get_dataset(record.dataset_id).data["splits"] == {"train": 1, "test": 1}
    assert service.get_dataset_placements(record.dataset_id).data["items"] == []
    assert not calls
    assert service.get_dataset("fixture@1").status == "error"
    assert service.get_dataset_members(record.dataset_id, "selected-cluster", 5, 10).status == "ok"
    assert calls == [("fixture@1", {"cluster": "selected-cluster", "offset": 5, "limit": 10})]
