"""Operational catalog/UI boundaries; tests never contact real execution targets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.Fleet import ClusterHealth, Fleet, FleetMember
from lambdaforge.controlplane.FleetResourceService import FleetResourceService
from lambdaforge.controlplane.ResourceSnapshot import ResourceSnapshot
from lambdaforge.ProjectContext import ProjectContext


def isolated_catalog(tmp_path: Path, monkeypatch: Any) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
    monkeypatch.delenv("LAMBDAFORGE_CLUSTERS", raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "lambdaforge.clusters.yaml"
    ClusterCatalog.add(path, ClusterProfile("A"))
    ClusterCatalog.add(path, ClusterProfile("B"))
    ClusterCatalog.add_fleet(path, Fleet("fleet", (FleetMember("A", max_gpus=2), FleetMember("B"))))
    return path


def test_catalog_roundtrip_and_atomic_fleet_override(tmp_path: Path, monkeypatch: Any) -> None:
    source = isolated_catalog(tmp_path, monkeypatch)
    catalog = ClusterCatalog.load(project=ProjectContext(tmp_path, "test"))
    assert catalog.fleet_names() == ("fleet",)
    assert catalog.fleet("fleet").member("A").max_gpus == 2
    assert catalog.definition("A").name == "A"
    explicit = tmp_path / "override.yaml"
    ClusterCatalog.add_fleet(explicit, Fleet("fleet", (FleetMember("B", max_runs=4),)))
    overlaid = ClusterCatalog.load(explicit, project=ProjectContext(tmp_path, "test"))
    assert tuple(member.cluster for member in overlaid.fleet("fleet").members) == ("B",)
    assert "A" in overlaid.names()
    assert yaml.safe_load(source.read_text())["fleets"]["fleet"]["members"][0]["max_gpus"] == 2


@pytest.mark.parametrize(
    "members", [[], [{"cluster": "A"}, {"cluster": "A"}], [{"cluster": "missing"}]]
)
def test_invalid_or_unknown_members_fail_locally(
    tmp_path: Path, monkeypatch: Any, members: Any
) -> None:
    source = isolated_catalog(tmp_path, monkeypatch)
    value = yaml.safe_load(source.read_text())
    value["fleets"]["fleet"]["members"] = members
    source.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        ClusterCatalog.load(source)


def test_cli_preview_apply_preserves_profiles_and_never_cancels_jobs(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    source = isolated_catalog(tmp_path, monkeypatch)
    before = source.read_bytes()
    assert (
        CommandLineInterface.main(
            ["fleets", "--catalog", str(source), "drain", "fleet", "A", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["apply"] is False
    assert source.read_bytes() == before
    assert (
        CommandLineInterface.main(
            ["fleets", "--catalog", str(source), "drain", "fleet", "A", "--apply", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["active_jobs_affected"] is False
    catalog = ClusterCatalog.load(source)
    assert catalog.fleet("fleet").member("A").state == ClusterHealth.DRAINING
    assert catalog.definition("A").to_dict() == ClusterProfile("A").to_dict()
    assert (
        CommandLineInterface.main(["fleets", "--catalog", str(source), "show", "fleet", "--json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["name"] == "fleet"


def test_observation_is_not_a_gpu_grant(tmp_path: Path, monkeypatch: Any) -> None:
    source = isolated_catalog(tmp_path, monkeypatch)
    catalog = ClusterCatalog.load(source)

    class Resources:
        def get(self, cluster: str) -> ResourceSnapshot:
            return ResourceSnapshot(
                cluster, cluster == "A", "local", observed={"gpus": [{"name": "H200", "index": 0}]}
            )

    service = FleetResourceService(catalog, resources=Resources())
    inspection = service.inspect(catalog.fleet("fleet"))
    assert inspection["dispatch_ready"] is False
    assert all(member["admissible_capacity"] is None for member in inspection["members"])
    assert inspection["members"][1]["health"] == "unreachable"


@pytest.mark.parametrize("limit", [True, -1, 0, 2.5, "2"])
def test_caps_are_positive_integers(limit: Any) -> None:
    with pytest.raises(ValueError):
        replace(FleetMember("A"), max_runs=limit)
