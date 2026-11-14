"""Inventory is metadata-only and must survive a broken scientific venv."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from lambdaforge.controlplane import ClusterProfile, CommandResult
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.errors import DatasetRegistryCorruptionError, DatasetResolutionError


class HostReader:
    def __init__(self, fail=False):
        self.commands = []
        self.fail = fail

    def run(self, command, **kwargs):
        self.commands.append(tuple(command))
        assert "active-environment" not in " ".join(command)
        assert command[0] == sys.executable
        if self.fail:
            return CommandResult(1, stderr="PermissionError: registry is not readable")
        result = subprocess.run(command, text=True, capture_output=True, timeout=5)
        return CommandResult(result.returncode, result.stdout, result.stderr)


def service(tmp_path, transport):
    profile = ClusterProfile.from_mapping(
        "remote",
        {
            "host": "test.invalid",
            "transport": "ssh",
            "environment": "managed",
            "workspace": str(tmp_path),
            "python": sys.executable,
        },
    )
    value = object.__new__(DatasetService)
    value.clusters = SimpleNamespace(get=lambda _: profile)
    value.factory = SimpleNamespace(transport=lambda _: transport)
    return value, profile


def test_inventory_uses_bootstrap_python_without_opening_dataset_bytes(tmp_path):
    transport = HostReader()
    value, profile = service(tmp_path, transport)
    path = tmp_path / "registry-source"
    record = DatasetRecord(
        "existing", "1", "dataset-sha", 3, {"train": 2, "validation": 1}, "2026-11-08T00:00:00Z"
    )
    registry = DatasetRegistry(profile.storage.state_root + "/datasets.json")
    registry.register(record)
    assert value._remote_records("remote")[0].dataset_id == "dataset-sha"
    assert "-I" in transport.commands[0] and "-S" in transport.commands[0]
    assert not path.exists()


def test_proven_missing_registry_is_empty(tmp_path):
    value, _ = service(tmp_path, HostReader())
    assert value._remote_records("remote") == ()


def test_permission_failure_is_not_reported_as_absence(tmp_path):
    value, _ = service(tmp_path, HostReader(fail=True))
    with pytest.raises(DatasetResolutionError, match="without proving absence"):
        value._remote_records("remote")


def test_corrupt_transported_registry_uses_native_validation():
    for raw in ["[]", "not json", json.dumps({"datasets": {"bad": {"name": "bad"}}})]:
        with pytest.raises(DatasetRegistryCorruptionError):
            DatasetRegistry.decode_inventory(raw, source="remote/datasets.json")
