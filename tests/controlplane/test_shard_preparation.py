"""Exact input relocation and owned provider entrypoint regressions; no cluster connections."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.PreparedWork import PreparedWork
from lambdaforge.controlplane.ShardPreparation import (
    prepared_input_bindings,
    relocate_file_inputs,
)
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.managed import CANONICAL_FINGERPRINT_ALGORITHM, canonical_fingerprint
from lambdaforge.work.shard import observe_worker, verify_shard_gpu_grant
from tests.work.test_concrete_shard import prepared_shard


def preparation(
    tmp_path: Path,
) -> tuple[Path, PreparedWork, ExecutionEquivalence, list[dict[str, Any]]]:
    source = tmp_path / "science.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "science",
                "run": "tests.work_cases.CompleteWork",
                "with": {"source": {"file": "source.txt"}, "count": 3},
                "seeds": [1, 2],
            }
        )
    )
    input_path = tmp_path / "source.txt"
    input_path.write_text("exact scientific input")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.yaml").write_text(
        yaml.safe_dump({"with": {"source": {"file": "inputs/0000-source.txt"}, "count": 3}})
    )
    code = {"provider": "explicit", "revision": "test-revision"}
    atomic_write_json(bundle_dir / "manifest.json", {"code_identity": code})
    bundle = ExecutionBundle(
        "test-bundle",
        bundle_dir,
        bundle_dir / "config.yaml",
        bundle_dir / "manifest.json",
        100,
        "env-immutable",
    )
    remote_root = tmp_path / "remote" / "work"
    (remote_root / "inputs").mkdir(parents=True)
    destination = remote_root / "inputs/0000-source.txt"
    destination.write_bytes(input_path.read_bytes())
    digest, size = canonical_fingerprint(input_path)
    identity = [
        {
            "configured": "source.txt",
            "sha256": digest,
            "size_bytes": size,
            "algorithm": CANONICAL_FINGERPRINT_ALGORITHM,
        }
    ]
    eq = ExecutionEquivalence(
        ScientificIdentity.from_payload(code).digest,
        "env-immutable",
        ScientificIdentity.from_payload({"file_inputs": identity}).digest,
        "fp32",
        "cpu",
    )
    prepared = PreparedWork(
        ClusterProfile("remote", environment="managed"),
        LocalTransport(),
        bundle,
        sys.executable,
        str(remote_root / "config.yaml"),
        str(remote_root),
        "job-exact",
        False,
    )
    return source, prepared, eq, [{**identity[0], "destination": str(destination)}]


def test_binding_rechecks_bytes_without_mutating_scientific_parameters(tmp_path: Path) -> None:
    source, prepared, eq, expected = preparation(tmp_path)
    bindings = prepared_input_bindings(source, prepared, eq)
    assert bindings == expected
    parameters = {"source": {"file": "source.txt"}, "count": 3, "label": "source.txt"}
    relocated = relocate_file_inputs(parameters, bindings)
    assert relocated["source"] == {"file": expected[0]["destination"]}
    assert relocated["count"] == 3 and relocated["label"] == "source.txt"
    assert parameters["source"] == {"file": "source.txt"}
    Path(expected[0]["destination"]).write_text("partial or stale")
    with pytest.raises(ValueError, match="bytes changed"):
        relocate_file_inputs(parameters, bindings)


@pytest.mark.parametrize("field", ["code", "environment", "inputs"])
def test_preparation_rejects_different_stratum(tmp_path: Path, field: str) -> None:
    source, prepared, eq, _bindings = preparation(tmp_path)
    with pytest.raises(ValueError, match="stratum|environment identity"):
        prepared_input_bindings(source, prepared, replace(eq, **{field: "different"}))


def test_binding_rejects_symlink_and_missing_binding(tmp_path: Path) -> None:
    _source, _prepared, _eq, bindings = preparation(tmp_path)
    with pytest.raises(ValueError, match="unbound"):
        relocate_file_inputs({"source": {"file": "another"}}, bindings)
    destination = Path(bindings[0]["destination"])
    destination.unlink()
    destination.symlink_to(tmp_path / "source.txt")
    with pytest.raises(ValueError, match="non-symlinked"):
        relocate_file_inputs({"source": {"file": "source.txt"}}, bindings)


def test_invocation_cannot_switch_prepared_work_class(tmp_path: Path) -> None:
    source, prepared, eq, _bindings = preparation(tmp_path)
    with pytest.raises(ValueError, match="prepared Work class"):
        prepared_input_bindings(
            source,
            prepared,
            eq,
            invocations={"untrusted": {"definition": {"work_class": "tests.work_cases.CudaWork"}}},
        )


def test_gpu_grant_preserves_opaque_tokens_and_rejects_foreign_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _control, shard, _invocations = prepared_shard(tmp_path)
    from lambdaforge.execution.ResourceRequest import ResourceRequest
    from lambdaforge.work import runner

    resources = ResourceRequest(gpu_count=2)
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-fleet-" + shard.shard_id)
    monkeypatch.setenv("LAMBDAFORGE_GPU_ACCESS_MODE", "command")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-owned-B,GPU-owned-A")
    monkeypatch.setattr(runner, "_initial_gpu_memory_inventory", lambda count: ((10, 20),) * count)
    monkeypatch.setattr(
        runner, "_gpu_hardware_labels", lambda count, memory: ("test|vram-20",) * count
    )
    hardware = ScientificIdentity.from_payload(
        {"gpu_stratum_version": 1, "model_capacity": "test|vram-20"}
    ).digest
    eq = replace(shard.leases[0].run.equivalence, hardware=hardware)
    assert verify_shard_gpu_grant(shard, resources, eq) == ("GPU-owned-B", "GPU-owned-A")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-owned-B,GPU-owned-A"
    with pytest.raises(ValueError, match="hardware differs"):
        verify_shard_gpu_grant(shard, resources, replace(eq, hardware="other"))
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-foreign")
    with pytest.raises(ValueError, match="exact granted provider"):
        verify_shard_gpu_grant(shard, resources, eq)


def test_observation_projects_only_requested_results_and_no_invocations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _control, shard, invocations = prepared_shard(tmp_path)
    keys = [lease.run.key for lease in shard.leases]
    state = {
        "manifest": {"shard": shard.to_dict(), "invocations": invocations},
        "results": {key: {"state": "completed", "result": {}} for key in keys},
    }
    atomic_write_json(tmp_path / "worker.json", state)
    observe_worker(tmp_path, keys[:1])
    payload = json.loads(capsys.readouterr().out)
    assert payload["shard_id"] == shard.shard_id
    assert list(payload["results"]) == keys[:1]
    assert "manifest" not in payload and "invocations" not in payload
    with pytest.raises(ValueError, match="64 exact"):
        observe_worker(tmp_path, keys * 17)
