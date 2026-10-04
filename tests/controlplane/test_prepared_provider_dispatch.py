"""Real local provider execution through preparation; not SSH/SLURM acceptance claims.

Only installation/runtime resolution are test fixtures. Bundle staging, JobService,
ProcessScheduler, the detached supervisor, native Work children and result observation run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor
from lambdaforge.controlplane.python_runtime import PythonRuntime
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.atomic import atomic_write_json, atomic_write_text
from tests.work.test_concrete_shard import prepared_shard


class LoopbackFactory(ControlPlaneFactory):
    """Use actual local transport/provider and the already installed test interpreter."""

    def transport(self, profile: ClusterProfile) -> LocalTransport:
        return LocalTransport()

    def environment_provider(self, profile: ClusterProfile) -> Any:
        class Prepared:
            def prepare(self, *args: Any, **kwargs: Any) -> PreparedEnvironment:
                assert profile.storage is not None
                atomic_write_text(
                    Path(profile.storage.state_root) / "active-environment", sys.executable
                )
                return PreparedEnvironment("env-test", sys.executable, True)

        return Prepared()


class RuntimeFixture:
    def resolve(self, *args: Any, **kwargs: Any) -> PythonRuntime:
        return PythonRuntime(
            "test-runtime",
            sys.executable,
            "3.11.1",
            "CPython",
            "Linux",
            "x86_64",
            "existing",
            None,
            False,
            True,
            "reuse",
        )

    def activate(self, *args: Any, **kwargs: Any) -> None:
        pass


class TorchFixture:
    def resolve(self, *args: Any, **kwargs: Any) -> TorchInstallationPlan:
        return TorchInstallationPlan(
            "cu121", "2.3.1", "https://download.pytorch.org/whl/cu121", "cuda"
        )


class BundleFixture:
    def __init__(self, bundle: ExecutionBundle) -> None:
        self.bundle = bundle

    def build(self, *args: Any, **kwargs: Any) -> ExecutionBundle:
        return self.bundle


@pytest.mark.skipif(os.name != "posix", reason="Detached direct provider needs POSIX")
@pytest.mark.parametrize("gpu", [False, True])
def test_prepared_provider_runs_native_work_and_observes_exact_results(
    tmp_path: Path, gpu: bool
) -> None:
    if gpu and not torch.cuda.is_available():
        pytest.skip("Actual local CUDA is unavailable")
    _control, shard, original = prepared_shard(tmp_path)
    source = tmp_path / "science.yaml"
    class_name = "tests.work_cases.CudaWork" if gpu else "tests.work_cases.ConfirmationFailureWork"
    resources = ResourceRequest(cpu_cores=1 if gpu else 4, gpu_count=int(gpu))
    source.write_text(
        yaml.safe_dump(
            {
                "name": "science",
                "run": class_name,
                "with": {} if gpu else {"quality": 0.1},
                "seeds": [101, 102],
                "resources": {"cpu": resources.cpu_cores, "gpu": resources.gpu_count},
            }
        )
    )
    code = {"provider": "explicit", "revision": "test-code"}
    hardware = "prepared-cpu"
    if gpu:
        from lambdaforge.work.runner import _gpu_hardware_labels, _initial_gpu_memory_inventory

        labels = _gpu_hardware_labels(1, _initial_gpu_memory_inventory(1))
        assert labels and not labels[0].startswith("unknown")
        hardware = ScientificIdentity.from_payload(
            {"gpu_stratum_version": 1, "model_capacity": labels[0]}
        ).digest
    eq = ExecutionEquivalence(
        ScientificIdentity.from_payload(code).digest,
        "env-test",
        ScientificIdentity.from_payload({"file_inputs": []}).digest,
        "fp32",
        hardware,
    )
    leases = shard.leases[:1] if gpu else shard.leases
    shard = replace(
        shard,
        leases=tuple(
            replace(
                lease,
                run=replace(
                    lease.run,
                    equivalence=eq,
                    requires_gpu=gpu,
                    parameters={} if gpu else lease.run.parameters,
                ),
            )
            for lease in leases
        ),
    )
    invocations = {}
    for lease in shard.leases:
        value = dict(original[lease.run.key])
        value["definition"] = {
            **value["definition"],
            "work_class": class_name,
            "resources": resources.to_dict(),
            "objective": {"metric": "cuda_score" if gpu else "score", "mode": "max"},
        }
        value["parameters"] = dict(lease.run.parameters)
        value["trial_parameters"] = {} if gpu else value["trial_parameters"]
        invocations[lease.run.key] = value
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.yaml").write_bytes(source.read_bytes())
    atomic_write_json(bundle_dir / "manifest.json", {"code_identity": code})
    bundle = ExecutionBundle(
        "bundle-test",
        bundle_dir,
        bundle_dir / "config.yaml",
        bundle_dir / "manifest.json",
        100,
        "env-test",
    )
    import_root = str(Path(__file__).resolve().parents[2])
    profile = ClusterProfile(
        "A",
        python=sys.executable,
        environment="managed",
        workspace=str(tmp_path / "provider"),
        command_prefix=("env", "PYTHONPATH=" + import_root),
        # A tiny CUDA tensor test may safely coexist with external local contexts; do not
        # override exclusive production policy just to make the test acquire a busy GPU.
        gpu_access=GpuAccessPolicy("shared") if gpu else GpuAccessPolicy(),
    )
    catalog = ClusterCatalog({"A": profile})
    factory = LoopbackFactory()
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"), factory)
    plane = ControlPlane(
        catalog, jobs, BundleFixture(bundle), factory, TorchFixture(), RuntimeFixture()
    )  # type: ignore[arg-type]
    executor = PreparedShardExecutor(
        profile,
        shard.member,
        root=tmp_path / "executor",
        resources=resources,
        equivalence=eq,
        invocation=lambda key: invocations[key],
        control_plane=plane,
    )
    # A launch path/hardware probe is still NOT an allocation offer.
    if gpu:
        assert executor.offer(()).admissible_gpus == 0
        assert executor.offer(()).slots == 0
    job_id = executor.submit(shard)
    try:
        deadline = time.monotonic() + 90
        while True:
            observations = executor.observe(shard, job_id)
            if all(item.result is not None for item in observations):
                break
            if time.monotonic() > deadline:
                pytest.fail(f"Provider did not publish concrete results: {jobs.logs(job_id)}")
            time.sleep(0.5)
        assert {item.state for item in observations} == (
            {"completed"} if gpu else {"completed", "failed"}
        )
        assert executor.submit(shard) == job_id
        receipt = json.loads((executor.root / shard.shard_id / "submission.json").read_text())
        assert receipt["acknowledged"] is True
        assert receipt["scheduler_id"] == jobs.get(job_id, refresh=False).scheduler_id
        assert receipt["worker_root"].startswith(str(tmp_path / "provider"))
        assert all(item.result["equivalence"] == eq.to_dict() for item in observations)
        assert len(jobs.store.records()) == 1
    finally:
        record = jobs.get(job_id, include_study=False)
        if not record.state.terminal:
            jobs.cancel(job_id)
