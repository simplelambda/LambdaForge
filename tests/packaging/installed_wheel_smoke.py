"""Smoke an installed LambdaForge wheel from outside its source checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import lambdaforge
from lambdaforge import Work


def smoke_inventory() -> None:
    """Run the wheel's complete shipped observer against disposable host metadata."""
    from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
    from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler

    with TemporaryDirectory(prefix="lf-wheel-observer-") as temporary:
        root = Path(temporary)
        for job_id in ("job-good", "job-bad"):
            directory = root / job_id
            directory.mkdir()
            identity = ProcessIdentity.create(
                os.getpid(),
                os.getpgrp() if os.name == "posix" else os.getpid(),
                [sys.executable],
                job_id,
            ).to_dict()
            (directory / "request.json").write_text(
                json.dumps({"job_id": job_id}), encoding="utf-8"
            )
            (directory / "state.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "state": "running",
                        "supervisor_identity": identity,
                        "process_identity": identity,
                    }
                )
                if job_id == "job-good"
                else "corrupt",
                encoding="utf-8",
            )
        result = subprocess.run(
            [sys.executable, "-c", ProcessScheduler._observation_code("inventory"), str(root)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        rows = {row["request"]["job_id"]: row for row in json.loads(result.stdout)}
        assert rows["job-good"]["state"]["state"] == "running"
        assert rows["job-good"]["state"]["provider_observation"] == "alive"
        assert rows["job-bad"]["observation_error"]


def main() -> None:
    """Verify imports, packaged resources and one minimal authoring operation."""
    installed = distribution("lambdaforge")
    assert lambdaforge.__version__ == installed.version
    assert files("lambdaforge").joinpath("schemas/work.schema.json").is_file()
    for relative in (
        "schemas/report-bundle-v1.json",
        "analysis/assets/report_bundle_bridge.js",
        "analysis/assets/report_project_tabs.js",
    ):
        assert files("lambdaforge").joinpath(relative).is_file()
    from lambdaforge.analysis.ReportViewer import ReportViewer
    from lambdaforge.work.report_bundle import ManagedReportBundle

    assert ReportViewer is not None and ManagedReportBundle is not None
    installed_files = tuple(installed.files or ())
    public_docs = {
        item.name
        for item in installed_files
        if "/share/lambdaforge/docs/" in "/" + item.as_posix()
        and "/ai/" not in item.as_posix()
        and item.suffix == ".md"
    }
    assert public_docs == {"MANUAL.md", "MANUAL.es.md"}, public_docs
    assert not any(item.as_posix().endswith("clustering/algorithms.py") for item in installed_files)
    for relative in (
        "share/lambdaforge/AGENTS.md",
        "share/lambdaforge/AGENTS.es.md",
        "share/lambdaforge/CHANGELOG.md",
        "share/lambdaforge/CHANGELOG.es.md",
        "share/lambdaforge/README.md",
        "share/lambdaforge/README.es.md",
        "share/lambdaforge/SECURITY.md",
        "share/lambdaforge/SECURITY.es.md",
        "share/lambdaforge/examples/work.yaml",
        "share/lambdaforge/examples/sequence.yaml",
        "share/lambdaforge/examples/fgw.py",
        "share/lambdaforge/docs/MANUAL.md",
        "share/lambdaforge/docs/MANUAL.es.md",
        "share/lambdaforge/docs/ai/IMPLEMENTATION.md",
        "share/lambdaforge/docs/ai/IMPLEMENTATION.es.md",
        "share/lambdaforge/docs/ai/FRAMEWORK_CONTRACTS.md",
        "share/lambdaforge/docs/ai/AGENT_GUIDE.es.md",
        "share/lambdaforge/examples/monitor.json",
    ):
        matches = tuple(
            item
            for item in installed_files
            if item.as_posix().endswith(relative) and installed.locate_file(item).is_file()
        )
        assert len(matches) == 1, relative
    scripts = {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["lf"] == scripts["lambdaforge"]
    assert lambdaforge.__all__ == ["Work", "__version__", "clustering"]
    from lambdaforge.distances import FusedGromovWassersteinDistance, StructuredMeasure

    assert files("lambdaforge").joinpath("clustering/algorithms/KMeans.py").is_file()
    assert FusedGromovWassersteinDistance is not None and StructuredMeasure is not None
    requirements = installed.requires or ()
    assert any("pot" in item.lower() and "optimal-transport" in item for item in requirements)
    assert Work.__module__ == "lambdaforge.work.Work"
    assert "mcp" not in sys.modules
    from lambdaforge.agents import AgentServices, IntegrationService
    from lambdaforge.agents.integration import example_assets, skill_names

    assert "mcp" not in sys.modules
    assert len(skill_names()) == 24 and "work.py" in example_assets()
    assert callable(AgentServices.framework_capabilities)
    assert callable(AgentServices.get_resource_report)
    assert callable(IntegrationService.preview)
    from lambdaforge.runtime.GPUObservation import GPUObservationCache
    from lambdaforge.runtime.HostLeaseStore import gpu_lease_name
    from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy

    assert "torch" not in sys.modules and "pynvml" not in sys.modules
    assert ResourceSafetyPolicy.from_value("shared-host").telemetry_ttl == 10
    assert gpu_lease_name("MIG-wheel-smoke").startswith("gpu-")
    with TemporaryDirectory(prefix="lf-wheel-resource-") as temporary:
        calls: list[bool] = []

        def probe() -> dict[str, object]:
            calls.append(True)
            return {"quality": "measured", "observed_at": 100.0, "devices": []}

        cache = GPUObservationCache(Path(temporary) / "cache", clock=lambda: 100.0, probe=probe)
        assert cache.read()["quality"] == "measured"
        assert cache.read()["statistics"]["real_queries"] == 1 and len(calls) == 1
    with TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "installed-agents"\nversion = "0.0.0"\n', encoding="utf-8"
        )
        agent = AgentServices(project)
        resource_report = agent.get_resource_report()
        assert resource_report.status == "ok", resource_report.error
        assert resource_report.data["job_count"] == 0
        contract = agent.get_model_contract("MLP")
        assert contract.status == "ok", contract.error
        assert contract.data["backend_imported"] is False
        assert agent.get_recipe("work.yaml").status == "ok"
        contract = agent.capability_describe("framework.version").data
        assert (
            contract["output_schema"]["allOf"][0]["then"]["properties"]["data"]["properties"][
                "version"
            ]["type"]
            == "string"
        )
        assert agent.capability_search("Quiero recuperar un estudio Fleet").status == "ok"
        cancel = agent.capability_describe("jobs.cancel.apply").data
        assert cancel["risk_annotations"]["destructive"] is True
        assert cancel["session_enabled"] is False
        assert agent.request_job_cancellation("job-not-launched").error.code == "PERMISSION_DENIED"
        assert agent.project_info().data["permissions"] == ["READ", "PLAN"]
        assert "mcp" not in sys.modules
    from lambdaforge.products import (
        ProductBundle,
        ProductContract,
        ProductInput,
        ProductPublication,
        ProductRegistry,
        StudyProduct,
        build_study_decision,
        publish_declared_products,
    )

    assert ProductContract("example/report:v1", ("sources",)).identifier == "example/report:v1"
    assert callable(ProductBundle.import_bundle)
    assert callable(ProductRegistry.resolve)
    assert callable(StudyProduct.from_dict)
    assert callable(ProductInput.artifact)
    assert callable(ProductPublication.from_mapping)
    assert callable(build_study_decision)
    assert callable(publish_declared_products)
    from lambdaforge.data import DatasetEquivalenceCertificate
    from lambdaforge.work import ResultStore

    assert callable(DatasetEquivalenceCertificate.build)
    assert callable(ResultStore.import_export)
    from lambdaforge.hpo.SequentialSweep import PairedSweepSequentialAnalyzer
    from lambdaforge.work.sweep_blocks import sweep_block_inventory

    decision = PairedSweepSequentialAnalyzer(mode="max", bounds=(0, 1)).evaluate(
        {1: {54: 0.5}, 2: {54: 0.6}}, seed_order=[54]
    )
    assert decision.policy_version == "paired-pm-eb-cs-v2"
    assert decision.evidence_seeds == (54,)
    assert callable(sweep_block_inventory)
    from lambdaforge.cli.parser import build_parser
    from lambdaforge.controlplane.JobService import JobService
    from lambdaforge.controlplane.MonitorService import MonitorService
    from lambdaforge.controlplane.ObservationGate import ObservationGate
    from lambdaforge.controlplane.OperationalHealth import ObservationPolicy
    from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents
    from lambdaforge.controlplane.RecoveryAudit import recovery_history
    from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation
    from lambdaforge.tui.screens.RecoveryWorkspace import RecoveryWorkspace
    from lambdaforge.work.checkpoint_policy import CheckpointPolicy
    from lambdaforge.work.checkpoints import CheckpointCollection

    assert ObservationPolicy().progress_seconds is None
    assert callable(MonitorService.start)
    assert callable(OperationalIncidents.deliver)
    assert callable(ObservationGate.__enter__)
    assert CheckpointPolicy().interval_steps is None
    assert callable(CheckpointCollection.inventory)
    assert callable(recovery_history)
    assert callable(JobService.inspect_recovery)
    assert callable(JobService.recovery_history)
    assert callable(RecoveryWorkspace.compose)
    assert not RecoveryObservation.eligible("unknown", {})
    parser = build_parser()
    assert parser.parse_args(["health", "--json"]).command == "health"
    assert parser.parse_args(["jobs", "recovery", "job-saved", "--json"]).job_command == "recovery"
    assert parser.parse_args(["monitor", "status", "--json"]).monitor_command == "status"
    assert (
        str(parser.parse_args(["retry", "study", "--config", "moved.yaml"]).config) == "moved.yaml"
    )
    smoke_inventory()
    print(lambdaforge.__version__)


if __name__ == "__main__":
    main()
