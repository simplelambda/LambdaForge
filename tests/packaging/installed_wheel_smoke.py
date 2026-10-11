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
    installed_files = tuple(installed.files or ())
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
        "share/lambdaforge/docs/MANUAL.md",
        "share/lambdaforge/docs/MANUAL.es.md",
        "share/lambdaforge/docs/STORAGE.md",
        "share/lambdaforge/docs/STORAGE.es.md",
        "share/lambdaforge/docs/DATASET_RECONSTRUCTION.md",
        "share/lambdaforge/docs/DATASET_RECONSTRUCTION.es.md",
        "share/lambdaforge/docs/ARCHITECTURAL_REFORM.md",
        "share/lambdaforge/docs/ARCHITECTURAL_REFORM.es.md",
        "share/lambdaforge/docs/SCIENTIFIC_AUDIT.md",
        "share/lambdaforge/docs/SCIENTIFIC_AUDIT.es.md",
        "share/lambdaforge/docs/PRODUCTS.md",
        "share/lambdaforge/docs/PRODUCTS.es.md",
        "share/lambdaforge/docs/OPERATIONAL_RELIABILITY.md",
        "share/lambdaforge/docs/OPERATIONAL_RELIABILITY.es.md",
        "share/lambdaforge/docs/ai/INTEGRATION.md",
        "share/lambdaforge/docs/ai/INTEGRATION.es.md",
        "share/lambdaforge/docs/ai/IMPLEMENTATION.md",
        "share/lambdaforge/docs/ai/IMPLEMENTATION.es.md",
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
    assert Work.__module__ == "lambdaforge.work.Work"
    assert "mcp" not in sys.modules
    from lambdaforge.agents import AgentServices, IntegrationService
    from lambdaforge.agents.integration import example_assets, skill_names

    assert "mcp" not in sys.modules
    assert len(skill_names()) == 24 and "work.py" in example_assets()
    assert callable(AgentServices.framework_capabilities)
    assert callable(IntegrationService.preview)
    with TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "installed-agents"\nversion = "0.0.0"\n', encoding="utf-8"
        )
        agent = AgentServices(project)
        contract = agent.get_model_contract("MLP")
        assert contract.status == "ok", contract.error
        assert contract.data["backend_imported"] is False
        assert agent.get_recipe("work.yaml").status == "ok"
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
