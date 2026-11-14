"""Offline native Attempts, Requests and lineage across repeated recovery staging roots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.staged_input_identity import original_file_bindings
from tests.work.test_single_work_recovery import snapshot

ROLES = (
    "legacy_train",
    "legacy_validation",
    "legacy_test",
    "legacy_positives",
    "legacy_negatives",
    "legacy_all",
)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from lambdaforge.work import runner

    revision = ["origin"]
    monkeypatch.setattr(
        runner,
        "_code_identity",
        lambda root: {"provider": "git", "revision": revision[0], "dirty": False},
    )
    profile = ClusterProfile(
        "fixture",
        transport="ssh",
        host="never-connect.invalid",
        workspace=str(tmp_path / "storage"),
    )
    jobs = JobService(
        ClusterCatalog({profile.name: profile}),
        JobStore(tmp_path / "index"),
        SimpleNamespace(
            transport=lambda _: LocalTransport(), scheduler=lambda *_: SimpleNamespace()
        ),
    )
    origin = []
    configs = []

    def stage(number):
        owner = f"job-stage-{number}"
        work = tmp_path / "jobs" / owner / "work"
        (work / "inputs").mkdir(parents=True)
        params = {}
        staged = []
        for index, role in enumerate(ROLES):
            name = (
                f"{index:04d}-{role}.txt"
                if number < 3
                else f"{number * 100 + index:04d}-renamed.txt"
            )
            path = work / "inputs" / name
            path.write_text(f"content-{role}")
            params[role] = {"file": f"inputs/{name}"}
            staged.append(
                [f"inputs/{name}", "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()]
            )
        raw = {
            "name": "six-role",
            "run": "tests.work_cases.ChainedBoundedRecoveryWork",
            "with": params,
        }
        source = work / "config.yaml"
        source.write_text(yaml.safe_dump(raw))
        (work / "manifest.json").write_text(
            json.dumps(
                {
                    "bundle_version": 4,
                    "config": raw,
                    "staged": staged,
                    "code_identity": {"provider": "git", "revision": revision[0], "dirty": False},
                }
            )
        )
        command = ["python", "-m", "lambdaforge", "run", str(source)]
        if origin:
            command += ["--resume-execution", str(origin[0].execution_dir), "--accept-code-change"]
        (work.parent / "request.json").write_text(
            json.dumps({"job_id": owner, "work_dir": str(work), "command": command})
        )
        config = WorkConfig.from_yaml(source)
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source_config_path": str(source),
            "study_expected": False,
            "submission_mode": "asynchronous",
            "run_arguments": [],
        }
        if origin:
            metadata.update(
                recovery_owner_job="job-stage-1",
                recovery_execution_dir=str(origin[0].execution_dir),
                recovery_dependencies=[f"job-stage-{n}" for n in range(1, number)],
            )
        jobs.store.write(
            JobRecord(
                owner,
                profile.name,
                "local",
                None,
                JobState.FAILED,
                (),
                str(work),
                config.resources.to_dict(),
                now,
                now,
                config_path=str(source),
                job_type="work",
                metadata=metadata,
            )
        )
        monkeypatch.setenv("LAMBDAFORGE_JOB_ID", owner)
        configs.append(config)
        return config

    first = WorkRunner().run(stage(1))
    origin.append(first)
    second_config = stage(2)
    revision[0] = "fix-two"
    second = WorkRunner().run(
        second_config, resume_execution=first.execution_dir, accept_code_change=True
    )
    assert second.status == "failed"
    return SimpleNamespace(
        first=first, second=second, jobs=jobs, configs=configs, stage=stage, revision=revision
    )


def test_native_chain_keeps_all_roles_checkpoints_and_origin(chain):
    first = chain.first
    before = snapshot(first.execution_dir)
    bindings = original_file_bindings(first.execution_dir)
    assert set(bindings["six-role"]) == set(ROLES)
    for owner in ("job-stage-1", "job-stage-2"):
        handle = chain.jobs.retry(owner, dry_run=True, accept_code_change=True)
        assert handle.recovery_plan["runs"][0]["next_attempt"] == 3
    assert snapshot(first.execution_dir) == before
    for number in (3, 4):
        chain.revision[0] = f"fix-{number}"
        current = chain.stage(number)
        plan = WorkRunner().run(
            current, resume_execution=first.execution_dir, accept_code_change=True, dry_run=True
        )
        assert plan.execution_id == first.execution_id
        result = WorkRunner().run(
            current, resume_execution=first.execution_dir, accept_code_change=True
        )
        assert result.status == ("failed" if number == 3 else "succeeded")
        assert result.runs[-1].run_id == first.runs[0].run_id
        assert result.runs[-1].attempt_number == number
        assert result.runs[-1].parameters == first.runs[0].parameters
        assert original_file_bindings(first.execution_dir) == bindings
    for name, content in before.items():
        if (
            "/attempt-0001/" in name
            or name.endswith("execution.json")
            or name.endswith("producers/0.bin")
        ):
            assert Path(name).read_bytes() == content


@pytest.mark.parametrize(
    "damage",
    [
        "bytes",
        "roles",
        "owner",
        "lineage",
        "symlink",
        "escape",
        "bundle",
        "origin",
        "history",
        "scalar",
        "duplicate-role",
        "request",
        "raw-receipt",
        "placement",
    ],
)
def test_later_foreign_or_corrupt_attempt_is_not_skipped(chain, damage):

    second = chain.second.runs[-1]
    receipt = second.run_dir / "result.json"
    result = json.loads(receipt.read_text())
    work = chain.configs[1].source.parent
    target = Path(result["inputs"][0]["path"])
    if damage == "bytes":
        content = target.read_bytes()
        target.write_bytes(b"!" * len(content))
    elif damage == "roles":
        result["inputs"][0]["name"], result["inputs"][1]["name"] = (
            result["inputs"][1]["name"],
            result["inputs"][0]["name"],
        )
        receipt.write_text(json.dumps(result))
    elif damage == "owner":
        result["job_id"] = "job-forged"
        receipt.write_text(json.dumps(result))
    elif damage == "lineage":
        path = work.parent / "request.json"
        data = json.loads(path.read_text())
        data["command"][data["command"].index("--resume-execution") + 1] = "/foreign/execution"
        path.write_text(json.dumps(data))
    elif damage == "symlink":
        other = work / "copy.txt"
        other.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(other)
    elif damage == "escape":
        result["inputs"][0]["path"] = str(work / "../escape.txt")
        receipt.write_text(json.dumps(result))
    elif damage == "bundle":
        (work / "manifest.json").write_text("broken")
    elif damage == "origin":
        (chain.first.runs[0].run_dir / "result.json").unlink()
    elif damage == "scalar":
        result["parameters"]["invented_science"] = 1
        receipt.write_text(json.dumps(result))
    elif damage == "duplicate-role":
        result["inputs"].append(result["inputs"][0])
        receipt.write_text(json.dumps(result))
    elif damage == "request":
        (work.parent / "request.json").write_text("broken")
    elif damage == "raw-receipt":
        path = work / "manifest.json"
        raw = json.loads(path.read_text())
        raw["staged"][0][1] = "sha256:" + "f" * 64
        path.write_text(json.dumps(raw))
    elif damage == "placement":
        path = work.parent / "request.json"
        raw = json.loads(path.read_text())
        raw["cluster"] = "foreign-placement"
        path.write_text(json.dumps(raw))
    else:
        (chain.first.execution_dir / "recovery-history.jsonl").write_text("{}\n")
    before = snapshot(chain.first.execution_dir)
    with pytest.raises(ValueError) as error:
        WorkRunner().run(
            chain.configs[1],
            resume_execution=chain.first.execution_dir,
            accept_code_change=True,
            dry_run=True,
        )
    assert "Attempt" in str(error.value)
    assert snapshot(chain.first.execution_dir) == before


def test_legacy_later_input_fingerprint_uses_own_basename_without_changing_origin(chain):
    from lambdaforge.work.managed import fingerprint

    current = chain.stage(3)
    third = WorkRunner().run(
        current, resume_execution=chain.first.execution_dir, accept_code_change=True
    )
    result = third.runs[-1]
    path = result.run_dir / "result.json"
    raw = json.loads(path.read_text())
    for record in raw["inputs"]:
        digest, size = fingerprint(Path(record["path"]))
        record["sha256"] = "sha256:" + digest
        assert size == record["size_bytes"]
    path.write_text(json.dumps(raw))  # disposable persisted-legacy receipt fixture only
    before = snapshot(chain.first.execution_dir)
    bindings = original_file_bindings(chain.first.execution_dir)
    assert all(
        binding["logical_name"].startswith("000") for binding in bindings["six-role"].values()
    )
    WorkRunner().run(
        current, resume_execution=chain.first.execution_dir, accept_code_change=True, dry_run=True
    )
    assert snapshot(chain.first.execution_dir) == before


@pytest.mark.parametrize("capability", ["missing", "old", "current"])
def test_prepared_worker_contract_is_checked_before_scientific_handoff(capability):
    from lambdaforge.controlplane.ControlPlane import ControlPlane
    from lambdaforge.work.staged_input_identity import RECOVERY_BINDING_CONTRACT

    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(
            returncode=1 if capability == "missing" else 0,
            stdout=RECOVERY_BINDING_CONTRACT if capability == "current" else "bounded-role-v1",
        )

    transport = SimpleNamespace(run=run)
    if capability == "current":
        ControlPlane._verify_recovery_binding_runtime(transport, "owned/python")
    else:
        with pytest.raises(ValueError, match="Prepare/bootstrap"):
            ControlPlane._verify_recovery_binding_runtime(transport, "owned/python")
    assert len(calls) == 1 and calls[0][0][0] == "owned/python"
