"""Remote project-mirror path contracts without contacting an external cluster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lambdaforge.controlplane import ClusterProfile, ControlPlane, LocalTransport
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.managed import (
    CANONICAL_FINGERPRINT_ALGORITHM,
    canonical_fingerprint,
    fingerprint,
)


def test_cluster_project_root_is_explicit_absolute_and_round_trips() -> None:
    profile = ClusterProfile.from_mapping(
        "gpu",
        {
            "transport": "ssh",
            "host": "gpu.invalid",
            "workspace": "/home/research/.lambdaforge",
            "project_root": "/home/research/project/",
        },
    )

    assert profile.project_root == "/home/research/project"
    assert profile.to_dict()["project_root"] == "/home/research/project"
    with pytest.raises(ValueError, match="non-root absolute"):
        ClusterProfile.from_mapping(
            "gpu",
            {
                "transport": "ssh",
                "host": "gpu.invalid",
                "workspace": "/home/research/.lambdaforge",
                "project_root": "project",
            },
        )


def test_large_project_input_maps_to_exact_remote_relative_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_dir = project / "experiments"
    data = project / "data" / "design"
    source_dir.mkdir(parents=True)
    data.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='consumer'\nversion='1'\n")
    (data / "first.txt").write_text("large-enough", encoding="utf-8")
    staged: list[tuple[Path, str]] = []
    shared: list[dict[str, object]] = []

    values = ExecutionBundleBuilder(max_inline_bytes=1)._stage_files(
        {"with": {"design": {"file": "../data/design"}}},
        source_dir,
        staged,
        project_root=project,
        remote_project_root="/scratch/research/project",
        shared_inputs=shared,
    )

    assert values["with"]["design"] == {"file": "/scratch/research/project/data/design"}
    assert staged == []
    assert shared[0]["project_relative"] == "data/design"
    assert shared[0]["fingerprint_algorithm"] == CANONICAL_FINGERPRINT_ALGORITHM
    assert shared[0]["sha256"] == canonical_fingerprint(data)[0]


def test_bundle_persists_remote_path_and_worker_identity_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiments = project / "experiments"
    data = project / "data" / "design"
    experiments.mkdir(parents=True)
    data.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    (data / "record.txt").write_text("shared input", encoding="utf-8")
    config = experiments / "work.yaml"
    config.write_text(
        "name: shared-bundle\n"
        "run: tests.work_cases.CompleteWork\n"
        "with:\n"
        "  source: {file: ../data/design}\n"
        "  count: 1\n",
        encoding="utf-8",
    )
    profile = ClusterProfile.from_mapping(
        "gpu",
        {
            "transport": "ssh",
            "host": "gpu.invalid",
            "workspace": "/remote/work",
            "environment": "existing",
            "project_root": "/scratch/research/project",
        },
    )

    bundle = ExecutionBundleBuilder(tmp_path / "bundles", max_inline_bytes=1).build(config, profile)

    path_context = json.loads(
        (bundle.directory / ".lambdaforge-paths.json").read_text(encoding="utf-8")
    )
    identities = json.loads(
        (bundle.directory / ".lambdaforge-shared-inputs.json").read_text(encoding="utf-8")
    )
    assert path_context == {
        "path_context_version": 1,
        "project_root": "/scratch/research/project",
        "source_relative": "experiments",
    }
    assert identities == [dict(bundle.shared_inputs[0])]
    assert "/scratch/research/project/data/design" in bundle.config_path.read_text()


def test_large_input_requires_a_project_mirror_or_managed_dataset(tmp_path: Path) -> None:
    data = tmp_path / "large.bin"
    data.write_bytes(b"too large")

    with pytest.raises(ValueError, match="project_root.*managed dataset"):
        ExecutionBundleBuilder(max_inline_bytes=1)._stage_files(
            {"with": {"data": {"file": str(data)}}},
            tmp_path,
            [],
            project_root=tmp_path,
            remote_project_root=None,
            shared_inputs=[],
        )


def test_shared_input_probe_accepts_exact_content_and_rejects_stale_content(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    item = data / "record.json"
    item.write_text('{"value": 1}', encoding="utf-8")
    digest, size = fingerprint(data)
    expected = (
        {
            "configured": "../data",
            "remote_path": str(data),
            "kind": "directory",
            "sha256": digest,
            "size_bytes": size,
        },
    )
    profile = ClusterProfile.from_mapping(
        "probe",
        {
            "transport": "ssh",
            "host": "localhost",
            "workspace": str(tmp_path / ".lambdaforge"),
            "python": sys.executable,
            "project_root": str(tmp_path),
        },
    )

    ControlPlane._verify_shared_inputs(LocalTransport(), profile, expected)
    item.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="differs.*stale or partial"):
        ControlPlane._verify_shared_inputs(LocalTransport(), profile, expected)


def test_shared_input_probe_uses_portable_versioned_directory_identity(tmp_path: Path) -> None:
    data = tmp_path / "design"
    (data / "nested").mkdir(parents=True)
    (data / "empty").mkdir()
    (data / "z.txt").write_bytes(b"last")
    (data / "nested" / "first.txt").write_bytes(b"first")
    digest, size = canonical_fingerprint(data)
    expected = (
        {
            "configured": "../data/design",
            "remote_path": str(data),
            "kind": "directory",
            "fingerprint_algorithm": CANONICAL_FINGERPRINT_ALGORITHM,
            "sha256": digest,
            "size_bytes": size,
        },
    )
    profile = ClusterProfile.from_mapping(
        "probe",
        {
            "transport": "ssh",
            "host": "localhost",
            "workspace": str(tmp_path / ".lambdaforge"),
            "python": sys.executable,
            "project_root": str(tmp_path),
        },
    )

    ControlPlane._verify_shared_inputs(LocalTransport(), profile, expected)


def test_worker_rechecks_shared_input_against_bundle_identity(tmp_path: Path) -> None:
    data = tmp_path / "shared.txt"
    data.write_text("original", encoding="utf-8")
    digest, size = fingerprint(data)
    marker = tmp_path / ".lambdaforge-shared-inputs.json"
    marker.write_text(
        json.dumps(
            [
                {
                    "configured": "../data/shared.txt",
                    "remote_path": str(data),
                    "kind": "file",
                    "sha256": digest,
                    "size_bytes": size,
                }
            ]
        ),
        encoding="utf-8",
    )
    config = WorkConfig.from_mapping(
        {
            "name": "shared-worker-check",
            "run": "tests.work_cases.CompleteWork",
            "with": {"source": {"file": str(data)}, "count": 1},
        },
        source=tmp_path / "config.yaml",
    )
    assert WorkRunner().plan(config).name == "shared-worker-check"

    data.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed before execution"):
        WorkRunner().plan(config)
