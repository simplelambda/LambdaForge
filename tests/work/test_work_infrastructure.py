"""Focused managed-infrastructure tests for scientific Work."""

from __future__ import annotations

import gzip
import io
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.cache import RateLimit, WorkCache
from lambdaforge.work.tools import ToolExecutionError, ToolService


def _config(tmp_path: Path, target: str, name: str) -> WorkConfig:
    path = tmp_path / "work.yaml"
    path.write_text(yaml.safe_dump({"name": name, "run": target}), encoding="utf-8")
    return WorkConfig.from_yaml(path)


def test_cache_build_integrity_reuse_and_same_key_lock(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")
    calls: list[int] = []

    def build(target: Path) -> None:
        calls.append(1)
        target.write_text("content", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = tuple(
            pool.map(lambda _index: cache.file("nested/value.txt", build=build), range(4))
        )
    assert calls == [1]
    assert {value.sha256 for value in files} == {files[0].sha256}
    assert files[0].read_text() == "content"
    assert pickle.loads(pickle.dumps(files[0])).read_text() == "content"

    files[0].path.write_text("corrupt", encoding="utf-8")
    rebuilt = cache.file("nested/value.txt", build=build)
    assert rebuilt.read_text() == "content"
    assert len(calls) == 2
    array = cache.file(
        "arrays/value.npy",
        build=lambda target: np.save(target, np.asarray([1, 2]), allow_pickle=False),
    )
    assert np.load(array, allow_pickle=False).tolist() == [1, 2]
    with pytest.raises(ValueError, match="relative path"):
        cache.file("../escape", build=build)


def test_cache_put_get_is_a_small_safe_key_value_api(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")

    cache.put("summary", {"count": 3, "labels": ["a", "b"]})
    cache.put("message", "ready")
    cache.put("payload", b"\x00\x01")

    assert cache.get("summary") == {"count": 3, "labels": ["a", "b"]}
    assert cache.get("message") == "ready"
    assert cache.get("payload") == b"\x00\x01"
    assert cache.get("absent") is None
    assert cache.get("absent", "fallback") == "fallback"

    cache.put("message", "updated")
    assert cache.get("message") == "updated"
    with pytest.raises(TypeError, match="strict JSON"):
        cache.put("unsafe", {"tuple": (1, 2)})


def test_cache_get_does_not_misinterpret_managed_files_as_values(tmp_path: Path) -> None:
    cache = WorkCache(tmp_path / "cache")
    cache.file("artifact.txt", build=lambda target: target.write_text("artifact"))

    with pytest.raises(TypeError, match=r"cache\.put"):
        cache.get("artifact.txt")


def test_cache_fetch_gzip_retry_and_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = gzip.compress(b"downloaded")
    attempts = 0

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def open_fixture(*args: object, **kwargs: object) -> Response:
        nonlocal attempts
        del args, kwargs
        attempts += 1
        if attempts == 1:
            raise OSError("temporary")
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", open_fixture)
    monkeypatch.setattr("time.sleep", lambda _value: None)
    cache = WorkCache(tmp_path / "cache")
    result = cache.fetch(
        "https://example.invalid/evidence.gz",
        key="downloads/evidence.txt",
        retries=1,
        retry_backoff=0,
        decompress="gzip",
    )
    assert result.read_bytes() == b"downloaded"
    assert attempts == 2

    now = [0.0]
    sleeps: list[float] = []

    def advance(value: float) -> None:
        sleeps.append(value)
        now[0] += value

    limiter = RateLimit(2.0, clock=lambda: now[0], sleep=advance)
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [0.5]


def test_storage_clean_previews_and_removes_only_reconstructible_work_cache(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache = WorkCache(cache_root / "work" / "scientific-identity")
    cache.file("entry.txt", build=lambda target: target.write_text("value"))
    descriptor = {
        "state_root": str(tmp_path / "state"),
        "cache_root": str(cache_root),
        "run_root": str(tmp_path / "jobs"),
        "dataset_root": str(tmp_path / "datasets"),
    }
    preview = StorageOperations.gc(descriptor, {}, apply=False)
    assert preview["candidates"][0]["category"] == "work_cache"
    assert preview["candidates"][0]["reason"] == "reconstructible"
    assert (cache_root / "work/scientific-identity").is_dir()
    applied = StorageOperations.gc(descriptor, {}, apply=True)
    assert applied["applied"] is True
    assert not (cache_root / "work/scientific-identity").exists()


def test_managed_outputs_checkpoint_tools_and_provenance(tmp_path: Path) -> None:
    result = WorkRunner().run(
        _config(tmp_path, "tests.work_cases.ManagedInfrastructureWork", "infrastructure")
    )
    assert result.status == "succeeded"
    run = result.runs[0]
    assert run.primary_result == {"returncode": 0, "cached": "cache"}
    assert {artifact.name for artifact in run.artifacts} == {"report", "evidence"}
    assert "external success" in (run.run_dir / "work.log").read_text(encoding="utf-8")
    environment = json.loads((run.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["external_tools"][0]["version"]


def test_managed_output_can_publish_a_safe_explicit_copy(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "report.txt"
    configured_destination = "results/report.txt"
    config = WorkConfig.from_mapping(
        {
            "name": "published-output",
            "run": "tests.work_cases.PublishedOutputWork",
            "with": {"destination": configured_destination, "value": "first"},
        },
        source=tmp_path / "work.yaml",
    )

    first = WorkRunner().run(config)
    assert first.status == "succeeded"
    assert destination.read_text(encoding="utf-8") == "first"
    assert first.runs[0].artifacts[0].metadata["published_to"] == str(destination)

    changed = WorkConfig.from_mapping(
        {
            "name": "published-output",
            "run": "tests.work_cases.PublishedOutputWork",
            "with": {"destination": configured_destination, "value": "second"},
        },
        source=tmp_path / "work.yaml",
    )
    refused = WorkRunner().run(changed)
    assert refused.status == "failed"
    assert "already exists" in refused.runs[0].failure["message"]
    assert destination.read_text(encoding="utf-8") == "first"

    replaced = WorkConfig.from_mapping(
        {
            "name": "published-output",
            "run": "tests.work_cases.PublishedOutputWork",
            "with": {
                "destination": configured_destination,
                "value": "second",
                "overwrite": True,
            },
        },
        source=tmp_path / "work.yaml",
    )
    assert WorkRunner().run(replaced).status == "succeeded"
    assert destination.read_text(encoding="utf-8") == "second"


def test_relative_output_uses_yaml_directory_locally_and_remote_project_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_project = tmp_path / "local-project"
    local_experiments = local_project / "experiments"
    local_experiments.mkdir(parents=True)
    (local_project / "pyproject.toml").write_text(
        '[project]\nname = "paths-test"\nversion = "1"\n', encoding="utf-8"
    )
    local = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "local-relative-output",
                "run": "tests.work_cases.PublishedOutputWork",
                "with": {"destination": "../data/report.txt"},
            },
            source=local_experiments / "work.yaml",
        )
    )
    assert local.status == "succeeded"
    assert (local_project / "data/report.txt").read_text(encoding="utf-8") == "published"

    bundle = tmp_path / "remote-job" / "work"
    remote_project = tmp_path / "remote-project"
    bundle.mkdir(parents=True)
    (bundle / "pyproject.toml").write_text(
        '[project]\nname = "paths-test"\nversion = "1"\n', encoding="utf-8"
    )
    (bundle / ".lambdaforge-paths.json").write_text(
        json.dumps(
            {
                "path_context_version": 1,
                "project_root": str(remote_project),
                "source_relative": "experiments",
            }
        ),
        encoding="utf-8",
    )
    remote = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "remote-relative-output",
                "run": "tests.work_cases.PublishedOutputWork",
                "with": {"destination": "../data/report.txt"},
            },
            source=bundle / "config.yaml",
        )
    )
    assert remote.status == "succeeded"
    assert (remote_project / "data/report.txt").read_text(encoding="utf-8") == "published"

    marker = bundle / ".lambdaforge-paths.json"
    marker.unlink()
    monkeypatch.setenv("LAMBDAFORGE_BUNDLE", "1")
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "unmapped-remote")
    unmapped = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "unmapped-relative-output",
                "run": "tests.work_cases.PublishedOutputWork",
                "with": {"destination": "relative/report.txt"},
            },
            source=bundle / "config.yaml",
        )
    )
    assert unmapped.status == "failed"
    assert "requires a cluster project_root" in unmapped.runs[0].failure["message"]


def test_managed_directory_can_publish_a_safe_explicit_copy(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "evidence"
    result = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "published-directory",
                "run": "tests.work_cases.PublishedDirectoryWork",
                "with": {"destination": str(destination)},
            },
            source=tmp_path / "work.yaml",
        )
    )

    assert result.status == "succeeded"
    assert (destination / "summary.txt").read_text(encoding="utf-8") == "complete"


def test_output_publication_rejects_symlink_ancestors(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    result = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "unsafe-publication",
                "run": "tests.work_cases.PublishedOutputWork",
                "with": {"destination": str(link / "report.txt")},
            },
            source=tmp_path / "work.yaml",
        )
    )

    assert result.status == "failed"
    assert "symbolic links" in result.runs[0].failure["message"]
    assert not (outside / "report.txt").exists()


def test_directory_publication_cannot_replace_its_project_owner(tmp_path: Path) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("safe", encoding="utf-8")
    result = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "unsafe-directory-publication",
                "run": "tests.work_cases.PublishedDirectoryWork",
                "with": {"destination": str(tmp_path), "overwrite": True},
            },
            source=tmp_path / "work.yaml",
        )
    )

    assert result.status == "failed"
    assert "authoritative source" in result.runs[0].failure["message"]
    assert marker.read_text(encoding="utf-8") == "safe"


def test_missing_managed_output_fails_finalization_without_registration(tmp_path: Path) -> None:
    result = WorkRunner().run(
        _config(tmp_path, "tests.work_cases.MissingManagedOutputWork", "missing-output")
    )
    assert result.status == "failed"
    assert result.runs[0].artifacts == ()
    assert "Managed path does not exist" in result.runs[0].failure["message"]


def test_tool_service_streams_bounded_output_and_scopes_threads() -> None:
    messages: list[tuple[str, str]] = []

    class Log:
        def emit(self, message: object, *, level: str = "info") -> None:
            messages.append((level, str(message)))

    service = ToolService(Log())
    python = service.require(sys.executable, version_args=["--version"])
    result = service.run(
        [
            python,
            "-c",
            "import os,sys; print(os.environ['OMP_NUM_THREADS']); "
            "print(os.environ['EXPLICIT'], file=sys.stderr)",
        ],
        threads=3,
        env={"EXPLICIT": "yes"},
    )
    assert result.stdout.strip() == "3"
    assert result.stderr.strip() == "yes"
    assert os.environ.get("EXPLICIT") is None
    assert ("info", "3") in messages
    assert ("warning", "yes") in messages
    with pytest.raises(ToolExecutionError, match="exited with code 4"):
        service.run([python, "-c", "raise SystemExit(4)"])


def test_map_recomputes_only_item_with_removed_managed_dependency(tmp_path: Path) -> None:
    config = _config(tmp_path, "tests.work_cases.SelectiveMapWork", "selective-map")
    first = WorkRunner().run(config)
    assert first.status == "failed"
    cache_files = tuple((tmp_path / ".lambdaforge/cache/work").rglob("items/b.txt"))
    assert len(cache_files) == 1
    cache_files[0].unlink()

    second = WorkRunner().run(config)
    assert second.status == "succeeded"
    assert second.runs[0].primary_result == {
        "calls": ["a", "b", "b"],
        "values": ["a", "b"],
    }
