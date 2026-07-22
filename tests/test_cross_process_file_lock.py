"""Focused cross-platform tests for reusable process file coordination."""

from __future__ import annotations

import multiprocessing
import pickle
from pathlib import Path

import pytest

from lambdaforge.data.cache.CacheFileLock import CacheFileLock
from lambdaforge.runtime import CrossProcessFileLock
from tests.fixtures.FileLockHolderJob import FileLockHolderJob


class TestCrossProcessFileLock:
    """Verify validation, compatibility, contention and crash release."""

    @pytest.mark.parametrize(
        ("argument", "value", "error_type"),
        [
            ("shared", 1, TypeError),
            ("timeout_seconds", True, ValueError),
            ("timeout_seconds", 0.0, ValueError),
            ("timeout_seconds", float("inf"), ValueError),
            ("timeout_seconds", float("nan"), ValueError),
            ("poll_interval_seconds", -0.1, ValueError),
            ("poll_interval_seconds", float("nan"), ValueError),
        ],
    )
    def test_invalid_configuration_is_rejected(
        self,
        tmp_path: Path,
        argument: str,
        value: object,
        error_type: type[Exception],
    ) -> None:
        arguments: dict[str, object] = {
            "shared": False,
            "timeout_seconds": 1.0,
            "poll_interval_seconds": 0.01,
        }
        arguments[argument] = value

        with pytest.raises(error_type):
            CrossProcessFileLock(tmp_path / "invalid.lock", **arguments)  # type: ignore[arg-type]

    def test_symlink_parent_is_rejected_without_writing_through_it(
        self,
        tmp_path: Path,
    ) -> None:
        external = tmp_path / "external"
        external.mkdir()
        linked = tmp_path / "linked"
        try:
            linked.symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are not available in this environment.")

        with pytest.raises(ValueError, match="safe directory"):
            CrossProcessFileLock(
                linked / "unsafe.lock",
                shared=False,
                timeout_seconds=1.0,
                poll_interval_seconds=0.01,
            ).acquire()

        assert not (external / "unsafe.lock").exists()

    def test_context_manager_and_cache_adapter_preserve_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lock = CacheFileLock(
            tmp_path / "cache.lock",
            shared=False,
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        )

        assert isinstance(lock, CrossProcessFileLock)
        assert not lock.acquired
        with lock as lease:
            assert lease is lock
            assert lock.acquired
            with pytest.raises(RuntimeError, match="already acquired"):
                lock.acquire()
            with pytest.raises(RuntimeError, match="cannot be serialized"):
                pickle.dumps(lock)
        assert not lock.acquired
        lock.release()

    def test_exclusive_lock_contends_across_spawned_process(
        self,
        tmp_path: Path,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        acquired_event = context.Event()
        release_event = context.Event()
        path = tmp_path / "exclusive.lock"
        process = context.Process(
            target=FileLockHolderJob(
                path,
                shared=False,
                acquired_event=acquired_event,
                release_event=release_event,
            )
        )
        try:
            process.start()
            assert acquired_event.wait(10.0)
            with pytest.raises(TimeoutError, match="Timed out acquiring file lock"):
                CrossProcessFileLock(
                    path,
                    shared=False,
                    timeout_seconds=0.2,
                    poll_interval_seconds=0.01,
                ).acquire()
            release_event.set()
            process.join(timeout=10.0)
            assert process.exitcode == 0
        finally:
            release_event.set()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)

        with CrossProcessFileLock(
            path,
            shared=False,
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        ):
            pass

    def test_shared_locks_coexist_but_exclude_a_writer(
        self,
        tmp_path: Path,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        acquired_event = context.Event()
        release_event = context.Event()
        path = tmp_path / "shared.lock"
        process = context.Process(
            target=FileLockHolderJob(
                path,
                shared=True,
                acquired_event=acquired_event,
                release_event=release_event,
            )
        )
        try:
            process.start()
            assert acquired_event.wait(10.0)
            with CrossProcessFileLock(
                path,
                shared=True,
                timeout_seconds=1.0,
                poll_interval_seconds=0.01,
            ):
                pass
            with pytest.raises(TimeoutError):
                CrossProcessFileLock(
                    path,
                    shared=False,
                    timeout_seconds=0.2,
                    poll_interval_seconds=0.01,
                ).acquire()
            release_event.set()
            process.join(timeout=10.0)
            assert process.exitcode == 0
        finally:
            release_event.set()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)

    def test_operating_system_releases_lock_after_process_crash(
        self,
        tmp_path: Path,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        acquired_event = context.Event()
        unused_release_event = context.Event()
        path = tmp_path / "crash.lock"
        process = context.Process(
            target=FileLockHolderJob(
                path,
                shared=False,
                acquired_event=acquired_event,
                release_event=unused_release_event,
                crash_exit_code=73,
            )
        )
        try:
            process.start()
            assert acquired_event.wait(10.0)
            process.join(timeout=10.0)
            assert process.exitcode == 73
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)

        with CrossProcessFileLock(
            path,
            shared=False,
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        ) as lock:
            assert lock.acquired
