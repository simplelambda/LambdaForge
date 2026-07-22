"""Real OS-signal integration tests for orchestrated process-tree cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard


@pytest.mark.process_integration
class TestProcessInterruptions:
    """Verify signal and launcher-death cleanup without leaving live descendants."""

    STARTUP_TIMEOUT_SECONDS = 20.0
    STOP_TIMEOUT_SECONDS = 12.0
    POLL_SECONDS = 0.05

    def test_group_interrupt_stops_worker_and_nested_descendant_cooperatively(
        self,
        tmp_path: Path,
    ) -> None:
        state_dir = tmp_path / "sigint"
        launcher = self._launch(state_dir, cooperative=True)
        identities: list[dict[str, Any]] = []
        try:
            self._wait_until_ready(launcher, state_dir)
            identities = self._all_process_identities(launcher, state_dir)

            self._send_interrupt(launcher, state_dir)
            return_code = launcher.wait(timeout=self.STOP_TIMEOUT_SECONDS)
            self._wait_until_dead(identities)

            assert return_code == 0
            result = json.loads((state_dir / "result.json").read_text(encoding="utf-8"))
            assert result["long-running"] == 0
            assert not (state_dir / "signal.request").exists()
            assert not list(state_dir.glob("*.tmp"))
        finally:
            self._emergency_cleanup(launcher, identities)

    def test_hard_launcher_termination_reaps_every_known_descendant(
        self,
        tmp_path: Path,
    ) -> None:
        state_dir = tmp_path / "launcher-death"
        launcher = self._launch(state_dir)
        identities: list[dict[str, Any]] = []
        try:
            self._wait_until_ready(launcher, state_dir)
            identities = self._all_process_identities(launcher, state_dir)
            assert {identity["role"] for identity in identities} >= {
                "launcher",
                "worker",
                "descendant",
            }

            self._hard_terminate(launcher)
            launcher.wait(timeout=self.STOP_TIMEOUT_SECONDS)
            self._wait_until_dead(identities)

            assert not list(state_dir.glob("*.tmp"))
        finally:
            self._emergency_cleanup(launcher, identities)

    @classmethod
    def _launch(
        cls,
        state_dir: Path,
        *,
        cooperative: bool = False,
    ) -> subprocess.Popen[bytes]:
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        python_paths = [str(repository / "src"), str(repository)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            "tests.fixtures.OrchestratorSignalLauncher",
            str(state_dir),
            "--grace-seconds",
            "2.0" if cooperative else "0.25",
        ]
        if cooperative:
            command.append("--cooperative")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        state_dir.mkdir(parents=True, exist_ok=True)
        with (state_dir / "launcher.log").open("wb") as log:
            return subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )

    @classmethod
    def _wait_until_ready(
        cls,
        launcher: subprocess.Popen[bytes],
        state_dir: Path,
    ) -> None:
        deadline = time.monotonic() + cls.STARTUP_TIMEOUT_SECONDS
        ready = state_dir / "ready"
        while time.monotonic() < deadline:
            if ready.exists():
                return
            return_code = launcher.poll()
            if return_code is not None:
                raise AssertionError(
                    f"Lifecycle-test launcher exited early with {return_code}: "
                    f"{cls._read_log(state_dir)}"
                )
            time.sleep(cls.POLL_SECONDS)
        raise TimeoutError(
            f"Lifecycle-test launcher did not become ready: {cls._read_log(state_dir)}"
        )

    @classmethod
    def _all_process_identities(
        cls,
        launcher: subprocess.Popen[bytes],
        state_dir: Path,
    ) -> list[dict[str, Any]]:
        identities = [
            json.loads((state_dir / name).read_text(encoding="utf-8"))
            for name in ("launcher.json", "worker.json", "descendant.json")
        ]
        known_pids = {int(identity["pid"]) for identity in identities}
        try:
            descendants = psutil.Process(launcher.pid).children(recursive=True)
        except psutil.Error:
            descendants = []
        for process in descendants:
            if process.pid in known_pids:
                continue
            try:
                identities.append(
                    {
                        "role": "auxiliary",
                        "pid": process.pid,
                        "create_time": process.create_time(),
                    }
                )
            except psutil.Error:
                pass
        return identities

    @staticmethod
    def _send_interrupt(launcher: subprocess.Popen[bytes], state_dir: Path) -> None:
        """Send group SIGINT on POSIX and targeted Python SIGBREAK on Windows.

        Windows console control events are group-wide and native numerical
        runtimes may abort on CTRL_BREAK before Python handles it. The external
        launcher therefore raises SIGBREAK in its own process after this test writes
        a request, exercising the real signal handler without touching workers.
        """
        if os.name == "nt":
            (state_dir / "signal.request").write_text("SIGBREAK\n", encoding="utf-8")
        else:
            process_group = os.getpgid(launcher.pid)  # type: ignore[attr-defined]
            os.killpg(process_group, signal.SIGINT)  # type: ignore[attr-defined]

    @staticmethod
    def _hard_terminate(launcher: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            launcher.terminate()
        else:
            launcher.send_signal(int(getattr(signal, "SIGKILL", 9)))

    @classmethod
    def _wait_until_dead(cls, identities: list[dict[str, Any]]) -> None:
        deadline = time.monotonic() + cls.STOP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            alive = [identity for identity in identities if cls._identity_is_alive(identity)]
            if not alive:
                return
            time.sleep(cls.POLL_SECONDS)
        alive = [identity for identity in identities if cls._identity_is_alive(identity)]
        raise AssertionError(f"Residual lifecycle-test processes: {alive}")

    @staticmethod
    def _identity_is_alive(identity: dict[str, Any]) -> bool:
        try:
            process = psutil.Process(int(identity["pid"]))
            if abs(process.create_time() - float(identity["create_time"])) > 0.02:
                return False
            return process.is_running()
        except psutil.Error:
            return False

    @classmethod
    def _emergency_cleanup(
        cls,
        launcher: subprocess.Popen[bytes],
        identities: list[dict[str, Any]],
    ) -> None:
        for identity in reversed(identities):
            if cls._identity_is_alive(identity):
                ProcessGuard().terminate_process_tree(
                    int(identity["pid"]),
                    grace_seconds=0.0,
                    include_parent=True,
                )
        if launcher.poll() is None:
            ProcessGuard().terminate_process_tree(
                launcher.pid,
                grace_seconds=0.0,
                include_parent=True,
            )
            try:
                launcher.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=2.0)

    @staticmethod
    def _read_log(state_dir: Path) -> str:
        path = state_dir / "launcher.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
