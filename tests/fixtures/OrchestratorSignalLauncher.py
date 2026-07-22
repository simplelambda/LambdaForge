"""External launcher for real signal and hard-termination integration tests."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import psutil

from lambdaforge.training import TrainingJob, TrainingOrchestrator
from tests.fixtures.LongRunningProcessJob import LongRunningProcessJob


class OrchestratorSignalLauncher:
    """Run one long CPU job in an isolated process suitable for OS signalling."""

    @classmethod
    def main(cls, arguments: Sequence[str] | None = None) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("state_dir", type=Path)
        parser.add_argument("--grace-seconds", type=float, default=0.25)
        parser.add_argument("--cooperative", action="store_true")
        parsed = parser.parse_args(arguments)
        parsed.state_dir.mkdir(parents=True, exist_ok=True)
        cls._write_json(
            parsed.state_dir / "launcher.json",
            {
                "role": "launcher",
                "pid": os.getpid(),
                "create_time": psutil.Process(os.getpid()).create_time(),
            },
        )
        cls._start_signal_relay(parsed.state_dir)

        orchestrator = TrainingOrchestrator(
            start_method="spawn",
            grace_seconds=parsed.grace_seconds,
            poll_seconds=0.02,
            cpu_threads_per_job=1,
            cpu_interop_threads_per_job=1,
            cpu_cores_per_job=None,
        )
        job = TrainingJob(
            "long-running",
            LongRunningProcessJob(
                parsed.state_dir,
                cooperative=parsed.cooperative,
            ),
        )
        try:
            exit_codes = orchestrator.run_scheduled([job], slots=[None])
        except BaseException as error:
            cls._write_json(
                parsed.state_dir / "error.json",
                {"type": type(error).__name__, "message": str(error)},
            )
            raise

        cls._write_json(parsed.state_dir / "result.json", exit_codes)
        return 0

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _start_signal_relay(state_dir: Path) -> None:
        """Deliver a targeted named signal when the test publishes a request."""
        request_path = state_dir / "signal.request"

        def relay() -> None:
            while True:
                if request_path.exists():
                    signal_name = request_path.read_text(encoding="utf-8").strip()
                    request_path.unlink(missing_ok=True)
                    signum = getattr(signal, signal_name)
                    signal.raise_signal(signum)
                time.sleep(0.02)

        threading.Thread(
            target=relay,
            name="lambdaforge-signal-test-relay",
            daemon=True,
        ).start()


if __name__ == "__main__":
    raise SystemExit(OrchestratorSignalLauncher.main())
