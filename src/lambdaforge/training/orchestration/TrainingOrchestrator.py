"""Multi-process training job orchestrator."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import torch.multiprocessing as mp

from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.WindowsJobObject import WindowsJobObject


class TrainingOrchestrator:
    r"""Orchestrates multiple independent training jobs in separate processes.

    Responsibilities
    ----------------
    - Start one subprocess per job.
    - Provide a shared stop event to every job.
    - Intercept SIGINT and SIGTERM, request a graceful stop, and force-kill
      processes that do not exit within the grace period.
    - Assign GPUs to each job without ever modifying the **parent**
      ``CUDA_VISIBLE_DEVICES``.

    Device assignment
    -----------------
    Each job can request specific GPUs via ``TrainingJob.devices``.
    Indices are always **logical** — relative to the ``CUDA_VISIBLE_DEVICES``
    of the parent process at orchestrator creation time.

    The orchestrator translates the requested logical indices into the
    physical GPU identifiers listed in ``CUDA_VISIBLE_DEVICES``, then passes
    that restricted string to the child process. The child sets its own
    ``CUDA_VISIBLE_DEVICES`` *before* CUDA is initialised, so the rest of the
    training code is unaffected and ``LightningTrainConfig.devices`` should
    use the logical indices ``[0, 1, ...]`` within the restricted set (or
    simply ``"auto"``).

    Examples — assume parent has ``CUDA_VISIBLE_DEVICES=4,7,9``:

    +---------------------+-------------------------------+
    | ``job.devices``     | child ``CUDA_VISIBLE_DEVICES`` |
    +=====================+===============================+
    | ``None``            | ``"4,7,9"`` (unchanged)       |
    +---------------------+-------------------------------+
    | ``[0]``             | ``"4"``                       |
    +---------------------+-------------------------------+
    | ``[1]``             | ``"7"``                       |
    +---------------------+-------------------------------+
    | ``[1, 2]``          | ``"7,9"``                     |
    +---------------------+-------------------------------+

    If ``CUDA_VISIBLE_DEVICES`` is not set in the parent, the indices are
    treated as physical GPU numbers directly.

    Parameters
    ----------
    start_method : str
        Multiprocessing start method. ``"spawn"`` is required for CUDA.
    grace_seconds : float
        Seconds to wait for a graceful shutdown before force-killing.
    poll_seconds : float
        Polling interval when waiting for processes to finish.
    cpu_threads_per_job : int | None
        CPU threads assigned to each training subprocess. ``None`` leaves the
        environment unchanged. Low values are important when many independent
        trainings share one node.
    cpu_interop_threads_per_job : int | None
        PyTorch inter-op threads assigned to each training subprocess.
    cpu_cores_per_job : int | None
        CPU affinity cores assigned to each training subprocess and inherited
        by descendants. ``None`` disables OS-level affinity.
    """

    def __init__(
        self,
        start_method: str = "spawn",
        grace_seconds: float = 15.0,
        poll_seconds: float = 0.5,
        cpu_threads_per_job: int | None = 1,
        cpu_interop_threads_per_job: int | None = 1,
        cpu_cores_per_job: int | None = 1,
    ) -> None:
        self.start_method = start_method
        self.grace_seconds = grace_seconds
        self.poll_seconds = poll_seconds
        self.cpu_threads_per_job = cpu_threads_per_job
        self.cpu_interop_threads_per_job = cpu_interop_threads_per_job
        self.cpu_cores_per_job = cpu_cores_per_job

        self.context: Any = mp.get_context(start_method)
        self.stop_event = self.context.Event()
        self.processes: list[tuple[str, mp.Process]] = []

        # Snapshot parent CUDA_VISIBLE_DEVICES once, before any subprocess
        # is spawned. This is the reference for all device-index translation.
        self._parent_cvd: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._available_cpu_ids = self._read_available_cpu_ids()
        self._original_cpu_affinity: list[int] | None = None
        self._windows_job: WindowsJobObject | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, jobs: Sequence[TrainingJob]) -> dict[str, int | None]:
        """Start **all** jobs at once and wait for them to complete.

        Each job uses its own :attr:`TrainingJob.devices`. Use this when the
        number of jobs already matches the available capacity. To cap
        concurrency (e.g. "N jobs per GPU"), use :meth:`run_scheduled`.

        Returns
        -------
        dict[str, int | None]
            Exit code per job name.
        """
        self._validate_jobs(jobs)
        self.stop_event.clear()
        self.processes = []
        self._windows_job = WindowsJobObject()
        self._install_signal_handlers()
        self._apply_parent_cpu_affinity(slot_count=len(jobs))

        try:
            for job_index, job in enumerate(jobs):
                self._launch(job, job.devices, slot_index=job_index)

            while any(p.is_alive() for _, p in self.processes):
                if self.stop_event.is_set():
                    self._stop_with_grace(self.processes)
                    break

                time.sleep(self.poll_seconds)

            for _, process in self.processes:
                process.join()

            return {name: process.exitcode for name, process in self.processes}

        finally:
            self._cleanup_running_processes()
            self._close_windows_job()
            self._restore_parent_cpu_affinity()
            self._restore_signal_handlers()

    def run_scheduled(
        self,
        jobs: Sequence[TrainingJob],
        slots: Sequence[list[int] | None],
        on_job_finished: Callable[[str, int | None], None] | None = None,
    ) -> dict[str, int | None]:
        """Run ``jobs`` across a fixed pool of ``slots``, one job per slot.

        A *slot* is a device assignment (a list of logical GPU indices, or
        ``None`` for the full parent set). At most ``len(slots)`` jobs run
        concurrently; each running job is bound to a specific slot, so its
        devices are the slot's — **not** ``job.devices``. This is what
        guarantees "at most K jobs per GPU": build ``K`` slots per GPU, and no
        GPU can ever host more than ``K`` jobs, unlike a global concurrency cap
        with round-robin assignment.

        When a job finishes, its slot is freed and the next queued job starts
        on it. On SIGINT/SIGTERM, launching stops, running jobs are asked to
        stop gracefully (and force-killed after ``grace_seconds``), and
        not-yet-started jobs get exit code ``None``.

        ``on_job_finished`` is called in the parent process after each launched
        job exits. It is used by the experiment framework to refresh aggregates
        and plots while a long sweep is still running.

        Returns
        -------
        dict[str, int | None]
            Exit code per job name (``None`` if never launched).
        """
        if not slots:
            raise ValueError("run_scheduled requires at least one slot.")

        self._validate_jobs(jobs)
        for slot in slots:
            if slot is not None and any(device < 0 for device in slot):
                raise ValueError("Scheduler slot device indices must be non-negative.")

        self.stop_event.clear()
        self.processes = []
        self._windows_job = WindowsJobObject()
        self._install_signal_handlers()
        self._apply_parent_cpu_affinity(slot_count=len(slots))

        pending = deque(jobs)
        free_slots = deque(range(len(slots)))
        running: dict[int, tuple[str, mp.Process]] = {}
        exit_codes: dict[str, int | None] = {}

        try:
            while pending or running:
                while free_slots and pending and not self.stop_event.is_set():
                    slot_index = free_slots.popleft()
                    job = pending.popleft()
                    process = self._launch(job, slots[slot_index], slot_index=slot_index)
                    running[slot_index] = (job.name, process)

                if self.stop_event.is_set():
                    self._stop_with_grace(list(running.values()))
                    for name, process in running.values():
                        process.join()
                        exit_codes[name] = process.exitcode
                        if on_job_finished is not None:
                            on_job_finished(name, process.exitcode)
                    for job in pending:
                        exit_codes[job.name] = None
                    return exit_codes

                finished = [
                    slot_index
                    for slot_index, (_, process) in running.items()
                    if not process.is_alive()
                ]

                for slot_index in finished:
                    name, process = running.pop(slot_index)
                    process.join()
                    exit_codes[name] = process.exitcode
                    if on_job_finished is not None:
                        on_job_finished(name, process.exitcode)
                    free_slots.append(slot_index)

                if not finished:
                    time.sleep(self.poll_seconds)

            return exit_codes

        finally:
            self._cleanup_running_processes()
            self._close_windows_job()
            self._restore_parent_cpu_affinity()
            self._restore_signal_handlers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_jobs(jobs: Sequence[TrainingJob]) -> None:
        names = [job.name for job in jobs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Training job names must be unique: {duplicates}.")

    def _install_signal_handlers(self) -> None:
        self._previous_sigint = signal.getsignal(signal.SIGINT)
        self._previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)

    def _restore_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._previous_sigint)
        signal.signal(signal.SIGTERM, self._previous_sigterm)

    def _launch(self, job: TrainingJob, devices: list[int] | None, slot_index: int) -> mp.Process:
        """Start one job subprocess bound to ``devices`` and return it."""
        child_cvd = self._resolve_cuda_visible_devices(devices)
        cpu_affinity = self._resolve_cpu_affinity(slot_index)
        env_updates = self._child_env_updates(child_cvd)

        process = self.context.Process(
            target=TrainingOrchestrator._run_job,
            args=(
                job,
                self.stop_event,
                child_cvd,
                self.cpu_threads_per_job,
                self.cpu_interop_threads_per_job,
                cpu_affinity,
                self.grace_seconds,
            ),
            name=job.name,
        )
        with self._temporary_env(env_updates):
            process.start()
        if self._windows_job is not None:
            self._windows_job.assign(process.pid)
        self._set_child_cpu_affinity(process.pid, cpu_affinity)
        self.processes.append((job.name, process))
        return process

    def _resolve_cuda_visible_devices(self, devices: list[int] | None) -> str | None:
        """Translate logical device indices into a CUDA_VISIBLE_DEVICES string.

        Returns ``None`` when no restriction is requested, meaning the child
        inherits the parent environment unchanged.
        """
        if devices is None:
            return None

        if self._parent_cvd:
            available = [token.strip() for token in self._parent_cvd.split(",")]

            out_of_range = [i for i in devices if i >= len(available)]
            if out_of_range:
                raise ValueError(
                    f"Job requested device indices {out_of_range} but "
                    f"CUDA_VISIBLE_DEVICES only has {len(available)} entries "
                    f"({self._parent_cvd})."
                )

            return ",".join(available[i] for i in devices)

        # No CUDA_VISIBLE_DEVICES in parent → treat indices as physical IDs
        return ",".join(str(i) for i in devices)

    def _resolve_cpu_affinity(self, slot_index: int) -> list[int] | None:
        """Assign a stable CPU-core slice to one scheduler slot."""
        if self.cpu_cores_per_job is None:
            return None

        cores_per_job = int(self.cpu_cores_per_job)
        if cores_per_job < 1 or not self._available_cpu_ids:
            return None

        available = self._available_cpu_ids
        width = min(cores_per_job, len(available))
        start = (int(slot_index) * width) % len(available)
        return [available[(start + offset) % len(available)] for offset in range(width)]

    def _child_env_updates(self, cuda_visible_devices: str | None) -> dict[str, str]:
        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        updates = ProcessGuard().cpu_thread_env_updates(
            torch_threads=self.cpu_threads_per_job,
            interop_threads=self.cpu_interop_threads_per_job,
        )
        if cuda_visible_devices is not None:
            updates["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        return updates

    @contextlib.contextmanager
    def _temporary_env(self, updates: dict[str, str]):
        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _read_available_cpu_ids() -> list[int]:
        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        return ProcessGuard().available_cpu_ids()

    @staticmethod
    def _set_child_cpu_affinity(pid: int | None, cpu_affinity: list[int] | None) -> None:
        if pid is None or not cpu_affinity:
            return

        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        ProcessGuard().set_cpu_affinity(cpu_affinity, pid=pid)

    def _parent_cpu_affinity(self, slot_count: int) -> list[int] | None:
        if self.cpu_cores_per_job is None:
            return None

        cores_per_job = int(self.cpu_cores_per_job)
        if cores_per_job < 1 or slot_count < 1 or not self._available_cpu_ids:
            return None

        total_cores = min(len(self._available_cpu_ids), int(slot_count) * cores_per_job)
        return self._available_cpu_ids[:total_cores]

    def _apply_parent_cpu_affinity(self, slot_count: int) -> None:
        cpu_affinity = self._parent_cpu_affinity(slot_count)
        if not cpu_affinity:
            return

        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        guard = ProcessGuard()
        self._original_cpu_affinity = guard.available_cpu_ids()
        guard.set_cpu_affinity(cpu_affinity)
        print(
            "Orchestrator resource guard: "
            f"pid={os.getpid()} "
            f"max_parallel_jobs={slot_count} "
            f"cpu_affinity={guard.available_cpu_ids()} "
            f"cpu_cores_per_job={self.cpu_cores_per_job}",
            flush=True,
        )

    def _restore_parent_cpu_affinity(self) -> None:
        if not self._original_cpu_affinity:
            return

        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        ProcessGuard().set_cpu_affinity(self._original_cpu_affinity)
        self._original_cpu_affinity = None

    def _request_stop(self, signum, frame) -> None:
        self.stop_event.set()

    def _stop_with_grace(self, processes: Sequence[tuple[str, mp.Process]]) -> None:
        """Wait ``grace_seconds`` for a graceful exit, then force-terminate.

        The shared ``stop_event`` is already set (via signal), so each job's
        ``_StopEventCallback`` will end its Lightning loop; this only bounds how
        long we wait before killing stragglers, leaving no residual processes.
        """
        deadline = time.time() + self.grace_seconds

        while time.time() < deadline:
            if not any(p.is_alive() for _, p in processes):
                return

            time.sleep(self.poll_seconds)

        for _, process in processes:
            if process.is_alive():
                from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

                ProcessGuard().terminate_process_tree(
                    process.pid,
                    grace_seconds=min(1.0, self.grace_seconds),
                )
                process.terminate()

        deadline = time.time() + min(5.0, max(1.0, self.grace_seconds))
        while time.time() < deadline:
            if not any(p.is_alive() for _, p in processes):
                return
            time.sleep(self.poll_seconds)

        for _, process in processes:
            if process.is_alive():
                from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

                ProcessGuard().terminate_process_tree(
                    process.pid,
                    grace_seconds=0.0,
                )
                kill = getattr(process, "kill", None)
                if kill is not None:
                    kill()
                else:
                    process.terminate()

    def _cleanup_running_processes(self) -> None:
        """Best-effort final cleanup for exceptions or abrupt parent shutdown."""
        alive = [(name, process) for name, process in self.processes if process.is_alive()]
        if not alive:
            return

        self.stop_event.set()
        self._stop_with_grace(alive)
        for _, process in alive:
            process.join(timeout=1.0)

    def _close_windows_job(self) -> None:
        if self._windows_job is not None:
            self._windows_job.close()
            self._windows_job = None

    @staticmethod
    def _run_job(
        job: TrainingJob,
        stop_event: Any,
        cuda_visible_devices: str | None,
        cpu_threads_per_job: int | None,
        cpu_interop_threads_per_job: int | None,
        cpu_affinity: list[int] | None,
        grace_seconds: float,
    ) -> None:
        # Die automatically if the parent (launcher) dies, even under `kill -9`,
        # so no training is left orphaned on the GPU.
        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        guard = ProcessGuard()
        guard.install_parent_death_guard()
        guard.install_child_process_cleanup(grace_seconds=grace_seconds)
        guard.set_cpu_affinity(cpu_affinity)

        # Set CUDA_VISIBLE_DEVICES before any CUDA initialisation so that
        # device indices inside the training function are relative to this
        # restricted set, not the full physical machine.
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        guard.configure_cpu_thread_limits(
            torch_threads=cpu_threads_per_job,
            interop_threads=cpu_interop_threads_per_job,
            override_env=True,
        )
        print(
            "Resource guard: "
            f"pid={os.getpid()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"cpu_affinity={guard.available_cpu_ids()} "
            f"torch_threads={os.environ.get('TORCH_NUM_THREADS', '')} "
            f"torch_interop={os.environ.get('TORCH_NUM_INTEROP_THREADS', '')}",
            flush=True,
        )

        job.run(stop_event)
