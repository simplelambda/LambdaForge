"""Multi-process training job orchestrator."""

from __future__ import annotations

import contextlib
import math
import os
import signal
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch.multiprocessing as mp

from lambdaforge.training.orchestration.DeviceAssignment import DeviceAssignment
from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.WindowsJobObject import WindowsJobObject


class TrainingOrchestrator:
    r"""Orchestrates multiple independent training jobs in separate processes.

    Responsibilities
    ----------------
    - Start one subprocess per job.
    - Provide a shared stop event to every job.
    - Expose :meth:`request_stop` for idempotent host-driven cancellation.
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
    | ``[]`` or ``()``      | ``""`` (explicit CPU mode)    |
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
    manage_signals : bool
        Install temporary SIGINT, SIGTERM and, on Windows, SIGBREAK handlers.
        Set to ``False`` when a host runs the orchestrator outside the main
        thread and calls :meth:`request_stop` itself.
    """

    def __init__(
        self,
        start_method: str = "spawn",
        grace_seconds: float = 15.0,
        poll_seconds: float = 0.5,
        cpu_threads_per_job: int | None = 1,
        cpu_interop_threads_per_job: int | None = 1,
        cpu_cores_per_job: int | None = 1,
        manage_signals: bool = True,
    ) -> None:
        if not isinstance(manage_signals, bool):
            raise TypeError("manage_signals must be a boolean.")
        self.start_method = start_method
        self.grace_seconds = self._validate_seconds(
            grace_seconds,
            "grace_seconds",
            allow_zero=True,
        )
        self.poll_seconds = self._validate_seconds(
            poll_seconds,
            "poll_seconds",
            allow_zero=False,
        )
        self.cpu_threads_per_job = cpu_threads_per_job
        self.cpu_interop_threads_per_job = cpu_interop_threads_per_job
        self.cpu_cores_per_job = cpu_cores_per_job
        self.manage_signals = manage_signals

        self.context: Any = mp.get_context(start_method)
        self.stop_event = self.context.Event()
        self.processes: list[tuple[str, mp.Process]] = []

        # Snapshot parent CUDA_VISIBLE_DEVICES once, before any subprocess
        # is spawned. This is the reference for all device-index translation.
        self._parent_cvd: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._available_cpu_ids = self._read_available_cpu_ids()
        self._original_cpu_affinity: list[int] | None = None
        self._windows_job: WindowsJobObject | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._signal_stop_requested = False
        self.process_isolation_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Request an idempotent cooperative stop for every active job."""
        self.stop_event.set()

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
        jobs = tuple(jobs)
        self._validate_jobs(jobs)
        exit_codes: dict[str, int | None] = {job.name: None for job in jobs}
        with self._execution_scope(slot_count=len(jobs)):
            for job_index, job in enumerate(jobs):
                if self.stop_event.is_set():
                    break
                self._launch(job, job.devices, slot_index=job_index)

            while any(p.is_alive() for _, p in self.processes):
                self._consume_signal_stop_request()
                if self.stop_event.is_set():
                    self._stop_with_grace(self.processes)
                    break

                time.sleep(self.poll_seconds)

            self._join_processes(self.processes)
            for name, process in self.processes:
                exit_codes[name] = process.exitcode
            self._raise_if_processes_alive(self.processes)
            return exit_codes

    def run_scheduled(
        self,
        jobs: Sequence[TrainingJob],
        slots: Sequence[Sequence[int] | None],
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
        jobs = tuple(jobs)
        self._validate_jobs(jobs)
        normalized_slots = self._normalize_slots(slots)

        pending = deque(jobs)
        free_slots = deque(range(len(normalized_slots)))
        running: dict[int, tuple[str, mp.Process]] = {}
        exit_codes: dict[str, int | None] = {}
        notified: set[str] = set()

        with self._execution_scope(slot_count=len(normalized_slots)):
            while pending or running:
                self._consume_signal_stop_request()
                while free_slots and pending and not self.stop_event.is_set():
                    self._consume_signal_stop_request()
                    if self.stop_event.is_set():
                        break
                    slot_index = free_slots.popleft()
                    job = pending.popleft()
                    process = self._launch(
                        job,
                        normalized_slots[slot_index],
                        slot_index=slot_index,
                    )
                    running[slot_index] = (job.name, process)

                if self.stop_event.is_set():
                    self._stop_with_grace(list(running.values()))
                    for name, process in running.values():
                        self._record_finished(
                            name,
                            process,
                            exit_codes,
                            notified,
                            on_job_finished,
                        )
                    for job in pending:
                        exit_codes[job.name] = None
                    self._raise_if_processes_alive(list(running.values()))
                    return exit_codes

                finished = [
                    slot_index
                    for slot_index, (_, process) in running.items()
                    if not process.is_alive()
                ]

                for slot_index in finished:
                    name, process = running.pop(slot_index)
                    self._record_finished(
                        name,
                        process,
                        exit_codes,
                        notified,
                        on_job_finished,
                    )
                    free_slots.append(slot_index)

                if not finished:
                    time.sleep(self.poll_seconds)

            self._raise_if_processes_alive(self.processes)
            return exit_codes

    def run_dynamic(
        self,
        slots: Sequence[Sequence[int] | None],
        next_job: Callable[[int, tuple[int, ...] | None], TrainingJob | None],
        on_job_finished: Callable[[str, int | None, int], None] | None = None,
    ) -> dict[str, int | None]:
        """Fill free slots from a live supplier until it declares no more work.

        The supplier is called in the parent process and may use every result observed so
        far to choose the next job. Returning ``None`` leaves that slot idle for the
        current scheduling pass. Scheduling terminates only when all slots return
        ``None`` while no process is running, which makes temporary resource admission
        failures safe while another job can still release capacity.
        """
        normalized_slots = self._normalize_slots(slots)
        running: dict[int, tuple[str, mp.Process]] = {}
        exit_codes: dict[str, int | None] = {}
        names: set[str] = set()

        with self._execution_scope(slot_count=len(normalized_slots)):
            while True:
                self._consume_signal_stop_request()
                if self.stop_event.is_set():
                    self._stop_with_grace(list(running.values()))

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
                        on_job_finished(name, process.exitcode, slot_index)

                if self.stop_event.is_set():
                    self._join_processes(list(running.values()))
                    for name, process in running.values():
                        exit_codes[name] = process.exitcode
                    self._raise_if_processes_alive(list(running.values()))
                    return exit_codes

                launched = False
                for slot_index, slot in enumerate(normalized_slots):
                    if slot_index in running:
                        continue
                    job = next_job(slot_index, slot)
                    if job is None:
                        continue
                    if job.name in names:
                        raise ValueError(f"Dynamic training job name is not unique: {job.name!r}.")
                    names.add(job.name)
                    process = self._launch(job, slot, slot_index=slot_index)
                    running[slot_index] = (job.name, process)
                    launched = True

                if not running and not launched:
                    self._raise_if_processes_alive(self.processes)
                    return exit_codes
                if not finished and not launched:
                    time.sleep(self.poll_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _execution_scope(self, slot_count: int) -> Iterator[None]:
        self.stop_event.clear()
        self.processes = []
        self.process_isolation_warnings = []
        self._previous_signal_handlers = {}
        self._signal_stop_requested = False
        self._original_cpu_affinity = None
        self._windows_job = None
        try:
            self._windows_job = WindowsJobObject()
            self._surface_windows_job_status()
            self._install_signal_handlers()
            self._apply_parent_cpu_affinity(slot_count=slot_count)
            yield
        finally:
            try:
                self._cleanup_running_processes()
            finally:
                try:
                    self._close_windows_job()
                finally:
                    try:
                        self._restore_parent_cpu_affinity()
                    finally:
                        self._restore_signal_handlers()

    @staticmethod
    def _validate_jobs(jobs: Sequence[TrainingJob]) -> None:
        names = [job.name for job in jobs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Training job names must be unique: {duplicates}.")

    @staticmethod
    def _normalize_slots(
        slots: Sequence[Sequence[int] | None],
    ) -> tuple[tuple[int, ...] | None, ...]:
        slots = tuple(slots)
        if not slots:
            raise ValueError("run_scheduled requires at least one slot.")
        return tuple(
            DeviceAssignment.normalize(slot, label=f"Scheduler slot {index}")
            for index, slot in enumerate(slots)
        )

    @staticmethod
    def _validate_seconds(value: float, name: str, *, allow_zero: bool) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number.")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite.")
        if resolved < 0 or (resolved == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}.")
        return resolved

    def _install_signal_handlers(self) -> None:
        if not self.manage_signals:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Signal management requires the main thread; use manage_signals=False "
                "for orchestrators run from worker threads."
            )
        installed: dict[int, Any] = {}
        try:
            for signum in self._managed_signals():
                installed[signum] = signal.getsignal(signum)
                signal.signal(signum, self._request_stop)
        except BaseException:
            for signum, previous in installed.items():
                signal.signal(signum, previous)
            raise
        self._previous_signal_handlers = installed

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers = {}

    @staticmethod
    def _managed_signals() -> tuple[int, ...]:
        names = ("SIGINT", "SIGTERM", "SIGBREAK")
        return tuple(
            int(signum) for name in names if (signum := getattr(signal, name, None)) is not None
        )

    def _surface_windows_job_status(self) -> None:
        if os.name != "nt" or self._windows_job is None or self._windows_job.active:
            return
        detail = self._windows_job.initialization_error or "native support unavailable"
        message = (
            "Windows Job Object isolation is unavailable; portable process-tree "
            f"cleanup remains active. Detail: {detail}."
        )
        self.process_isolation_warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _record_finished(
        self,
        name: str,
        process: mp.Process,
        exit_codes: dict[str, int | None],
        notified: set[str],
        callback: Callable[[str, int | None], None] | None,
    ) -> None:
        if name in notified:
            return
        if process.is_alive():
            raise RuntimeError(
                f"Cannot record running training process {name!r} (pid={process.pid})."
            )
        process.join(timeout=0.0)
        exit_codes[name] = process.exitcode
        notified.add(name)
        if callback is not None:
            callback(name, process.exitcode)

    def _join_processes(
        self,
        processes: Sequence[tuple[str, mp.Process]],
        timeout_seconds: float | None = None,
    ) -> None:
        timeout = (
            max(1.0, min(5.0, self.grace_seconds + 1.0))
            if timeout_seconds is None
            else max(0.0, timeout_seconds)
        )
        deadline = time.monotonic() + timeout
        for _, process in processes:
            if process.pid is None:
                continue
            process.join(timeout=max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _raise_if_processes_alive(
        processes: Sequence[tuple[str, mp.Process]],
    ) -> None:
        alive = [f"{name} (pid={process.pid})" for name, process in processes if process.is_alive()]
        if alive:
            raise RuntimeError(
                "Training processes remained alive after bounded shutdown: " + ", ".join(alive)
            )

    def _launch(
        self,
        job: TrainingJob,
        devices: Sequence[int] | None,
        slot_index: int,
    ) -> mp.Process:
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
                os.getpid(),
            ),
            name=job.name,
        )
        with self._temporary_env(env_updates):
            process.start()
        self.processes.append((job.name, process))
        if self._windows_job is not None:
            assigned = self._windows_job.assign(process.pid)
            if self._windows_job.active and not assigned:
                from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

                ProcessGuard().terminate_process_tree(
                    process.pid,
                    grace_seconds=0.0,
                    include_parent=True,
                )
                process.join(timeout=1.0)
                detail = self._windows_job.last_error or "unknown native error"
                raise RuntimeError(
                    f"Could not assign worker {job.name!r} to the Windows Job Object: {detail}."
                )
        self._set_child_cpu_affinity(process.pid, cpu_affinity)
        return process

    def _resolve_cuda_visible_devices(self, devices: Sequence[int] | None) -> str | None:
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

    def _request_stop(self, signum: int, frame: Any) -> None:
        del signum, frame
        # multiprocessing.Event.set() acquires a semaphore-backed lock. A
        # Python signal handler can interrupt Event.wait()/is_set() while that
        # same lock is held, so setting it here can self-deadlock. Store only a
        # GIL-protected flag and let the ordinary orchestration loop publish
        # the shared event immediately after the handler returns.
        self._signal_stop_requested = True

    def _consume_signal_stop_request(self) -> None:
        if self._signal_stop_requested:
            self._signal_stop_requested = False
            self.request_stop()

    def _stop_with_grace(self, processes: Sequence[tuple[str, mp.Process]]) -> None:
        """Wait ``grace_seconds`` for a graceful exit, then force-terminate.

        The shared ``stop_event`` is already set (via signal), so each job's
        ``_StopEventCallback`` will end its Lightning loop; this only bounds how
        long we wait before killing stragglers, leaving no residual processes.
        """
        self.request_stop()
        deadline = time.monotonic() + self.grace_seconds
        while True:
            alive = [(name, process) for name, process in processes if process.is_alive()]
            if not alive:
                self._join_processes(processes, timeout_seconds=0.0)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(self.poll_seconds, remaining))

        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        guard = ProcessGuard()
        termination_grace = min(1.0, self.grace_seconds)
        tree_cleanup_available = guard.terminate_process_trees(
            [process.pid for _, process in alive],
            grace_seconds=termination_grace,
            include_parents=True,
        )
        if not tree_cleanup_available:
            for _, process in alive:
                if process.is_alive():
                    process.terminate()

        self._join_processes(
            alive,
            timeout_seconds=max(1.0, termination_grace),
        )
        survivors = [(name, process) for name, process in alive if process.is_alive()]
        if survivors:
            for _, process in survivors:
                kill = getattr(process, "kill", None)
                if kill is not None:
                    kill()
                else:
                    process.terminate()
            guard.terminate_process_trees(
                [process.pid for _, process in survivors],
                grace_seconds=0.0,
                include_parents=True,
            )
            self._join_processes(survivors, timeout_seconds=1.0)

        self._raise_if_processes_alive(processes)

    def _cleanup_running_processes(self) -> None:
        """Best-effort final cleanup for exceptions or abrupt parent shutdown."""
        alive = [(name, process) for name, process in self.processes if process.is_alive()]
        if not alive:
            self._join_processes(self.processes, timeout_seconds=0.0)
            return

        self.request_stop()
        self._stop_with_grace(alive)
        self._join_processes(alive, timeout_seconds=1.0)
        self._raise_if_processes_alive(alive)

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
        expected_parent_pid: int,
    ) -> None:
        # Die automatically if the parent (launcher) dies, even under `kill -9`,
        # so no training is left orphaned on the GPU.
        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        guard = ProcessGuard()
        guard.install_child_process_cleanup(
            grace_seconds=grace_seconds,
            stop_event=stop_event,
        )
        guard.install_parent_death_guard(expected_parent_pid=expected_parent_pid)
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
