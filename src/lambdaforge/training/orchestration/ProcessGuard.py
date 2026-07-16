"""Guard worker resources and make them die when their parent dies.

When the launcher is stopped with an uncatchable signal (``kill -9``), Python's
signal handlers never run, so any worker subprocesses spawned by
:class:`~lambdaforge.training.orchestration.TrainingOrchestrator.TrainingOrchestrator` would be
reparented to init and keep running on the GPU. This installs a parent-death
guard **inside each worker** so it terminates itself as soon as the parent is
gone, regardless of how the parent died:

- Linux: ``prctl(PR_SET_PDEATHSIG, SIGKILL)`` asks the kernel to signal the
  worker the instant its parent dies (immediate, no polling);
- any POSIX: a lightweight daemon thread also watches ``os.getppid()`` and exits
  if the parent changes/disappears (covers the prctl race and non-Linux POSIX).

On non-POSIX platforms it is a safe no-op. This is generic process management, so
it lives in ``core``.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from typing import Any

# Exit code used when a worker self-terminates because its parent vanished
# (128 + SIGKILL, mirroring how the shell reports a killed process).
ORPHAN_EXIT_CODE = 137

_PR_SET_PDEATHSIG = 1
# SIGKILL is not defined on Windows; resolve it lazily/portably (POSIX value 9).
_SIGKILL = int(getattr(signal, "SIGKILL", 9))

_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "VECLIB_MAXIMUM_THREADS",
)

_CLEANUP_INSTALLED = False
_CLEANUP_RUNNING = False
_CLEANUP_GRACE_SECONDS = 5.0
_PREVIOUS_SIGNAL_HANDLERS: dict[int, Any] = {}
_THREADPOOL_CONTROLLERS: list[Any] = []


class ProcessGuard:
    """Apply parent-death, CPU-resource and descendant-cleanup safeguards.

    Instances are lightweight; cleanup coordination remains process-global so
    repeated installation by framework components is idempotent.
    """

    def _set_linux_parent_death_signal(self, sig: int = _SIGKILL) -> bool:
        """Ask the kernel (Linux) to send ``sig`` to this process when its parent dies."""
        if not sys.platform.startswith("linux"):
            return False
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            return libc.prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0) == 0
        except Exception:
            return False

    def _start_orphan_watchdog(self, poll_seconds: float) -> None:
        """Start a daemon thread that exits the process if its parent disappears."""
        if os.name != "posix":
            return
        try:
            original_ppid = os.getppid()
        except Exception:
            return

        def _watch() -> None:
            while True:
                try:
                    ppid = os.getppid()
                except Exception:
                    return
                # Parent gone: reparented to init (1) or to a different pid.
                if ppid != original_ppid or ppid == 1:
                    os._exit(ORPHAN_EXIT_CODE)
                time.sleep(poll_seconds)

        threading.Thread(target=_watch, name="parent-death-watchdog", daemon=True).start()

    def install_parent_death_guard(self, poll_seconds: float = 1.0) -> None:
        """Terminate this process automatically when its parent dies.

        Safe to call once per worker at startup. No-op on non-POSIX platforms.
        """
        self._set_linux_parent_death_signal(_SIGKILL)

        # Handle the race where the parent already died between spawn and here.
        if os.name == "posix":
            try:
                if os.getppid() == 1:
                    os._exit(ORPHAN_EXIT_CODE)
            except Exception:
                pass

        self._start_orphan_watchdog(poll_seconds)

    def configure_cpu_thread_limits(
        self,
        torch_threads: int | None = None,
        interop_threads: int | None = None,
        *,
        override_env: bool = False,
    ) -> None:
        """Cap CPU thread pools for one training or dataloader process.

        Parallel experiments run many independent Python processes. Leaving every
        process free to create full BLAS/OpenMP/Torch thread pools can oversubscribe
        the node badly, so the orchestrator calls this at child startup and the
        dataloader calls it again inside each worker.
        """
        torch_threads = self._resolve_positive_int(torch_threads, "TORCH_NUM_THREADS")
        interop_threads = self._resolve_positive_int(interop_threads, "TORCH_NUM_INTEROP_THREADS")

        for name, value in self.cpu_thread_env_updates(torch_threads, interop_threads).items():
            if override_env or name not in os.environ:
                os.environ[name] = value

        self._apply_threadpoolctl_limit(torch_threads)

        try:
            import torch
        except Exception:
            return

        if torch_threads is not None:
            try:
                torch.set_num_threads(torch_threads)
            except RuntimeError:
                pass

        if interop_threads is not None:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                # PyTorch only allows setting inter-op threads before parallel work
                # starts. The env var still helps freshly spawned descendants.
                pass

    def cpu_thread_env_updates(
        self,
        torch_threads: int | None = None,
        interop_threads: int | None = None,
    ) -> dict[str, str]:
        """Environment updates that cap native CPU thread pools before process start."""
        torch_threads = self._resolve_positive_int(torch_threads, "TORCH_NUM_THREADS")
        interop_threads = self._resolve_positive_int(interop_threads, "TORCH_NUM_INTEROP_THREADS")

        updates: dict[str, str] = {
            # Avoid idle OpenMP workers spinning and consuming CPU between batches.
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_BLOCKTIME": "0",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            # Tokenizers/Rust libraries otherwise create their own thread pools.
            "TOKENIZERS_PARALLELISM": "false",
            "HF_ENABLE_PARALLEL_LOADING": "false",
            # Keep glibc from creating many malloc arenas in process-heavy sweeps.
            "MALLOC_ARENA_MAX": "2",
        }

        if torch_threads is not None:
            for name in (*_CPU_THREAD_ENV_VARS, "TORCH_NUM_THREADS"):
                updates[name] = str(torch_threads)

        if interop_threads is not None:
            updates["TORCH_NUM_INTEROP_THREADS"] = str(interop_threads)

        return updates

    def _apply_threadpoolctl_limit(self, torch_threads: int | None) -> None:
        if torch_threads is None:
            return

        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            return

        try:
            controller = threadpool_limits(limits=int(torch_threads))
        except Exception:
            return

        # Keep the controller alive for the process lifetime. The limit is restored
        # only when the controller exits/restores, which we intentionally avoid in
        # long-running training workers.
        _THREADPOOL_CONTROLLERS.append(controller)

    def available_cpu_ids(
        self,
    ) -> list[int]:
        """Return the CPU ids available to the current process/cgroup."""
        if hasattr(os, "sched_getaffinity"):
            try:
                return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
            except OSError:
                pass

        try:
            import psutil

            cpus = psutil.Process(os.getpid()).cpu_affinity()
            if cpus:
                return sorted(int(cpu) for cpu in cpus)
        except Exception:
            pass

        return list(range(os.cpu_count() or 1))

    def set_cpu_affinity(
        self, cpu_ids: list[int] | tuple[int, ...] | None, pid: int | None = None
    ) -> bool:
        """Restrict one process to ``cpu_ids`` when the platform supports it."""
        if not cpu_ids:
            return False

        cpuset = {int(cpu) for cpu in cpu_ids}
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0 if pid is None else int(pid), cpuset)
                return True
            except OSError:
                pass

        try:
            import psutil

            process = psutil.Process(os.getpid() if pid is None else int(pid))
            process.cpu_affinity(sorted(cpuset))
            return True
        except Exception:
            return False

    def terminate_child_process_tree(self, grace_seconds: float = 5.0) -> None:
        """Terminate all descendants of the current process.

        ``psutil`` gives a recursive process-tree view when installed. The fallback
        handles children created through Python multiprocessing, which is still
        enough for the framework's orchestrator and dataloaders.
        """
        global _CLEANUP_RUNNING

        if _CLEANUP_RUNNING:
            return
        _CLEANUP_RUNNING = True
        try:
            if self._terminate_with_psutil(grace_seconds):
                return
            self._terminate_multiprocessing_children(grace_seconds)
        finally:
            _CLEANUP_RUNNING = False

    def terminate_process_tree(
        self,
        pid: int | None,
        grace_seconds: float = 1.0,
        include_parent: bool = False,
    ) -> bool:
        """Terminate descendants of an arbitrary PID using ``psutil``.

        This parent-side fallback is especially important on Windows, where a
        forced ``multiprocessing.Process.terminate`` does not execute cleanup
        handlers inside the child process.
        """
        if pid is None:
            return False
        try:
            import psutil

            parent = psutil.Process(int(pid))
            descendants = parent.children(recursive=True)
        except Exception:
            return False
        targets = descendants + ([parent] if include_parent else [])
        for process in reversed(targets):
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(targets, timeout=max(0.0, grace_seconds))
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=1.0)
        return True

    def install_child_process_cleanup(self, grace_seconds: float = 5.0) -> None:
        """Clean descendants on normal exit, SIGINT, SIGTERM and SIGHUP.

        This complements :func:`install_parent_death_guard`: the parent-death guard
        makes this process die with its parent, while this cleanup makes any
        dataloader/sub-worker processes die with this process.
        """
        global _CLEANUP_GRACE_SECONDS, _CLEANUP_INSTALLED

        _CLEANUP_GRACE_SECONDS = float(grace_seconds)
        if _CLEANUP_INSTALLED:
            return

        atexit.register(self.terminate_child_process_tree, _CLEANUP_GRACE_SECONDS)

        for sig in self._available_cleanup_signals():
            _PREVIOUS_SIGNAL_HANDLERS[sig] = signal.getsignal(sig)
            try:
                signal.signal(sig, self._cleanup_signal_handler)
            except (OSError, ValueError):
                pass

        _CLEANUP_INSTALLED = True

    def _resolve_positive_int(self, value: int | None, env_name: str) -> int | None:
        if value is None:
            raw = os.environ.get(env_name)
            if raw in (None, ""):
                return None
            try:
                value = int(raw)
            except ValueError:
                return None

        value = int(value)
        return value if value > 0 else None

    def _terminate_with_psutil(self, grace_seconds: float) -> bool:
        try:
            import psutil
        except Exception:
            return False

        try:
            current = psutil.Process(os.getpid())
            children = current.children(recursive=True)
        except Exception:
            return False

        children = [child for child in children if child.pid != os.getpid()]
        if not children:
            return True

        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass

        _, alive = psutil.wait_procs(children, timeout=max(0.0, float(grace_seconds)))

        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass

        if alive:
            psutil.wait_procs(alive, timeout=1.0)
        return True

    def _terminate_multiprocessing_children(self, grace_seconds: float) -> None:
        children = [
            child for child in mp.active_children() if child.pid is not None and child.is_alive()
        ]
        if not children:
            return

        for child in children:
            child.terminate()

        deadline = time.time() + max(0.0, float(grace_seconds))
        for child in children:
            timeout = max(0.0, deadline - time.time())
            child.join(timeout=timeout)

        for child in children:
            if child.is_alive():
                kill = getattr(child, "kill", None)
                if kill is not None:
                    kill()
                else:
                    child.terminate()

        for child in children:
            child.join(timeout=1.0)

    def _available_cleanup_signals(
        self,
    ) -> tuple[int, ...]:
        names = ("SIGTERM", "SIGINT", "SIGHUP")
        signals = []
        for name in names:
            sig = getattr(signal, name, None)
            if sig is not None:
                signals.append(int(sig))
        return tuple(signals)

    def _cleanup_signal_handler(self, signum: int, frame) -> None:
        self.terminate_child_process_tree(_CLEANUP_GRACE_SECONDS)

        previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
        if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
            previous(signum, frame)
            return

        raise SystemExit(128 + int(signum))
