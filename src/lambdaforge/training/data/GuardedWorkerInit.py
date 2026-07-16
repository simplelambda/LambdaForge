"""Safe DataLoader worker initialization."""

from __future__ import annotations

from collections.abc import Callable


class GuardedWorkerInit:
    """Install process safeguards before invoking a user worker initializer."""

    def __init__(self, user_worker_init_fn: Callable[[int], None] | None = None) -> None:
        self.user_worker_init_fn = user_worker_init_fn

    def __call__(self, worker_id: int) -> None:
        from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard

        guard = ProcessGuard()
        guard.install_parent_death_guard()
        guard.configure_cpu_thread_limits()
        if self.user_worker_init_fn is not None:
            self.user_worker_init_fn(worker_id)
