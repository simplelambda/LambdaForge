"""Spawn-safe probe for expected-parent PID validation."""

from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard


class ExpectedParentGuardProbe:
    """Install a parent-death guard with a deliberately supplied parent PID."""

    def __init__(self, expected_parent_pid: int) -> None:
        self.expected_parent_pid = expected_parent_pid

    def __call__(self) -> None:
        ProcessGuard().install_parent_death_guard(
            poll_seconds=0.02,
            expected_parent_pid=self.expected_parent_pid,
        )
