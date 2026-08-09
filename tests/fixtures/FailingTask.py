"""External-style task that always fails for result-path tests."""

from lambdaforge.tasks import TaskContext


class FailingTask:
    """Raise one deterministic exception from the task lifecycle."""

    def __init__(self, message: str = "expected task failure") -> None:
        self.message = message

    def run(self, context: TaskContext) -> None:
        """Raise after proving the context was supplied."""
        assert context.name
        raise RuntimeError(self.message)
