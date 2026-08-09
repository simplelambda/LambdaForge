"""Reusable-style generic task fixture for entry-point contract tests."""

from lambdaforge.tasks import Task, TaskContext, TaskOutput


class UserPluginTask(Task):
    """Satisfy the strict reusable task-plugin inheritance boundary."""

    def run(self, context: TaskContext) -> TaskOutput:
        """Return the owning task identity without creating an artifact."""
        return TaskOutput(outputs={"name": context.name})
