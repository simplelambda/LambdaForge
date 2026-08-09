"""External-style duck-typed generic task used by YAML integration tests."""

from __future__ import annotations

from lambdaforge.tasks import ArtifactDeclaration, ArtifactType, TaskContext, TaskOutput


class UserTask:
    """Write one configured artifact and return structured task data."""

    def __init__(self, message: str, artifact_name: str = "output.txt") -> None:
        self.message = message
        self.artifact_name = artifact_name

    def run(self, context: TaskContext) -> TaskOutput:
        """Write the configured message below the safe task run directory."""
        path = context.output_path(self.artifact_name, create_parent=True)
        path.write_text(self.message, encoding="utf-8")
        return TaskOutput(
            outputs={"message": self.message},
            metrics={"message_length": len(self.message)},
            artifacts=[
                ArtifactDeclaration(
                    self.artifact_name,
                    kind=ArtifactType.REPORT,
                    media_type="text/plain",
                )
            ],
            metadata={"fixture": True},
        )
