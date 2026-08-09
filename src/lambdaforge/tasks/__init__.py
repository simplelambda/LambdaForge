"""Public contracts and execution objects for generic LambdaForge tasks."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
    from lambdaforge.tasks.ArtifactType import ArtifactType
    from lambdaforge.tasks.Task import Task
    from lambdaforge.tasks.TaskArtifact import TaskArtifact
    from lambdaforge.tasks.TaskConfig import TaskConfig
    from lambdaforge.tasks.TaskContext import TaskContext
    from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
    from lambdaforge.tasks.TaskInput import TaskInput
    from lambdaforge.tasks.TaskOutput import TaskOutput
    from lambdaforge.tasks.TaskPlanAction import TaskPlanAction
    from lambdaforge.tasks.TaskResult import TaskResult
    from lambdaforge.tasks.TaskRun import TaskRun
    from lambdaforge.tasks.TaskRunner import TaskRunner
    from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
    from lambdaforge.tasks.TaskStatus import TaskStatus
    from lambdaforge.tasks.TaskValidationReport import TaskValidationReport
    from lambdaforge.tasks.TaskValidator import TaskValidator

LazyExports.install(
    __name__,
    {
        "ArtifactDeclaration": (
            "lambdaforge.tasks.ArtifactDeclaration",
            "ArtifactDeclaration",
        ),
        "ArtifactType": ("lambdaforge.tasks.ArtifactType", "ArtifactType"),
        "Task": ("lambdaforge.tasks.Task", "Task"),
        "TaskArtifact": ("lambdaforge.tasks.TaskArtifact", "TaskArtifact"),
        "TaskConfig": ("lambdaforge.tasks.TaskConfig", "TaskConfig"),
        "TaskContext": ("lambdaforge.tasks.TaskContext", "TaskContext"),
        "TaskExecutionPlan": (
            "lambdaforge.tasks.TaskExecutionPlan",
            "TaskExecutionPlan",
        ),
        "TaskInput": ("lambdaforge.tasks.TaskInput", "TaskInput"),
        "TaskOutput": ("lambdaforge.tasks.TaskOutput", "TaskOutput"),
        "TaskPlanAction": ("lambdaforge.tasks.TaskPlanAction", "TaskPlanAction"),
        "TaskResult": ("lambdaforge.tasks.TaskResult", "TaskResult"),
        "TaskRun": ("lambdaforge.tasks.TaskRun", "TaskRun"),
        "TaskRunner": ("lambdaforge.tasks.TaskRunner", "TaskRunner"),
        "TaskSchemaCatalog": (
            "lambdaforge.tasks.TaskSchemaCatalog",
            "TaskSchemaCatalog",
        ),
        "TaskStatus": ("lambdaforge.tasks.TaskStatus", "TaskStatus"),
        "TaskValidationReport": (
            "lambdaforge.tasks.TaskValidationReport",
            "TaskValidationReport",
        ),
        "TaskValidator": ("lambdaforge.tasks.TaskValidator", "TaskValidator"),
    },
)

__all__ = [
    "ArtifactDeclaration",
    "ArtifactType",
    "Task",
    "TaskArtifact",
    "TaskConfig",
    "TaskContext",
    "TaskExecutionPlan",
    "TaskInput",
    "TaskOutput",
    "TaskPlanAction",
    "TaskResult",
    "TaskRun",
    "TaskRunner",
    "TaskSchemaCatalog",
    "TaskStatus",
    "TaskValidationReport",
    "TaskValidator",
]
