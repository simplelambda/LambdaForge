"""Public contracts and execution objects for generic LambdaForge tasks."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.tasks.artifacts import ArtifactDeclaration, ArtifactType, TaskArtifact
    from lambdaforge.tasks.Task import Task
    from lambdaforge.tasks.TaskConfig import TaskConfig
    from lambdaforge.tasks.TaskContext import TaskContext
    from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan, TaskPlanAction
    from lambdaforge.tasks.TaskInput import TaskInput
    from lambdaforge.tasks.TaskOutput import TaskOutput
    from lambdaforge.tasks.TaskResult import TaskResult, TaskStatus
    from lambdaforge.tasks.TaskRun import TaskRun
    from lambdaforge.tasks.TaskRunner import TaskRunner
    from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
    from lambdaforge.tasks.TaskValidator import TaskValidationReport, TaskValidator

LazyExports.install(
    __name__,
    {
        "ArtifactDeclaration": (
            "lambdaforge.tasks.artifacts",
            "ArtifactDeclaration",
        ),
        "ArtifactType": ("lambdaforge.tasks.artifacts", "ArtifactType"),
        "Task": ("lambdaforge.tasks.Task", "Task"),
        "TaskArtifact": ("lambdaforge.tasks.artifacts", "TaskArtifact"),
        "TaskConfig": ("lambdaforge.tasks.TaskConfig", "TaskConfig"),
        "TaskContext": ("lambdaforge.tasks.TaskContext", "TaskContext"),
        "TaskExecutionPlan": (
            "lambdaforge.tasks.TaskExecutionPlan",
            "TaskExecutionPlan",
        ),
        "TaskInput": ("lambdaforge.tasks.TaskInput", "TaskInput"),
        "TaskOutput": ("lambdaforge.tasks.TaskOutput", "TaskOutput"),
        "TaskPlanAction": ("lambdaforge.tasks.TaskExecutionPlan", "TaskPlanAction"),
        "TaskResult": ("lambdaforge.tasks.TaskResult", "TaskResult"),
        "TaskRun": ("lambdaforge.tasks.TaskRun", "TaskRun"),
        "TaskRunner": ("lambdaforge.tasks.TaskRunner", "TaskRunner"),
        "TaskSchemaCatalog": (
            "lambdaforge.tasks.TaskSchemaCatalog",
            "TaskSchemaCatalog",
        ),
        "TaskStatus": ("lambdaforge.tasks.TaskResult", "TaskStatus"),
        "TaskValidationReport": (
            "lambdaforge.tasks.TaskValidator",
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
