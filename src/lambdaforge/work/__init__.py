"""Lazy public Work exports; metadata consumers must not initialize the execution engine."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.work.cache import HttpCooldownPolicy, RateLimit, WorkCache
    from lambdaforge.work.config import WorkConfig, WorkValidationReport
    from lambdaforge.work.managed import ManagedFile, ManagedOutput
    from lambdaforge.work.models import (
        WorkArtifact,
        WorkConfiguration,
        WorkInput,
        WorkResources,
        WorkResult,
        WorkTrial,
    )
    from lambdaforge.work.recovery_dependencies import (
        recovery_dependency_preview,
        repair_recovery_dependencies,
    )
    from lambdaforge.work.ResultStore import ResultStore
    from lambdaforge.work.runner import WorkExecutionPlan, WorkExecutionResult, WorkRunner
    from lambdaforge.work.tools import Tool, ToolExecutionError, ToolResult, ToolService
    from lambdaforge.work.Work import Work

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.work.{module}", name)
        for module, names in {
            "cache": ("RateLimit", "WorkCache", "HttpCooldownPolicy"),
            "recovery_dependencies": (
                "recovery_dependency_preview",
                "repair_recovery_dependencies",
            ),
            "config": ("WorkConfig", "WorkValidationReport"),
            "managed": ("ManagedFile", "ManagedOutput"),
            "models": (
                "WorkArtifact",
                "WorkConfiguration",
                "WorkInput",
                "WorkResources",
                "WorkResult",
                "WorkTrial",
            ),
            "ResultStore": ("ResultStore",),
            "ResultInput": ("ResultInput", "ResultRequirement"),
            "runner": ("WorkExecutionPlan", "WorkExecutionResult", "WorkRunner"),
            "tools": ("Tool", "ToolExecutionError", "ToolResult", "ToolService"),
            "Work": ("Work",),
        }.items()
        for name in names
    },
)

__all__ = [
    "Work",
    "WorkArtifact",
    "WorkConfig",
    "WorkConfiguration",
    "WorkExecutionPlan",
    "WorkExecutionResult",
    "WorkInput",
    "WorkCache",
    "ManagedFile",
    "ManagedOutput",
    "RateLimit",
    "HttpCooldownPolicy",
    "recovery_dependency_preview",
    "repair_recovery_dependencies",
    "WorkResources",
    "WorkResult",
    "WorkRunner",
    "ResultStore",
    "ResultInput",
    "ResultRequirement",
    "WorkTrial",
    "WorkValidationReport",
    "Tool",
    "ToolExecutionError",
    "ToolResult",
    "ToolService",
]
