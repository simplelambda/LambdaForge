"""Auditable Research Console action support; an inventory is never fake parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionStatus = Literal["IMPLEMENTED", "CLI-ONLY BY DESIGN", "NOT SUPPORTED"]


@dataclass(frozen=True, slots=True)
class ConsoleAction:
    """One public operation with an explicit, testable support classification."""

    name: str
    family: str
    operation: str
    screen: str
    status: ActionStatus
    handler: str | None = None
    service_method: str | None = None
    destructive: bool = False
    requires_selection: bool = False


def _action(
    family: str,
    operation: str,
    screen: str,
    *,
    status: ActionStatus = "CLI-ONLY BY DESIGN",
    handler: str | None = None,
    service: str | None = None,
    destructive: bool = False,
    selection: bool = False,
) -> ConsoleAction:
    return ConsoleAction(
        f"{operation.title()} {family}",
        family,
        operation,
        screen,
        status,
        handler,
        service,
        destructive,
        selection,
    )


# Ctrl+P contains only IMPLEMENTED entries. Contextual entity workspaces expose the other
# service-backed operations beside their exact targets. CLI-only entries remain for an honest audit.
CONSOLE_ACTIONS = (
    _action("project/work", "list", "work", status="IMPLEMENTED", handler="navigate"),
    _action(
        "project/work",
        "run",
        "work",
        status="IMPLEMENTED",
        handler="work_launch",
        service="submit_work",
    ),
    _action(
        "project/work",
        "validate",
        "work",
        status="IMPLEMENTED",
        handler="work_launch",
        service="validate_work",
    ),
    _action(
        "project/work",
        "explain",
        "work",
        status="IMPLEMENTED",
        handler="work_launch",
        service="explain_work",
    ),
    _action("project/work", "cancel", "work", service="cancel_work", selection=True),
    _action(
        "project/work", "delete", "work", service="delete_work", destructive=True, selection=True
    ),
    _action("system", "overview", "overview", status="IMPLEMENTED", handler="navigate"),
    _action("system", "doctor", "overview"),
    _action("system", "resources", "overview"),
    _action("system", "clean", "overview", destructive=True),
    _action("job", "list", "work"),
    _action("job", "show", "work", selection=True),
    _action("job", "logs", "work", selection=True),
    _action("job", "retry", "work", service="retry_job", selection=True),
    _action("job", "cancel", "work", destructive=True, selection=True),
    _action("job", "delete", "work", destructive=True, selection=True),
    _action("cluster", "list", "clusters", status="IMPLEMENTED", handler="navigate"),
    _action(
        "cluster",
        "add",
        "clusters",
        status="IMPLEMENTED",
        handler="cluster_add",
    ),
    _action("cluster", "bootstrap", "clusters", service="bootstrap", selection=True),
    _action("cluster", "doctor", "clusters", service="doctor", selection=True),
    _action(
        "cluster", "credentials set", "clusters", service="set_cluster_credential", selection=True
    ),
    _action(
        "cluster",
        "remove",
        "clusters",
        service="remove_cluster",
        destructive=True,
        selection=True,
    ),
    _action("dataset", "list", "datasets", status="IMPLEMENTED", handler="navigate"),
    _action("dataset", "verify", "datasets", service="dataset_verify", selection=True),
    _action("dataset", "stats", "datasets", service="dataset_stats", selection=True),
    _action("dataset", "members", "datasets", service="dataset_members", selection=True),
    _action("dataset", "delete", "datasets", destructive=True, selection=True),
    _action("result", "list", "results", status="IMPLEMENTED", handler="navigate"),
    _action(
        "result",
        "analyze",
        "results",
        status="IMPLEMENTED",
        handler="result_analyze",
        service="analyze",
        selection=True,
    ),
    _action(
        "result",
        "report",
        "results",
        status="IMPLEMENTED",
        handler="result_report",
        service="report",
        selection=True,
    ),
    _action("result", "compare", "results", service="compare_results", selection=True),
)

CLI_PARITY = {
    family: tuple(
        {"operation": action.operation, "status": action.status}
        for action in CONSOLE_ACTIONS
        if action.family == family
    )
    for family in {action.family for action in CONSOLE_ACTIONS}
}

PALETTE_ACTIONS = tuple(action for action in CONSOLE_ACTIONS if action.status == "IMPLEMENTED")

__all__ = [
    "CLI_PARITY",
    "CONSOLE_ACTIONS",
    "PALETTE_ACTIONS",
    "ActionStatus",
    "ConsoleAction",
]
