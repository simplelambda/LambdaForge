"""Stable Research Console navigation and CLI parity inventory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConsoleAction:
    """One explicit interactive counterpart of a public CLI domain operation."""

    name: str
    family: str
    operation: str
    screen: str
    destructive: bool = False


def _actions(
    family: str, screen: str, operations: tuple[str, ...], destructive: tuple[str, ...] = ()
) -> tuple[ConsoleAction, ...]:
    return tuple(
        ConsoleAction(
            f"{operation.title()} {family}", family, operation, screen, operation in destructive
        )
        for operation in operations
    )


CONSOLE_ACTIONS = (
    *_actions(
        "project/work",
        "work",
        ("init", "validate", "explain", "run", "show", "logs", "cancel", "retry", "delete"),
        ("cancel", "delete"),
    ),
    *_actions("system", "overview", ("doctor", "overview", "resources", "clean"), ("clean",)),
    *_actions(
        "cluster",
        "clusters",
        (
            "add",
            "list",
            "show",
            "inspect",
            "test",
            "resources",
            "storage",
            "set",
            "unset",
            "remove",
            "export",
            "credentials set",
            "credentials delete",
            "bootstrap",
        ),
        ("remove", "credentials delete"),
    ),
    *_actions(
        "job",
        "work",
        (
            "list",
            "status",
            "show",
            "cancel",
            "pause",
            "resume",
            "delete",
            "logs",
            "retry",
            "reconcile",
            "clear",
        ),
        ("cancel", "delete", "clear"),
    ),
    *_actions(
        "dataset",
        "datasets",
        (
            "list",
            "show",
            "locations",
            "stats",
            "verify",
            "lineage",
            "remove",
            "reconcile",
            "delete",
            "materialize",
            "members",
            "member",
            "diff",
            "add",
            "replicate",
        ),
        ("remove", "delete"),
    ),
    *_actions("result", "results", ("list", "show", "compare", "analyze", "report")),
)

CLI_PARITY = {
    family: tuple(action.operation for action in CONSOLE_ACTIONS if action.family == family)
    for family in {action.family for action in CONSOLE_ACTIONS}
}

__all__ = ["CLI_PARITY", "CONSOLE_ACTIONS", "ConsoleAction"]
