"""Read-only native result queries with explicit project and store-factory injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from lambdaforge.agents.contracts import page
from lambdaforge.agents.scope import ProjectScope

if TYPE_CHECKING:
    from lambdaforge.work.ResultStore import ResultStore


class ResultQueries:
    """Own result-query validation, not scientific records, caches, execution or providers."""

    def __init__(self, scope: ProjectScope, store_factory: Callable[[], ResultStore]):
        self.scope = scope
        self._store_factory = store_factory

    def _store(self) -> ResultStore:
        self.scope.assert_current()
        return self._store_factory()

    @staticmethod
    def _execution(execution_id: str) -> None:
        if not execution_id.startswith("execution-"):
            raise ValueError("Use an exact Execution ID.")

    def catalog(self, offset: int, limit: int) -> dict[str, Any]:
        page([], offset, limit)
        values = [
            {key: value for key, value in row.items() if key != "_manifest_path"}
            for row in self._store().catalog()
        ]
        return page(values, offset, limit)

    def view(
        self,
        execution_id: str,
        view: Literal["overview", "metrics", "outputs", "logs", "provenance"],
        run_id: str | None,
        attempt: int | None,
        points: int,
    ) -> dict[str, Any]:
        self._execution(execution_id)
        if not 1 <= points <= 500:
            raise ValueError("Use 1–500 curve points.")
        if view not in {"overview", "metrics", "outputs", "logs", "provenance"}:
            raise ValueError("Unknown result view.")
        return self._store().view(
            execution_id, view=view, run_id=run_id, attempt=attempt, points=points, tail=100
        )

    def reference(
        self, execution_id: str, run_id: str | None, attempt: int | None
    ) -> dict[str, Any]:
        self._execution(execution_id)
        return self._store().reference(execution_id, run=run_id, attempt=attempt)

    def artifact(
        self, execution_id: str, name: str, run_id: str | None, attempt: int | None
    ) -> dict[str, Any]:
        self._execution(execution_id)
        return self._store().output_preview(execution_id, name, run_id=run_id, attempt=attempt)

    def compare(
        self, execution_ids: list[str], metric: str | None, mode: Literal["min", "max"]
    ) -> dict[str, Any]:
        if not 2 <= len(execution_ids) <= 8:
            raise ValueError("Compare 2–8 exact Execution IDs.")
        for identifier in execution_ids:
            self._execution(identifier)
        return self._store().compare(tuple(execution_ids), metric=metric, mode=mode)
