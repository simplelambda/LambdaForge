"""Series navigation is a presentation of domain evidence, not a second planner."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

from textual.app import App
from textual.widgets import DataTable, Static

from lambdaforge.tui.screens.StudySeries import StudySeriesWorkspace
from lambdaforge.tui.viewmodels import entity_key


def view():
    runs = [
        {
            "cell_id": "cell-origin",
            "candidate_id": "candidate-a",
            "presentation_trial": 1,
            "original_trial": 7,
            "origin_revision": 0,
            "seed": 11,
            "status": "succeeded",
            "metrics": {"score": 0.6},
            "execution_id": "execution-origin",
            "run_id": "run-original",
            "attempt_id": "attempt-0001",
        },
        {
            "cell_id": "cell-added",
            "candidate_id": "candidate-a",
            "presentation_trial": 1,
            "origin_revision": 1,
            "seed": 23,
            "status": "preparing",
            "metrics": {},
        },
    ]
    return {
        "study_series_id": "series-" + "a" * 24,
        "state": "preparing",
        "name": "regularization",
        "objective": {"metric": "score", "mode": "max"},
        "runs": runs,
        "candidates": [
            {
                "trial": 1,
                "parameters": {"strength": 0.1},
                "runs": runs,
                "state": "incomplete",
                "selection_objective": None,
            }
        ],
        "run_states": {"succeeded": 1, "preparing": 1},
        "counts": {"completed": 1, "required": 2, "pending": 1, "candidates": 1},
        "physical_cost": {"spent_attempts": 1, "elapsed_seconds": 2},
        "revisions": [
            {"number": 0, "execution_id": "execution-origin", "evidence": [runs[0]]},
            {"number": 1, "new_cells": [runs[1]]},
        ],
        "operations": [{"revision": 1, "native_state": "preparing", "latest_job_id": "job-delta"}],
    }


def test_series_screen_visible_loading_runs_provenance_and_scroll(tmp_path):
    asyncio.run(exercise(tmp_path))


async def exercise(tmp_path):
    data = view()
    calls = []

    def load(series, refresh=False, include_live=False):
        calls.append((series, refresh))
        return deepcopy(data)

    services = SimpleNamespace(
        study_series_overview=load, study_series_history=lambda series: {"history": []}
    )
    screen = StudySeriesWorkspace(
        {"study_series_id": data["study_series_id"], "name": data["name"], "cluster": "local"},
        services,
    )

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(120, 42)) as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if screen.value is not None:
                break
        assert screen.value is not None
        assert calls == [(data["study_series_id"], True)]
        assert "revision 1" in str(screen.query_one("#series-status", Static).render())
        table = screen.query_one("#series-runs", DataTable)
        assert table.row_count == 2
        assert "run-original" in [str(value) for value in table.get_row_at(0)]
        assert "Score" in [str(column.label) for column in table.columns.values()]
        assert screen.query_one("#series-revisions", DataTable).row_count == 2
        assert screen.query_one("#series-actions").region.right <= 120
        app.save_screenshot("study-series.svg", path=str(tmp_path))


def test_series_selection_uses_series_identity_not_latest_job():
    assert entity_key({"study_series_id": "series-a", "primary_job_id": "job-one"}) == entity_key(
        {"study_series_id": "series-a", "primary_job_id": "job-two"}
    )
