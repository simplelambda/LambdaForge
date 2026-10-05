"""Host-side, view-specific Study reads and transport paging, without real providers."""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobService, JobState, JobStore
from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ResearchWork import study_overview
from lambdaforge.study_projection import (
    analysis_panel,
    panel_detail,
    projection_page,
    study_table,
    trial_detail,
)


class RecordingTransport(LocalTransport):
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.response_sizes: list[int] = []

    def run(self, command, **kwargs):
        result = super().run(command, **kwargs)
        self.requests.append(json.loads(command[-1]))
        self.response_sizes.append(len(result.stdout.encode("utf-8")))
        return result


def _service(tmp_path: Path) -> tuple[JobService, JobRecord, Path, RecordingTransport]:
    work = tmp_path / "jobs" / "job-read" / "work"
    work.mkdir(parents=True)
    root = work.parent / "study"
    root.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-read",
        "local",
        "local",
        "provider-read",
        JobState.RUNNING,
        (sys.executable, "-m", "lambdaforge", "run", "config.yaml"),
        str(work),
        {},
        now,
        now,
        metadata={"name": "test-study"},
        job_type="work",
    )
    store = JobStore(tmp_path / "records")
    store.write(record)
    service = JobService(ClusterCatalog({"local": ClusterProfile("local")}), store)
    return service, record, root, RecordingTransport()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _snapshot() -> dict:
    return {
        "study_telemetry_version": 1,
        "objective": {"metric": "accuracy", "mode": "max"},
        "counts": {"candidates": 100, "active_runs": 100},
        "hpo_analysis": {"parameters": [{"parameter": "width", "confidence": 0.3}]},
        "controller": {"recent": [{"action": "ADD_SEED"}]},
        "admission": {"current": {"devices": [{"gpu": 0}], "status": "waiting"}},
        "candidates": [
            {
                "trial": i,
                "parameters": {"width": i},
                "state": "running",
                "selection_objective": i / 100,
                "selection_seed_count": 0,
                "runs": [
                    {
                        "key": f"trial-{i:05d}-seed-4",
                        "state": "running",
                        "latest_step": 23,
                        "gpu_index": i % 2,
                        "failure": {"traceback": "x" * 100_000},
                        "latest_metrics": {"accuracy": 0.5},
                    }
                ],
            }
            for i in range(1, 101)
        ],
    }


def test_oversized_legacy_summary_is_projected_before_transfer(tmp_path: Path) -> None:
    service, record, root, transport = _service(tmp_path)
    rich = _snapshot()
    _write(root / "summary.json", rich)
    assert (root / "summary.json").stat().st_size > 8 * 1024 * 1024
    table = service._load_study_summary(record, transport)
    assert len(table["candidates"]) == 100
    assert table["candidates"][9]["latest_step"] == 23
    assert table["candidates"][9]["gpu_indices"] == [0]
    assert table["run_states"] == {"running": 100}
    assert table["leader_parameters"] == {"width": 100}
    assert not {"hpo_analysis", "controller", "admission"} & table.keys()
    assert not {"parameters", "runs", "failure"} & table["candidates"][0].keys()
    assert sum(transport.response_sizes) < 40_000
    # A refresh of this view transfers only the unchanged-generation marker.
    assert service._load_study_summary(record, transport) == table
    assert transport.response_sizes[-1] < 200
    trial = service._load_study_summary(record, transport, view="trial", trial=10)
    assert trial["parameters"] == {"width": 10}
    assert len(trial["runs"]) == 1
    assert "failure" not in trial["runs"][0]
    hpo = service._load_study_summary(record, transport, view="hpo")
    assert hpo["hpo_analysis"] == rich["hpo_analysis"]
    assert "candidates" not in hpo and "admission" not in hpo


def test_prepared_views_do_not_read_authoritative_summary(tmp_path: Path) -> None:
    service, record, root, transport = _service(tmp_path)
    rich = _snapshot()
    (root / "summary.json").write_text("corrupt and must not be opened", encoding="utf-8")
    _write(root / "interactive.json", study_table(rich))
    _write(root / "hpo.json", panel_detail(rich, "hpo"))
    _write(root / "resources.json", panel_detail(rich, "resources"))
    _write(root / "trials" / "trial-00010.json", trial_detail(rich["candidates"][9]))
    _write(root / "overview.json", study_overview(rich))
    _write(
        root / "runs" / "trial-00010-seed-4.json",
        {
            "key": "trial-00010-seed-4",
            "state": "running",
            "seed": 4,
        },
    )
    _write(root / "observations" / "trial-00010-seed-4.json", {"latest_step": 23})
    assert len(service._load_study_summary(record, transport)["candidates"]) == 100
    trial = service._load_study_summary(record, transport, view="trial", trial=10)
    assert trial["parameters"] == {"width": 10}
    selected = service._load_study_summary(
        record, transport, view="run", trial=10, run="trial-00010-seed-4"
    )
    assert selected["parameters"] == {"width": 10}
    assert selected["run"]["latest_step"] == 23
    _write(root / "observations" / "trial-00010-seed-4.json", {"latest_step": 24})
    selected = service._load_study_summary(
        record, transport, view="run", trial=10, run="trial-00010-seed-4"
    )
    assert selected["run"]["latest_step"] == 24
    assert service._load_study_summary(record, transport, view="resources") == {
        "admission": rich["admission"]
    }
    assert service._load_study_summary(record, transport, view="hpo")["hpo_analysis"]
    assert all(size < 1024 * 1024 for size in transport.response_sizes)


def test_panels_exclude_unshown_evidence_and_admission_history() -> None:
    value = {
        "design": {
            "goal": "balanced",
            "replication": "adaptive",
            "seed_source": {"role": "search"},
            "space": {"width": {"values": [16, 32]}},
            "evidence": {
                "required_run_count": 100_000,
                "optional_run_count": 0,
                "requirements": [{"candidate": 1, "seed": 4}] * 100_000,
            },
        },
        "admission": {
            "current": {"status": "waiting"},
            "updated_at_utc": "now",
            "recent": [{"devices": [{"gpu": 0}]}] * 100_000,
        },
    }
    hpo = panel_detail(value, "hpo")
    assert hpo["design"]["goal"] == "balanced"
    assert hpo["design"]["evidence"] == {
        "required_run_count": 100_000,
        "optional_run_count": 0,
    }
    assert "space" not in hpo["design"]
    assert "admission" not in hpo
    resources = panel_detail(value, "resources")
    assert resources == {
        "admission": {"current": {"status": "waiting"}, "updated_at_utc": "now"}
    }
    assert len(json.dumps(hpo)) < 500
    assert len(value["design"]["evidence"]["requirements"]) == 100_000


def test_open_study_refreshes_only_selected_job_state_and_fetches_table_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, record, root, transport = _service(tmp_path)
    _write(root / "interactive.json", study_table(_snapshot()))
    scheduler = SimpleNamespace(
        state=lambda scheduler_id: JobState.SUCCEEDED,
        details=lambda scheduler_id: {"state": "succeeded"},
    )
    monkeypatch.setattr(
        service,
        "_provider",
        lambda selected: (
            service.catalog.get("local"),
            transport,
            scheduler,
        ),
    )
    value = service.study(record.job_id)
    assert value["job_state"] == "succeeded"
    assert len(transport.requests) == 1
    assert transport.requests[0]["view"] == "interactive"


def test_historical_reader_falls_back_to_active_managed_python_without_gpu_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, record, root, transport = _service(tmp_path)
    record = record.with_updates(command=("/missing/old-env/bin/python", "-m", "lambdaforge"))
    _write(root / "interactive.json", study_table(_snapshot()))
    state = tmp_path / "state"
    state.mkdir()
    (state / "active-environment").write_text(sys.executable + "\n", encoding="utf-8")
    profile = SimpleNamespace(
        transport="ssh",
        python="/missing/old-system-python",
        workspace=str(tmp_path),
        storage=SimpleNamespace(state_root=str(state)),
    )
    monkeypatch.setattr(service.catalog, "get", lambda name: profile)
    assert len(service._load_study_summary(record, transport)["candidates"]) == 100


def test_legitimate_analysis_above_8_mib_is_paged_losslessly(tmp_path: Path) -> None:
    service, record, root, transport = _service(tmp_path)
    analysis = {"status": "final", "observations": ["métrica ñ" * 1500] * 650}
    _write(root / "analysis.json", analysis)
    assert (root / "analysis.json").stat().st_size > 8 * 1024 * 1024
    loaded = service._load_study_summary(record, transport, view="analysis-report")
    assert loaded == analysis
    assert len(transport.requests) > 16
    assert max(transport.response_sizes) < 1024 * 1024
    assert [r["offset"] for r in transport.requests] == list(
        range(0, (root / "analysis.json").stat().st_size, 512 * 1024)
    )
    assert service._load_study_summary(record, transport, view="analysis-report") == loaded
    assert transport.response_sizes[-1] < 200


def test_analysis_console_omits_unrequested_run_and_html_details(tmp_path: Path) -> None:
    service, record, root, transport = _service(tmp_path)
    analysis = {
        "source": {"status": "final"},
        "parameter_importance": {"width": {"importance": 0.3}},
        "candidates": [
            {
                "trial": 1,
                "parameters": {"width": 64},
                "mean": 0.7,
                "runs": [{"failure": {"traceback": "x" * 9_000_000}}],
            }
        ],
        "research": {"detailed_findings": ["not needed for this console view"]},
    }
    _write(root / "analysis.json", analysis)
    loaded = service._load_study_summary(record, transport, view="analysis")
    assert loaded == analysis_panel(analysis)
    assert loaded["candidates"] == [{"trial": 1, "parameters": {"width": 64}, "mean": 0.7}]
    assert "research" not in loaded
    assert sum(transport.response_sizes) < 2000
    _write(root / "analysis-panel.json", loaded)
    (root / "analysis.json").write_text("must not be read", encoding="utf-8")
    assert service._load_study_summary(record, transport, view="analysis") == loaded


def test_large_trial_table_has_no_total_transfer_ceiling(tmp_path: Path) -> None:
    service, record, root, transport = _service(tmp_path)
    candidate = {
        "trial": 1,
        "state": "succeeded",
        "selection_objective": 0.512312321321,
        "current_objective": 0.512312321321,
        "best_objective": 0.512312321321,
        "selection_seed_count": 3,
        "selection_standard_error": 0.012321321321,
        "latest_step": 120,
        "gpu_indices": [0, 1],
    }
    table = study_table(
        {
            "study_telemetry_version": 1,
            "candidates": [{**candidate, "trial": trial} for trial in range(1, 40_001)],
        }
    )
    _write(root / "interactive.json", table)
    assert (root / "interactive.json").stat().st_size > 8 * 1024 * 1024
    loaded = service._load_study_summary(record, transport)
    assert len(loaded["candidates"]) == 40_000
    assert loaded["candidates"][-1]["trial"] == 40_000
    assert len(transport.requests) > 16
    assert max(transport.response_sizes) < 1024 * 1024


def test_changed_snapshot_restarts_instead_of_mixing_pages(tmp_path: Path) -> None:
    _service_instance, _record, root, _transport = _service(tmp_path)
    _write(root / "hpo.json", {"hpo_analysis": {"data": "x" * 600_000}})
    first = projection_page(str(root), {"view": "hpo"})
    _write(root / "hpo.json", {"hpo_analysis": {"data": "new"}})
    assert projection_page(
        str(root),
        {
            "view": "hpo",
            "offset": first["next_offset"],
            "fingerprint": first["fingerprint"],
        },
    ) == {"restart": True}


def test_provisional_analysis_runs_once_on_host_and_pins_paged_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

    _service_instance, _record, root, _transport = _service(tmp_path)
    _write(root / "summary.json", {"study_telemetry_version": 1})
    calls = []

    def compute(value, **kwargs):
        calls.append(value)
        return {"status": "provisional", "data": "x" * 600_000}

    monkeypatch.setattr(StudyAnalysis, "compute", compute)
    first = projection_page(str(root), {"view": "analysis-report"})
    assert len(calls) == 1
    # The running worker keeps updating summary while the explicit post-hoc read is paged.
    _write(root / "summary.json", {"study_telemetry_version": 1, "updated": True})
    second = projection_page(
        str(root),
        {
            "view": "analysis-report",
            "fingerprint": first["fingerprint"],
            "offset": first["next_offset"],
        },
    )
    assert second["eof"] and len(calls) == 1
    assert json.loads(base64.b64decode(first["data"]) + base64.b64decode(second["data"])) == {
        "status": "provisional",
        "data": "x" * 600_000,
    }
    assert sorted(p.name for p in root.glob("analysis-read-*")) == [
        "analysis-read-cache.json",
        "analysis-read-cache.lock",
    ]


def test_host_reader_passes_sweep_geometry_to_an_older_analysis_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

    _service_instance, _record, root, _transport = _service(tmp_path)
    space = {
        "family": {"values": ["max", "attention"]},
        "variant": {"values": ["simple", "gated"], "when": {"family": "attention"}},
    }
    _write(
        root / "summary.json",
        {"study_telemetry_version": 1, "design": {"type": "sweep", "space": space}},
    )
    calls = []

    def old_compute(value, *, authored_space=None, **kwargs):
        assert authored_space == space
        calls.append(value)
        return {"search_space": authored_space, "status": "provisional"}

    monkeypatch.setattr(StudyAnalysis, "compute", old_compute)
    page = projection_page(str(root), {"view": "analysis"})
    value = json.loads(base64.b64decode(page["data"]))
    assert value["search_space"] == space
    assert len(calls) == 1
    assert projection_page(
        str(root), {"view": "analysis", "fingerprint": page["fingerprint"]}
    )["unchanged"]


def test_remote_analysis_failure_preserves_the_terminal_exception(tmp_path: Path) -> None:
    service, record, _root, _transport = _service(tmp_path)
    stderr = (
        "Traceback (most recent call last):\n"
        + "  File remote/Effects.py, in response_curve\n" * 200
        + "statistics.StatisticsError: fmean requires at least one data point"
    )
    transport = SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=stderr)
    )
    with pytest.raises(RuntimeError) as caught:
        service._load_study_summary(record, transport, view="analysis")
    message = str(caught.value)
    assert message.startswith("Could not read Study analysis: Traceback")
    assert message.endswith("statistics.StatisticsError: fmean requires at least one data point")
    assert len(message) < 2000


def test_parameter_filter_is_host_owned_and_returns_only_table_columns(tmp_path: Path) -> None:
    _service_instance, _record, root, _transport = _service(tmp_path)
    rich = _snapshot()
    rich["candidates"][1]["parameters"] = {"pooling": "negative-log"}
    _write(root / "interactive.json", study_table(rich))
    _write(root / "index.json", {"candidates": rich["candidates"]})
    page = projection_page(str(root), {"query": "negative"})
    table = json.loads(base64.b64decode(page["data"]))
    assert len(table["candidates"]) == 1
    assert table["candidates"][0]["trial"] == 2
    assert "parameters" not in table["candidates"][0]


def test_overview_reprojects_old_overview_without_device_bulk() -> None:
    value = study_overview(_snapshot())
    value["admission"]["current"]["large_device_diagnostics"] = "x" * 100_000
    value["accidental_details"] = {"runs": [1, 2, 3]}
    projected = study_overview(value)
    assert "accidental_details" not in projected
    assert "large_device_diagnostics" not in projected["admission"]["current"]
    assert projected["leader"] == value["leader"]


@pytest.mark.parametrize(
    "view,options",
    [
        ("trial", {"trial": -1}),
        ("run", {"trial": 1, "run": "../../outside"}),
    ],
)
def test_invalid_view_selectors_fail_before_reading(
    tmp_path: Path, view: str, options: dict
) -> None:
    with pytest.raises(ValueError):
        projection_page(str(tmp_path), {"view": view, **options})


def test_view_symlink_is_rejected_instead_of_falling_back(tmp_path: Path) -> None:
    _write(tmp_path / "summary.json", {"study_telemetry_version": 1})
    (tmp_path / "hpo.json").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="Symlinked"):
        projection_page(str(tmp_path), {"view": "hpo"})


@pytest.mark.parametrize(
    "response, message",
    [
        ("[]", "Invalid Study page"),
        (json.dumps({"unchanged": True, "fingerprint": "unknown"}), "Invalid unchanged"),
        (
            json.dumps({"offset": 0, "next_offset": 0, "data": "", "eof": False}),
            "no forward progress",
        ),
        ("x" * (1024 * 1024 + 1), "per-page safety limit"),
    ],
)
def test_corrupt_or_oversized_packets_are_rejected(
    tmp_path: Path,
    response: str,
    message: str,
) -> None:
    from lambdaforge.controlplane.CommandResult import CommandResult

    service, record, _root, _transport = _service(tmp_path)

    class BrokenTransport:
        def run(self, *args, **kwargs):
            return CommandResult(0, response, "")

    with pytest.raises(RuntimeError, match=message):
        service._load_study_summary(record, BrokenTransport())
