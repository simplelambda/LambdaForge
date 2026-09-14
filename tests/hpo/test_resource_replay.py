"""Trace replay distinguishes observed history from counterfactual simulation."""

from __future__ import annotations

import json
from pathlib import Path

from lambdaforge.cli.parser import build_parser
from lambdaforge.hpo.ResourceReplay import ResourceSchedulerReplay, ResourceTraceEvent

GIB = 1024**3


def _event(
    elapsed: float,
    *,
    active: int,
    state: str,
    admitted: bool,
) -> ResourceTraceEvent:
    return ResourceTraceEvent.from_mapping(
        {
            "elapsed_seconds": elapsed,
            "devices": [
                {
                    "gpu": 0,
                    "total_bytes": 80 * GIB,
                    "physical_free_bytes": (70 - active * 10) * GIB,
                    "active_runs": active,
                    "active_candidates": [f"active-{index}" for index in range(active)],
                    "active_current_bytes": [8 * GIB] * active,
                    "resource_states": [state] * active,
                    "active_throughputs": [1.0] * active,
                }
            ],
            "frontier": [
                {
                    "candidate": f"candidate-{elapsed}",
                    "scientific_value": 1.0,
                    "value_contract": {"normalized_value": 0.8},
                }
            ],
            "decisions": (
                [
                    {
                        "candidate_key": f"candidate-{elapsed}",
                        "target_gpu": 0,
                        "state": "ADMITTED",
                        "admission_mode": "EXPLORATORY_ADMISSION",
                        "scientific_value": 1.0,
                        "rollback_cost_seconds": 2.0,
                    }
                ]
                if admitted
                else [
                    {
                        "candidate_key": f"candidate-{elapsed}",
                        "state": "RESOURCE_BLOCKED",
                        "scientific_value": 1.0,
                    }
                ]
            ),
        }
    )


def test_replay_is_exact_until_divergence_and_labels_simulated_tail() -> None:
    replay = ResourceSchedulerReplay(
        (
            _event(0.0, active=1, state="RAMPING", admitted=False),
            _event(10.0, active=1, state="RAMPING", admitted=True),
            _event(20.0, active=2, state="PROVISIONALLY_STABLE", admitted=True),
        )
    )
    recorded = replay.replay("recorded")
    legacy = replay.replay("ari-v2-compat")

    assert recorded["counterfactual_divergence"] is None
    assert recorded["segments"]["factual_events"] == 3
    assert legacy["counterfactual_divergence"]["event_index"] == 1
    assert legacy["segments"]["factual_events"] == 1
    assert legacy["segments"]["counterfactual_simulated_events"] == 2
    assert legacy["segments"]["post_divergence_is_observed"] is False
    assert legacy["metrics"]["completed_runs_per_hour"] is None
    assert legacy["segments"]["terminal_outcomes_source"].startswith("unavailable")


def test_v31_wait_regret_can_diverge_from_v3_compatibility_policy() -> None:
    event = ResourceTraceEvent.from_mapping(
        {
            "elapsed_seconds": 10.0,
            "devices": [
                {
                    "gpu": 0,
                    "total_bytes": 80 * GIB,
                    "physical_free_bytes": 50 * GIB,
                    "active_runs": 1,
                    "active_candidates": ["resident"],
                    "active_current_bytes": [10 * GIB],
                    "active_throughputs": [1.0],
                }
            ],
            "frontier": [{"candidate": "probe", "scientific_value": 0.8}],
            "decisions": [
                {
                    "candidate_key": "probe",
                    "target_gpu": 0,
                    "state": "ADMITTED",
                    "admission_mode": "EXPLORATORY_ADMISSION",
                }
            ],
            "exploration": [
                {
                    "candidate": "probe",
                    "final_delta_value": 0.4,
                    "wait_regret": 0.6,
                    "plan": "EXPLORE",
                }
            ],
        }
    )
    replay = ResourceSchedulerReplay((event,))

    assert replay.replay("ari-v3.1")["counterfactual_divergence"] is None
    legacy = replay.replay("ari-v3-compat")
    assert legacy["counterfactual_divergence"]["event_index"] == 0
    assert "WaitRegret" in legacy["policy_assumptions"]["rule"]


def test_replay_loads_versioned_trace_and_reports_calibration(tmp_path: Path) -> None:
    resources = tmp_path / "hpo-control" / "resources"
    resources.mkdir(parents=True)
    event = _event(10.0, active=1, state="PROVISIONALLY_STABLE", admitted=True)
    (resources / "scheduler-trace.jsonl").write_text(
        json.dumps(
            {
                "trace_version": 1,
                "elapsed_seconds": event.elapsed_seconds,
                "devices": list(event.devices),
                "frontier": list(event.frontier),
                "decisions": list(event.decisions),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (resources / "observations.jsonl").write_text(
        json.dumps(
            {
                "state": "completed",
                "candidate_key": "candidate-10.0",
                "predicted_fit_probability": 0.8,
                "placement_succeeded": True,
                "admission_mode": "SAFE_ADMISSION",
                "predicted_peak_bytes": 12 * GIB,
                "observed_peak_bytes": 10 * GIB,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = ResourceSchedulerReplay.from_execution(tmp_path).replay("ari-v3.1")
    assert payload["events"] == 1
    assert payload["metrics"]["fit_probability_calibration"]["observations"] == 1
    assert payload["metrics"]["false_safe_admission_rate"] == 0.0
    assert payload["metrics"]["peak_prediction_mae_bytes"] == 2 * GIB


def test_resource_replay_is_exposed_by_the_shared_cli_parser() -> None:
    arguments = build_parser().parse_args(
        ["results", "replay", "execution-1", "--policy", "ari-v3-compat", "--json"]
    )

    assert arguments.result_command == "replay"
    assert arguments.selector == "execution-1"
    assert arguments.policy == "ari-v3-compat"
    assert arguments.json is True
