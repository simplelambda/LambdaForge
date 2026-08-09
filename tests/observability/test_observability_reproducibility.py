"""Focused structured observability and reproducibility tests."""

import json
from pathlib import Path

from lambdaforge.observability import EventLogger, ResourceMonitor
from lambdaforge.reproducibility import ReproducibilityProfile, SeedDeriver


def test_events_monitor_seed_and_fingerprint_contracts(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    EventLogger(path).write("batch_finished", {"items": 32})
    assert json.loads(path.read_text(encoding="utf-8"))["event"] == "batch_finished"
    assert (
        ResourceMonitor(min_interval_seconds=0.001).sample(processed_items=10, elapsed_seconds=2)[
            "items_per_second"
        ]
        == 5
    )
    deriver = SeedDeriver(7)
    assert deriver.derive("loader", 0) == deriver.derive("loader", 0)
    fingerprints = ReproducibilityProfile.fingerprints(
        {"model": {"width": 8}, "execution": {"mode": "parallel"}}
    )
    assert fingerprints["scientific"] != fingerprints["infrastructure"]
