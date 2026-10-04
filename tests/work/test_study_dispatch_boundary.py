"""One scientific planner may use a different dispatcher without copying its algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.work import WorkConfig, WorkRunner, runner


@pytest.mark.parametrize("adaptive", [False, True])
def test_native_science_uses_injected_dispatch_and_keeps_identity(
    tmp_path: Path,
    adaptive: bool,
) -> None:
    source = tmp_path / "study.yaml"
    value: dict[str, Any] = {
        "name": "boundary",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": [4, 7],
        "resources": {"cpu": 2},
        "objective": {"metric": "score", "mode": "max"},
    }
    if adaptive:
        value["search"] = {
            "trials": 2,
            "quality": {"values": [0.2, 0.8]},
            "min_seeds": 1,
            "confirmation_seeds": [],
            "early_stopping": False,
        }
    else:
        value["sweep"] = {"space": {"quality": {"values": [0.2, 0.8]}}}
    source.write_text(yaml.safe_dump(value))
    config = WorkConfig.from_yaml(source)
    calls = []
    native = runner._execute_adaptive_dispatch

    def dispatcher(specifications: Any, **options: Any) -> Any:
        calls.append((specifications, options))
        return native(specifications, **options)

    plain = WorkRunner().plan(config)
    injected = WorkRunner(dispatcher=dispatcher)
    assert injected.plan(config).scientific_fingerprint == plain.scientific_fingerprint
    result = injected.run(config)
    assert result.status == "succeeded"
    assert calls
    assert all(call[1]["on_result"] is not None for call in calls)
    assert len({(item.trial["index"], item.seed) for item in result.runs}) == len(result.runs)
    if adaptive:
        decisions = result.execution_dir / "hpo-control" / "decisions.jsonl"
        assert decisions.is_file()
        assert len(list(result.execution_dir.rglob("decisions.jsonl"))) == 1


def test_injected_dispatch_rejects_composition_before_state_is_created(tmp_path: Path) -> None:
    config = WorkConfig.from_mapping(
        {"name": "ordinary", "run": "tests.work_cases.Producer"}, source=tmp_path / "work.yaml"
    )
    with pytest.raises(ValueError, match="one Study"):
        WorkRunner(dispatcher=lambda *args, **kwargs: ()).run(config)
    assert not (tmp_path / ".lambdaforge").exists()
