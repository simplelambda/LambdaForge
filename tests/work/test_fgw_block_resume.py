"""Real native Work lifecycle: reuse good blocks, repair corrupt/missing/incomplete ones."""

import json

import pytest
import yaml

from lambdaforge.work import WorkConfig, WorkRunner

pytest.importorskip("ot")


@pytest.mark.parametrize("corruption", [None, "missing", "malformed", "checksum", "incomplete"])
def test_native_work_resume_and_selective_block_repair(tmp_path, corruption):
    yaml_path = tmp_path / "work.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"name": "fgw-blocks", "run": "tests.fgw_cases.BlockMatrixWork"})
    )
    config = WorkConfig.from_yaml(yaml_path)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    checkpoints = sorted(tmp_path.rglob("maps/fgw-blocks-*/*.json"))
    assert len(checkpoints) == 2  # completed blocks survive interruption of the third
    if corruption == "missing":
        checkpoints[0].unlink()
    elif corruption == "malformed":
        checkpoints[0].write_text("not valid JSON")
    elif corruption in {"checksum", "incomplete"}:
        content = json.loads(checkpoints[0].read_text())
        content["result"]["values"] = [12345.0] if corruption == "checksum" else []
        checkpoints[0].write_text(json.dumps(content))
    second = WorkRunner().run(config)
    assert second.status == "succeeded", second.runs[0].failure
    result = second.runs[0].primary_result
    assert result["calls"] == 7 + int(corruption is not None)
    assert len(result["values"]) == 3


def test_native_work_does_not_reuse_recorded_failed_block(tmp_path):
    path = tmp_path / "work.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "fgw-partial",
                "run": "tests.fgw_cases.BlockMatrixWork",
                "with": {"error_policy": "record"},
            }
        )
    )
    config = WorkConfig.from_yaml(path)
    assert WorkRunner().run(config).status == "failed"
    second = WorkRunner().run(config)
    assert second.status == "succeeded"
    # Six comparisons were attempted initially; only the failed one is repeated.
    assert second.runs[0].primary_result["calls"] == 7
