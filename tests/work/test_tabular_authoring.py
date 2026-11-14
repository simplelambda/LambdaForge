"""Typed table preview uses the sole Work lifecycle and durable report evidence."""

import json

from lambdaforge.work import WorkConfig, WorkRunner


def test_tabular_preview_recipe_is_an_ordinary_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "pyproject.toml").write_text('[project]\nname="table-recipe"\nversion="1"\n')
    (tmp_path / "rows.csv").write_text("id,value\na,2\nb,NA\nc,1\n")
    config = WorkConfig.from_mapping(
        {
            "name": "table-preview",
            "run": "examples.tabular_preview.TableReport",
            "with": {"source": {"file": "rows.csv"}, "limit": 2},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "work.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    run = result.runs[0]
    assert run.primary_result == {"preview_rows": 2, "scope": "prefix", "complete": False}
    artifact = next(a for a in run.artifacts if a.name == "preview")
    report = json.loads((run.run_dir / artifact.path).read_text())
    assert report["rows"][1]["value"] is None
    assert report["visible_sorted_rows"] == [{"id": "a", "value": 2.0}]
