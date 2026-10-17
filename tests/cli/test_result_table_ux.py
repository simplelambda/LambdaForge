"""Small derived exports retain native provenance and never read artifact payloads."""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.cli.result_tables import render_table
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner
from lambdaforge.work.result_tables import PROVENANCE_COLUMNS, result_table


def evidence():
    def attempt(run, number, state, metric, **extra):
        return {
            "execution_id": "execution-table",
            "scientific_fingerprint": "sha256:run-evidence",
            "name": "fixture",
            "run_id": f"run-{run}",
            "attempt_id": f"attempt-{number:03}",
            "attempt_number": number,
            "status": state,
            "metrics": metric,
            "seed": 11 if run == "a" else 23,
            "trial": {"index": 1 if run == "a" else 2},
            **extra,
        }

    return {
        "execution_id": "execution-table",
        "scientific_fingerprint": "sha256:execution-evidence",
        "status": "completed_with_failures",
        "runs": [
            attempt("a", 1, "failed", {"accuracy": 0.3}),
            attempt("a", 2, "succeeded", {"accuracy": 0.8, "loss": 0.2}),
            attempt("b", 1, "pruned", {"accuracy": 0.4}, pruned=True),
        ],
    }


def test_latest_attempt_table_never_duplicates_completed_evidence_or_invents_final_scores():
    selected = evidence()
    before = copy.deepcopy(selected)
    value = result_table(selected, metrics=("accuracy",))
    assert len(value["rows"]) == 2
    assert value["rows"][0]["value"] == 0.8
    assert value["rows"][0]["attempt_number"] == 2
    assert value["rows"][1]["evidence_kind"] == "censored_snapshot"
    assert value["rows"][1]["value"] == 0.4
    assert selected == before
    assert "not inferred best/final/selection" in value["notice"]
    assert len(result_table(selected, metrics=("accuracy",), attempts="all")["rows"]) == 3


def test_table_filters_exact_runs_candidates_columns_and_keeps_required_provenance():
    selected = evidence()
    rows = result_table(selected, trials=(2,), metrics=("accuracy",), columns=("value",))["rows"]
    assert len(rows) == 1 and set(rows[0]) == set(PROVENANCE_COLUMNS) | {"value"}
    assert rows[0]["run_id"] == "run-b"
    assert result_table(selected, runs=("run-a",), metrics=("loss",))["rows"][0]["value"] == 0.2
    with pytest.raises(KeyError, match="Unknown selected Run"):
        result_table(selected, runs=("a",))
    with pytest.raises(KeyError, match="Unknown selected Trials"):
        result_table(selected, trials=(99,))
    with pytest.raises(ValueError, match="Unknown table columns"):
        result_table(selected, columns=("artifacts",))


def test_table_missing_metric_is_explicit_and_unchanged_paging_is_stable():
    selected = evidence()
    first = result_table(selected, metrics=("loss",), limit=1)
    second = result_table(
        selected,
        metrics=("loss",),
        offset=first["next_offset"],
        fingerprint=first["evidence_fingerprint"],
    )
    assert second["rows"][0]["value"] is None
    assert second["next_offset"] is None
    selected["runs"][-1]["metrics"]["loss"] = 0.9
    with pytest.raises(ValueError, match="evidence changed"):
        result_table(selected, fingerprint=first["evidence_fingerprint"])
    with pytest.raises(KeyError, match="Metrics absent"):
        result_table(selected, metrics=("invented-metric",))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "broken"])
def test_table_refuses_corrupt_non_scalar_metric(value):
    selected = evidence()
    selected["runs"][-1]["metrics"]["accuracy"] = value
    with pytest.raises(ValueError):
        result_table(selected)


def test_csv_and_json_escape_authored_content_without_exporting_secrets():
    selected = evidence()
    selected["runs"][1]["name"] = '=HYPERLINK("malicious")'
    selected["runs"][1]["parameters"] = {"api_key": "private-token", "normal": "λ\ncomma,value"}
    value = result_table(selected, metrics=("accuracy",), columns=("name", "parameters", "value"))
    csv_text = render_table(value, "csv")
    assert "private-token" not in csv_text and "private-token" not in render_table(value, "json")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert rows[0]["name"].startswith("'=")
    assert json.loads(rows[0]["parameters"])["normal"] == "λ\ncomma,value"
    assert rows[0]["execution_id"] == "execution-table"


def test_table_bounds_large_optional_columns_and_duplicate_attempts():
    selected = evidence()
    selected["runs"][1]["parameters"] = {"large": "x" * (490 * 1024)}
    with pytest.raises(ValueError, match="One result table row"):
        result_table(selected, columns=("parameters",))
    assert result_table(selected, columns=("value",))["rows"]
    selected["runs"].append(selected["runs"][0])
    with pytest.raises(ValueError, match="Duplicate native"):
        result_table(selected)


def test_native_cpu_result_cli_json_csv_ownership_and_no_payload_reads(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="table-fixture"\nversion="1"\n')
    config = WorkConfig.from_mapping(
        {
            "name": "table-fixture",
            "run": "tests.work_cases.SeedWork",
            "seeds": [11, 23],
            "resources": {"cpu": 1},
        },
        source=tmp_path / "work.yaml",
    )
    execution = WorkRunner().run(config)
    store = ResultStore()
    before = {p: p.read_bytes() for p in execution.execution_dir.rglob("*") if p.is_file()}
    original_read = Path.read_text

    def metadata_only(path, *args, **kwargs):
        assert path.suffix != ".jsonl" and path.name != "work.log"
        assert "checkpoints" not in path.parts and "artifacts" not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", metadata_only)
    value = store.table(execution.execution_id, metrics=("score",))
    assert sorted(row["value"] for row in value["rows"]) == [11.0, 23.0]
    assert before == {p: p.read_bytes() for p in execution.execution_dir.rglob("*") if p.is_file()}
    base = ["results", "table", execution.execution_id, "--metric", "score"]
    capsys.readouterr()
    assert CommandLineInterface.main(base + ["--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["rows"]) == 2
    output = tmp_path / "scalars.csv"
    assert CommandLineInterface.main(base + ["--format", "csv", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 2
    assert len(list(csv.DictReader(io.StringIO(output.read_text())))) == 2
    with pytest.raises(FileExistsError):
        CommandLineInterface._dispatch(base + ["--output", str(output)], json_output=False)
    with pytest.raises(ValueError, match="outside the native"):
        CommandLineInterface._dispatch(
            base + ["--output", str(execution.execution_dir / "unsafe.csv")], json_output=False
        )
    link = tmp_path / "linked"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        CommandLineInterface._dispatch(
            base + ["--output", str(link / "unsafe.csv")], json_output=False
        )
