"""Typed terminal and aggregate result compatibility contracts."""

import json

from lambdaforge.experiments import (
    AggregateResult,
    RunResult,
    RunStatus,
    VariantAggregateResult,
)


class TestTypedResults:
    """Keep attributes, Mapping behavior and persisted JSON mutually compatible."""

    def test_run_result_round_trip_preserves_known_and_extra_fields(self, tmp_path) -> None:
        result = RunResult(
            name="demo",
            run_dir=tmp_path / "demo",
            variant="width=8",
            seed=7,
            status=RunStatus.OK,
            seconds=1.25,
            final_metrics={"val_loss": 0.5},
            extra={"future_field": {"enabled": True}},
        )

        assert result.status is RunStatus.OK
        assert result["status"] == "ok"
        assert result.get("final_metrics") == {"val_loss": 0.5}
        assert json.loads(json.dumps(result))["result_version"] == 1
        path = result.write_json(tmp_path / "result.json")
        restored = RunResult.read_json(path)
        assert restored.to_dict() == result.to_dict()
        assert restored["future_field"] == {"enabled": True}
        assert not list(tmp_path.glob(".*.tmp"))

    def test_run_result_defensively_copies_nested_payloads(self) -> None:
        metrics = {"loss": 1.0}
        result = RunResult(
            name="demo",
            run_dir="run",
            status="future_status",
            final_metrics=metrics,
        )
        metrics["loss"] = 9.0
        exported = result.to_dict()
        exported["final_metrics"]["loss"] = 3.0

        assert result.status is RunStatus.UNKNOWN
        assert result["status"] == "future_status"
        assert result.final_metrics == {"loss": 1.0}

    def test_results_reject_mapping_and_attribute_mutation(self) -> None:
        result = RunResult(name="demo", run_dir="run", status=RunStatus.OK)

        try:
            result["status"] = "failed"
        except TypeError:
            pass
        else:
            raise AssertionError("RunResult accepted item assignment.")

        try:
            result.name = "changed"
        except AttributeError:
            pass
        else:
            raise AssertionError("RunResult accepted attribute assignment.")

    def test_aggregate_result_keeps_legacy_nested_mapping_access(self, tmp_path) -> None:
        result = AggregateResult.from_mapping(
            {
                "base": {
                    "variant": "base",
                    "complete": True,
                    "terminal": True,
                    "expected_n": 2,
                    "n_seeds": 2,
                    "metrics": {"val_loss": {"mean": 0.25}},
                }
            }
        )

        assert result["base"]["n_seeds"] == 2
        assert isinstance(result.variant("base"), VariantAggregateResult)
        assert result.variant("base").completed_runs == 2
        path = result.write_json(tmp_path / "aggregate.json")
        assert json.loads(path.read_text(encoding="utf-8"))["base"]["complete"] is True

    def test_reads_complete_summary_without_losing_metadata(self, tmp_path) -> None:
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps(
                {
                    "experiment": "demo",
                    "expected_runs": 1,
                    "variants": {
                        "base": {
                            "variant": "base",
                            "expected_n": 1,
                            "n_seeds": 1,
                            "metrics": {},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        restored = AggregateResult.read_json(path)
        assert restored.to_summary_dict()["experiment"] == "demo"
        assert restored.variant("base").expected_runs == 1
