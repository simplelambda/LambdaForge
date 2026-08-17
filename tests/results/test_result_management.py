"""Scientific run identity and attempt-catalog regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus


class TestResultManagement:
    """Prevent stale reuse and make repeated attempts explicitly discoverable."""

    @staticmethod
    def _config(root: Path, *, width: int = 8) -> dict[str, object]:
        return {
            "schema_version": "1.1",
            "experiment": {
                "name": "identity_demo__base",
                "base_name": "identity_demo",
                "variant": "base",
                "seed": 7,
                "output_root": str(root),
            },
            "data": {},
            "model": {"target": "types.SimpleNamespace", "params": {"width": width}},
            "losses": [{"target": "types.SimpleNamespace"}],
            "trainer": {"checkpoint_policy": "none"},
        }

    def test_fingerprint_ignores_storage_and_retry_controls(self, tmp_path: Path) -> None:
        first = self._config(tmp_path / "first")
        second = copy.deepcopy(first)
        second["experiment"]["name"] = "renamed__variant"
        second["experiment"]["output_root"] = str(tmp_path / "second")
        second["experiment"]["rerun_completed"] = True
        second["execution"] = {"mode": "parallel", "gpus": [0]}
        second["retention"] = {"mode": "apply"}

        assert RunFingerprint.digest(first) == RunFingerprint.digest(second)
        assert RunFingerprint.digest(first) != RunFingerprint.digest(
            self._config(tmp_path / "first", width=16)
        )

    def test_completed_result_must_match_materialized_scientific_config(
        self,
        tmp_path: Path,
    ) -> None:
        old = self._config(tmp_path, width=8)
        changed = self._config(tmp_path, width=16)
        runner = ExperimentRunner()
        run_dir = runner.experiment_run_dir(old)
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
        RunResult(
            name="identity_demo__base",
            run_dir=run_dir,
            variant="base",
            seed=7,
            status=RunStatus.OK,
        ).write_json(run_dir / "result.json")
        checkpoint = run_dir / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"old model")

        assert runner._completed_result(old, run_dir) is not None
        assert runner._completed_result(changed, run_dir) is None
        assert runner._prepare_rerun_config(changed, run_dir)["experiment"].get("ckpt_path") is None

    def test_catalog_keeps_attempts_grouped_and_selectable(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        runner = ExperimentRunner()
        run_dir = runner.experiment_run_dir(config)
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        fingerprint = RunFingerprint.digest(config)
        for attempt_id, score in (("attempt-one", 0.4), ("attempt-two", 0.3)):
            RunResult(
                result_version=2,
                name="identity_demo__base",
                run_dir=run_dir,
                variant="base",
                seed=7,
                status=RunStatus.OK,
                attempt_id=attempt_id,
                config_fingerprint=fingerprint,
                final_metrics={"val_loss": score},
            ).write_json(run_dir / "result.json")
            if attempt_id == "attempt-one":
                runner._archive_previous_result(run_dir)

        catalog = ResultCatalog(tmp_path / "identity_demo")
        records = catalog.records()

        assert len(records) == 2
        assert len(catalog.duplicate_groups()[fingerprint]) == 2
        assert len(catalog.ambiguous_successes()[fingerprint]) == 2
        assert catalog.select(attempt_id="attempt-one").archived
        index = json.loads(catalog.write_index().read_text(encoding="utf-8"))
        assert index["ambiguous_success_fingerprints"] == [fingerprint]

    def test_results_cli_can_fail_ci_on_ambiguous_successes(
        self,
        tmp_path: Path,
        capsys,
    ) -> None:
        self.test_catalog_keeps_attempts_grouped_and_selectable(tmp_path)
        root = tmp_path / "identity_demo"

        exit_code = CommandLineInterface.main(
            ["results", str(root), "--json", "--fail-on-ambiguous"]
        )
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 2
        assert len(payload["records"]) == 2

    def test_results_cli_rejects_a_missing_yaml_instead_of_treating_it_as_a_root(
        self,
        tmp_path: Path,
        capsys,
    ) -> None:
        exit_code = CommandLineInterface.main(["results", str(tmp_path / "missing.yaml")])

        assert exit_code == 2
        assert "CONFIGURATION ERROR" in capsys.readouterr().err
