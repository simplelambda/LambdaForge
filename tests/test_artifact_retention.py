"""Safety, lifecycle and transaction tests for artifact retention."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest
import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments import (
    ArtifactCompressionOptions,
    ArtifactRetentionAction,
    ArtifactRetentionMode,
    ArtifactRetentionPolicy,
    ArtifactRetentionRule,
    CheckpointChoice,
    CheckpointRetention,
    Experiment,
    ExperimentAggregator,
    ExperimentConfig,
    ExperimentRunner,
    ExperimentValidator,
    RunLoader,
)
from lambdaforge.experiments.ExperimentExecutor import ExperimentExecutor
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import (
    ExperimentSchemaCatalog,
)
from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
from lambdaforge.experiments.retention.ArtifactRetentionJournal import (
    ArtifactRetentionJournal,
)
from lambdaforge.experiments.retention.ArtifactRetentionManager import (
    ArtifactRetentionManager,
)
from lambdaforge.experiments.retention.ArtifactRetentionPhase import (
    ArtifactRetentionPhase,
)
from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
from lambdaforge.experiments.retention.ArtifactRetentionResult import (
    ArtifactRetentionResult,
)
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus
from lambdaforge.experiments.retention.ArtifactRetentionTransaction import (
    ArtifactRetentionTransaction,
)
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from tests.fixtures.RetentionApplyJob import RetentionApplyJob


class TestArtifactRetention:
    """Verify preview-first retention and fail-closed filesystem behavior."""

    @staticmethod
    def _config(
        tmp_path: Path,
        *,
        mode: str = "preview",
        keep: str = "best",
        prune_checkpoints: bool = True,
        rules: list[dict[str, object]] | None = None,
        protect: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": "1.1",
            "experiment": {
                "name": "retention_demo",
                "output_root": str(tmp_path),
                "seeds": [7],
                "required_artifacts": ["final/model-card.txt"],
            },
            "data": {},
            "model": {"target": "types.SimpleNamespace"},
            "losses": [{"target": "types.SimpleNamespace"}],
            "trainer": {"checkpoint_policy": "last_and_best"},
            "retention": {
                "mode": mode,
                "checkpoints": {
                    "keep": keep,
                    "prune_unselected": prune_checkpoints,
                },
                "protect": protect or ["artifacts/keep.bin"],
                "rules": rules
                if rules is not None
                else [
                    {
                        "action": "compress",
                        "include": ["artifacts/*.bin"],
                        "exclude": [],
                        "min_size_bytes": 1,
                        "compression": {"level": 9, "only_if_smaller": True},
                    },
                    {
                        "action": "prune",
                        "include": ["scratch/*.tmp"],
                        "exclude": [],
                        "min_size_bytes": 0,
                    },
                ],
                "archive": {
                    "name": "intermediate-artifacts.zip",
                    "compression_level": 6,
                },
                "lock_timeout_seconds": 5,
            },
        }

    @classmethod
    def _materialize(
        cls,
        tmp_path: Path,
        *,
        config: dict[str, object] | None = None,
        status: RunStatus = RunStatus.OK,
        compressible: bool = True,
    ) -> tuple[ExperimentConfig, Path]:
        normalized = ExperimentConfig(config or cls._config(tmp_path))
        run = normalized.expand()[0]
        run_dir = normalized.suite_dir / "base" / "seed=7"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "artifacts").mkdir()
        (run_dir / "scratch").mkdir()
        (run_dir / "final").mkdir()
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(run, sort_keys=False),
            encoding="utf-8",
        )
        (run_dir / "environment.json").write_text('{"python": "test"}\n', encoding="utf-8")
        (run_dir / "hparams.json").write_text('{"width": 8}\n', encoding="utf-8")
        (run_dir / "train.log").write_text("training complete\n", encoding="utf-8")
        (run_dir / "metrics.csv").write_text(
            "epoch,val_loss\n1,0.5\n",
            encoding="utf-8",
        )
        (run_dir / "final" / "model-card.txt").write_text("keep me\n", encoding="utf-8")
        payload = b"A" * 8192 if compressible else os.urandom(8192)
        (run_dir / "artifacts" / "large.bin").write_bytes(payload)
        (run_dir / "artifacts" / "keep.bin").write_bytes(b"protected")
        (run_dir / "scratch" / "temporary.tmp").write_bytes(b"discard")
        best = run_dir / "checkpoints" / "best-001.ckpt"
        last = run_dir / "checkpoints" / "last.ckpt"
        extra = run_dir / "checkpoints" / "epoch-000.ckpt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        extra.write_bytes(b"extra")
        RunResult(
            name="retention_demo__base",
            run_dir=run_dir,
            variant="base",
            seed=7,
            status=status,
            best_model_path=best,
            last_model_path=last,
            best_epoch_metrics={"epoch": 1, "metrics": {"val_loss": 0.5}},
            final_metrics={"val_loss": 0.5},
        ).write_json(run_dir / "result.json")
        ExperimentAggregator().write(normalized, make_plots=False, final=True)
        return normalized, run_dir

    def test_policy_defaults_are_safe_and_typed(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        config["retention"] = {}

        policy = ArtifactRetentionPolicy.from_config(ExperimentConfig(config))

        assert policy.mode is ArtifactRetentionMode.DISABLED
        assert policy.checkpoint_keep is CheckpointRetention.ALL
        assert not policy.prune_unselected_checkpoints
        assert not policy.rules
        assert policy.archive_compression_level == 6

    @pytest.mark.parametrize(
        ("retention", "error_type"),
        [
            (None, TypeError),
            (
                {
                    "rules": [
                        {
                            "action": "prune",
                            "include": ["scratch/**"],
                            "unknown": True,
                        }
                    ]
                },
                ValueError,
            ),
            (
                {
                    "rules": [
                        {
                            "action": "compress",
                            "include": ["artifacts/**"],
                            "compression": {"unknown": True},
                        }
                    ]
                },
                ValueError,
            ),
            (
                {
                    "rules": [
                        {
                            "action": "compress",
                            "include": ["artifacts/**"],
                            "compression": False,
                        }
                    ]
                },
                TypeError,
            ),
            ({"protect": [7]}, TypeError),
            (
                {
                    "rules": [
                        {
                            "action": "prune",
                            "include": [7],
                        }
                    ]
                },
                TypeError,
            ),
            (
                {
                    "rules": [
                        {
                            "action": "prune",
                            "include": ["scratch/**"],
                            "compression": None,
                        }
                    ]
                },
                ValueError,
            ),
            ({"archive": {"name": "nested.name.zip"}}, ValueError),
        ],
    )
    def test_runtime_parser_rejects_every_schema_invalid_retention_mapping(
        self,
        tmp_path: Path,
        retention: object,
        error_type: type[Exception],
    ) -> None:
        config = self._config(tmp_path)
        config["retention"] = retention

        assert ExperimentSchemaCatalog().validation_errors(config)
        with pytest.raises(error_type):
            ArtifactRetentionPolicy.from_config(config)

    @pytest.mark.parametrize("timeout", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_lock_timeouts_fail_semantic_validation(
        self,
        tmp_path: Path,
        timeout: float,
    ) -> None:
        config = self._config(tmp_path)
        config["retention"] = {"lock_timeout_seconds": timeout}

        with pytest.raises(ValueError, match="finite positive"):
            ArtifactRetentionPolicy.from_config(config)
        report = ExperimentValidator().validate(config, check_imports=False)
        assert not report.is_valid
        assert any("lock_timeout_seconds" in error for error in report.errors)

    def test_direct_value_objects_enforce_the_same_invariants(self) -> None:
        import lambdaforge.experiments as experiments_api

        assert experiments_api.ArtifactCompressionOptions is ArtifactCompressionOptions
        assert experiments_api.ArtifactRetentionAction is ArtifactRetentionAction
        assert experiments_api.ArtifactRetentionRule is ArtifactRetentionRule
        assert {
            "ArtifactCompressionOptions",
            "ArtifactRetentionAction",
            "ArtifactRetentionRule",
        } <= set(experiments_api.__all__)

        compression = ArtifactCompressionOptions()
        rule = ArtifactRetentionRule(
            action=ArtifactRetentionAction.COMPRESS,
            include=("artifacts/**",),
        )
        policy = ArtifactRetentionPolicy(rules=(rule,))

        assert rule.compression == compression
        assert policy.rules == (rule,)

        with pytest.raises(TypeError):
            ArtifactCompressionOptions(level=True)
        with pytest.raises(ValueError):
            ArtifactCompressionOptions(level=10)
        with pytest.raises(TypeError):
            ArtifactCompressionOptions(only_if_smaller=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ArtifactRetentionRule(  # type: ignore[arg-type]
                action="compress",
                include=("artifacts/**",),
            )
        with pytest.raises(ValueError):
            ArtifactRetentionRule(
                action=ArtifactRetentionAction.PRUNE,
                include=("../outside",),
            )
        with pytest.raises(TypeError):
            ArtifactRetentionRule(  # type: ignore[arg-type]
                action=ArtifactRetentionAction.PRUNE,
                include=["scratch/**"],
            )
        with pytest.raises(ValueError):
            ArtifactRetentionRule(
                action=ArtifactRetentionAction.PRUNE,
                include=("scratch/**",),
                compression=compression,
            )
        with pytest.raises(TypeError):
            ArtifactRetentionPolicy(mode="preview")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ArtifactRetentionPolicy(lock_timeout_seconds=float("nan"))
        with pytest.raises(ValueError):
            ArtifactRetentionPolicy(archive_name="nested.name.zip")
        with pytest.raises(TypeError):
            ArtifactRetentionPolicy(rules=[rule])  # type: ignore[arg-type]

    def test_policy_round_trip_is_schema_valid_and_fingerprint_stable(
        self,
        tmp_path: Path,
    ) -> None:
        source = {
            "mode": "preview",
            "rules": [
                {
                    "action": "compress",
                    "include": ["artifacts/**"],
                }
            ],
            "archive": {
                "name": "research-artifacts.zip",
                "compression_level": 9,
            },
        }
        policy = ArtifactRetentionPolicy.from_mapping(source)
        payload = policy.to_dict()

        assert "level" not in payload["rules"][0]["compression"]
        assert policy.compression_level_for(policy.rules[0]) == 9
        config = self._config(tmp_path)
        config["retention"] = payload
        assert ExperimentSchemaCatalog().validation_errors(config) == ()

        reconstructed = ArtifactRetentionPolicy.from_mapping(payload)
        assert reconstructed == policy
        assert reconstructed.fingerprint == policy.fingerprint

        changed = dict(payload)
        changed["archive"] = {
            "name": "research-artifacts.zip",
            "compression_level": 8,
        }
        assert ArtifactRetentionPolicy.from_mapping(changed).fingerprint != policy.fingerprint

    @pytest.mark.parametrize(
        "pattern",
        ["../outside", "/absolute", r"C:/drive", r"folder\file", "safe/\0bad"],
    )
    def test_semantic_pattern_validation_rejects_escape(
        self,
        tmp_path: Path,
        pattern: str,
    ) -> None:
        config = self._config(
            tmp_path,
            rules=[{"action": "prune", "include": [pattern]}],
        )

        with pytest.raises(ValueError):
            ArtifactRetentionPolicy.from_config(ExperimentConfig(config))

    def test_preview_is_strictly_read_only(self, tmp_path: Path) -> None:
        config, _ = self._materialize(tmp_path)
        before = {
            path.relative_to(config.suite_dir).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in config.suite_dir.rglob("*")
            if path.is_file()
        }

        plan = ArtifactRetentionManager().preview(config)

        after = {
            path.relative_to(config.suite_dir).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in config.suite_dir.rglob("*")
            if path.is_file()
        }
        assert plan.status is ArtifactRetentionStatus.PREVIEW
        assert {operation.action.value for operation in plan.operations} == {
            "compress",
            "prune",
            "prune_checkpoint",
        }
        assert before == after

    def test_apply_archives_prunes_and_preserves_contract_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)

        result = ArtifactRetentionManager().apply(config)

        assert result.status is ArtifactRetentionStatus.APPLIED
        assert result.reclaimed_bytes > 0
        assert not (run_dir / "artifacts" / "large.bin").exists()
        assert not (run_dir / "scratch" / "temporary.tmp").exists()
        assert (run_dir / "artifacts" / "keep.bin").read_bytes() == b"protected"
        assert (run_dir / "final" / "model-card.txt").exists()
        for canonical in (
            "config.yaml",
            "environment.json",
            "hparams.json",
            "train.log",
            "metrics.csv",
            "result.json",
        ):
            assert (run_dir / canonical).exists()
        assert (run_dir / "checkpoints" / "best-001.ckpt").exists()
        assert not (run_dir / "checkpoints" / "last.ckpt").exists()
        assert not (run_dir / "checkpoints" / "epoch-000.ckpt").exists()
        persisted = RunResult.read_json(run_dir / "result.json")
        assert persisted.best_model_path is not None
        assert persisted.last_model_path is None

        assert len(result.archives) == 1
        archive_path = run_dir / str(result.archives[0]["path"])
        with ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            assert archive.read("artifacts/large.bin") == b"A" * 8192
        assert not (config.suite_dir / "aggregate" / ".retention-transaction.json").exists()
        assert not (config.suite_dir / "aggregate" / ".retention-quarantine").exists()
        assert AggregationReceipt.read_json(AggregationReceipt.path_for(config)).is_current(config)

    def test_second_apply_is_idempotent(self, tmp_path: Path) -> None:
        config, _ = self._materialize(tmp_path)
        manager = ArtifactRetentionManager()

        first = manager.apply(config)
        second = manager.apply(config)

        assert first.status is ArtifactRetentionStatus.APPLIED
        assert second.status is ArtifactRetentionStatus.ALREADY_APPLIED
        assert second.plan_id == first.plan_id
        assert len(list((config.suite_dir / "aggregate" / "retention").glob("*.json"))) == 2

    def test_incompressible_source_is_preserved(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path, compressible=False)

        result = ArtifactRetentionManager().apply(config)

        assert result.status is ArtifactRetentionStatus.APPLIED
        assert (run_dir / "artifacts" / "large.bin").exists()
        compression = [
            operation for operation in result.operations if operation["action"] == "compress"
        ]
        assert compression[0]["state"] == "preserved_not_smaller"
        assert not result.archives

    def test_failed_run_never_becomes_retention_eligible(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path, status=RunStatus.FAILED)
        source = run_dir / "scratch" / "temporary.tmp"

        plan = ArtifactRetentionManager().preview(config)
        result = ArtifactRetentionManager().apply(config)

        assert plan.status is ArtifactRetentionStatus.NOT_READY
        assert result.status is ArtifactRetentionStatus.NOT_READY
        assert source.exists()

    def test_stale_receipt_fails_closed(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path)
        (run_dir / "metrics.csv").write_text(
            "epoch,val_loss\n1,99\n",
            encoding="utf-8",
        )

        plan = ArtifactRetentionManager().preview(config)

        assert plan.status is ArtifactRetentionStatus.NOT_READY
        assert "stale" in str(plan.reason).lower()
        assert (run_dir / "scratch" / "temporary.tmp").exists()

    def test_overlapping_rules_abort_before_mutation(self, tmp_path: Path) -> None:
        config = self._config(
            tmp_path,
            rules=[
                {"action": "compress", "include": ["artifacts/*.bin"]},
                {"action": "prune", "include": ["artifacts/large.bin"]},
            ],
            protect=[],
        )
        normalized, run_dir = self._materialize(tmp_path, config=config)

        with pytest.raises(ValueError, match="overlap"):
            ArtifactRetentionManager().preview(normalized)

        assert (run_dir / "artifacts" / "large.bin").exists()

    def test_changed_source_rolls_transaction_back(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path)
        policy = ArtifactRetentionPolicy.from_config(config)
        plan = ArtifactRetentionManager().preview(config)
        changed = run_dir / "scratch" / "temporary.tmp"
        changed.write_bytes(b"changed after preview")

        result = ArtifactRetentionTransaction(config, policy).execute(plan)

        assert result.status is ArtifactRetentionStatus.ROLLED_BACK
        assert changed.read_bytes() == b"changed after preview"
        assert (run_dir / "artifacts" / "large.bin").exists()
        assert not list(run_dir.glob("intermediate-artifacts-*.zip"))

    def test_mid_quarantine_failure_restores_every_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        policy = ArtifactRetentionPolicy.from_config(config)
        plan = ArtifactRetentionManager().preview(config)
        transaction = ArtifactRetentionTransaction(config, policy)
        original = transaction._quarantine
        calls = 0

        def fail_after_first(operation, plan_id) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError("injected quarantine failure")
            original(operation, plan_id)

        monkeypatch.setattr(transaction, "_quarantine", fail_after_first)
        result = transaction.execute(plan)

        assert result.status is ArtifactRetentionStatus.ROLLED_BACK
        assert (run_dir / "artifacts" / "large.bin").exists()
        assert (run_dir / "scratch" / "temporary.tmp").exists()
        assert (run_dir / "checkpoints" / "last.ckpt").exists()
        assert not list(run_dir.glob("intermediate-artifacts-*.zip"))
        assert not transaction.journal_path.exists()

        retry = ArtifactRetentionManager().apply(config)
        assert retry.status is ArtifactRetentionStatus.APPLIED
        assert not (run_dir / "scratch" / "temporary.tmp").exists()

    def test_committing_journal_recovers_forward_after_crash(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        policy = ArtifactRetentionPolicy.from_config(config)
        plan = ArtifactRetentionManager().preview(config)
        transaction = ArtifactRetentionTransaction(config, policy)

        def crash_after_commit_marker(journal) -> None:
            del journal
            raise RuntimeError("simulated process death")

        monkeypatch.setattr(transaction, "_finish_commit", crash_after_commit_marker)
        with pytest.raises(RuntimeError, match="simulated process death"):
            transaction.execute(plan)
        assert transaction.journal_path.exists()
        assert not (run_dir / "scratch" / "temporary.tmp").exists()

        recovered = ArtifactRetentionTransaction(config, policy).recover_pending()

        assert recovered is not None
        assert recovered.status is ArtifactRetentionStatus.APPLIED
        assert not transaction.journal_path.exists()
        assert not (run_dir / "artifacts" / "large.bin").exists()
        assert not (config.suite_dir / "aggregate" / ".retention-quarantine").exists()

    @pytest.mark.parametrize(
        "tamper",
        [
            "omitted_operation",
            "malformed_operation",
            "changed_plan_id",
            "invalid_archives",
            "plan_version",
            "journal_version",
        ],
    )
    def test_tampered_committing_journal_fails_closed_without_mutation(
        self,
        tmp_path: Path,
        tamper: str,
    ) -> None:
        config, _ = self._materialize(tmp_path)
        policy = ArtifactRetentionPolicy.from_config(config)
        plan = ArtifactRetentionManager().preview(config)
        transaction = ArtifactRetentionTransaction(config, policy)
        removable = [
            operation
            for operation in plan.operations
            if operation.action is not ArtifactRetentionAction.COMPRESS
        ]
        quarantined, untouched = removable[:2]
        quarantined_source = transaction._source(quarantined)
        untouched_source = transaction._source(untouched)
        quarantined_bytes = quarantined_source.read_bytes()
        ArtifactRetentionJournal(
            plan=plan,
            phase=ArtifactRetentionPhase.COMMITTING,
        ).write_json(transaction.journal_path)
        transaction._quarantine(quarantined, plan.plan_id)
        quarantine_path = transaction._quarantine_path(quarantined, plan.plan_id)

        payload = json.loads(transaction.journal_path.read_text(encoding="utf-8"))
        if tamper == "omitted_operation":
            payload["plan"]["operations"].pop()
        elif tamper == "malformed_operation":
            payload["plan"]["operations"][0] = "silently filtered before hardening"
        elif tamper == "changed_plan_id":
            payload["plan"]["plan_id"] = "0" * 64
        elif tamper == "invalid_archives":
            payload["archives"] = [{"run_relative": quarantined.run_relative}]
        elif tamper == "plan_version":
            payload["plan"]["retention_plan_version"] = 2
        else:
            payload["retention_journal_version"] = 2
        transaction.journal_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        recovered = ArtifactRetentionTransaction(config, policy).recover_pending()

        assert recovered is not None
        assert recovered.status is ArtifactRetentionStatus.CONFLICT
        assert transaction.journal_path.exists()
        assert quarantine_path.read_bytes() == quarantined_bytes
        assert not quarantined_source.exists()
        assert untouched_source.exists()
        assert not (transaction.retention_dir / f"{plan.plan_id}.json").exists()

    @pytest.mark.parametrize(
        "identity",
        ["base_dir", "policy_fingerprint", "receipt_id"],
    )
    def test_recovery_rejects_recomputed_plan_with_wrong_suite_identity(
        self,
        tmp_path: Path,
        identity: str,
    ) -> None:
        config, _ = self._materialize(tmp_path)
        policy = ArtifactRetentionPolicy.from_config(config)
        original = ArtifactRetentionManager().preview(config)
        values = {
            "base_dir": original.base_dir,
            "policy_fingerprint": original.policy_fingerprint,
            "receipt_id": original.receipt_id,
        }
        if identity == "base_dir":
            values[identity] = str(tmp_path / "different-suite")
        elif identity == "policy_fingerprint":
            values[identity] = "0" * 64
        else:
            values[identity] = "1" * 64
        plan = ArtifactRetentionPlan(
            status=original.status,
            base_dir=values["base_dir"],
            receipt_id=values["receipt_id"],
            policy_fingerprint=values["policy_fingerprint"],
            archive_name=original.archive_name,
            operations=original.operations,
            warnings=original.warnings,
            reason=original.reason,
        )
        transaction = ArtifactRetentionTransaction(config, policy)
        operation = next(
            item for item in plan.operations if item.action is not ArtifactRetentionAction.COMPRESS
        )
        source = transaction._source(operation)
        source_bytes = source.read_bytes()
        ArtifactRetentionJournal(
            plan=plan,
            phase=ArtifactRetentionPhase.COMMITTING,
        ).write_json(transaction.journal_path)
        transaction._quarantine(operation, plan.plan_id)
        quarantine = transaction._quarantine_path(operation, plan.plan_id)

        recovered = ArtifactRetentionTransaction(config, policy).recover_pending()

        assert recovered is not None
        assert recovered.status is ArtifactRetentionStatus.CONFLICT
        assert transaction.journal_path.exists()
        assert quarantine.read_bytes() == source_bytes
        assert not source.exists()

    def test_retention_result_parser_rejects_malformed_durable_entries(
        self,
        tmp_path: Path,
    ) -> None:
        config, _ = self._materialize(tmp_path)
        result = ArtifactRetentionManager().apply(config)
        payloads = []
        for _ in range(4):
            payloads.append(json.loads(json.dumps(result.to_dict())))
        payloads[0]["retention_result_version"] = 2
        payloads[1]["plan_id"] = "not-a-sha256"
        payloads[2]["operations"].append("silently filtered before hardening")
        payloads[3]["archives"].append("silently filtered before hardening")

        for payload in payloads:
            with pytest.raises((TypeError, ValueError)):
                ArtifactRetentionResult.from_mapping(payload)

    def test_concurrent_apply_is_serialized_and_idempotent(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path)
        context = multiprocessing.get_context("spawn")
        ready_queue = context.Queue()
        result_queue = context.Queue()
        start_event = context.Event()
        processes = [
            context.Process(
                target=RetentionApplyJob(
                    config.as_dict(),
                    ready_queue,
                    start_event,
                    result_queue,
                )
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            assert ready_queue.get(timeout=10.0)
            assert ready_queue.get(timeout=10.0)
            start_event.set()
            for process in processes:
                process.join(timeout=20.0)
                assert process.exitcode == 0
            outcomes = [result_queue.get(timeout=5.0) for _ in processes]
        finally:
            start_event.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
            ready_queue.close()
            result_queue.close()

        assert {status for status, _ in outcomes} == {"applied", "already_applied"}
        assert len({plan_id for _, plan_id in outcomes}) == 1
        assert not (run_dir / "scratch" / "temporary.tmp").exists()
        assert not (config.suite_dir / "aggregate" / ".retention-transaction.json").exists()

    def test_checkpoint_ambiguity_keeps_every_checkpoint(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, keep="best")
        normalized, run_dir = self._materialize(tmp_path, config=config)
        result_path = run_dir / "result.json"
        result = RunResult.read_json(result_path)
        result.with_updates(
            best_model_path=None,
            best_epoch_metrics=None,
        ).write_json(result_path)
        (run_dir / "checkpoints" / "best-002.ckpt").write_bytes(b"other")
        ExperimentAggregator().write(normalized, make_plots=False, final=True)

        plan = ArtifactRetentionManager().preview(normalized)

        checkpoint_operations = [
            operation
            for operation in plan.operations
            if operation.action.value == "prune_checkpoint"
        ]
        assert not checkpoint_operations
        assert any("unambiguously" in warning for warning in plan.warnings)

    def test_checkpoint_policy_all_is_recognized_as_complete(self, tmp_path: Path) -> None:
        raw = self._config(tmp_path)
        raw["trainer"] = {"checkpoint_policy": "all"}
        config, run_dir = self._materialize(tmp_path, config=raw)
        result_path = run_dir / "result.json"
        RunResult.read_json(result_path).with_updates(
            best_model_path=None,
            last_model_path=None,
        ).write_json(result_path)

        completed = ExperimentRunner()._completed_result(config.expand()[0], run_dir)

        assert completed is not None
        assert completed.status is RunStatus.OK

    def test_nonzero_worker_exit_never_reuses_an_old_ok_result(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path)
        run_config = config.expand()[0]

        observed = ExperimentExecutor()._collect_result(run_config, 137)

        assert RunResult.read_json(run_dir / "result.json").status is RunStatus.OK
        assert observed.status is RunStatus.FAILED
        assert observed.exit_code == 137

    def test_failed_rerun_archives_previous_result_without_incompatible_resume(
        self,
        tmp_path: Path,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        run_config = config.expand()[0]
        run_config["experiment"]["rerun_completed"] = True
        run_config["model"] = {"target": "missing_package.missing_module.MissingModel"}

        results = ExperimentExecutor()._run_sequential([run_config], None)

        assert results[0].status is RunStatus.FAILED
        assert RunResult.read_json(run_dir / "result.json").status is RunStatus.FAILED
        archived = sorted((run_dir / ".lambdaforge" / "attempts").glob("result-*.json"))
        assert len(archived) == 1
        assert RunResult.read_json(archived[0]).status is RunStatus.OK
        materialized = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert "ckpt_path" not in materialized["experiment"]

    def test_scheduled_preparation_preserves_completed_runs_that_will_skip(
        self,
        tmp_path: Path,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)

        prepared = ExperimentRunner()._prepare_scheduled_attempts(config.expand())

        assert len(prepared) == 1
        assert (run_dir / "result.json").exists()
        assert not (run_dir / ".lambdaforge" / "attempts").exists()

    def test_dead_worker_before_runner_cannot_leave_stale_success_for_aggregation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        raw = config.as_dict()
        raw["experiment"]["rerun_completed"] = True

        def die_before_runner(
            executor: ExperimentExecutor,
            run_configs: list[dict[str, object]],
            execution: object,
            on_run_finished: object = None,
        ) -> list[RunResult]:
            del execution, on_run_finished
            assert len(run_configs) == 1
            run_config = run_configs[0]
            assert str(run_config["experiment"]["ckpt_path"]).endswith("last.ckpt")
            assert not (run_dir / "result.json").exists()
            return [executor._collect_result(run_config, 137)]

        monkeypatch.setattr(ExperimentExecutor, "run", die_before_runner)

        results = ExperimentRunner().run_experiment_config(raw, aggregate_plots=False)
        standalone = ExperimentAggregator().write(raw, make_plots=False)

        assert results[0].status is RunStatus.FAILED
        assert not (run_dir / "result.json").exists()
        archived = sorted((run_dir / ".lambdaforge" / "attempts").glob("result-*.json"))
        assert len(archived) == 1
        assert RunResult.read_json(archived[0]).status is RunStatus.OK
        assert not standalone["base"]["terminal"]
        assert standalone["base"]["pending_seeds"] == [7]

    def test_public_single_run_takes_activity_lease_and_invalidates_receipt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        run_config = config.expand()[0]
        run_config["experiment"]["rerun_completed"] = True
        receipt_path = AggregationReceipt.path_for(config)
        lease = MagicMock()
        lease.__enter__.return_value = lease
        monkeypatch.setattr(
            ArtifactRetentionManager,
            "activity_lock",
            staticmethod(lambda *_args, **_kwargs: lease),
        )
        runner = ExperimentRunner()
        delegated = MagicMock(
            return_value=RunResult(
                name="retention_demo",
                run_dir=run_dir,
                variant="base",
                seed=7,
                status=RunStatus.DRY_RUN,
            )
        )
        monkeypatch.setattr(
            runner,
            "_run_single_experiment_unlocked",
            delegated,
        )

        result = runner.run_single_experiment(run_config)

        assert result.status is RunStatus.DRY_RUN
        lease.__enter__.assert_called_once()
        lease.__exit__.assert_called_once()
        delegated.assert_called_once()
        assert isinstance(delegated.call_args.args[0], ExperimentConfig)
        assert not receipt_path.exists()

    def test_attempt_archive_rejects_linked_metadata_before_creating_children(
        self,
        tmp_path: Path,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        metadata_dir = run_dir / ".lambdaforge"
        outside = tmp_path / "outside-attempt-metadata"
        outside.mkdir()
        try:
            metadata_dir.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are not available in this environment.")

        with pytest.raises(ValueError, match="Unsafe run-attempt archive directory"):
            ExperimentRunner()._archive_previous_result(run_dir)

        assert (run_dir / "result.json").exists()
        assert not (outside / "attempts").exists()

    def test_receipt_allows_an_explicitly_disabled_epoch_metrics_csv(
        self,
        tmp_path: Path,
    ) -> None:
        raw = self._config(tmp_path)
        raw["trainer"] = {
            "checkpoint_policy": "last_and_best",
            "write_epoch_metrics_csv": False,
        }
        config, run_dir = self._materialize(tmp_path, config=raw)
        (run_dir / "metrics.csv").unlink()

        ExperimentAggregator().write(config, make_plots=False, final=True)
        receipt = AggregationReceipt.read_json(AggregationReceipt.path_for(config))

        assert receipt.complete
        assert receipt.is_current(config)
        assert ArtifactRetentionManager().preview(config).is_ready

    def test_receipt_commits_all_framework_aggregate_outputs(self, tmp_path: Path) -> None:
        config, _ = self._materialize(tmp_path)
        receipt = AggregationReceipt.read_json(AggregationReceipt.path_for(config))

        assert {
            "summary.csv",
            "base/aggregate.json",
            "aggregate/baseline_comparisons.csv",
            "aggregate/reliability.json",
        } <= set(receipt.output_fingerprints)
        assert receipt.is_current(config)

        variant_aggregate = config.suite_dir / "base" / "aggregate.json"
        variant_aggregate.write_text("{}\n", encoding="utf-8")

        assert not receipt.is_current(config)
        assert not ArtifactRetentionManager().preview(config).is_ready

    def test_summary_links_a_manifest_only_after_a_committed_transaction(
        self,
        tmp_path: Path,
    ) -> None:
        config, _ = self._materialize(tmp_path)
        summary_path = config.suite_dir / "aggregate" / "summary.json"
        before = json.loads(summary_path.read_text(encoding="utf-8"))

        assert before["retention"] == {
            "mode": "preview",
            "status": "not_applied",
            "latest_manifest": None,
        }

        result = ArtifactRetentionManager().apply(config)
        after = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = AggregationReceipt.read_json(AggregationReceipt.path_for(config))

        assert result.status is ArtifactRetentionStatus.APPLIED
        assert after["retention"]["status"] == "applied"
        assert after["retention"]["plan_id"] == result.plan_id
        assert Path(after["retention"]["latest_manifest"]).is_file()
        assert receipt.is_current(config)

    def test_sequential_keyboard_interrupt_persists_a_resumable_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_config = ExperimentConfig(self._config(tmp_path)).expand()[0]

        def interrupt(*_args: object, **_kwargs: object) -> RunResult:
            raise KeyboardInterrupt

        monkeypatch.setattr(ExperimentRunner, "_run_single_experiment_unlocked", interrupt)

        with pytest.raises(KeyboardInterrupt):
            ExperimentExecutor()._run_sequential([run_config], None)

        run_dir = ExperimentRunner().experiment_run_dir(run_config)
        persisted = RunResult.read_json(run_dir / "result.json")
        assert persisted.status is RunStatus.INTERRUPTED
        assert persisted.error == "KeyboardInterrupt: execution interrupted."

    def test_loader_auto_prefers_best_and_exact_choices_do_not_cross(
        self,
        tmp_path: Path,
    ) -> None:
        config, run_dir = self._materialize(tmp_path)

        assert RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.AUTO).name == "best-001.ckpt"
        assert RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.BEST).name == "best-001.ckpt"
        assert RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.LAST).name == "last.ckpt"

        (run_dir / "checkpoints" / "best-001.ckpt").unlink()
        result_path = run_dir / "result.json"
        RunResult.read_json(result_path).with_updates(best_model_path=None).write_json(result_path)
        with pytest.raises(FileNotFoundError, match="best"):
            RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.BEST)
        assert RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.AUTO).name == "last.ckpt"

    def test_exact_last_never_falls_back_to_a_best_checkpoint(self, tmp_path: Path) -> None:
        _, run_dir = self._materialize(tmp_path)
        (run_dir / "checkpoints" / "last.ckpt").unlink()
        (run_dir / "checkpoints" / "epoch-000.ckpt").unlink()
        result_path = run_dir / "result.json"
        RunResult.read_json(result_path).with_updates(last_model_path=None).write_json(result_path)

        with pytest.raises(FileNotFoundError, match="last"):
            RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.LAST)
        assert RunLoader.resolve_checkpoint(run_dir, CheckpointChoice.AUTO).name == "best-001.ckpt"

    def test_symlink_candidates_are_never_followed(self, tmp_path: Path) -> None:
        config, run_dir = self._materialize(tmp_path)
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        link = run_dir / "artifacts" / "linked.bin"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks are not available in this environment.")

        plan = ArtifactRetentionManager().preview(config)
        result = ArtifactRetentionManager().apply(config)

        assert all(
            operation.relative_path != "artifacts/linked.bin" for operation in plan.operations
        )
        assert result.status is ArtifactRetentionStatus.APPLIED
        assert outside.read_bytes() == b"outside"
        assert link.is_symlink()

    def test_public_experiment_api_exposes_preview_and_apply(self, tmp_path: Path) -> None:
        config, _ = self._materialize(tmp_path)
        experiment = Experiment(config)

        assert experiment.preview_retention().status is ArtifactRetentionStatus.PREVIEW
        assert experiment.apply_retention().status is ArtifactRetentionStatus.APPLIED

    def test_cli_is_preview_first_and_requires_apply_flag(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config, run_dir = self._materialize(tmp_path)
        source = tmp_path / "retention.yaml"
        source.write_text(
            yaml.safe_dump(config.as_dict(), sort_keys=False),
            encoding="utf-8",
        )

        assert CommandLineInterface.main(["retain", str(source), "--json"]) == 0
        preview = json.loads(capsys.readouterr().out)
        assert preview["status"] == "preview"
        assert (run_dir / "scratch" / "temporary.tmp").exists()

        assert CommandLineInterface.main(["retain", str(source), "--apply", "--json"]) == 0
        applied = json.loads(capsys.readouterr().out)
        assert applied["status"] == "applied"
        assert not (run_dir / "scratch" / "temporary.tmp").exists()

    def test_cli_retention_errors_keep_json_output_machine_readable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        raw = self._config(tmp_path)
        raw["retention"] = None
        source = tmp_path / "invalid-retention.yaml"
        source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        assert CommandLineInterface.main(["retain", str(source), "--json"]) == 1
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert payload["status"] == "error"
        assert payload["error_type"]
        assert payload["message"]
        assert captured.err == ""

    def test_automatic_apply_runs_only_after_final_aggregation(self, tmp_path: Path) -> None:
        config = self._config(tmp_path, mode="apply")
        normalized = ExperimentConfig(config)
        # Build through the common helper with preview first, then switch only the
        # in-memory config for the final aggregate to isolate this lifecycle check.
        preview_config = dict(config)
        preview_config["retention"] = dict(config["retention"])  # type: ignore[arg-type]
        preview_config["retention"]["mode"] = "preview"  # type: ignore[index]
        _, run_dir = self._materialize(tmp_path, config=preview_config)
        assert (run_dir / "scratch" / "temporary.tmp").exists()

        ExperimentAggregator().write(normalized, make_plots=False, final=False)
        assert (run_dir / "scratch" / "temporary.tmp").exists()

        ExperimentAggregator().write(normalized, make_plots=False, final=True)
        assert not (run_dir / "scratch" / "temporary.tmp").exists()
        latest = json.loads(
            (normalized.suite_dir / "aggregate" / "retention" / "latest.json").read_text(
                encoding="utf-8"
            )
        )
        assert latest["status"] == "applied"
