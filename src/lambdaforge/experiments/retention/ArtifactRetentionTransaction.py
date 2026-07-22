"""Crash-recoverable transaction for an approved retention plan."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.experiments.AggregateResult import AggregateResult
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
from lambdaforge.experiments.retention.ArtifactArchive import ArtifactArchive
from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.ArtifactRetentionAction import ArtifactRetentionAction
from lambdaforge.experiments.retention.ArtifactRetentionJournal import ArtifactRetentionJournal
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)
from lambdaforge.experiments.retention.ArtifactRetentionPhase import ArtifactRetentionPhase
from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
from lambdaforge.experiments.retention.ArtifactRetentionPolicy import ArtifactRetentionPolicy
from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus
from lambdaforge.experiments.RunResult import RunResult


class ArtifactRetentionTransaction:
    """Archive, quarantine, commit and recover without guessing on conflicts."""

    def __init__(self, config: ExperimentConfig, policy: ArtifactRetentionPolicy) -> None:
        self.config = config
        self.policy = policy
        self.base_dir = config.suite_dir
        self.aggregate_dir = self.base_dir / "aggregate"
        self.journal_path = self.aggregate_dir / ".retention-transaction.json"
        self.retention_dir = self.aggregate_dir / "retention"

    def recover_pending(self) -> ArtifactRetentionResult | None:
        """Roll back reversible work or finish a transaction already committing."""
        if not self.journal_path.exists():
            return None
        try:
            journal = ArtifactRetentionJournal.read_json(self.journal_path)
        except (OSError, TypeError, ValueError) as error:
            return ArtifactRetentionResult(
                plan_id="unknown",
                status=ArtifactRetentionStatus.CONFLICT,
                receipt_id=None,
                policy_fingerprint=self.policy.fingerprint,
                errors=(f"Unreadable retention journal: {type(error).__name__}: {error}",),
            )
        identity_errors = self._identity_errors(
            journal,
            require_current_receipt=journal.phase is not ArtifactRetentionPhase.COMMITTING,
        )
        if identity_errors:
            return self._result(
                journal,
                status=ArtifactRetentionStatus.CONFLICT,
                errors=identity_errors,
            )
        if journal.phase is ArtifactRetentionPhase.COMMITTING:
            return self._finish_commit(journal)
        errors = self._rollback(journal)
        status = (
            ArtifactRetentionStatus.ROLLED_BACK if not errors else ArtifactRetentionStatus.CONFLICT
        )
        result = self._result(journal, status=status, errors=errors)
        if not errors:
            self._publish_result(result)
        return result

    def execute(self, plan: ArtifactRetentionPlan) -> ArtifactRetentionResult:
        """Apply a ready plan through a durable two-way transaction."""
        if not plan.is_ready:
            raise ValueError(f"Cannot apply retention plan with status {plan.status.value!r}.")
        try:
            plan = ArtifactRetentionPlan.from_mapping(plan.to_dict())
        except (TypeError, ValueError) as error:
            return ArtifactRetentionResult(
                plan_id="unknown",
                status=ArtifactRetentionStatus.CONFLICT,
                receipt_id=None,
                policy_fingerprint=self.policy.fingerprint,
                errors=(f"Invalid retention plan: {type(error).__name__}: {error}",),
            )
        journal = ArtifactRetentionJournal(
            plan=plan,
            phase=ArtifactRetentionPhase.PREPARED,
        )
        identity_errors = self._identity_errors(
            journal,
            require_current_receipt=True,
        )
        if identity_errors:
            return self._result(
                journal,
                status=ArtifactRetentionStatus.CONFLICT,
                errors=identity_errors,
            )
        journal.write_json(self.journal_path)
        archives: list[Mapping[str, Any]] = []
        try:
            for operations in self._compression_groups(plan).values():
                first = operations[0]
                run_dir = self._run_dir(first.run_relative)
                archive = ArtifactArchive(
                    run_dir,
                    configured_name=plan.archive_name,
                    plan_id=plan.plan_id,
                    compression_level=int(first.compression_level or 0),
                )
                metadata, _ = archive.write(operations)
                if metadata is not None:
                    archives.append(metadata)
                journal = journal.with_phase(
                    ArtifactRetentionPhase.PREPARED,
                    archives=archives,
                )
                journal.write_json(self.journal_path)

            journal = journal.with_phase(
                ArtifactRetentionPhase.ARCHIVED,
                archives=archives,
            )
            journal.write_json(self.journal_path)
            active = self._active_operations(journal)
            for operation in active:
                self._verify_operation(operation)
            for operation in active:
                self._quarantine(operation, plan.plan_id)
            journal = journal.with_phase(ArtifactRetentionPhase.QUARANTINED)
            journal.write_json(self.journal_path)
        except BaseException as error:
            rollback_errors = self._rollback(journal)
            errors = (f"{type(error).__name__}: {error}", *rollback_errors)
            status = (
                ArtifactRetentionStatus.ROLLED_BACK
                if not rollback_errors
                else ArtifactRetentionStatus.CONFLICT
            )
            result = self._result(journal, status=status, errors=errors)
            if not rollback_errors:
                self._publish_result(result)
            return result

        journal = journal.with_phase(ArtifactRetentionPhase.COMMITTING)
        journal.write_json(self.journal_path)
        return self._finish_commit(journal)

    def _finish_commit(self, journal: ArtifactRetentionJournal) -> ArtifactRetentionResult:
        immutable_path = self.retention_dir / f"{journal.plan.plan_id}.json"
        expected = self._result(journal, status=ArtifactRetentionStatus.APPLIED)
        if immutable_path.exists():
            try:
                existing = ArtifactRetentionResult.read_json(str(immutable_path))
            except (OSError, TypeError, ValueError):
                existing = None
            if existing is not None and existing.status is ArtifactRetentionStatus.APPLIED:
                if existing.to_dict() != expected.to_dict():
                    return self._result(
                        journal,
                        status=ArtifactRetentionStatus.CONFLICT,
                        errors=("Published retention result does not match the pending journal.",),
                    )
                self._remove_quarantine(journal.plan.plan_id)
                self._publish_result(existing)
                self._update_summary(existing)
                self._refresh_receipt()
                self.journal_path.unlink(missing_ok=True)
                return existing

        active = self._active_operations(journal)
        conflicts: list[str] = []
        for operation in active:
            source = self._source(operation)
            quarantine = self._quarantine_path(operation, journal.plan.plan_id)
            if source.exists():
                conflicts.append(f"Source unexpectedly exists during commit: {operation.key}.")
            elif not quarantine.exists():
                # A crash can occur after the quarantine was purged but before the
                # manifest was published. The committing journal is the durable proof.
                continue
        if conflicts:
            return self._result(
                journal,
                status=ArtifactRetentionStatus.CONFLICT,
                errors=tuple(conflicts),
            )

        try:
            self._update_checkpoint_results(active)
            self._remove_quarantine(journal.plan.plan_id)
            result = self._result(journal, status=ArtifactRetentionStatus.APPLIED)
            self._publish_result(result)
            self._update_summary(result)
            self._refresh_receipt()
            self.journal_path.unlink(missing_ok=True)
            return result
        except BaseException as error:
            return self._result(
                journal,
                status=ArtifactRetentionStatus.PARTIAL,
                errors=(f"{type(error).__name__}: {error}",),
            )

    def _identity_errors(
        self,
        journal: ArtifactRetentionJournal,
        *,
        require_current_receipt: bool,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            plan_base = Path(journal.plan.base_dir).resolve()
            current_base = self.base_dir.resolve()
        except OSError as error:
            errors.append(f"Could not resolve retention base_dir identity: {error}.")
        else:
            if plan_base != current_base:
                errors.append("Retention plan base_dir does not identify the configured suite.")
        if journal.plan.policy_fingerprint != self.policy.fingerprint:
            errors.append("Retention plan policy fingerprint does not match the current policy.")
        if journal.plan.archive_name != self.policy.archive_name:
            errors.append("Retention plan archive name does not match the current policy.")

        try:
            receipt = AggregationReceipt.read_json(AggregationReceipt.path_for(self.config))
        except (OSError, TypeError, ValueError) as error:
            errors.append(
                f"Could not validate retention receipt identity: {type(error).__name__}: {error}."
            )
            return tuple(errors)
        try:
            receipt_base = Path(receipt.base_dir).resolve()
            current_base = self.base_dir.resolve()
        except OSError as error:
            errors.append(f"Could not resolve aggregation receipt base_dir: {error}.")
        else:
            if receipt_base != current_base:
                errors.append(
                    "Aggregation receipt base_dir does not identify the configured suite."
                )

        receipt_matches = (
            journal.plan.receipt_id is not None and receipt.receipt_id == journal.plan.receipt_id
        )
        if not receipt_matches:
            post_commit_receipt = (
                journal.phase is ArtifactRetentionPhase.COMMITTING
                and self._has_exact_applied_manifest(journal)
                and receipt.is_current(self.config)
            )
            if not post_commit_receipt:
                errors.append(
                    "Retention plan receipt_id does not match the suite aggregation receipt."
                )
        elif require_current_receipt and not receipt.is_current(self.config):
            errors.append("Retention plan aggregation receipt is stale.")

        run_dirs = set(receipt.run_dirs)
        for operation in journal.plan.operations:
            if operation.run_relative not in run_dirs:
                errors.append(
                    f"Retention operation run {operation.run_relative!r} is absent "
                    "from the aggregation receipt."
                )
                continue
            required = receipt.required_artifacts.get(operation.run_relative, ())
            if any(
                operation.relative_path == path
                or operation.relative_path.startswith(f"{path.rstrip('/')}/")
                for path in required
            ):
                errors.append(f"Retention operation targets required artifact {operation.key!r}.")
        return tuple(errors)

    def _has_exact_applied_manifest(self, journal: ArtifactRetentionJournal) -> bool:
        path = self.retention_dir / f"{journal.plan.plan_id}.json"
        try:
            existing = ArtifactRetentionResult.read_json(str(path))
        except (OSError, TypeError, ValueError):
            return False
        expected = self._result(journal, status=ArtifactRetentionStatus.APPLIED)
        return (
            existing.status is ArtifactRetentionStatus.APPLIED
            and existing.to_dict() == expected.to_dict()
        )

    def _rollback(self, journal: ArtifactRetentionJournal) -> tuple[str, ...]:
        errors: list[str] = []
        for operation in reversed(journal.plan.operations):
            source = self._source(operation)
            quarantine = self._quarantine_path(operation, journal.plan.plan_id)
            if not quarantine.exists():
                continue
            if source.exists():
                errors.append(f"Rollback preserved both conflicting copies for {operation.key}.")
                continue
            try:
                source_parent = PurePosixPath(operation.relative_path).parent
                if source_parent != PurePosixPath("."):
                    ArtifactPathGuard.ensure_directory(
                        self._run_dir(operation.run_relative),
                        source_parent.as_posix(),
                    )
                quarantine.replace(source)
            except OSError as error:
                errors.append(f"Rollback failed for {operation.key}: {error}.")
        if errors:
            return tuple(errors)

        for archive_path in self._all_archive_paths(journal):
            try:
                archive_path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"Could not remove rollback archive {archive_path}: {error}.")
        if errors:
            return tuple(errors)
        try:
            self._remove_quarantine(journal.plan.plan_id)
            self._cleanup_temporaries(journal.plan)
            self.journal_path.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"Could not clean rolled-back transaction: {error}.")
        return tuple(errors)

    def _compression_groups(
        self,
        plan: ArtifactRetentionPlan,
    ) -> dict[tuple[str, int], tuple[ArtifactRetentionOperation, ...]]:
        groups: dict[tuple[str, int], list[ArtifactRetentionOperation]] = {}
        for operation in plan.operations:
            if operation.action is not ArtifactRetentionAction.COMPRESS:
                continue
            level = int(operation.compression_level or 0)
            groups.setdefault((operation.run_relative, level), []).append(operation)
        return {
            key: tuple(sorted(operations, key=lambda item: item.relative_path))
            for key, operations in sorted(groups.items())
        }

    def _active_operations(
        self,
        journal: ArtifactRetentionJournal,
    ) -> tuple[ArtifactRetentionOperation, ...]:
        archived = {
            f"{archive['run_relative']}/{member}"
            for archive in journal.archives
            for member in archive.get("members", ())
        }
        return tuple(
            operation
            for operation in journal.plan.operations
            if operation.action is not ArtifactRetentionAction.COMPRESS or operation.key in archived
        )

    def _verify_operation(self, operation: ArtifactRetentionOperation) -> None:
        source = self._source(operation)
        relative = ArtifactPathGuard.relative_regular_file(
            self._run_dir(operation.run_relative),
            source,
        )
        if relative != operation.relative_path:
            raise ValueError(f"Unsafe or replaced retention source: {operation.key}.")
        metadata = source.stat()
        if metadata.st_size != operation.size_bytes or metadata.st_mtime_ns != operation.mtime_ns:
            raise ValueError(f"Retention source changed after preview: {operation.key}.")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != operation.sha256:
            raise ValueError(f"Retention source hash changed after preview: {operation.key}.")

    def _quarantine(self, operation: ArtifactRetentionOperation, plan_id: str) -> None:
        source = self._source(operation)
        destination = self._quarantine_path(operation, plan_id)
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Retention quarantine collision: {operation.key}.")
        parent_relative = destination.parent.relative_to(self.aggregate_dir).as_posix()
        ArtifactPathGuard.ensure_directory(self.aggregate_dir, parent_relative)
        source.replace(destination)

    def _update_checkpoint_results(
        self,
        operations: tuple[ArtifactRetentionOperation, ...],
    ) -> None:
        by_run: dict[str, set[str]] = {}
        for operation in operations:
            if operation.action is ArtifactRetentionAction.PRUNE_CHECKPOINT:
                by_run.setdefault(operation.run_relative, set()).add(
                    Path(operation.relative_path).name
                )
        for run_relative, removed_names in by_run.items():
            result_path = self._run_dir(run_relative) / "result.json"
            result = RunResult.read_json(result_path)
            updates: dict[str, Any] = {}
            if result.best_model_path and Path(result.best_model_path).name in removed_names:
                updates["best_model_path"] = None
            if result.last_model_path and Path(result.last_model_path).name in removed_names:
                updates["last_model_path"] = None
            if updates:
                result.with_updates(**updates).write_json(result_path)

    def _result(
        self,
        journal: ArtifactRetentionJournal,
        *,
        status: ArtifactRetentionStatus,
        errors: tuple[str, ...] = (),
    ) -> ArtifactRetentionResult:
        active_keys = {operation.key for operation in self._active_operations(journal)}
        operation_rows: list[dict[str, Any]] = []
        reclaimed = 0
        for operation in journal.plan.operations:
            if operation.key not in active_keys:
                state = "preserved_not_smaller"
            elif operation.action is ArtifactRetentionAction.COMPRESS:
                state = "archived_and_removed"
                reclaimed += operation.size_bytes
            else:
                state = "pruned"
                reclaimed += operation.size_bytes
            row = operation.to_dict()
            row["state"] = state
            operation_rows.append(row)
        if status is not ArtifactRetentionStatus.APPLIED:
            reclaimed = 0
        return ArtifactRetentionResult(
            plan_id=journal.plan.plan_id,
            status=status,
            receipt_id=journal.plan.receipt_id,
            policy_fingerprint=journal.plan.policy_fingerprint,
            operations=operation_rows,
            archives=journal.archives,
            selected_bytes=sum(operation.size_bytes for operation in journal.plan.operations),
            reclaimed_bytes=reclaimed,
            warnings=journal.plan.warnings,
            errors=errors,
        )

    def _publish_result(self, result: ArtifactRetentionResult) -> None:
        ArtifactPathGuard.ensure_directory(self.aggregate_dir, "retention")
        suffix = (
            "" if result.status is ArtifactRetentionStatus.APPLIED else f"-{result.status.value}"
        )
        immutable = self.retention_dir / f"{result.plan_id}{suffix}.json"
        if immutable.exists():
            existing = ArtifactRetentionResult.read_json(str(immutable))
            if existing.to_dict() != result.to_dict():
                raise FileExistsError(f"Immutable retention result conflict: {immutable}")
        else:
            result.write_json(immutable)
        result.write_json(self.retention_dir / "latest.json")

    def _refresh_receipt(self) -> None:
        receipt = AggregationReceipt.build(self.config)
        receipt.write_json(AggregationReceipt.path_for(self.config))

    def _update_summary(self, result: ArtifactRetentionResult) -> None:
        summary_path = self.aggregate_dir / "summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("Aggregate summary JSON must contain an object.")
        summary = dict(payload)
        summary["retention"] = {
            "mode": self.policy.mode.value,
            "status": result.status.value,
            "plan_id": result.plan_id,
            "latest_manifest": str(self.retention_dir / "latest.json"),
        }
        AggregateResult.from_mapping(summary).write_summary_json(summary_path)

    def _all_archive_paths(
        self,
        journal: ArtifactRetentionJournal,
    ) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for archive in journal.archives:
            run_relative = str(archive.get("run_relative", ""))
            name = str(archive.get("path", ""))
            pure = PurePosixPath(name)
            if (
                name
                and not pure.is_absolute()
                and ".." not in pure.parts
                and pure.parts[:2] == (".lambdaforge", "retention")
            ):
                paths.add(self._run_dir(run_relative).joinpath(*pure.parts))
        for (run_relative, level), _ in self._compression_groups(journal.plan).items():
            paths.add(
                ArtifactArchive(
                    self._run_dir(run_relative),
                    configured_name=journal.plan.archive_name,
                    plan_id=journal.plan.plan_id,
                    compression_level=level,
                ).path
            )
        return tuple(sorted(paths, key=str))

    def _cleanup_temporaries(self, plan: ArtifactRetentionPlan) -> None:
        marker = plan.plan_id[:12]
        for run_relative in {operation.run_relative for operation in plan.operations}:
            archive_dir = self._run_dir(run_relative) / ".lambdaforge" / "retention"
            for path in archive_dir.glob(f".*{marker}*.zip.*"):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)

    def _remove_quarantine(self, plan_id: str) -> None:
        path = self._quarantine_root(plan_id)
        try:
            path.lstat()
        except FileNotFoundError:
            return
        parent = path.parent.resolve()
        resolved = path.resolve()
        if (
            path.name != plan_id
            or path.parent.name != ".retention-quarantine"
            or not resolved.is_relative_to(parent)
            or not parent.is_relative_to(self.aggregate_dir.resolve())
        ):
            raise ValueError(f"Refusing to remove unsafe quarantine path: {path}")
        ArtifactPathGuard.validate_regular_tree(path)
        shutil.rmtree(path)
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def _run_dir(self, run_relative: str) -> Path:
        pure = PurePosixPath(run_relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe run-relative path: {run_relative!r}.")
        path = self.base_dir.joinpath(*pure.parts)
        if not path.resolve().is_relative_to(self.base_dir.resolve()):
            raise ValueError(f"Run path escapes suite directory: {run_relative!r}.")
        return path

    def _source(self, operation: ArtifactRetentionOperation) -> Path:
        return ArtifactPathGuard.safe_destination(
            self._run_dir(operation.run_relative),
            operation.relative_path,
        )

    def _quarantine_root(self, plan_id: str) -> Path:
        if len(plan_id) != 64 or any(character not in "0123456789abcdef" for character in plan_id):
            raise ValueError("Retention plan_id is not a lowercase SHA-256.")
        return self.aggregate_dir / ".retention-quarantine" / plan_id

    def _quarantine_path(
        self,
        operation: ArtifactRetentionOperation,
        plan_id: str,
    ) -> Path:
        root = self._quarantine_root(plan_id)
        relative = PurePosixPath(operation.run_relative) / operation.relative_path
        return root.joinpath(*relative.parts)
