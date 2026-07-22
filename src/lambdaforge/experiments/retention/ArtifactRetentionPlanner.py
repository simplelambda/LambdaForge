"""Read-only planner for safe artifact-retention operations."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.ArtifactRetentionAction import ArtifactRetentionAction
from lambdaforge.experiments.retention.ArtifactRetentionMode import ArtifactRetentionMode
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)
from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
from lambdaforge.experiments.retention.ArtifactRetentionPolicy import ArtifactRetentionPolicy
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus
from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver
from lambdaforge.experiments.retention.CheckpointRetention import CheckpointRetention
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus


class ArtifactRetentionPlanner:
    """Build deterministic plans without creating locks, reports or directories."""

    _CORE_PROTECTED = frozenset(
        {
            "config.yaml",
            "environment.json",
            "hparams.json",
            "train.log",
            "metrics.csv",
            "result.json",
        }
    )

    def plan(
        self,
        config: ExperimentConfig,
        policy: ArtifactRetentionPolicy,
        *,
        explicit: bool,
    ) -> ArtifactRetentionPlan:
        """Return a fully read-only preview gated by a current receipt."""
        if policy.mode is ArtifactRetentionMode.DISABLED and not explicit:
            return self._empty_plan(
                config,
                policy,
                ArtifactRetentionStatus.DISABLED,
                "Retention is disabled.",
            )
        receipt_path = AggregationReceipt.path_for(config)
        try:
            receipt = AggregationReceipt.read_json(receipt_path)
        except (OSError, TypeError, ValueError):
            return self._empty_plan(
                config,
                policy,
                ArtifactRetentionStatus.NOT_READY,
                "No valid aggregation receipt is available.",
            )
        if not receipt.complete:
            return self._empty_plan(
                config,
                policy,
                ArtifactRetentionStatus.NOT_READY,
                "The latest aggregation is incomplete.",
                receipt_id=receipt.receipt_id,
                warnings=receipt.reasons,
            )
        if not receipt.is_current(config):
            return self._empty_plan(
                config,
                policy,
                ArtifactRetentionStatus.NOT_READY,
                "The aggregation receipt is stale.",
                receipt_id=receipt.receipt_id,
            )

        operations: list[ArtifactRetentionOperation] = []
        warnings: list[str] = []
        for run_relative in receipt.run_dirs:
            run_dir = config.suite_dir.joinpath(*PurePosixPath(run_relative).parts)
            result = self._read_ok_result(run_dir)
            if result is None:
                return self._empty_plan(
                    config,
                    policy,
                    ArtifactRetentionStatus.NOT_READY,
                    f"Run {run_relative!r} is no longer complete.",
                    receipt_id=receipt.receipt_id,
                )
            required = receipt.required_artifacts.get(run_relative, ())
            operations.extend(
                self._generic_operations(
                    config.suite_dir,
                    run_dir,
                    run_relative,
                    required,
                    policy,
                )
            )
            checkpoint_operations, checkpoint_warning = self._checkpoint_operations(
                config.suite_dir,
                run_dir,
                run_relative,
                result,
                policy,
            )
            operations.extend(checkpoint_operations)
            if checkpoint_warning is not None:
                warnings.append(checkpoint_warning)

        operations.sort(key=lambda item: (item.run_relative, item.relative_path, item.action.value))
        keys = [operation.key for operation in operations]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"Retention plan contains overlapping operations: {duplicates}.")
        if not receipt.is_current(config):
            return self._empty_plan(
                config,
                policy,
                ArtifactRetentionStatus.NOT_READY,
                "Aggregation inputs changed while retention was being planned.",
                receipt_id=receipt.receipt_id,
            )
        return ArtifactRetentionPlan(
            status=ArtifactRetentionStatus.PREVIEW,
            base_dir=config.suite_dir,
            receipt_id=receipt.receipt_id,
            policy_fingerprint=policy.fingerprint,
            archive_name=policy.archive_name,
            operations=operations,
            warnings=warnings,
        )

    def _generic_operations(
        self,
        suite_dir: Path,
        run_dir: Path,
        run_relative: str,
        required: tuple[str, ...],
        policy: ArtifactRetentionPolicy,
    ) -> list[ArtifactRetentionOperation]:
        operations: list[ArtifactRetentionOperation] = []
        archive_stem = Path(policy.archive_name).stem
        for path in self._regular_files(run_dir):
            relative = ArtifactPathGuard.relative_regular_file(run_dir, path)
            if relative is None or self._is_protected(relative, required, policy, archive_stem):
                continue
            metadata = path.stat()
            matches = [rule for rule in policy.rules if rule.matches(relative, metadata.st_size)]
            if len(matches) > 1:
                raise ValueError(
                    f"Retention rules overlap for {run_relative}/{relative}; "
                    "make include/exclude patterns disjoint."
                )
            if not matches:
                continue
            rule = matches[0]
            fingerprint = self._fingerprint(path)
            if fingerprint is None:
                raise ValueError(f"Artifact changed while planning: {run_relative}/{relative}.")
            operations.append(
                ArtifactRetentionOperation(
                    run_relative=run_relative,
                    relative_path=relative,
                    action=rule.action,
                    size_bytes=int(fingerprint["size_bytes"]),
                    sha256=str(fingerprint["sha256"]),
                    mtime_ns=int(fingerprint["mtime_ns"]),
                    compression_level=(
                        policy.compression_level_for(rule)
                        if rule.action is ArtifactRetentionAction.COMPRESS
                        else None
                    ),
                    only_if_smaller=(
                        bool(rule.compression.only_if_smaller)
                        if rule.compression is not None
                        else False
                    ),
                )
            )
        return operations

    def _checkpoint_operations(
        self,
        suite_dir: Path,
        run_dir: Path,
        run_relative: str,
        result: RunResult,
        policy: ArtifactRetentionPolicy,
    ) -> tuple[list[ArtifactRetentionOperation], str | None]:
        del suite_dir
        if (
            not policy.prune_unselected_checkpoints
            or policy.checkpoint_keep is CheckpointRetention.ALL
        ):
            return [], None
        resolver = CheckpointResolver(run_dir)
        candidates = resolver.candidates()
        if not candidates:
            return [], None
        retained = resolver.retained(policy.checkpoint_keep, result)
        if retained is None or not retained:
            return (
                [],
                f"Checkpoint pruning skipped for {run_relative}: requested roles "
                "could not be resolved unambiguously.",
            )
        retained_set = set(retained)
        operations: list[ArtifactRetentionOperation] = []
        for path in candidates:
            if path in retained_set:
                continue
            relative = path.relative_to(run_dir).as_posix()
            fingerprint = self._fingerprint(path)
            if fingerprint is None:
                return (
                    [],
                    f"Checkpoint pruning skipped for {run_relative}: a candidate changed.",
                )
            operations.append(
                ArtifactRetentionOperation(
                    run_relative=run_relative,
                    relative_path=relative,
                    action=ArtifactRetentionAction.PRUNE_CHECKPOINT,
                    size_bytes=int(fingerprint["size_bytes"]),
                    sha256=str(fingerprint["sha256"]),
                    mtime_ns=int(fingerprint["mtime_ns"]),
                )
            )
        return operations, None

    def _is_protected(
        self,
        relative: str,
        required: tuple[str, ...],
        policy: ArtifactRetentionPolicy,
        archive_stem: str,
    ) -> bool:
        if relative in self._CORE_PROTECTED:
            return True
        first = PurePosixPath(relative).parts[0]
        if first in {"checkpoints", ".lambdaforge"}:
            return True
        if relative == policy.archive_name:
            return True
        if relative.startswith(f"{archive_stem}-l") and relative.endswith(".zip"):
            return True
        if ArtifactPathGuard.matches(relative, policy.protect):
            return True
        return any(
            relative == protected or relative.startswith(f"{protected.rstrip('/')}/")
            for protected in required
        )

    def _regular_files(self, root: Path) -> tuple[Path, ...]:
        output: list[Path] = []
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError:
                continue
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                attributes = getattr(metadata, "st_file_attributes", 0)
                reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                if entry.is_symlink() or attributes & reparse:
                    continue
                path = Path(entry.path)
                if stat.S_ISDIR(metadata.st_mode):
                    stack.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    output.append(path)
        return tuple(sorted(output, key=lambda item: item.relative_to(root).as_posix()))

    @staticmethod
    def _read_ok_result(run_dir: Path) -> RunResult | None:
        try:
            result = RunResult.read_json(run_dir / "result.json")
        except (OSError, TypeError, ValueError):
            return None
        return result if result.status is RunStatus.OK else None

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, int | str] | None:
        try:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            after = path.stat()
        except OSError:
            return None
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            return None
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }

    @staticmethod
    def _empty_plan(
        config: ExperimentConfig,
        policy: ArtifactRetentionPolicy,
        status: ArtifactRetentionStatus,
        reason: str,
        *,
        receipt_id: str | None = None,
        warnings: tuple[str, ...] = (),
    ) -> ArtifactRetentionPlan:
        return ArtifactRetentionPlan(
            status=status,
            base_dir=config.suite_dir,
            receipt_id=receipt_id,
            policy_fingerprint=policy.fingerprint,
            archive_name=policy.archive_name,
            warnings=warnings,
            reason=reason,
        )
