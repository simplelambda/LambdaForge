"""Public coordinator for retention preview, locking and application."""

from __future__ import annotations

from collections.abc import Mapping

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
from lambdaforge.experiments.retention.ArtifactRetentionMode import ArtifactRetentionMode
from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
from lambdaforge.experiments.retention.ArtifactRetentionPlanner import ArtifactRetentionPlanner
from lambdaforge.experiments.retention.ArtifactRetentionPolicy import ArtifactRetentionPolicy
from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus
from lambdaforge.experiments.retention.ArtifactRetentionTransaction import (
    ArtifactRetentionTransaction,
)
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class ArtifactRetentionManager:
    """Expose preview/apply APIs and serialize lifecycle mutations across processes."""

    def __init__(self, planner: ArtifactRetentionPlanner | None = None) -> None:
        self.planner = planner or ArtifactRetentionPlanner()

    def preview(
        self,
        config: ExperimentConfig | Mapping[str, object],
    ) -> ArtifactRetentionPlan:
        """Create a read-only plan even when automatic retention is disabled."""
        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        policy = ArtifactRetentionPolicy.from_config(normalized)
        return self.planner.plan(normalized, policy, explicit=True)

    def apply(
        self,
        config: ExperimentConfig | Mapping[str, object],
        *,
        explicit: bool = True,
    ) -> ArtifactRetentionResult:
        """Apply one eligible plan, or return a typed no-op/conflict result."""
        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        policy = ArtifactRetentionPolicy.from_config(normalized)
        if not explicit and policy.mode is not ArtifactRetentionMode.APPLY:
            plan = self.planner.plan(normalized, policy, explicit=False)
            return self._plan_result(plan)

        with self.activity_lock(normalized, policy, shared=False):
            with self.aggregation_lock(normalized, policy):
                with self.retention_lock(normalized, policy):
                    transaction = ArtifactRetentionTransaction(normalized, policy)
                    recovered = transaction.recover_pending()
                    if recovered is not None:
                        if recovered.status in {
                            ArtifactRetentionStatus.CONFLICT,
                            ArtifactRetentionStatus.PARTIAL,
                            ArtifactRetentionStatus.APPLIED,
                        }:
                            return recovered

                    plan = self.planner.plan(normalized, policy, explicit=True)
                    if not plan.is_ready:
                        return self._plan_result(plan)
                    latest = self.latest_result(normalized)
                    if latest is not None:
                        if (
                            latest.plan_id == plan.plan_id
                            and latest.status is ArtifactRetentionStatus.APPLIED
                        ):
                            return self._already_applied(latest)
                        if (
                            not plan.operations
                            and latest.policy_fingerprint == plan.policy_fingerprint
                            and latest.status is ArtifactRetentionStatus.APPLIED
                        ):
                            return self._already_applied(latest)
                    return transaction.execute(plan)

    @staticmethod
    def activity_lock(
        config: ExperimentConfig,
        policy: ArtifactRetentionPolicy | None = None,
        *,
        shared: bool,
    ) -> CrossProcessFileLock:
        """Return the suite activity lock used by training and retention."""
        policy = policy or ArtifactRetentionPolicy.from_config(config)
        return CrossProcessFileLock(
            config.suite_dir / ".lambdaforge" / "activity.lock",
            shared=shared,
            timeout_seconds=policy.lock_timeout_seconds,
            poll_interval_seconds=0.05,
        )

    @staticmethod
    def aggregation_lock(
        config: ExperimentConfig,
        policy: ArtifactRetentionPolicy | None = None,
    ) -> CrossProcessFileLock:
        """Return the exclusive lock that serializes aggregate publication."""
        policy = policy or ArtifactRetentionPolicy.from_config(config)
        return CrossProcessFileLock(
            config.suite_dir / ".lambdaforge" / "aggregation.lock",
            shared=False,
            timeout_seconds=policy.lock_timeout_seconds,
            poll_interval_seconds=0.05,
        )

    @staticmethod
    def retention_lock(
        config: ExperimentConfig,
        policy: ArtifactRetentionPolicy | None = None,
    ) -> CrossProcessFileLock:
        """Return the exclusive lock that serializes retention transactions."""
        policy = policy or ArtifactRetentionPolicy.from_config(config)
        return CrossProcessFileLock(
            config.suite_dir / ".lambdaforge" / "retention.lock",
            shared=False,
            timeout_seconds=policy.lock_timeout_seconds,
            poll_interval_seconds=0.05,
        )

    @staticmethod
    def invalidate_receipt(config: ExperimentConfig) -> None:
        """Invalidate an old aggregation commit before launching new training."""
        AggregationReceipt.path_for(config).unlink(missing_ok=True)

    @staticmethod
    def latest_result(config: ExperimentConfig) -> ArtifactRetentionResult | None:
        """Load the latest published transaction result when it is readable."""
        path = config.suite_dir / "aggregate" / "retention" / "latest.json"
        try:
            return ArtifactRetentionResult.read_json(str(path))
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _plan_result(plan: ArtifactRetentionPlan) -> ArtifactRetentionResult:
        return ArtifactRetentionResult(
            plan_id=plan.plan_id,
            status=plan.status,
            receipt_id=plan.receipt_id,
            policy_fingerprint=plan.policy_fingerprint,
            selected_bytes=sum(operation.size_bytes for operation in plan.operations),
            warnings=plan.warnings,
            errors=(plan.reason,) if plan.reason else (),
        )

    @staticmethod
    def _already_applied(result: ArtifactRetentionResult) -> ArtifactRetentionResult:
        return ArtifactRetentionResult(
            plan_id=result.plan_id,
            status=ArtifactRetentionStatus.ALREADY_APPLIED,
            receipt_id=result.receipt_id,
            policy_fingerprint=result.policy_fingerprint,
            operations=result.operations,
            archives=result.archives,
            selected_bytes=result.selected_bytes,
            reclaimed_bytes=result.reclaimed_bytes,
            warnings=result.warnings,
            errors=result.errors,
        )
