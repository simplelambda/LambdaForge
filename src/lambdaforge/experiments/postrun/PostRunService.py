"""Execution, provenance and resume semantics for post-run actions."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.postrun.PostRunActionReceipt import PostRunActionReceipt
from lambdaforge.experiments.postrun.PostRunActionSpec import PostRunActionSpec
from lambdaforge.experiments.postrun.PostRunCheckpoint import PostRunCheckpoint
from lambdaforge.experiments.postrun.PostRunContext import PostRunContext
from lambdaforge.experiments.postrun.PostRunResult import PostRunResult
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class PostRunService:
    """Run bounded rank-zero actions after successful training and verify their receipts."""

    ROOT = Path(".lambdaforge/post-run")
    _SCOPES = frozenset({"confirmed_runs", "all_runs"})

    def configuration_fingerprint(self, config: Mapping[str, Any]) -> str | None:
        """Hash only current post-run policy; return ``None`` when none is configured."""
        scope, actions = self._configuration(config)
        if not actions:
            return None
        payload = {
            "post_run_version": 1,
            "scope": scope,
            "actions": [action.to_dict() for action in actions],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def run(
        self,
        config: Mapping[str, Any],
        training_result: RunResult,
        *,
        plugins: PluginRegistry | None = None,
        is_global_zero: bool = True,
        stop_event: Any | None = None,
    ) -> RunResult:
        """Execute missing actions and return the canonical whole-run result."""
        if training_result.status is not RunStatus.OK:
            raise ValueError("Post-run actions require a successful training result.")
        if not is_global_zero:
            return training_result
        _, actions = self._configuration(config)
        fingerprint = self.configuration_fingerprint(config)
        if not actions:
            completed = training_result.with_updates(
                post_run_fingerprint=None,
                post_run_actions=[],
                post_run_warnings=[],
                artifacts=[],
            )
            return self._apply_required_artifacts(config, completed)

        registry = plugins or PluginRegistry.default()
        receipts = [
            self._execute(config, training_result, action, registry, stop_event)
            for action in actions
        ]
        failed_required = [
            receipt.name for receipt in receipts if receipt.required and receipt.status != "ok"
        ]
        failed_optional = [
            receipt.name for receipt in receipts if not receipt.required and receipt.status != "ok"
        ]
        artifacts = [artifact.to_dict() for receipt in receipts for artifact in receipt.artifacts]
        updates: dict[str, Any] = {
            "status": RunStatus.FAILED.value if failed_required else RunStatus.OK.value,
            "post_run_fingerprint": fingerprint,
            "post_run_actions": [receipt.to_dict() for receipt in receipts],
            "post_run_warnings": [
                f"Optional post-run action {name!r} failed; inspect its receipt."
                for name in failed_optional
            ],
            "artifacts": artifacts,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if failed_required:
            updates["error"] = (
                "Required post-run action(s) failed after successful training: "
                + ", ".join(failed_required)
            )
        else:
            updates["error"] = None
        return self._apply_required_artifacts(config, training_result.with_updates(**updates))

    def is_complete(self, config: Mapping[str, Any], result: RunResult) -> bool:
        """Verify current action identities, failure policy and artifact bytes."""
        _, actions = self._configuration(config)
        expected = self.configuration_fingerprint(config)
        observed = result.get("post_run_fingerprint")
        if expected is None:
            return observed is None
        if observed != expected:
            return False
        for action in actions:
            selection = self._selection(action, result)
            identity = self._action_identity(
                action.static_identity(RunFingerprint.digest(config)), selection[3]
            )
            receipt = self._read_receipt(self._receipt_path(result.run_dir, action.name))
            if receipt is None or not self._receipt_valid(
                receipt, action, identity, result.run_dir
            ):
                return False
        return True

    def should_run_for_adaptive(
        self,
        config: Mapping[str, Any],
        *,
        phase: str,
        trial_status: str,
    ) -> bool:
        """Gate costly adaptive analysis to explicit successful terminal scopes."""
        scope, actions = self._configuration(config)
        if not actions or trial_status not in {"completed", "early_stopped"}:
            return False
        return scope == "all_runs" or phase == "confirmation"

    def should_run_for_materialized_adaptive(
        self, config: Mapping[str, Any], result: RunResult
    ) -> bool:
        """Decide inside the trial process while its resource allocation is still active."""
        metadata = ExperimentConfig.get_value(config, "metadata.adaptive")
        if not isinstance(metadata, Mapping):
            return True
        if bool(result.get("trainer_stopped_early", False)):
            trial_status = "early_stopped"
        else:
            target = metadata.get("target_budget")
            maximum = metadata.get("max_budget")
            trial_status = (
                "completed"
                if isinstance(target, int) and isinstance(maximum, int) and target >= maximum
                else "paused"
            )
        return self.should_run_for_adaptive(
            config,
            phase=str(metadata.get("phase", "search")),
            trial_status=trial_status,
        )

    def _execute(
        self,
        config: Mapping[str, Any],
        result: RunResult,
        action: PostRunActionSpec,
        plugins: PluginRegistry,
        stop_event: Any | None,
    ) -> PostRunActionReceipt:
        best, last, selected, checkpoint_sha = self._selection(action, result)
        static_identity = action.static_identity(RunFingerprint.digest(config))
        identity = self._action_identity(static_identity, checkpoint_sha)
        receipt_path = self._receipt_path(result.run_dir, action.name)
        existing = self._read_receipt(receipt_path)
        if existing is not None and self._receipt_valid(existing, action, identity, result.run_dir):
            return existing.with_updates(skipped_existing=True)

        started = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        try:
            if action.checkpoint is not PostRunCheckpoint.NONE and selected is None:
                raise FileNotFoundError(
                    f"No unambiguous {action.checkpoint.value!r} checkpoint exists for "
                    f"post-run action {action.name!r}."
                )
            state_dir = self._state_dir(result.run_dir, action.name, identity)
            state_dir.mkdir(parents=True, exist_ok=True)
            context = PostRunContext(
                run_dir=Path(result.run_dir).resolve(),
                config=FrozenJsonMapping(config),
                result=result,
                seed=result.seed,
                variant=result.variant,
                best_checkpoint=best,
                last_checkpoint=last,
                selected_checkpoint=selected,
                selected_checkpoint_role=action.checkpoint.value,
                selected_checkpoint_sha256=checkpoint_sha,
                model_identity=self._model_identity(config),
                action_name=action.name,
                action_identity=identity,
                state_dir=state_dir,
                _stop_requested=(stop_event.is_set if stop_event is not None else lambda: False),
            )
            instance = ObjectFactory.build(
                {"target": action.target, "params": dict(action.params)}, plugins=plugins
            )
            method = getattr(instance, "run", None)
            if not callable(method):
                raise TypeError("A post-run action must expose run(context).")
            output = PostRunResult.from_value(method(context))
            declarations = self._declarations(action, output, identity)
            artifacts = tuple(
                TaskArtifact.materialize(declaration, result.run_dir)
                for declaration in declarations
            )
            receipt = PostRunActionReceipt(
                name=action.name,
                target=action.target,
                action_identity=identity,
                required=action.required,
                status="ok",
                checkpoint_role=action.checkpoint.value,
                checkpoint_path=str(selected) if selected is not None else None,
                checkpoint_sha256=checkpoint_sha,
                seconds=time.perf_counter() - started_clock,
                outputs=output.outputs,
                metrics=output.metrics,
                artifacts=artifacts,
                metadata=output.metadata,
                started_at_utc=started.isoformat(),
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as error:
            receipt = PostRunActionReceipt(
                name=action.name,
                target=action.target,
                action_identity=identity,
                required=action.required,
                status="failed",
                checkpoint_role=action.checkpoint.value,
                checkpoint_path=str(selected) if selected is not None else None,
                checkpoint_sha256=checkpoint_sha,
                seconds=time.perf_counter() - started_clock,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                started_at_utc=started.isoformat(),
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        receipt.write_json(receipt_path)
        return receipt

    @staticmethod
    def _declarations(
        action: PostRunActionSpec,
        output: PostRunResult,
        identity: str,
    ) -> tuple[ArtifactDeclaration, ...]:
        by_path = {artifact.path: artifact for artifact in action.artifacts}
        by_path.update({artifact.path: artifact for artifact in output.artifacts})
        declarations: list[ArtifactDeclaration] = []
        names: set[str] = set()
        for artifact in by_path.values():
            logical_name = artifact.name or Path(artifact.path).name
            if logical_name in names:
                raise ValueError(
                    f"Post-run action {action.name!r} declares logical artifact name "
                    f"{logical_name!r} more than once."
                )
            names.add(logical_name)
            declarations.append(
                ArtifactDeclaration(
                    path=artifact.path,
                    name=logical_name,
                    kind=artifact.kind,
                    media_type=artifact.media_type,
                    metadata={
                        **dict(artifact.metadata),
                        "producer": action.target,
                        "action_name": action.name,
                        "action_identity": identity,
                    },
                )
            )
        return tuple(declarations)

    @staticmethod
    def _selection(
        action: PostRunActionSpec,
        result: RunResult,
    ) -> tuple[Path | None, Path | None, Path | None, str | None]:
        resolver = CheckpointResolver(result.run_dir)
        best = resolver.best(result)
        last = resolver.last(result)
        if action.checkpoint is PostRunCheckpoint.BEST:
            selected = best
        elif action.checkpoint in {PostRunCheckpoint.LAST, PostRunCheckpoint.CURRENT}:
            selected = last
        else:
            selected = None
        digest = PostRunService._checkpoint_sha256(selected) if selected is not None else None
        return best, last, selected, digest

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _model_identity(config: Mapping[str, Any]) -> FrozenJsonMapping:
        extensions = config.get("extensions", {})
        code_identity = (
            extensions.get("_lambdaforge_code_identity")
            if isinstance(extensions, Mapping)
            else None
        )
        return FrozenJsonMapping(
            {
                "model": config.get("model"),
                "code_identity": code_identity,
            }
        )

    @staticmethod
    def _action_identity(static_identity: str, checkpoint_sha: str | None) -> str:
        encoded = json.dumps(
            {"static_identity": static_identity, "checkpoint_sha256": checkpoint_sha},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def _configuration(cls, config: Mapping[str, Any]) -> tuple[str, tuple[PostRunActionSpec, ...]]:
        raw = config.get("post_run")
        if raw is None:
            return "confirmed_runs", ()
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            scope = "confirmed_runs"
            values = raw
        elif isinstance(raw, Mapping):
            scope = str(raw.get("scope", "confirmed_runs"))
            unexpected = set(raw) - {"scope", "actions"}
            if unexpected:
                raise ValueError(f"Unexpected post_run keys: {sorted(unexpected)}.")
            values = raw.get("actions", ())
        else:
            raise TypeError("post_run must be a list or a mapping with scope/actions.")
        if scope not in cls._SCOPES:
            raise ValueError(f"post_run.scope must be one of {sorted(cls._SCOPES)}.")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise TypeError("post_run.actions must be a list.")
        actions = tuple(
            PostRunActionSpec.from_mapping(value, index=index) for index, value in enumerate(values)
        )
        names = [action.name for action in actions]
        if len(names) != len(set(names)):
            raise ValueError("Every post_run action must have a unique name.")
        return scope, actions

    @classmethod
    def _receipt_path(cls, run_dir: str | Path, name: str) -> Path:
        root = Path(run_dir).resolve()
        path = Path(run_dir) / cls.ROOT / "actions" / name / "result.json"
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
            raise ValueError(f"Unsafe post-run action receipt path: {path}")
        return path

    @classmethod
    def _state_dir(cls, run_dir: str | Path, name: str, identity: str) -> Path:
        path = Path(run_dir) / cls.ROOT / "actions" / name / "state" / identity[-16:]
        root = Path(run_dir).resolve()
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
            raise ValueError(f"Unsafe post-run action state path: {path}")
        return path

    @staticmethod
    def _read_receipt(path: Path) -> PostRunActionReceipt | None:
        if path.is_symlink():
            return None
        try:
            return PostRunActionReceipt.read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _receipt_valid(
        receipt: PostRunActionReceipt,
        action: PostRunActionSpec,
        identity: str,
        run_dir: str | Path,
    ) -> bool:
        if receipt.action_identity != identity or receipt.required != action.required:
            return False
        if receipt.status != "ok":
            return not action.required
        for artifact in receipt.artifacts:
            try:
                current = TaskArtifact.materialize(
                    ArtifactDeclaration(
                        path=artifact.path,
                        name=artifact.name,
                        kind=artifact.kind,
                        media_type=artifact.media_type,
                        metadata=artifact.metadata,
                    ),
                    run_dir,
                )
            except (OSError, TypeError, ValueError):
                return False
            if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
                return False
        return True

    @staticmethod
    def _apply_required_artifacts(config: Mapping[str, Any], result: RunResult) -> RunResult:
        raw = ExperimentConfig.get_value(config, "experiment.required_artifacts", [])
        if not isinstance(raw, list):
            raise TypeError("experiment.required_artifacts must be a list of relative paths.")
        root = Path(result.run_dir).resolve()
        missing: list[str] = []
        for value in raw:
            relative = Path(str(value))
            path = (root / relative).resolve(strict=False)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or not path.is_relative_to(root)
            ):
                raise ValueError("Required artifact paths must remain inside the run directory.")
            if not path.exists() or path.is_symlink():
                missing.append(relative.as_posix())
        if not missing or result.status is RunStatus.FAILED:
            return result
        return result.with_updates(
            status=RunStatus.FAILED.value,
            error="Required artifact(s) missing after successful training/post-run: "
            + ", ".join(missing),
        )
