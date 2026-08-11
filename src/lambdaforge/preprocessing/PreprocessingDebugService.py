"""Sample-first preprocessing debugger."""

from __future__ import annotations

import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.preprocessing.PreprocessingDebugResult import PreprocessingDebugResult
from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskContext import TaskContext


class PreprocessingDebugService:
    """Run only N records through transforms and expose stage-level evidence."""

    def debug(
        self,
        config_path: str | Path,
        *,
        records: int = 1,
        intermediates: str | Path | None = None,
    ) -> PreprocessingDebugResult:
        """Sample deterministically without invoking or finalizing the production sink."""
        if records < 1 or records > 10_000:
            raise ValueError("Debug records must be between 1 and 10000.")
        config = TaskConfig.from_yaml(config_path)
        task = ObjectFactory.build(config["task"])
        source = getattr(task, "source", None)
        transforms = getattr(task, "transforms", None)
        if not callable(getattr(source, "records", None)) or transforms is None:
            raise TypeError("Debug preprocessing requires a PreprocessingTask-compatible object.")
        assert source is not None
        output_root = Path(intermediates).resolve() if intermediates is not None else None
        with tempfile.TemporaryDirectory(prefix="lambdaforge-debug-") as temporary:
            context = TaskContext(
                name=f"{config.name}-debug",
                run_dir=Path(temporary),
                source_dir=config.source_dir,
                attempt_id=f"debug-{config.fingerprint.removeprefix('sha256:')[:12]}",
                config_fingerprint=config.fingerprint,
                resume=False,
                inputs=tuple(value.to_dict() for value in config.resolved_inputs),
                outputs=config.outputs,
            )
            reports: list[dict[str, Any]] = []
            for index, record in enumerate(source.records(context)):
                if index >= records:
                    break
                if not isinstance(record, PreprocessingRecord):
                    raise TypeError("Preprocessing source yielded a non-PreprocessingRecord value.")
                reports.append(self._record(record, tuple(transforms), context, output_root, index))
        return PreprocessingDebugResult(
            config.fingerprint,
            f"debug:{config.fingerprint}:{records}",
            records,
            tuple(reports),
        )

    def _record(
        self,
        record: PreprocessingRecord,
        transforms: tuple[Any, ...],
        context: TaskContext,
        output_root: Path | None,
        index: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        current = record
        stages: list[dict[str, Any]] = []
        exception: dict[str, str] | None = None
        for stage_index, transform in enumerate(transforms):
            stage_started = time.perf_counter()
            try:
                current = transform.transform(current, context)
                if not isinstance(current, PreprocessingRecord):
                    raise TypeError("Transform did not return PreprocessingRecord.")
                if current.key != record.key:
                    raise ValueError("Transform changed the stable record key.")
                artifact = self._intermediate(output_root, record.key, stage_index, current.value)
                stages.append(
                    {
                        "index": stage_index,
                        "transform": f"{type(transform).__module__}.{type(transform).__name__}",
                        "duration_seconds": time.perf_counter() - stage_started,
                        "output_type": (
                            f"{type(current.value).__module__}.{type(current.value).__name__}"
                        ),
                        "preview": self._preview(current.value),
                        "debug_artifact": str(artifact) if artifact is not None else None,
                    }
                )
            except Exception as error:
                exception = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                }
                break
        return {
            "source_key": record.key,
            "source_type": f"{type(record.value).__module__}.{type(record.value).__name__}",
            "transform_stages": stages,
            "output_type": f"{type(current.value).__module__}.{type(current.value).__name__}",
            "output_artifact": None,
            "duration_seconds": time.perf_counter() - started,
            "exception": exception,
            "sample_index": index,
        }

    @staticmethod
    def _preview(value: Any, limit: int = 500) -> str:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            rendered = repr(value)
        return rendered if len(rendered) <= limit else f"{rendered[:limit]}…"

    @staticmethod
    def _intermediate(root: Path | None, key: str, stage: int, value: Any) -> Path | None:
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        safe_key = "".join(character if character.isalnum() else "-" for character in key)[:80]
        path = root / f"{safe_key}-stage-{stage}.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path
