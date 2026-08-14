"""Generic resumable record preprocessing task."""

from __future__ import annotations

import hashlib
import multiprocessing
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
from lambdaforge.preprocessing.PreprocessingErrorPolicy import PreprocessingErrorPolicy
from lambdaforge.preprocessing.PreprocessingManifest import PreprocessingManifest
from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.preprocessing.PreprocessingWorker import PreprocessingWorker
from lambdaforge.preprocessing.PreprocessingWorkload import PreprocessingWorkload
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
from lambdaforge.tasks.ArtifactType import ArtifactType
from lambdaforge.tasks.Task import Task
from lambdaforge.tasks.TaskArtifact import TaskArtifact
from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskOutput import TaskOutput


class PreprocessingTask(Task):
    """Run source records through transforms into a resumable sink.

    The pipeline is deliberately domain-neutral. Sources own record discovery,
    transforms own scientific logic and sinks own serialization. Stable keys and
    the task fingerprint provide deterministic sharding and safe resume.
    """

    def __init__(
        self,
        *,
        source: Any,
        transforms: Sequence[Any],
        sink: Any,
        shard_count: int = 1,
        shard_index: int = 0,
        on_error: PreprocessingErrorPolicy | str = PreprocessingErrorPolicy.FAIL,
        checkpoint_interval: int = 1,
        workers: int = 1,
        workload: PreprocessingWorkload | str = PreprocessingWorkload.AUTO,
        publish_dataset: bool = False,
        dataset_name: str | None = None,
        dataset_version: str = "1",
        dataset_splits: Mapping[str, int] | None = None,
        dataset_source: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(shard_count, bool) or int(shard_count) < 1:
            raise ValueError("Preprocessing shard_count must be at least 1.")
        if isinstance(shard_index, bool) or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("Preprocessing shard_index must be within [0, shard_count).")
        if isinstance(checkpoint_interval, bool) or int(checkpoint_interval) < 1:
            raise ValueError("Preprocessing checkpoint_interval must be at least 1.")
        if isinstance(workers, bool) or int(workers) < 1:
            raise ValueError("Preprocessing workers must be at least 1.")
        if not callable(getattr(source, "records", None)):
            raise TypeError("Preprocessing source must expose records(context).")
        if not callable(getattr(sink, "write", None)):
            raise TypeError("Preprocessing sink must expose write(record, context).")
        for transform in transforms:
            if not callable(getattr(transform, "transform", None)):
                raise TypeError(
                    "Every preprocessing transform must expose transform(record, context)."
                )
        self.source = source
        self.transforms = tuple(transforms)
        self.sink = sink
        self.shard_count = int(shard_count)
        self.shard_index = int(shard_index)
        self.on_error = PreprocessingErrorPolicy(on_error)
        self.checkpoint_interval = int(checkpoint_interval)
        self.workers = int(workers)
        self.workload = PreprocessingWorkload(workload)
        if not isinstance(publish_dataset, bool):
            raise TypeError("Preprocessing publish_dataset must be a bool.")
        self.publish_dataset = publish_dataset or dataset_name is not None
        if self.workload is PreprocessingWorkload.GPU and self.workers != 1:
            raise ValueError(
                "GPU preprocessing requires workers=1. Use explicit task sharding and "
                "resource planning instead of creating uncoordinated GPU workers."
            )
        self.dataset_name = dataset_name
        self.dataset_version = str(dataset_version)
        self.dataset_splits = dict(dataset_splits or {})
        self.dataset_source = dict(dataset_source or {})
        self.dataset_metadata = dict(dataset_metadata or {})

    def run(self, context: TaskContext) -> TaskOutput:
        """Process the selected shard and return manifests as task artifacts."""
        manifest_path = context.output_path("preprocessing-manifest.json", create_parent=True)
        started = datetime.now(timezone.utc).isoformat()
        records: dict[str, Mapping[str, Any]] = {}
        if context.resume and manifest_path.is_file():
            previous = PreprocessingManifest.read_json(manifest_path)
            if (
                previous.config_fingerprint == context.config_fingerprint
                and previous.shard_count == self.shard_count
                and previous.shard_index == self.shard_index
            ):
                started = previous.started_at_utc
                records = dict(previous.records)

        seen: set[str] = set()
        processed = 0
        resumed = 0
        failures = 0
        selected = 0
        since_checkpoint = 0
        pending: dict[Future[Any], PreprocessingRecord] = {}
        executor = self._executor()
        try:
            for record in self.source.records(context):
                if not isinstance(record, PreprocessingRecord):
                    raise TypeError("Preprocessing sources must yield PreprocessingRecord objects.")
                if record.key in seen:
                    raise ValueError(f"Duplicate preprocessing record key: {record.key!r}.")
                seen.add(record.key)
                if not self._belongs_to_shard(record.key):
                    continue
                selected += 1
                if context.stop_requested:
                    self._checkpoint(manifest_path, context, started, records, complete=False)
                    raise KeyboardInterrupt("Preprocessing was cancelled.")
                prior = records.get(record.key, {})
                complete = prior.get("status") == "ok" and self._sink_is_complete(
                    record.key, context
                )
                if complete:
                    resumed += 1
                    continue
                if executor is None:
                    try:
                        self._transform_and_write(record, context)
                        records[record.key] = self._success_record()
                        processed += 1
                    except Exception as error:
                        failures += 1
                        records[record.key] = self._failed_record(error)
                        self._checkpoint(manifest_path, context, started, records, complete=False)
                        if self.on_error is PreprocessingErrorPolicy.FAIL:
                            raise
                    since_checkpoint += 1
                else:
                    pending[self._submit(executor, record, context)] = record
                    if len(pending) >= self.workers * 2:
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            current = pending.pop(future)
                            try:
                                self._complete_future(future, context)
                                records[current.key] = self._success_record()
                                processed += 1
                            except Exception as error:
                                failures += 1
                                records[current.key] = self._failed_record(error)
                                if self.on_error is PreprocessingErrorPolicy.FAIL:
                                    for waiting in pending:
                                        waiting.cancel()
                                    self._checkpoint(
                                        manifest_path, context, started, records, complete=False
                                    )
                                    raise
                            since_checkpoint += 1
                if since_checkpoint >= self.checkpoint_interval:
                    self._checkpoint(manifest_path, context, started, records, complete=False)
                    since_checkpoint = 0
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    current = pending.pop(future)
                    try:
                        self._complete_future(future, context)
                        records[current.key] = self._success_record()
                        processed += 1
                    except Exception as error:
                        failures += 1
                        records[current.key] = self._failed_record(error)
                        if self.on_error is PreprocessingErrorPolicy.FAIL:
                            for waiting in pending:
                                waiting.cancel()
                            self._checkpoint(
                                manifest_path, context, started, records, complete=False
                            )
                            raise
                    since_checkpoint += 1
                    if since_checkpoint >= self.checkpoint_interval:
                        self._checkpoint(manifest_path, context, started, records, complete=False)
                        since_checkpoint = 0
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        final_declarations = self._finalize_sink(context)
        self._checkpoint(manifest_path, context, started, records, complete=True)
        content_artifacts = tuple(
            TaskArtifact.materialize(declaration, context.run_dir)
            for declaration in final_declarations
        )
        successful = sum(value.get("status") == "ok" for value in records.values())
        splits = self.dataset_splits or {"all": successful}
        declarations: tuple[ArtifactDeclaration, ...] = (
            *final_declarations,
            ArtifactDeclaration(
                "preprocessing-manifest.json",
                kind=ArtifactType.REPORT,
                media_type="application/json",
            ),
        )
        outputs: dict[str, Any] = {
            "preprocessing_manifest": "preprocessing-manifest.json",
        }
        if self.publish_dataset:
            dataset = DatasetArtifact.create(
                name=self.dataset_name or context.name,
                version=self.dataset_version,
                sample_count=successful,
                splits=splits,
                preprocessing_fingerprint=context.config_fingerprint,
                source=self.dataset_source
                or {
                    "type": f"{type(self.source).__module__}.{type(self.source).__name__}",
                    "inputs": [
                        {
                            "name": value["name"],
                            "path": value["path"],
                            "sha256": value["sha256"],
                            "size_bytes": value["size_bytes"],
                        }
                        for value in context.inputs
                    ],
                },
                artifacts=content_artifacts,
                metadata={
                    **self.dataset_metadata,
                    "shard_count": self.shard_count,
                    "shard_index": self.shard_index,
                },
            )
            dataset_path = context.output_path("dataset-artifact.json", create_parent=True)
            dataset.write_json(dataset_path)
            outputs.update(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_manifest": "dataset-artifact.json",
                }
            )
            declarations = (
                *declarations,
                ArtifactDeclaration(
                    "dataset-artifact.json",
                    kind=ArtifactType.DATASET,
                    media_type="application/json",
                    metadata={"dataset_id": dataset.dataset_id},
                ),
            )
        return TaskOutput(
            outputs=outputs,
            metrics={
                "records_selected": selected,
                "records_processed": processed,
                "records_resumed": resumed,
                "records_failed": failures,
            },
            artifacts=declarations,
            metadata={
                "shard_count": self.shard_count,
                "shard_index": self.shard_index,
                "on_error": self.on_error.value,
                "workers": self.workers,
                "workload": self.workload.value,
            },
        )

    def _transform_and_write(self, record: PreprocessingRecord, context: TaskContext) -> None:
        transformed = PreprocessingWorker.transform(self.transforms, record, context)
        self.sink.write(transformed, context)

    def _executor(self) -> Executor | None:
        if self.workers == 1:
            return None
        if self.workload is PreprocessingWorkload.CPU:
            return ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        # AUTO deliberately prefers the safer thread protocol. Consumers opt in
        # to process serialization costs and restrictions with workload=cpu.
        return ThreadPoolExecutor(max_workers=self.workers)

    def _submit(
        self,
        executor: Executor,
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> Future[Any]:
        if self.workload is PreprocessingWorkload.CPU:
            return executor.submit(
                PreprocessingWorker.transform,
                self.transforms,
                record,
                replace(context, stop_event=None),
            )
        return executor.submit(self._transform_and_write, record, context)

    def _complete_future(self, future: Future[Any], context: TaskContext) -> None:
        result = future.result()
        if self.workload is PreprocessingWorkload.CPU:
            if not isinstance(result, PreprocessingRecord):
                raise TypeError("A preprocessing process returned an invalid record.")
            self.sink.write(result, context)

    @staticmethod
    def _success_record() -> Mapping[str, Any]:
        return {"status": "ok", "updated_at_utc": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def _failed_record(error: Exception) -> Mapping[str, Any]:
        return {
            "status": "failed",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            },
        }

    def _checkpoint(
        self,
        path: Path,
        context: TaskContext,
        started: str,
        records: Mapping[str, Mapping[str, Any]],
        *,
        complete: bool,
    ) -> None:
        PreprocessingManifest(
            config_fingerprint=context.config_fingerprint,
            shard_count=self.shard_count,
            shard_index=self.shard_index,
            started_at_utc=started,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            complete=complete,
            records=records,
        ).write_json(path)

    def _belongs_to_shard(self, key: str) -> bool:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.shard_count == self.shard_index

    def _sink_is_complete(self, key: str, context: TaskContext) -> bool:
        method = getattr(self.sink, "is_complete", None)
        return bool(method(key, context)) if callable(method) else True

    def _finalize_sink(self, context: TaskContext) -> tuple[ArtifactDeclaration, ...]:
        method = getattr(self.sink, "finalize", None)
        values = method(context) if callable(method) else ()
        if values is None:
            return ()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise TypeError("Preprocessing sink finalize() must return artifact declarations.")
        return tuple(ArtifactDeclaration.from_value(value) for value in values)
