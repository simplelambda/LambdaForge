"""State-on-disk preprocessing transform used to test resumable attempts."""

from __future__ import annotations

from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingTransform
from lambdaforge.tasks import TaskContext


class FailOnceRecordTransform(PreprocessingTransform):
    """Fail once for one key, then succeed across a new task instance."""

    def __init__(self, key: str) -> None:
        self.key = key

    def transform(
        self,
        record: PreprocessingRecord,
        context: TaskContext,
    ) -> PreprocessingRecord:
        """Use a run-local marker to make only the first matching call fail."""
        marker = context.output_path(".lambdaforge/fail-once.marker", create_parent=True)
        if record.key == self.key and not marker.exists():
            marker.write_text("failed\n", encoding="utf-8")
            raise RuntimeError("intentional first-attempt record failure")
        return record
