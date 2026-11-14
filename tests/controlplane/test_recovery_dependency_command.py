"""The public metadata probe reuses native ownership and never prepares or launches a Job."""

import sys

from tests.controlplane.test_single_work_retry import single_job  # noqa: F401
from tests.work.test_single_work_recovery import snapshot


def test_dependency_command_is_read_only_and_bounded(single_job):  # noqa: F811 - imported pytest fixture
    jobs, record, first, _ = single_job
    jobs.factory.reader_command = lambda profile: (sys.executable,)
    before = snapshot(jobs.store.root), snapshot(first.execution_dir)
    plan = jobs.recovery_dependencies(record.job_id, limit=1)
    assert plan["unavailable"] == 0 and plan["entry_count"] == 0
    assert (snapshot(jobs.store.root), snapshot(first.execution_dir)) == before
