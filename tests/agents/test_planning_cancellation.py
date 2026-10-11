"""Permitted project imports stay cancellable without retaining a planning child."""

import asyncio
import multiprocessing
from pathlib import Path

import pytest

from lambdaforge.agents.planning import invoke_planning


def test_cancel_native_planning_reaps_the_fresh_child(tmp_path: Path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="slow"\nversion="0.1"\n')
    (tmp_path / "slow_work.py").write_text(
        'import time\nfrom pathlib import Path\nPath(__file__).with_suffix(".started").touch()\n'
        "time.sleep(20)\nfrom lambdaforge import Work\n"
        "class Example(Work):\n    def run(self):\n        return {}\n"
    )
    (tmp_path / "work.yaml").write_text("name: slow\nrun: slow_work.Example\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    async def scenario():
        previous = {child.pid for child in multiprocessing.active_children()}
        task = asyncio.create_task(
            invoke_planning(tmp_path, "validate_work", {"config": "work.yaml"})
        )
        try:
            for _ in range(100):
                if (tmp_path / "slow_work.started").exists():
                    break
                await asyncio.sleep(0.05)
            assert (tmp_path / "slow_work.started").exists()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert {child.pid for child in multiprocessing.active_children()} == previous

    asyncio.run(scenario())
