"""Logical dataset and code identity tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from lambdaforge.reproducibility import CodeIdentity, IdentityExplainer
from lambdaforge.tasks import TaskConfig


class TestDataAndCodeIdentity:
    """Ensure storage paths do not masquerade as scientific identity."""

    @staticmethod
    def task(path: Path, output: Path) -> TaskConfig:
        return TaskConfig(
            {
                "kind": "task",
                "schema_version": "1.0",
                "name": "portable",
                "output_root": str(output),
                "inputs": [
                    {
                        "name": "data",
                        "path": str(path),
                        "identity": {
                            "strategy": "version",
                            "namespace": "project/raw",
                            "version": "2026-08-11",
                        },
                    }
                ],
                "task": {"target": "tests.fixtures.UserTask.UserTask", "params": {"value": 1}},
                "extensions": {"authoring": {"code_version": "release-test"}},
            }
        )

    def test_same_logical_version_at_two_paths_has_same_fingerprint(self, tmp_path: Path) -> None:
        """Physical relocation must not invalidate a logical dataset version."""
        first = tmp_path / "a/data.txt"
        second = tmp_path / "b/data.txt"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_text("one", encoding="utf-8")
        second.write_text("different bytes managed externally", encoding="utf-8")
        assert (
            self.task(first, tmp_path / "runs-a").fingerprint
            == self.task(second, tmp_path / "runs-b").fingerprint
        )

    def test_git_identity_distinguishes_clean_and_dirty_code(self, tmp_path: Path) -> None:
        """A dirty worktree must carry the base revision plus a change digest."""
        repository = tmp_path / "project"
        repository.mkdir()
        subprocess.run(("git", "init", "-q", str(repository)), check=True)
        subprocess.run(
            ("git", "-C", str(repository), "config", "user.email", "test@example.com"), check=True
        )
        subprocess.run(("git", "-C", str(repository), "config", "user.name", "Test"), check=True)
        code = repository / "model.py"
        code.write_text("WIDTH = 8\n", encoding="utf-8")
        subprocess.run(("git", "-C", str(repository), "add", "model.py"), check=True)
        subprocess.run(("git", "-C", str(repository), "commit", "-qm", "base"), check=True)
        clean = CodeIdentity.capture(repository)
        code.write_text("WIDTH = 16\n", encoding="utf-8")
        dirty = CodeIdentity.capture(repository)
        assert not clean.dirty
        assert dirty.dirty
        assert dirty.revision == clean.revision
        assert dirty.changes_digest is not None

    def test_identity_explanation_names_exact_changed_path(self) -> None:
        """A user should see the scientific field, not only two opaque hashes."""
        explanation = IdentityExplainer().compare(
            {"model": {"width": 32}, "seed": 7},
            {"model": {"width": 16}, "seed": 7},
        )
        assert not explanation.same
        assert [change["path"] for change in explanation.changes] == ["model.width"]
