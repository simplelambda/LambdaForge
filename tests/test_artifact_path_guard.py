"""Filesystem containment tests for retention-owned paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard


class TestArtifactPathGuard:
    """Verify internal retention writes and removals never follow links."""

    @staticmethod
    def _symlink(source: Path, destination: Path, *, directory: bool) -> None:
        try:
            destination.symlink_to(source, target_is_directory=directory)
        except OSError:
            pytest.skip("Symlinks are not available in this environment.")

    def test_internal_directory_cannot_escape_through_symlink(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        self._symlink(external, root / ".lambdaforge", directory=True)

        with pytest.raises(ValueError, match="Unsafe retention directory"):
            ArtifactPathGuard.ensure_directory(root, ".lambdaforge/retention")

        assert tuple(external.iterdir()) == ()

    def test_destination_rejects_linked_parent_and_dangling_link(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "run"
        root.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        self._symlink(external, root / "linked", directory=True)

        with pytest.raises(ValueError, match="Unsafe retention destination parent"):
            ArtifactPathGuard.safe_destination(root, "linked/value.bin")

        missing = external / "missing.bin"
        dangling = root / "dangling.bin"
        self._symlink(missing, dangling, directory=False)
        with pytest.raises(ValueError, match="Unsafe retention destination"):
            ArtifactPathGuard.safe_destination(root, "dangling.bin")

    def test_recursive_removal_preflight_rejects_link_entries(self, tmp_path: Path) -> None:
        root = tmp_path / "quarantine"
        root.mkdir()
        external = tmp_path / "external.bin"
        external.write_bytes(b"preserve")
        self._symlink(external, root / "linked.bin", directory=False)

        with pytest.raises(ValueError, match="unsafe link"):
            ArtifactPathGuard.validate_regular_tree(root)

        assert external.read_bytes() == b"preserve"
