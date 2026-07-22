"""Focused tests for safe disk and memory-mapped dataset adapters."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lambdaforge.data import FileDataset, NumpyMemmapDataset


class TestDataAdapters:
    """Verify lazy I/O, isolation, process reopening and deterministic cleanup."""

    def test_file_dataset_loads_explicit_paths_lazily(self, tmp_path: Path) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("one", encoding="utf-8")
        second.write_text("two", encoding="utf-8")
        loader = Mock(side_effect=lambda path: path.read_text(encoding="utf-8"))

        dataset = FileDataset(["first.txt", "second.txt"], loader=loader, root=tmp_path)

        assert len(dataset) == 2
        loader.assert_not_called()
        assert dataset[1] == "two"
        loader.assert_called_once_with(second.resolve())

    def test_file_dataset_rejects_escapes_and_missing_samples(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        loader = Mock()

        with pytest.raises(ValueError, match="escapes root"):
            FileDataset(["../outside.bin"], loader=loader, root=root)
        with pytest.raises(TypeError, match="sequence of paths"):
            FileDataset("sample.bin", loader=loader)

        dataset = FileDataset(["missing.bin"], loader=loader, root=root)
        with pytest.raises(FileNotFoundError, match="missing.bin"):
            dataset[0]
        loader.assert_not_called()

    def test_memmap_samples_are_lazy_writable_copies(self, tmp_path: Path) -> None:
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        targets = np.array([0, 1, 0], dtype=np.int64)
        np.save(features_path, features)
        np.save(targets_path, targets)
        dataset = NumpyMemmapDataset({"x": features_path, "target": targets_path})

        assert not dataset.is_open
        try:
            assert len(dataset) == 3
            assert dataset.is_open
            sample = dataset[1]
            assert torch.equal(sample["x"], torch.from_numpy(features[1]))
            assert torch.equal(sample["target"], torch.tensor(targets[1]))

            sample["x"].fill_(-1)
            assert torch.equal(dataset[1]["x"], torch.from_numpy(features[1]))
        finally:
            dataset.close()
            dataset.close()

        assert not dataset.is_open
        features_path.unlink()
        targets_path.unlink()

    def test_memmap_numpy_mode_still_returns_independent_arrays(self, tmp_path: Path) -> None:
        path = tmp_path / "values.npy"
        values = np.arange(6, dtype=np.float64).reshape(3, 2)
        np.save(path, values)
        dataset = NumpyMemmapDataset({"value": path}, as_tensors=False)

        try:
            sample = dataset[0]["value"]
            assert isinstance(sample, np.ndarray)
            assert sample.flags.writeable
            sample[:] = -1
            np.testing.assert_array_equal(dataset[0]["value"], values[0])
        finally:
            dataset.close()

    def test_memmap_refuses_pickled_objects_and_misaligned_arrays(self, tmp_path: Path) -> None:
        object_path = tmp_path / "objects.npy"
        short_path = tmp_path / "short.npy"
        long_path = tmp_path / "long.npy"
        np.save(object_path, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)
        np.save(short_path, np.zeros((2, 1), dtype=np.float32))
        np.save(long_path, np.zeros((3, 1), dtype=np.float32))

        objects = NumpyMemmapDataset({"value": object_path})
        with pytest.raises(ValueError, match="pickled data are refused"):
            len(objects)
        objects.close()

        misaligned = NumpyMemmapDataset({"short": short_path, "long": long_path})
        with pytest.raises(ValueError, match="has 3 samples; expected 2"):
            len(misaligned)
        misaligned.close()

        object_path.unlink()
        short_path.unlink()
        long_path.unlink()

    def test_memmap_pickle_drops_live_handles_and_reopens(self, tmp_path: Path) -> None:
        path = tmp_path / "values.npy"
        values = np.arange(8, dtype=np.float32).reshape(4, 2)
        np.save(path, values)
        dataset = NumpyMemmapDataset({"x": path})
        assert len(dataset) == 4
        assert dataset.is_open

        restored = pickle.loads(pickle.dumps(dataset))
        assert not restored.is_open
        try:
            assert torch.equal(restored[2]["x"], torch.from_numpy(values[2]))
            assert restored.is_open
        finally:
            restored.close()
            dataset.close()

        path.unlink()

    def test_memmap_works_in_a_spawned_dataloader_worker(self, tmp_path: Path) -> None:
        path = tmp_path / "spawn.npy"
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        np.save(path, values)
        dataset = NumpyMemmapDataset({"x": path})
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            multiprocessing_context="spawn",
        )

        try:
            batches = list(loader)
            reconstructed = torch.cat([batch["x"] for batch in batches])
            assert torch.equal(reconstructed, torch.from_numpy(values))
        finally:
            dataset.close()

        path.unlink()
