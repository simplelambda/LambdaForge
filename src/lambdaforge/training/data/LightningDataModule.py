"""Implementation of the LightningDataModule object."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch.utils.data import DataLoader, Dataset

from lambdaforge.integrations.Lightning import LightningDataModuleBase
from lambdaforge.training.data.GuardedWorkerInit import GuardedWorkerInit


class LightningDataModule(LightningDataModuleBase):
    r"""LightningDataModule for already-created datasets.

    This class wraps standard PyTorch datasets and exposes train, validation
    and test dataloaders to Lightning.

    Parameters
    ----------
    train : Dataset
        Training dataset.
    val : Dataset | None
        Optional validation dataset.
    test : Dataset | None
        Optional test dataset.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Whether DataLoader should use pinned memory.
    persistent_workers : bool
        Whether workers should stay alive between epochs.
    prefetch_factor : int | None
        Number of batches prefetched by each worker. Only used when
        ``num_workers > 0``.
    collate_fn : Callable | None
        Optional custom collate function.
    worker_init_fn : Callable | None
        Optional custom DataLoader worker initializer. It is wrapped so every
        worker still installs the framework process guard and CPU thread caps.
    drop_last : bool
        Whether to drop the last incomplete training batch.
    dataloader_kwargs : Mapping[str, Any] | None
        Extra ``DataLoader`` keyword arguments shared by every split.
    train_dataloader_kwargs, val_dataloader_kwargs, test_dataloader_kwargs
        Split-specific extra keyword arguments. Framework-managed keys such as
        ``dataset``, ``shuffle`` and ``worker_init_fn`` cannot be overridden.
    """

    def __init__(
        self,
        train: Dataset,
        val: Dataset | None = None,
        test: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = 2,
        collate_fn: Callable[[Any], Any] | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        drop_last: bool = False,
        dataloader_kwargs: Mapping[str, Any] | None = None,
        train_dataloader_kwargs: Mapping[str, Any] | None = None,
        val_dataloader_kwargs: Mapping[str, Any] | None = None,
        test_dataloader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.drop_last = drop_last
        self.dataloader_kwargs = dict(dataloader_kwargs or {})
        self.train_dataloader_kwargs = dict(train_dataloader_kwargs or {})
        self.val_dataloader_kwargs = dict(val_dataloader_kwargs or {})
        self.test_dataloader_kwargs = dict(test_dataloader_kwargs or {})

        managed_keys = {
            "dataset",
            "batch_size",
            "shuffle",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "collate_fn",
            "worker_init_fn",
            "drop_last",
        }
        for name, values in (
            ("dataloader_kwargs", self.dataloader_kwargs),
            ("train_dataloader_kwargs", self.train_dataloader_kwargs),
            ("val_dataloader_kwargs", self.val_dataloader_kwargs),
            ("test_dataloader_kwargs", self.test_dataloader_kwargs),
        ):
            conflicts = sorted(managed_keys & values.keys())
            if conflicts:
                raise ValueError(
                    f"{name} cannot override framework-managed keys: " + ", ".join(conflicts)
                )

    def make_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self.collate_fn,
            "drop_last": drop_last,
        }

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["worker_init_fn"] = GuardedWorkerInit(self.worker_init_fn)

        kwargs.update(self.dataloader_kwargs)
        kwargs.update(extra_kwargs or {})

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(
            dataset=self.train,
            shuffle=True,
            drop_last=self.drop_last,
            extra_kwargs=self.train_dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val is None:
            return None

        return self.make_dataloader(
            dataset=self.val,
            shuffle=False,
            drop_last=False,
            extra_kwargs=self.val_dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test is None:
            return None

        return self.make_dataloader(
            dataset=self.test,
            shuffle=False,
            drop_last=False,
            extra_kwargs=self.test_dataloader_kwargs,
        )
