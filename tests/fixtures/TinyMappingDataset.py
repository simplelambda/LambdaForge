"""Small deterministic mapping dataset for Lightning smoke tests."""

import torch
from torch.utils.data import Dataset


class TinyMappingDataset(Dataset):
    """Return four-feature samples and binary column-vector targets."""

    def __init__(self, size: int = 16) -> None:
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(size, 4, generator=generator)
        self.targets = (self.features.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.features[index], "target": self.targets[index]}
