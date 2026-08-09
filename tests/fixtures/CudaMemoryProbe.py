"""Representative CUDA operation used by isolated HPO preflight tests."""

import torch


class CudaMemoryProbe:
    """Allocate a small differentiable CUDA workload."""

    def __call__(self) -> None:
        value = torch.randn(256, 256, device="cuda", requires_grad=True)
        (value.square().mean()).backward()
