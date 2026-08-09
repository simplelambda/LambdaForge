"""Child-process defensive PyTorch CUDA allocator limit."""

from __future__ import annotations

import torch


class CudaMemoryLimiter:
    """Apply a public per-process caching-allocator fraction on a visible logical GPU."""

    def apply(
        self, budget_bytes: int, *, device: int = 0, headroom_fraction: float = 0.01
    ) -> float:
        """Apply and return the validated fraction, or zero for CPU/unlimited operation."""
        if budget_bytes <= 0 or not torch.cuda.is_available():
            return 0.0
        if not 0.0 <= headroom_fraction < 1.0:
            raise ValueError("CUDA memory headroom fraction must be in [0, 1).")
        total = int(torch.cuda.get_device_properties(device).total_memory)
        if total <= 0:
            raise RuntimeError("Visible CUDA device reported no usable total memory.")
        fraction = min(float(budget_bytes) / total, 1.0 - headroom_fraction)
        if fraction <= 0:
            raise ValueError("CUDA allocator budget must produce a positive fraction.")
        torch.cuda.memory.set_per_process_memory_fraction(fraction, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        return fraction
