"""Input type shared by non-differentiable clustering adapters."""

import numpy as np
import torch

ArrayInput = np.ndarray | torch.Tensor
