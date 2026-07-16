"""Shape, mask and gradient smoke tests for every built-in pooling object."""

import torch

from lambdaforge.nn.pooling import (
    AttentionPooling,
    AutoPool,
    FractionalTopKMeanPooling,
    GatedAttentionPooling,
    LogSumExpPooling,
    MaxPooling,
    MeanPooling,
    MinPooling,
    MomentPooling,
    MultiHeadGatedAttentionPooling,
    NoisyOrPooling,
    ProbabilityGeMPooling,
    SoftmaxPooling,
    SumPooling,
    TopKMeanPooling,
    TopKPooling,
)


class TestPooling:
    """Exercise every pooling implementation under one shared masked contract."""

    def test_all_poolings_accept_mask_and_preserve_batch(self) -> None:
        base = torch.randn(2, 5, 4, requires_grad=True)
        values = torch.sigmoid(base)
        mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
        poolings = [
            AttentionPooling(4),
            AutoPool(),
            FractionalTopKMeanPooling(0.5),
            GatedAttentionPooling(4),
            LogSumExpPooling(),
            MaxPooling(),
            MeanPooling(),
            MinPooling(),
            MomentPooling(),
            MultiHeadGatedAttentionPooling(4),
            NoisyOrPooling(),
            ProbabilityGeMPooling(),
            SoftmaxPooling(),
            SumPooling(),
            TopKMeanPooling(2),
            TopKPooling(4, 2),
        ]

        outputs = [pooling(values, mask) for pooling in poolings]
        assert all(output.shape[0] == 2 for output in outputs)
        assert outputs[8].shape == (2, 8)
        assert all(torch.isfinite(output).all() for output in outputs)

        sum(output.sum() for output in outputs).backward()
        assert base.grad is not None
        assert torch.isfinite(base.grad).all()
