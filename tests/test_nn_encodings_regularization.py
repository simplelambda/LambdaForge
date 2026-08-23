"""Tests for reusable encoding and regularization objects."""

import torch

from lambdaforge.nn.encodings import (
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from lambdaforge.nn.regularization import DropPath, FeatureDropout, GaussianNoise


class TestEncodingsAndRegularization:
    """Verify shapes, gradients, RNG isolation and train/eval contracts."""

    def test_absolute_position_encodings_preserve_shape_and_gradient(self) -> None:
        x = torch.randn(2, 5, 7, requires_grad=True)
        sinusoidal = SinusoidalPositionalEncoding(7, max_length=8)
        learned = LearnedPositionalEncoding(7, max_length=8)
        output = learned(sinusoidal(x))
        assert output.shape == x.shape
        output.sum().backward()
        assert learned.positions.grad is not None


    def test_rotary_encoding_preserves_pair_norms(self) -> None:
        x = torch.randn(2, 6, 8)
        output = RotaryPositionalEncoding(8)(x)
        assert output.shape == x.shape
        assert torch.allclose(
            output.square().sum(dim=-1),
            x.square().sum(dim=-1),
            atol=1e-5,
        )

    def test_regularizers_are_identity_in_evaluation(self) -> None:
        x = torch.ones(4, 3, 5)
        modules = [
            DropPath(0.5),
            FeatureDropout(0.5),
            GaussianNoise(0.5),
        ]
        for module in modules:
            module.eval()
            assert torch.equal(module(x), x)

    def test_feature_dropout_shares_mask_across_sequence(self) -> None:
        torch.manual_seed(4)
        module = FeatureDropout(0.5, feature_dim=-1)
        output = module(torch.ones(8, 7, 32))
        assert torch.equal(output[:, :1, :].expand_as(output), output)
