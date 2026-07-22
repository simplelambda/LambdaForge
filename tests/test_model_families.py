"""Behavioral tests for sequence, set, tabular and vision model families."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.sequence import (
    GRUModel,
    LSTMModel,
    RNNModel,
    TemporalConvNet,
    TransformerEncoderModel,
)
from lambdaforge.nn.models.sets import DeepSets, SetTransformer
from lambdaforge.nn.models.tabular import FTTransformer, ResidualMLP
from lambdaforge.nn.models.vision import ConvNeXt2D, ConvNeXtBlock2D, ResNet2D


class TestModelFamilies:
    """Verify masks, invariances, gradients and configuration entry points."""

    @pytest.mark.parametrize("model_cls", [RNNModel, GRUModel, LSTMModel])
    def test_recurrent_models_ignore_right_padding_and_backpropagate(self, model_cls) -> None:
        model = model_cls(5, hidden_size=8, out_features=3, bidirectional=True).eval()
        x = torch.randn(2, 6, 5, requires_grad=True)
        padding_mask = torch.tensor([[False] * 6, [False] * 3 + [True] * 3])
        changed = x.detach().clone()
        changed[padding_mask] = 1000.0

        first = model(x, padding_mask=padding_mask)
        second = model(changed, padding_mask=padding_mask)

        assert first.shape == (2, 3)
        assert torch.allclose(first, second, atol=1e-6)
        first.sum().backward()
        assert x.grad is not None

    def test_transformer_ignores_masked_tokens(self) -> None:
        model = TransformerEncoderModel(
            4,
            d_model=16,
            num_heads=4,
            num_layers=1,
            out_features=2,
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 5, 4, requires_grad=True)
        padding_mask = torch.tensor([[False] * 5, [False] * 2 + [True] * 3])
        changed = x.detach().clone()
        changed[padding_mask] = -1000.0

        first = model(x, padding_mask=padding_mask)
        second = model(changed, padding_mask=padding_mask)

        assert first.shape == (2, 2)
        assert torch.allclose(first, second, atol=1e-6)
        first.sum().backward()
        assert x.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_transformer_moves_cpu_masks_to_cuda_input(self) -> None:
        model = TransformerEncoderModel(
            4,
            d_model=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ).cuda()
        x = torch.randn(2, 4, 4, device="cuda")
        padding_mask = torch.tensor([[False, False, False, False], [False, False, True, True]])
        attention_mask = torch.zeros(4, 4, dtype=torch.bool)
        assert model(x, padding_mask, attention_mask).device.type == "cuda"

    def test_temporal_conv_is_causal_and_preserves_sequence_length(self) -> None:
        model = TemporalConvNet(
            4,
            channels=[6, 8],
            out_features=3,
            dropout=0.0,
            causal=True,
            output_mode="sequence",
        ).eval()
        x = torch.randn(2, 7, 4, requires_grad=True)
        changed = x.detach().clone()
        changed[:, 5:] += 1000.0

        first = model(x)
        second = model(changed)

        assert first.shape == (2, 7, 3)
        assert torch.allclose(first[:, :5], second[:, :5], atol=1e-6)
        first.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("kind", ["deep_sets", "set_transformer"])
    def test_set_models_are_permutation_invariant(self, kind: str) -> None:
        if kind == "deep_sets":
            model = DeepSets(5, 3, embedding_dim=8)
        else:
            model = SetTransformer(
                5,
                3,
                d_model=16,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
            )
        model.eval()
        x = torch.randn(2, 6, 5, requires_grad=True)
        valid_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])

        first = model(x, valid_mask)
        second = model(x[:, permutation], valid_mask[:, permutation])

        assert first.shape == (2, 3)
        assert torch.allclose(first, second, atol=2e-5)
        first.sum().backward()
        assert x.grad is not None

    def test_ft_transformer_masks_missing_values_before_embedding_lookup(self) -> None:
        model = FTTransformer(
            2,
            [3, 4],
            3,
            d_model=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        continuous = torch.randn(3, 2, requires_grad=True)
        categorical = torch.tensor([[0, 1], [1, 2], [2, 3]])
        continuous_mask = torch.tensor([[True, False], [True, True], [True, True]])
        categorical_mask = torch.tensor([[False, True], [True, True], [True, True]])

        first = model(continuous, categorical, continuous_mask, categorical_mask)
        changed_continuous = continuous.detach().clone()
        changed_categorical = categorical.clone()
        changed_continuous[0, 1] = 1000.0
        changed_categorical[0, 0] = 1000
        second = model(
            changed_continuous,
            changed_categorical,
            continuous_mask,
            categorical_mask,
        )

        assert first.shape == (3, 3)
        assert torch.allclose(first, second, atol=1e-6)
        first.sum().backward()
        assert continuous.grad is not None

    def test_ft_transformer_builds_from_yaml_compatible_spec(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.tabular.FTTransformer.FTTransformer",
                "params": {
                    "num_continuous_features": 2,
                    "categorical_cardinalities": [3],
                    "out_features": 1,
                    "d_model": 8,
                    "num_heads": 2,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
            }
        )
        output = model(torch.randn(2, 2), torch.tensor([[0], [2]]))
        assert output.shape == (2, 1)

    def test_residual_mlp_supports_per_block_configuration_and_gradients(self) -> None:
        model = ResidualMLP(
            5,
            2,
            hidden_features=12,
            num_blocks=2,
            expansion_factor=[1.5, 2.0],
            activation=["relu", "gelu"],
            dropout=[0.0, 0.1],
        )
        x = torch.randn(4, 5, requires_grad=True)
        output = model(x)
        assert output.shape == (4, 2)
        output.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("architecture", ["resnet", "convnext"])
    def test_vision_models_support_custom_stages_and_gradients(self, architecture: str) -> None:
        if architecture == "resnet":
            model = ResNet2D(
                3,
                5,
                stage_channels=[8, 16],
                blocks_per_stage=[1, 1],
                stage_strides=[1, 2],
                stem_channels=8,
                stem_kernel_size=3,
                stem_stride=1,
                use_stem_pooling=False,
            )
        else:
            model = ConvNeXt2D(
                3,
                5,
                stage_channels=[8, 16],
                blocks_per_stage=[1, 1],
                stem_kernel_size=2,
                stem_stride=2,
                block_kernel_size=[3, 5],
                expansion_ratio=[2.0, 3.0],
                drop_path_probabilities=[0.0, 0.1],
                layer_scale_init=[1e-6, None],
            )
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        output = model(x)
        assert output.shape == (2, 5)
        output.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize(
        "model",
        [
            ResNet2D(
                3,
                stage_channels=[8, 12, 16],
                blocks_per_stage=[1, 1, 1],
                stage_strides=[1, 2, 2],
                stem_channels=8,
                stem_kernel_size=3,
                stem_stride=1,
                use_stem_pooling=False,
            ),
            ConvNeXt2D(
                3,
                stage_channels=[8, 12, 16],
                blocks_per_stage=[1, 1, 1],
                stem_kernel_size=2,
                stem_stride=2,
            ),
        ],
    )
    def test_hierarchical_vision_backbones_expose_ordered_feature_maps(
        self,
        model: torch.nn.Module,
    ) -> None:
        x = torch.randn(2, 3, 31, 47, requires_grad=True)

        feature_maps = model.forward_feature_maps(x)
        pooled = model.forward_features(x)
        pooled.sum().backward()

        assert len(feature_maps) == len(model.feature_channels)
        assert tuple(feature.shape[1] for feature in feature_maps) == model.feature_channels
        assert all(
            feature_maps[index].shape[-2] >= feature_maps[index + 1].shape[-2]
            and feature_maps[index].shape[-1] >= feature_maps[index + 1].shape[-1]
            for index in range(len(feature_maps) - 1)
        )
        assert pooled.shape == (2, model.feature_channels[-1])
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_convnext_block_preserves_shape_and_uses_depthwise_convolution(self) -> None:
        block = ConvNeXtBlock2D(8, kernel_size=5, expansion_ratio=3.0).eval()
        x = torch.randn(2, 8, 9, 7)
        output = block(x)
        assert output.shape == x.shape
        assert block.depthwise_convolution.groups == 8

    def test_resnet_adapts_or_rejects_normalization_by_image_layout(self) -> None:
        model = ResNet2D(
            3,
            stage_channels=[8],
            blocks_per_stage=[1],
            stage_strides=[1],
            stem_channels=8,
            stem_kernel_size=3,
            stem_stride=1,
            use_stem_pooling=False,
            normalization="channel-layer-norm",
        )
        assert model(torch.randn(2, 3, 8, 8)).shape == (2, 8)

        with pytest.raises(ValueError, match="ChannelLayerNorm"):
            ResNet2D(
                3,
                stage_channels=[8],
                blocks_per_stage=[1],
                stage_strides=[1],
                normalization="layernorm",
            )
