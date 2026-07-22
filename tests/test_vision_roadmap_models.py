"""Tests for dense, patch-based and efficient vision roadmap models."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.vision import (
    FeaturePyramidNetwork2D,
    InvertedResidualBlock2D,
    MobileNetV2,
    ResNet2D,
    UNet2D,
    VisionTransformer2D,
)


class TestVisionRoadmapModels:
    """Verify shapes, gradients, policies and recursive YAML construction."""

    @pytest.mark.parametrize(
        ("output_mode", "expected_shape"),
        [
            ("class_token", (2, 5)),
            ("mean", (2, 5)),
            ("tokens", (2, 24, 5)),
            ("feature_map", (2, 5, 4, 6)),
        ],
    )
    def test_vision_transformer_supports_every_output_mode(
        self,
        output_mode: str,
        expected_shape: tuple[int, ...],
    ) -> None:
        model = VisionTransformer2D(
            3,
            patch_size=8,
            image_size=(32, 48),
            d_model=16,
            num_heads=4,
            num_layers=1,
            out_features=5,
            output_mode=output_mode,
            remainder_policy="pad",
            dropout=0.0,
        )
        x = torch.randn(2, 3, 31, 47, requires_grad=True)

        output = model(x)
        output.sum().backward()

        assert output.shape == expected_shape
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_vision_transformer_rejects_partial_patches_when_requested(self) -> None:
        model = VisionTransformer2D(
            3,
            patch_size=8,
            image_size=32,
            d_model=16,
            num_heads=4,
            num_layers=1,
        )
        with pytest.raises(ValueError, match="divisible"):
            model(torch.randn(1, 3, 31, 32))
        with pytest.raises(ValueError, match="class_token"):
            VisionTransformer2D(
                3,
                d_model=16,
                num_heads=4,
                num_layers=1,
                output_mode="class_token",
                use_class_token=False,
            )

    def test_unet_preserves_odd_spatial_dimensions_and_backpropagates(self) -> None:
        model = UNet2D(
            3,
            4,
            stage_channels=[8, 16],
            blocks_per_stage=[1, 2],
            bottleneck_channels=24,
        )
        x = torch.randn(2, 3, 31, 47, requires_grad=True)

        output = model(x)
        output.mean().backward()

        assert output.shape == (2, 4, 31, 47)
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_mobile_backbone_exposes_stages_and_depthwise_blocks(self) -> None:
        model = MobileNetV2(
            3,
            6,
            stage_channels=[8, 16, 24],
            blocks_per_stage=[1, 2, 1],
            stage_strides=[1, 2, 2],
            expansion_ratios=[1.0, 2.0, 3.0],
            stem_channels=8,
            final_channels=32,
            head_dropout=0.0,
        )
        x = torch.randn(2, 3, 32, 48, requires_grad=True)

        maps = model.forward_feature_maps(x)
        output = model(x)
        output.sum().backward()

        first_block = model.stages[0][0]
        assert isinstance(first_block, InvertedResidualBlock2D)
        assert first_block.depthwise_convolution.groups == 8
        assert tuple(feature.shape[1] for feature in maps) == model.feature_channels
        assert output.shape == (2, 6)
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_feature_pyramid_uses_the_generic_backbone_contract(self) -> None:
        backbone = ResNet2D(
            3,
            stage_channels=[8, 16, 24],
            blocks_per_stage=[1, 1, 1],
            stage_strides=[1, 2, 2],
            stem_channels=8,
            stem_kernel_size=3,
            stem_stride=1,
            use_stem_pooling=False,
        )
        model = FeaturePyramidNetwork2D(backbone, out_channels=7, extra_levels=1)

        maps = model(torch.randn(2, 3, 31, 47))

        assert len(maps) == 4
        assert tuple(feature.shape[1] for feature in maps) == (7, 7, 7, 7)
        assert maps[0].shape[-2:] == (31, 47)
        assert all(
            maps[index].shape[-2] >= maps[index + 1].shape[-2]
            and maps[index].shape[-1] >= maps[index + 1].shape[-1]
            for index in range(len(maps) - 1)
        )

    def test_feature_pyramid_builds_recursively_from_yaml_compatible_spec(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.FeaturePyramidNetwork2D",
                "params": {
                    "backbone": {
                        "target": "lambdaforge.nn.models.MobileNetV2",
                        "params": {
                            "in_channels": 3,
                            "stage_channels": [8, 16],
                            "blocks_per_stage": [1, 1],
                            "stage_strides": [1, 2],
                            "expansion_ratios": [1.0, 2.0],
                            "stem_channels": 8,
                            "final_channels": 24,
                        },
                    },
                    "out_channels": 6,
                },
            }
        )

        assert isinstance(model, FeaturePyramidNetwork2D)
        assert [feature.shape[1] for feature in model(torch.randn(1, 3, 32, 32))] == [6, 6]
