"""Numerical and integration tests for LambdaForge's expanded NN component catalog."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.activations import (
    CELU,
    GEGLU,
    GLU,
    SELU,
    Hardsigmoid,
    Hardswish,
    Mish,
    PReLU,
    ReGLU,
    ReLU6,
    Sine,
    Snake,
    Softplus,
    Softsign,
    SquarePlus,
    SwiGLU,
)
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.distances import (
    AngularDistance,
    ChebyshevDistance,
    CosineDistance,
    MahalanobisDistance,
    ManhattanDistance,
    MinkowskiDistance,
)
from lambdaforge.nn.kernels import LaplacianKernel, PolynomialKernel, RBFKernel
from lambdaforge.nn.losses import (
    BinaryCrossEntropyWithLogitsLoss,
    BinaryFocalLoss,
    ContrastiveLoss,
    CrossEntropyLoss,
    DiceLoss,
    HuberLoss,
    InfoNCELoss,
    MeanAbsoluteErrorLoss,
    MeanSquaredErrorLoss,
    MulticlassFocalLoss,
    SmoothL1Loss,
    TripletMarginLoss,
    TverskyLoss,
)
from lambdaforge.nn.normalizations import (
    ChannelLayerNorm,
    GroupNorm,
    InstanceNorm,
    L2Norm,
    ScaleNorm,
)
from lambdaforge.nn.pooling import (
    ConcatMeanMaxPooling,
    GeneralizedMeanPooling,
    MultiheadAttentionPooling,
    StatisticsPooling,
)
from lambdaforge.nn.similarities import (
    BilinearSimilarity,
    CosineSimilarity,
    DotProductSimilarity,
)


class TestNeuralComponentCatalog:
    """Check reference formulas, gradients, validation and YAML construction."""

    def test_activation_catalog_shapes_references_and_gradients(self) -> None:
        x = torch.linspace(-2.0, 2.0, 24).reshape(3, 8).requires_grad_()
        pointwise = [
            PReLU(),
            ReLU6(),
            SELU(),
            CELU(alpha=0.7),
            Mish(),
            Softplus(beta=1.5, threshold=10.0),
            Softsign(),
            Hardsigmoid(),
            Hardswish(),
            Sine(frequency=2.0, amplitude=0.5, phase=0.1),
            Snake(alpha=1.2),
            SquarePlus(b=3.0),
        ]
        pointwise_outputs = [activation(x) for activation in pointwise]
        gated_outputs = [activation(x) for activation in (GLU(), GEGLU(), SwiGLU(), ReGLU())]

        assert all(output.shape == x.shape for output in pointwise_outputs)
        assert all(output.shape == (3, 4) for output in gated_outputs)
        assert torch.allclose(gated_outputs[0], F.glu(x, dim=-1))
        sum(output.sum() for output in pointwise_outputs + gated_outputs).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_distances_match_reference_formulas_and_backpropagate(self) -> None:
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        y = torch.tensor([[[1.0, 1.0], [2.0, 0.0]]], requires_grad=True)

        assert torch.allclose(ManhattanDistance()(x, y), torch.cdist(x, y, p=1.0))
        assert torch.allclose(MinkowskiDistance(p=3.0)(x, y), torch.cdist(x, y, p=3.0))
        expected_chebyshev = (x.unsqueeze(2) - y.unsqueeze(1)).abs().amax(dim=-1)
        assert torch.allclose(ChebyshevDistance()(x, y), expected_chebyshev)
        expected_cosine = 1.0 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(1, 2)
        assert torch.allclose(CosineDistance()(x, y), expected_cosine)
        angles = AngularDistance()(x, x)
        assert torch.allclose(angles[0].diagonal(), torch.zeros(2))
        assert torch.allclose(angles[0, 0, 1], torch.tensor(torch.pi / 2.0))
        assert torch.allclose(
            MahalanobisDistance(2, squared=True)(x, y), torch.cdist(x, y).square()
        )

        total = sum(
            distance(x, y).sum()
            for distance in (ManhattanDistance(), MinkowskiDistance(), CosineDistance())
        )
        total.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_similarities_and_kernels_match_closed_forms(self) -> None:
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        y = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
        dot = torch.matmul(x, y.transpose(-1, -2))
        assert torch.allclose(DotProductSimilarity()(x, y), dot)
        assert torch.allclose(CosineSimilarity()(x, y), torch.full((1, 2, 1), 2**-0.5))

        bilinear = BilinearSimilarity(2, bias=False)
        with torch.no_grad():
            bilinear.weight.copy_(torch.eye(2))
        assert torch.allclose(bilinear(x, y), dot)
        squared = torch.cdist(x, y).square()
        assert torch.allclose(RBFKernel(length_scale=2.0)(x, y), torch.exp(-squared / 8.0))
        assert torch.allclose(
            LaplacianKernel(length_scale=2.0)(x, y),
            torch.exp(-torch.cdist(x, y, p=1.0) / 2.0),
        )
        assert torch.allclose(
            PolynomialKernel(2, gamma=0.5, offset=1.0)(x, y), (0.5 * dot + 1.0).square()
        )
        (bilinear(x, y).sum() + RBFKernel(learnable=True)(x, y).sum()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

        with pytest.raises(TypeError, match="integer"):
            PolynomialKernel(degree=2.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite"):
            RBFKernel(length_scale=float("inf"))
        with pytest.raises(ValueError, match="same dtype"):
            DotProductSimilarity()(x.detach(), y.detach().double())
        with pytest.raises(TypeError, match="floating"):
            RBFKernel()(
                torch.ones(1, 1, 2, dtype=torch.long), torch.ones(1, 1, 2, dtype=torch.long)
            )

    def test_losses_are_scalar_and_match_pytorch(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
        target = torch.tensor([0, 1])
        outputs = {"logits": logits}
        batch = {"target": target}
        cross_entropy = CrossEntropyLoss()(outputs, batch)
        assert torch.allclose(cross_entropy, F.cross_entropy(logits, target))
        assert torch.allclose(MulticlassFocalLoss(gamma=0.0)(outputs, batch), cross_entropy)

        prediction = torch.tensor([[1.0, 3.0]], requires_grad=True)
        regression_target = torch.tensor([[2.0, 1.0]])
        regression_outputs = {"prediction": prediction}
        regression_batch = {"target": regression_target}
        regression_losses = [
            MeanSquaredErrorLoss()(regression_outputs, regression_batch),
            MeanAbsoluteErrorLoss()(regression_outputs, regression_batch),
            SmoothL1Loss(beta=0.5)(regression_outputs, regression_batch),
            HuberLoss(delta=0.5)(regression_outputs, regression_batch),
        ]
        binary_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        binary_target = torch.tensor([[1.0, 0.0]])
        binary_outputs = {"logits": binary_logits}
        binary_batch = {"target": binary_target}
        binary_reference = F.binary_cross_entropy_with_logits(binary_logits, binary_target)
        assert torch.allclose(
            BinaryCrossEntropyWithLogitsLoss(pos_weight=[1.0])(binary_outputs, binary_batch),
            binary_reference,
        )
        assert torch.allclose(
            BinaryFocalLoss(alpha=None, gamma=0.0)(binary_outputs, binary_batch),
            binary_reference,
        )
        overlap_losses = [
            DiceLoss()(binary_outputs, binary_batch),
            TverskyLoss()(binary_outputs, binary_batch),
        ]
        losses = [cross_entropy, *regression_losses, *overlap_losses]
        assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses)
        sum(losses).backward()
        assert (
            logits.grad is not None
            and prediction.grad is not None
            and binary_logits.grad is not None
        )

    def test_embedding_losses_reduction_and_yaml_factory(self) -> None:
        first = torch.eye(4, requires_grad=True)
        second = torch.eye(4, requires_grad=True)
        contrastive = ContrastiveLoss()(
            {"embedding_a": first, "embedding_b": second}, {"target": torch.ones(4)}
        )
        triplet = TripletMarginLoss()(
            {"anchor": first, "positive": second, "negative": -second}, {}
        )
        info_nce = InfoNCELoss(temperature=0.1)({"embedding_a": first, "embedding_b": second}, {})
        assert contrastive < 1e-8
        assert triplet.ndim == 0 and info_nce.ndim == 0
        (contrastive + triplet + info_nce).backward()
        assert first.grad is not None and torch.isfinite(first.grad).all()
        with pytest.raises(ValueError, match="matrices"):
            ContrastiveLoss()(
                {"embedding_a": first.unsqueeze(0), "embedding_b": second.unsqueeze(0)},
                {"target": torch.ones(1, 4)},
            )
        with pytest.raises(ValueError, match="mean.*sum"):
            HuberLoss(reduction="none")

        configured = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.losses.HuberLoss.HuberLoss",
                "params": {"delta": 2.0, "reduction": "sum"},
            }
        )
        assert isinstance(configured, HuberLoss)

    def test_normalizations_have_expected_shapes_and_norms(self) -> None:
        vectors = torch.randn(3, 4, requires_grad=True)
        l2 = L2Norm(4)(vectors)
        scaled = ScaleNorm(4, scale=2.0)(vectors)
        assert torch.allclose(torch.linalg.vector_norm(l2, dim=-1), torch.ones(3))
        assert torch.allclose(torch.linalg.vector_norm(scaled, dim=-1), torch.full((3,), 2.0))

        image = torch.randn(2, 4, 3, 3, requires_grad=True)
        channel = ChannelLayerNorm(4)(image)
        grouped = GroupNorm(4, num_groups=2)(image)
        instance = InstanceNorm(4, dim=2)(image)
        assert channel.shape == grouped.shape == instance.shape == image.shape
        assert torch.allclose(channel.mean(dim=1), torch.zeros(2, 3, 3), atol=1e-5)
        (l2.sum() + scaled.sum() + channel.sum() + grouped.sum() + instance.sum()).backward()
        assert vectors.grad is not None and image.grad is not None

    def test_poolings_handle_empty_masks_and_have_finite_gradients(self) -> None:
        x = torch.rand(2, 4, 4, requires_grad=True)
        mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
        attention = MultiheadAttentionPooling(4, num_heads=2, num_queries=2)
        poolings = [
            GeneralizedMeanPooling(p=1.0, min_p=1e-4, learnable=False),
            ConcatMeanMaxPooling(),
            StatisticsPooling(include_min=True, include_max=True),
            attention,
        ]
        outputs = [pooling(x, mask) for pooling in poolings]
        assert [tuple(output.shape) for output in outputs] == [(2, 4), (2, 8), (2, 16), (2, 4)]
        assert torch.allclose(outputs[0][0], x[0, :2].mean(dim=0), atol=1e-5)
        assert all(torch.allclose(output[1], torch.zeros_like(output[1])) for output in outputs)
        weights = attention.attention_weights(x, mask)
        assert torch.allclose(weights[1], torch.zeros_like(weights[1]))
        sum(output.sum() for output in outputs).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_component_registry_rejects_invalid_or_accidental_replacement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ComponentRegistry, "_activations", ComponentRegistry._activations.copy()
        )
        assert ComponentRegistry.resolve_activation("hard-swish") is Hardswish
        with pytest.raises(ValueError, match="already registered"):
            ComponentRegistry.register_activation("relu", ReLU6)
        with pytest.raises(ValueError, match="empty"):
            ComponentRegistry.register_activation(" --_ ", ReLU6)
        with pytest.raises(TypeError, match="subclass"):
            ComponentRegistry.register_activation("linear", nn.Linear)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="boolean"):
            ComponentRegistry.register_activation("catalog_test", ReLU6, replace=1)  # type: ignore[arg-type]
        ComponentRegistry.register_activation("catalog_test", ReLU6)
        with pytest.raises(ValueError, match="already registered"):
            ComponentRegistry.register_activation("catalog-test", Hardswish)
        ComponentRegistry.register_activation("catalog-test", Hardswish, replace=True)
        assert ComponentRegistry.resolve_activation("catalog_test") is Hardswish
