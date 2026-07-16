"""Core model and sparse-operation smoke tests."""

import pytest
import torch

from lambdaforge.nn import CNN2D, MLP, Aggregation, BatchedKNN, Scatter


class TestModels:
    """Verify generic shapes, YAML-friendly aliases and validation."""

    def test_mlp_accepts_registered_string_components(self) -> None:
        model = MLP(4, 2, hidden=[8, 8], activation="gelu", normalization="layernorm")
        output = model(torch.randn(3, 4))
        assert output.shape == (3, 2)

    def test_cnn_default_normalization_accepts_nchw(self) -> None:
        model = CNN2D(3, 5, hidden_channels=[7])
        output = model(torch.randn(2, 3, 8, 8))
        assert output.shape == (2, 5, 8, 8)

    def test_model_predict_restores_training_mode(self) -> None:
        model = MLP(2, 1)
        model.train()
        model.predict(torch.ones(1, 2))
        assert model.training is True

    def test_scatter_mean_and_softmax(self) -> None:
        index = torch.tensor([0, 0, 1], dtype=torch.long)
        values = torch.tensor([[1.0], [3.0], [8.0]])
        assert torch.equal(
            Scatter.reduce(values, index, 2, Aggregation.MEAN), torch.tensor([[2.0], [8.0]])
        )
        weights = Scatter.segment_softmax(torch.tensor([0.0, 0.0, 1.0]), index, 2)
        assert torch.allclose(weights, torch.tensor([0.5, 0.5, 1.0]))

    def test_batched_knn_has_fixed_k_shape(self) -> None:
        points = torch.tensor([[[0.0], [1.0], [3.0]]])
        indices, distances = BatchedKNN(k=4)(points, points)
        assert indices.shape == distances.shape == (1, 3, 4)

    def test_rejects_per_layer_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            MLP(4, 1, hidden=[8, 8], activation=["relu"])
