"""Tests for composable and implicit neural models."""

import math

import pytest
import torch
import yaml
from torch import nn

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.models.composition import (
    AutoEncoder,
    EnsembleModel,
    EnsembleReduction,
    MixtureOfExperts,
    MultiTaskModel,
    SiameseMerge,
    SiameseModel,
    VariationalAutoEncoder,
)
from lambdaforge.nn.models.implicit import SIREN


class TestCompositionModels:
    """Verify tensor contracts, gradients, routing and YAML composition."""

    def test_autoencoder_shape_stages_and_gradients(self) -> None:
        model = AutoEncoder(
            encoder=nn.Linear(4, 2),
            decoder=nn.Linear(2, 4),
            latent_transform=nn.Tanh(),
        )
        x = torch.randn(5, 4)
        assert model.encode(x).shape == (5, 2)
        output = model(x)
        assert output.shape == x.shape
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_vae_train_eval_reparameterization_and_gradients(self) -> None:
        model = VariationalAutoEncoder(
            encoder=nn.Linear(4, 6),
            decoder=nn.Linear(3, 4),
            encoder_features=6,
            latent_features=3,
        )
        x = torch.randn(7, 4)

        model.eval()
        first = model(x)
        second = model(x)
        assert set(first) == {
            "reconstruction",
            "mean",
            "log_variance",
            "latent",
            "kl_divergence",
        }
        assert first["reconstruction"].shape == (7, 4)
        assert first["mean"].shape == first["log_variance"].shape == (7, 3)
        assert first["kl_divergence"].shape == (7,)
        assert torch.equal(first["latent"], first["mean"])
        assert torch.equal(first["latent"], second["latent"])

        model.train()
        generator = torch.Generator().manual_seed(17)
        sampled = model(x, generator=generator)
        assert not torch.equal(sampled["latent"], sampled["mean"])
        deterministic = model(x, sample=False)
        assert torch.equal(deterministic["latent"], deterministic["mean"])
        loss = sampled["reconstruction"].square().mean() + sampled["kl_divergence"].mean()
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    def test_weighted_ensemble_exposes_normalized_trainable_weights(self) -> None:
        first = nn.Linear(2, 1, bias=False)
        second = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            first.weight.fill_(1.0)
            second.weight.fill_(3.0)
        model = EnsembleModel(
            [first, second],
            reduction=EnsembleReduction.WEIGHTED_MEAN,
            weights=[0.25, 0.75],
            learnable_weights=True,
        )
        x = torch.ones(4, 2)
        output = model(x)
        assert torch.allclose(output, torch.full((4, 1), 5.0))
        assert torch.allclose(model.model_weights(), torch.tensor([0.25, 0.75]))
        assert torch.allclose(model.model_weights().sum(), torch.tensor(1.0))
        output.sum().backward()
        assert model.weight_logits.grad is not None

        stacked = EnsembleModel(
            [nn.Identity(), nn.Identity()],
            reduction=EnsembleReduction.STACK,
            stack_dimension=1,
        )(x)
        concatenated = EnsembleModel(
            [nn.Identity(), nn.Identity()],
            reduction="concatenate",
            concatenate_dimension=-1,
        )(x)
        assert stacked.shape == (4, 2, 2)
        assert concatenated.shape == (4, 4)

    def test_mixture_of_experts_dense_sparse_balance_and_gradients(self) -> None:
        dense_gate = nn.Linear(2, 3, bias=False)
        experts = [nn.Linear(2, 1, bias=False) for _ in range(3)]
        with torch.no_grad():
            dense_gate.weight.zero_()
            for index, expert in enumerate(experts, start=1):
                expert.weight.fill_(float(index))
        dense = MixtureOfExperts(experts, dense_gate)
        x = torch.ones(5, 2)
        assert torch.allclose(dense.routing_weights(x), torch.full((5, 3), 1.0 / 3.0))
        assert torch.allclose(dense(x), torch.full((5, 1), 4.0))
        assert torch.allclose(dense.load_balance_loss(x, apply_weight=False), torch.tensor(1.0))

        sparse_gate = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            sparse_gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]))
        sparse = MixtureOfExperts(
            [nn.Linear(2, 2) for _ in range(3)],
            sparse_gate,
            temperature=0.7,
            learnable_temperature=True,
            top_k=2,
            balance_loss_weight=0.1,
        )
        inputs = torch.tensor([[2.0, 1.0], [-1.0, 2.0], [0.5, -0.25]])
        weights = sparse.routing_weights(inputs)
        assert weights.shape == (3, 3)
        assert torch.equal((weights > 0).sum(dim=-1), torch.full((3,), 2))
        assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
        output = sparse(inputs)
        balance = sparse.load_balance_loss(inputs)
        assert output.shape == (3, 2)
        assert balance.ndim == 0 and torch.isfinite(balance)
        (output.square().mean() + balance).backward()
        assert sparse_gate.weight.grad is not None
        assert sparse.log_temperature.grad is not None

    def test_top_one_routing_keeps_main_loss_gate_gradient(self) -> None:
        gate = nn.Linear(2, 2, bias=False)
        experts = [nn.Linear(2, 1, bias=False), nn.Linear(2, 1, bias=False)]
        with torch.no_grad():
            gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
            experts[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            experts[1].weight.copy_(torch.tensor([[0.0, 2.0]]))
        model = MixtureOfExperts(experts, gate, top_k=1)
        inputs = torch.tensor([[2.0, 1.0], [-2.0, 1.0]])

        weights = model.routing_weights(inputs)
        assert torch.equal((weights > 0).sum(dim=-1), torch.ones(2, dtype=torch.long))
        model(inputs).square().mean().backward()
        assert gate.weight.grad is not None
        assert bool((gate.weight.grad.abs() > 0).any())

    def test_weighted_composition_preserves_prediction_dtype(self) -> None:
        inputs = torch.ones(3, 2, dtype=torch.float16)
        ensemble = EnsembleModel(
            [nn.Identity(), nn.Identity()],
            reduction="weighted_mean",
            weights=[0.25, 0.75],
        )
        assert ensemble(inputs).dtype is torch.float16

        gate = nn.Linear(2, 2, bias=False, dtype=torch.float16)
        mixture = MixtureOfExperts([nn.Identity(), nn.Identity()], gate, top_k=1)
        assert mixture(inputs).dtype is torch.float16

    def test_multitask_mapping_and_detached_task(self) -> None:
        model = MultiTaskModel(
            backbone=nn.Linear(4, 5),
            heads={"classification": nn.Linear(5, 3), "regression": nn.Linear(5, 1)},
            detach_backbone_for=["regression"],
            return_features=True,
            features_key="shared",
        )
        x = torch.randn(6, 4)
        output = model(x)
        assert output["classification"].shape == (6, 3)
        assert output["regression"].shape == (6, 1)
        assert output["shared"].shape == (6, 5)

        model.zero_grad(set_to_none=True)
        model.forward_task("regression", x).sum().backward()
        assert all(parameter.grad is None for parameter in model.backbone.parameters())
        assert all(
            parameter.grad is not None for parameter in model.heads["regression"].parameters()
        )

    def test_siamese_merges_head_and_gradients(self) -> None:
        model = SiameseModel(
            encoder=nn.Linear(3, 2),
            merge=SiameseMerge.CONCATENATE,
            head=nn.Linear(4, 1),
        )
        left = torch.randn(5, 3)
        right = torch.randn(5, 3)
        output = model(left, right)
        assert output.shape == (5, 1)
        output.sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

        cosine = SiameseModel(nn.Identity(), merge="cosine", keep_distance_dimension=True)
        identical = cosine(left, left)
        assert identical.shape == (5, 1)
        assert torch.allclose(identical, torch.ones_like(identical), atol=1e-6)

    def test_nested_yaml_builds_injected_mixture(self) -> None:
        spec = yaml.safe_load(
            """
            target: lambdaforge.nn.models.composition.MixtureOfExperts.MixtureOfExperts
            params:
              experts:
                - target: torch.nn.Linear
                  params: {in_features: 3, out_features: 2}
                - target: torch.nn.Linear
                  params: {in_features: 3, out_features: 2}
              gate:
                target: torch.nn.Linear
                params: {in_features: 3, out_features: 2}
              temperature: 0.8
              top_k: 1
              balance_loss_weight: 0.02
            """
        )
        model = ObjectFactory.build(spec)
        assert isinstance(model, MixtureOfExperts)
        assert model(torch.randn(4, 3)).shape == (4, 2)
        assert torch.equal(
            (model.routing_weights(torch.randn(4, 3)) > 0).sum(dim=-1),
            torch.ones(4, dtype=torch.long),
        )

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: EnsembleModel([], reduction="mean"),
            lambda: EnsembleModel([nn.Identity()], reduction="mean", weights=[1.0]),
            lambda: MixtureOfExperts([nn.Identity()], nn.Identity(), top_k=2),
            lambda: VariationalAutoEncoder(
                nn.Identity(),
                nn.Identity(),
                mean_head=nn.Identity(),
            ),
            lambda: MultiTaskModel(nn.Identity(), {"invalid.name": nn.Identity()}),
            lambda: SiameseModel(nn.Identity(), merge="unknown"),
        ],
    )
    def test_validation_rejects_invalid_composition(self, constructor: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            constructor()  # type: ignore[operator]


class TestSIREN:
    """Verify SIREN architecture, initialization and differentiability."""

    def test_shapes_activations_and_coordinate_gradients(self) -> None:
        model = SIREN(
            in_features=2,
            out_features=3,
            hidden=[8, 6],
            first_omega=15.0,
            hidden_omega=[20.0],
        )
        coordinates = torch.randn(7, 2, requires_grad=True)
        output = model(coordinates)
        activations = model.activations(coordinates)
        assert output.shape == (7, 3)
        assert [value.shape for value in activations] == [(7, 8), (7, 6), (7, 3)]
        output.square().mean().backward()
        assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    def test_paper_initialization_bounds_and_configured_reset(self) -> None:
        model = SIREN(
            in_features=2,
            out_features=1,
            hidden=[4, 5],
            first_omega=10.0,
            hidden_omega=[20.0],
            bias=False,
        )
        expected = [0.5, math.sqrt(6.0 / 4.0) / 20.0, math.sqrt(6.0 / 5.0) / 20.0]
        for linear, bound in zip(model.linears, expected, strict=True):
            assert float(linear.weight.detach().abs().max()) <= bound + 1e-7

        configured = SIREN(
            2,
            1,
            hidden=[3, 3],
            hidden_omega=[5.0],
            first_weight_bound=0.1,
            hidden_weight_bounds=[0.2],
            output_weight_bound=0.3,
            bias_bounds=0.0,
        )
        generator = torch.Generator().manual_seed(99)
        configured.reset_parameters(generator)
        first_state = [parameter.detach().clone() for parameter in configured.parameters()]
        configured.reset_parameters(torch.Generator().manual_seed(99))
        assert all(
            torch.equal(before, after)
            for before, after in zip(first_state, configured.parameters(), strict=True)
        )
        assert all(
            linear.bias is None or torch.count_nonzero(linear.bias) == 0
            for linear in configured.linears
        )

    def test_nonlinear_output_and_yaml_factory(self) -> None:
        spec = yaml.safe_load(
            """
            target: lambdaforge.nn.models.implicit.SIREN.SIREN
            params:
              in_features: 2
              out_features: 1
              hidden: [12, 10]
              first_omega: 25.0
              hidden_omega: [18.0]
              output_omega: 7.0
              outermost_linear: false
              output_transform:
                target: torch.nn.Tanh
            """
        )
        model = ObjectFactory.build(spec)
        assert isinstance(model, SIREN)
        output = model(torch.randn(9, 2))
        assert output.shape == (9, 1)
        assert bool((output.abs() <= 1.0).all())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"in_features": 0, "out_features": 1},
            {"in_features": 2, "out_features": 1, "hidden": 0},
            {"in_features": 2, "out_features": 1, "first_omega": 0.0},
            {
                "in_features": 2,
                "out_features": 1,
                "hidden": [4, 4, 4],
                "hidden_omega": [30.0],
            },
        ],
    )
    def test_validation_rejects_invalid_siren(self, kwargs: dict[str, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            SIREN(**kwargs)  # type: ignore[arg-type]
