"""Executable contracts for the tabular-through-conformance roadmap families."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from lambdaforge.data import CategoricalFeatureEncoder
from lambdaforge.experiments import ObjectFactory
from lambdaforge.nn import (
    SAINT,
    ArchitectureConformanceCase,
    ArchitectureConformancePack,
    AutoInt,
    ConformalPredictionInterval,
    ConformerModel,
    DeepFM,
    DeepONet,
    DiffusionSchedule,
    FourierNeuralOperator1D,
    GaussianDiffusion,
    NeuralCDE,
    NeuralODE,
    StateSpaceAdapter,
    TabNet,
    TemperatureScaler,
    TensorFieldNetwork,
    TransformerDecoderModel,
    TransformerSeq2Seq,
    VariationalAutoEncoderLoss,
    VectorQuantizedAutoEncoder,
)


class ExponentialDynamics(nn.Module):
    """Tiny autonomous ODE field used by the scientific-model tests."""

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return state


class ConstantCDEField(nn.Module):
    """Return a constant learned-free CDE vector field."""

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return torch.ones(state.shape[0], state.shape[1], 2, device=state.device)


class ZeroDenoiser(nn.Module):
    """Shape-preserving denoiser fixture with one trainable scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return torch.zeros_like(sample) + self.offset


class SequenceFirstProvider(nn.Module):
    """Sequence-first provider fixture for the optional state-space adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.projection(sequence)


class TestRoadmapThirteenToSeventeen:
    """Keep every completed roadmap family public, differentiable and shape-safe."""

    def test_categorical_preprocessing_is_stable_and_handles_unknowns(self) -> None:
        encoder = CategoricalFeatureEncoder()
        encoded = encoder.fit_transform([["red", None], ["blue", "small"]])
        state = encoder.state_dict()
        restored = CategoricalFeatureEncoder()
        restored.load_state_dict(state)

        assert encoded.tolist() == [[2, 0], [1, 1]]
        assert encoder.cardinalities == (3, 2)
        assert restored.transform([["missing", "small"]]).tolist() == [[0, 1]]

    def test_tabular_research_models_are_yaml_buildable_and_differentiable(self) -> None:
        continuous = torch.randn(4, 3, requires_grad=True)
        categorical = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])
        tabnet = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.TabNet",
                "params": {"in_features": 3, "out_features": 2, "hidden_features": 8},
            }
        )
        models = (
            tabnet,
            SAINT(3, [3, 3], 2, d_model=8, num_heads=2),
            AutoInt(3, [3, 3], 2, embedding_features=8, num_heads=2),
            DeepFM(3, [3, 3], 2, embedding_features=4, hidden_features=(8,)),
        )

        for model in models:
            output = (
                model(continuous) if isinstance(model, TabNet) else model(continuous, categorical)
            )
            assert output.shape == (4, 2)
            output.sum().backward(retain_graph=True)
        _, masks = tabnet.forward_with_masks(continuous)
        assert len(masks) == 3
        assert torch.allclose(masks[0].sum(dim=-1), torch.ones(4))

    def test_long_sequence_models_and_optional_adapter_preserve_contracts(self) -> None:
        source = torch.randn(2, 7, 3, requires_grad=True)
        target = torch.randn(2, 5, 4, requires_grad=True)
        decoder = TransformerDecoderModel(
            4, 3, d_model=8, num_heads=2, num_layers=2, out_features=6
        )
        seq2seq = TransformerSeq2Seq(
            3, 4, 6, d_model=8, num_heads=2, encoder_layers=2, decoder_layers=2
        )
        conformer = ConformerModel(3, d_model=8, num_heads=2, num_layers=2, out_features=6)
        adapter = StateSpaceAdapter(SequenceFirstProvider(), module_batch_first=False)

        outputs = (decoder(target, source), seq2seq(source, target), conformer(source))
        assert [tuple(output.shape) for output in outputs] == [(2, 5, 6), (2, 5, 6), (2, 7, 6)]
        assert adapter(source).shape == (2, 7, 5)
        sum(output.sum() for output in outputs).backward()

    def test_generative_objectives_diffusion_and_uncertainty(self) -> None:
        autoencoder = VectorQuantizedAutoEncoder(
            nn.Linear(4, 3), nn.Linear(3, 4), num_codes=8, code_features=3
        )
        vq_output = autoencoder(torch.randn(5, 4))
        objective = VariationalAutoEncoderLoss(beta=0.5)
        vae_loss = objective(
            {
                "reconstruction": torch.zeros(2, 4, requires_grad=True),
                "mean": torch.zeros(2, 3, requires_grad=True),
                "log_variance": torch.zeros(2, 3, requires_grad=True),
            },
            {"target": torch.ones(2, 4)},
        )
        total = vq_output["reconstruction"].mean() + vq_output["quantization_loss"] + vae_loss
        total.backward()
        assert vq_output["code_indices"].shape == (5,)
        assert vq_output["perplexity"].ndim == 0

        diffusion = GaussianDiffusion(ZeroDenoiser(), DiffusionSchedule(4, kind="linear"))
        diffusion_output = diffusion(torch.randn(2, 3))
        assert diffusion_output["prediction"].shape == (2, 3)
        assert diffusion.sample((2, 3), method="ddpm").shape == (2, 3)
        assert diffusion.sample((2, 3), method="ddim").shape == (2, 3)

        scaler = TemperatureScaler().fit(
            torch.tensor([[4.0, -1.0], [-1.0, 4.0]]),
            torch.tensor([0, 1]),
            max_iterations=2,
        )
        assert scaler.temperature.item() > 0
        intervals = ConformalPredictionInterval().fit(
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[1.1], [2.2], [2.7]]),
        )(torch.tensor([[4.0]]))
        assert intervals["lower"].item() <= 4 <= intervals["upper"].item()

    def test_scientific_models_have_numerical_and_equivariance_contracts(self) -> None:
        ode = NeuralODE(ExponentialDynamics(), steps_per_interval=8)
        trajectory = ode(torch.ones(2, 3), torch.tensor([0.0, 0.5, 1.0]))
        assert trajectory.shape == (2, 3, 3)
        assert math.isclose(trajectory[0, -1, 0].item(), math.e, rel_tol=1e-5)

        cde = NeuralCDE(ConstantCDEField(), 2, 3, initial_encoder=nn.Linear(2, 3))
        assert cde(torch.randn(2, 5, 2), torch.arange(5.0)).shape == (2, 3)
        operator = DeepONet(nn.Linear(4, 6), nn.Linear(2, 6), 3, out_features=2)
        assert operator(torch.randn(2, 4), torch.randn(7, 2)).shape == (2, 7, 2)
        fno = FourierNeuralOperator1D(2, 3, width=8, modes=4, num_layers=2)
        field = fno(torch.randn(2, 16, 2, requires_grad=True))
        field.sum().backward()
        assert field.shape == (2, 16, 3)

        tensor_field = TensorFieldNetwork(2, 3, 1, 2).eval()
        scalars = torch.randn(4, 2)
        vectors = torch.randn(4, 1, 3)
        coordinates = torch.randn(4, 3)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        original = tensor_field(scalars, vectors, edges, coordinates)
        transformed = tensor_field(
            scalars,
            vectors @ rotation.T,
            edges,
            coordinates @ rotation.T + torch.tensor([2.0, -1.0, 0.5]),
        )
        assert torch.allclose(original["scalars"], transformed["scalars"], atol=1e-5)
        assert torch.allclose(original["vectors"] @ rotation.T, transformed["vectors"], atol=1e-5)

    def test_architecture_conformance_reference_round_trip(self, tmp_path: Path) -> None:
        inputs = (torch.tensor([[1.0, 2.0, 3.0]]),)
        factory = lambda: nn.Linear(3, 2)  # noqa: E731
        case = ArchitectureConformanceCase.capture(
            name="tiny-linear",
            model_factory=factory,
            inputs=inputs,
            source="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html",
            seed=7,
        )
        reference = case.write_reference(tmp_path / "tiny-linear.pt")
        restored = ArchitectureConformanceCase.from_reference(
            reference,
            model_factory=factory,
            inputs=inputs,
        )
        pack = ArchitectureConformancePack("tiny-reference", [restored])

        result = restored.run()
        pack.assert_conformant()

        assert result.passed
        assert result.parameter_count == 8
        assert result.state_checksum.startswith("sha256:")
