"""Feature Tokenizer Transformer for heterogeneous tabular data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model


class FTTransformer(Model):
    """Tokenize continuous and categorical features before Transformer encoding."""

    def __init__(
        self,
        num_continuous_features: int,
        categorical_cardinalities: Sequence[int],
        out_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        activation: type[Activation] | str = "gelu",
        activation_kwargs: dict[str, Any] | None = None,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        tokenizer_bias: bool = True,
        initialization_std: float = 0.02,
        enable_nested_tensor: bool = False,
    ) -> None:
        super().__init__()
        cardinalities = [int(value) for value in categorical_cardinalities]
        if num_continuous_features < 0 or any(value < 1 for value in cardinalities):
            raise ValueError("Feature count must be non-negative and cardinalities positive.")
        if num_continuous_features + len(cardinalities) == 0:
            raise ValueError("At least one continuous or categorical feature is required.")
        if min(out_features, d_model, num_heads, num_layers) < 1:
            raise ValueError("Output size, d_model, num_heads and num_layers must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if feedforward_dim is not None and feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        if layer_norm_eps <= 0 or initialization_std <= 0:
            raise ValueError("layer_norm_eps and initialization_std must be positive.")
        self.num_continuous_features = num_continuous_features
        self.categorical_cardinalities = tuple(cardinalities)
        self.continuous_weight = (
            nn.Parameter(torch.empty(num_continuous_features, d_model))
            if num_continuous_features
            else None
        )
        self.continuous_bias = (
            nn.Parameter(torch.empty(num_continuous_features, d_model))
            if num_continuous_features and tokenizer_bias
            else None
        )
        self.continuous_missing = (
            nn.Parameter(torch.empty(num_continuous_features, d_model))
            if num_continuous_features
            else None
        )
        self.category_embedding: nn.Embedding | None
        if cardinalities:
            offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(dim=0)
            self.register_buffer("categorical_offsets", offsets, persistent=True)
            self.category_embedding = nn.Embedding(sum(cardinalities), d_model)
            self.categorical_bias = (
                nn.Parameter(torch.empty(len(cardinalities), d_model)) if tokenizer_bias else None
            )
            self.categorical_missing = nn.Parameter(torch.empty(len(cardinalities), d_model))
        else:
            self.register_buffer("categorical_offsets", None, persistent=False)
            self.category_embedding = None
            self.register_parameter("categorical_bias", None)
            self.register_parameter("categorical_missing", None)
        self.class_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        activation_cls = ComponentRegistry.resolve_activation(activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=feedforward_dim or 4 * d_model,
            dropout=dropout,
            activation=activation_cls(**(activation_kwargs or {})),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias),
            enable_nested_tensor=enable_nested_tensor,
        )
        self.head_normalization = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.head = nn.Linear(d_model, out_features, bias=bias)
        self.initialization_std = initialization_std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize tokenizer parameters without altering Transformer defaults."""
        for parameter in (
            self.continuous_weight,
            self.continuous_bias,
            self.continuous_missing,
            self.categorical_bias,
            self.categorical_missing,
            self.class_token,
        ):
            if parameter is not None:
                nn.init.normal_(parameter, std=self.initialization_std)
        if self.category_embedding is not None:
            nn.init.normal_(self.category_embedding.weight, std=self.initialization_std)

    def forward(
        self,
        continuous: torch.Tensor | None = None,
        categorical: torch.Tensor | None = None,
        continuous_mask: torch.Tensor | None = None,
        categorical_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict from continuous floats and zero-based categorical indices."""
        tokens: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.num_continuous_features:
            if continuous is None:
                raise ValueError("continuous is required for configured continuous features.")
            if continuous.ndim != 2 or continuous.shape[1] != self.num_continuous_features:
                raise ValueError("continuous must have shape (batch, num_continuous_features).")
            if self.continuous_weight is None or self.continuous_missing is None:
                raise RuntimeError("Continuous tokenizer parameters are unavailable.")
            batch_size = continuous.shape[0]
            values = continuous
            if continuous_mask is not None:
                valid = self._mask(
                    continuous_mask, continuous.shape, continuous.device, "continuous_mask"
                )
                values = values.masked_fill(~valid, 0.0)
            continuous_tokens = values.unsqueeze(-1) * self.continuous_weight
            if self.continuous_bias is not None:
                continuous_tokens = continuous_tokens + self.continuous_bias
            if continuous_mask is not None:
                continuous_tokens = torch.where(
                    valid.unsqueeze(-1), continuous_tokens, self.continuous_missing.unsqueeze(0)
                )
            tokens.append(continuous_tokens)
        elif continuous is not None:
            raise ValueError("continuous was supplied but no continuous features are configured.")

        if self.categorical_cardinalities:
            if categorical is None:
                raise ValueError("categorical is required for configured categorical features.")
            if categorical.ndim != 2:
                raise ValueError("categorical must have shape (batch, categorical_features).")
            expected = (categorical.shape[0], len(self.categorical_cardinalities))
            if categorical.shape != expected:
                raise ValueError("categorical has an incompatible feature dimension.")
            category_embedding = cast(nn.Embedding, self.category_embedding)
            categorical_offsets = cast(torch.Tensor | None, self.categorical_offsets)
            if categorical_offsets is None:
                raise RuntimeError("Categorical tokenizer parameters are unavailable.")
            if self.categorical_missing is None:
                raise RuntimeError("Categorical missing-value parameters are unavailable.")
            if categorical.dtype not in (torch.int32, torch.int64):
                raise TypeError("categorical must use an integer dtype.")
            if batch_size is not None and categorical.shape[0] != batch_size:
                raise ValueError("continuous and categorical batch sizes must match.")
            batch_size = categorical.shape[0]
            valid_categories = (
                self._mask(
                    categorical_mask, categorical.shape, categorical.device, "categorical_mask"
                )
                if categorical_mask is not None
                else torch.ones_like(categorical, dtype=torch.bool)
            )
            safe_categories = categorical.masked_fill(~valid_categories, 0).to(torch.long)
            for index, cardinality in enumerate(self.categorical_cardinalities):
                values = safe_categories[:, index][valid_categories[:, index]]
                if values.numel() and bool(((values < 0) | (values >= cardinality)).any()):
                    raise ValueError(f"categorical feature {index} contains an out-of-range index.")
            categorical_tokens = category_embedding(
                safe_categories + categorical_offsets.unsqueeze(0)
            )
            if self.categorical_bias is not None:
                categorical_tokens = categorical_tokens + self.categorical_bias
            if categorical_mask is not None:
                categorical_tokens = torch.where(
                    valid_categories.unsqueeze(-1),
                    categorical_tokens,
                    self.categorical_missing.unsqueeze(0),
                )
            tokens.append(categorical_tokens)
        elif categorical is not None:
            raise ValueError("categorical was supplied but no categorical features are configured.")

        if batch_size is None:
            raise RuntimeError("No configured input tensor was supplied.")
        feature_tokens = torch.cat(tokens, dim=1)
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoded = self.encoder(
            self.embedding_dropout(torch.cat([class_token, feature_tokens], dim=1))
        )
        return self.head(self.head_normalization(encoded[:, 0]))

    @staticmethod
    def _mask(
        mask: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if mask.shape != shape:
            raise ValueError(f"{name} must match its input tensor shape.")
        return mask.to(device=device, dtype=torch.bool)
