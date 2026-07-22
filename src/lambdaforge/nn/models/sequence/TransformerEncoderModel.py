"""Configurable Transformer encoder for batch-first sequences."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.PositionalEncodingType import PositionalEncodingType
from lambdaforge.nn.models.sequence.SequenceOutput import SequenceOutput
from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode


class TransformerEncoderModel(Model):
    """Transformer encoder with masks, positional encoding and output reduction."""

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        out_features: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        activation: type[Activation] | str = "gelu",
        activation_kwargs: dict[str, Any] | None = None,
        positional_encoding: PositionalEncodingType | str = PositionalEncodingType.SINUSOIDAL,
        max_sequence_length: int = 2048,
        use_class_token: bool = False,
        output_mode: SequenceOutputMode | str = SequenceOutputMode.MEAN,
        norm_first: bool = True,
        final_normalization: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        causal: bool = False,
        enable_nested_tensor: bool = False,
    ) -> None:
        super().__init__()
        if in_features < 1 or d_model < 1 or num_layers < 1 or num_heads < 1:
            raise ValueError("Feature sizes, num_heads and num_layers must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if feedforward_dim is not None and feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive.")

        self.position_type = PositionalEncodingType(positional_encoding)
        self.output_mode = SequenceOutputMode(output_mode)
        self.max_sequence_length = max_sequence_length
        self.use_class_token = bool(use_class_token)
        self.causal = bool(causal)
        self.input = nn.Linear(in_features, d_model, bias=bias)
        activation_cls = ComponentRegistry.resolve_activation(activation)
        activation_module = activation_cls(**(activation_kwargs or {}))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 4 * d_model,
            dropout=dropout,
            activation=activation_module,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )
        final_norm = (
            nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias) if final_normalization else None
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=final_norm,
            enable_nested_tensor=enable_nested_tensor,
        )
        self.class_token = nn.Parameter(torch.empty(1, 1, d_model)) if use_class_token else None
        total_positions = max_sequence_length + int(use_class_token)
        if self.position_type is PositionalEncodingType.LEARNED:
            self.position = nn.Parameter(torch.empty(1, total_positions, d_model))
        else:
            self.register_parameter("position", None)
        if self.position_type is PositionalEncodingType.SINUSOIDAL:
            self.register_buffer(
                "sinusoidal_position",
                self._sinusoidal(total_positions, d_model),
                persistent=True,
            )
        else:
            self.register_buffer("sinusoidal_position", None, persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.output = (
            nn.Linear(d_model, out_features, bias=bias)
            if out_features is not None
            else nn.Identity()
        )
        self.reset_parameters()

    @staticmethod
    def _sinusoidal(length: int, features: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, features, 2, dtype=torch.float32) * (-math.log(10000.0) / features)
        )
        encoding = torch.zeros(1, length, features)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        if features > 1:
            encoding[0, :, 1::2] = torch.cos(positions * frequencies[: features // 2])
        return encoding

    def reset_parameters(self) -> None:
        """Initialize optional learned sequence tokens."""
        if self.class_token is not None:
            nn.init.normal_(self.class_token, std=0.02)
        if self.position is not None:
            nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``x`` shaped ``(batch, length, in_features)``."""
        input_padding_mask = SequenceOutput.resolve_padding_mask(x, padding_mask)
        SequenceOutput.lengths(x, input_padding_mask, None)
        batch_size, sequence_length, _ = x.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError("Input sequence exceeds max_sequence_length.")
        encoded = self.input(x)
        encoder_padding_mask = input_padding_mask
        if self.class_token is not None:
            encoded = torch.cat([self.class_token.expand(batch_size, -1, -1), encoded], dim=1)
            if input_padding_mask is not None:
                prefix = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
                encoder_padding_mask = torch.cat([prefix, input_padding_mask], dim=1)

        if self.position is not None:
            encoded = encoded + self.position[:, : encoded.shape[1]].to(dtype=encoded.dtype)
        elif self.sinusoidal_position is not None:
            encoded = encoded + self.sinusoidal_position[:, : encoded.shape[1]].to(
                device=encoded.device, dtype=encoded.dtype
            )
        encoded = self.dropout(encoded)
        resolved_attention_mask = self._attention_mask(encoded, attention_mask)
        encoded = self.encoder(
            encoded,
            mask=resolved_attention_mask,
            src_key_padding_mask=encoder_padding_mask,
        )

        if self.class_token is not None and self.output_mode is SequenceOutputMode.FIRST:
            return self.output(encoded[:, 0])
        if self.class_token is not None:
            encoded = encoded[:, 1:]
        selected = SequenceOutput.select(encoded, self.output_mode, input_padding_mask)
        return self.output(selected)

    def _attention_mask(
        self, encoded: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        total_length = encoded.shape[1]
        resolved = attention_mask
        if resolved is not None:
            if resolved.dtype is not torch.bool and not torch.is_floating_point(resolved):
                raise TypeError("attention_mask must use boolean or floating-point dtype.")
            expected_without_token = total_length - int(self.class_token is not None)
            if (
                resolved.shape == (expected_without_token, expected_without_token)
                and self.class_token is not None
            ):
                expanded = torch.zeros(
                    (total_length, total_length), dtype=resolved.dtype, device=resolved.device
                )
                expanded[1:, 1:] = resolved
                resolved = expanded
            elif resolved.shape != (total_length, total_length):
                raise ValueError("attention_mask has an incompatible shape.")
            resolved = resolved.to(
                device=encoded.device,
                dtype=encoded.dtype if torch.is_floating_point(resolved) else torch.bool,
            )
        if self.causal:
            causal = torch.triu(
                torch.ones(total_length, total_length, dtype=torch.bool, device=encoded.device),
                diagonal=1,
            )
            if self.class_token is not None:
                causal[0] = False
            if resolved is None:
                resolved = causal
            elif resolved.dtype is torch.bool:
                resolved = resolved | causal
            else:
                resolved = resolved.masked_fill(causal, float("-inf"))
        return resolved
