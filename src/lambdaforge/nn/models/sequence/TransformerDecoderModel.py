"""Batch-first Transformer decoder with cross-attention and causal masking."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.PositionalEncodingType import PositionalEncodingType


class TransformerDecoderModel(Model):
    """Decode target features against encoded memory with explicit mask contracts."""

    def __init__(
        self,
        target_features: int,
        memory_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        out_features: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        positional_encoding: PositionalEncodingType | str = PositionalEncodingType.SINUSOIDAL,
        max_sequence_length: int = 2048,
        norm_first: bool = True,
        causal: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(target_features, memory_features, d_model, num_heads, num_layers) < 1:
            raise ValueError("Feature sizes, num_heads and num_layers must be positive.")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if feedforward_dim is not None and feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        if max_sequence_length < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("max_sequence_length must be positive and dropout in [0, 1).")
        self.position_type = PositionalEncodingType(positional_encoding)
        self.max_sequence_length = max_sequence_length
        self.causal = causal
        self.target_projection = nn.Linear(target_features, d_model, bias=bias)
        self.memory_projection = nn.Linear(memory_features, d_model, bias=bias)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers,
            norm=nn.LayerNorm(d_model, bias=bias),
        )
        if self.position_type is PositionalEncodingType.LEARNED:
            self.position = nn.Parameter(torch.empty(1, max_sequence_length, d_model))
        else:
            self.register_parameter("position", None)
        if self.position_type is PositionalEncodingType.SINUSOIDAL:
            self.register_buffer(
                "sinusoidal_position",
                self._sinusoidal(max_sequence_length, d_model),
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
        """Initialize an optional learned positional table."""
        if self.position is not None:
            nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``target=(B,Lt,Ft)`` against ``memory=(B,Ls,Fs)``."""
        if target.ndim != 3 or memory.ndim != 3:
            raise ValueError("target and memory must be rank-three batch-first tensors.")
        if target.shape[0] != memory.shape[0]:
            raise ValueError("target and memory batch sizes must match.")
        if target.shape[1] > self.max_sequence_length:
            raise ValueError("Target sequence exceeds max_sequence_length.")
        target_padding_mask = self._padding_mask(
            target_padding_mask, target.shape[:2], target.device, "target_padding_mask"
        )
        memory_padding_mask = self._padding_mask(
            memory_padding_mask, memory.shape[:2], memory.device, "memory_padding_mask"
        )
        decoded = self.target_projection(target)
        decoded = self.dropout(decoded + self._position(decoded))
        encoded_memory = self.memory_projection(memory)
        mask = self._attention_mask(decoded, target_attention_mask)
        result = self.decoder(
            decoded,
            encoded_memory,
            tgt_mask=mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output(result)

    def _position(self, target: torch.Tensor) -> torch.Tensor:
        if self.position_type is PositionalEncodingType.NONE:
            return torch.zeros_like(target)
        table = self.position if self.position is not None else self.sinusoidal_position
        if table is None:
            raise RuntimeError("Configured positional encoding is unavailable.")
        return table[:, : target.shape[1]].to(device=target.device, dtype=target.dtype)

    def _attention_mask(
        self,
        target: torch.Tensor,
        supplied: torch.Tensor | None,
    ) -> torch.Tensor | None:
        length = target.shape[1]
        if supplied is not None and supplied.shape != (length, length):
            raise ValueError(
                "target_attention_mask must have shape (target_length, target_length)."
            )
        mask = supplied.to(target.device) if supplied is not None else None
        if self.causal:
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=target.device), diagonal=1
            )
            if mask is None:
                mask = causal
            elif mask.dtype is torch.bool:
                mask = mask | causal
            elif torch.is_floating_point(mask):
                mask = mask.to(target.dtype).masked_fill(causal, float("-inf"))
            else:
                raise TypeError("target_attention_mask must be boolean or floating point.")
        elif (
            mask is not None and mask.dtype is not torch.bool and not torch.is_floating_point(mask)
        ):
            raise TypeError("target_attention_mask must be boolean or floating point.")
        return mask

    @staticmethod
    def _padding_mask(
        mask: torch.Tensor | None,
        shape: torch.Size | tuple[int, ...],
        device: torch.device,
        name: str,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != shape:
            raise ValueError(f"{name} must match the first two input dimensions.")
        return mask.to(device=device, dtype=torch.bool)
