"""Convolution-augmented Transformer encoder for long local/global context."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.SequenceOutput import SequenceOutput
from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode


class ConformerModel(Model):
    """Apply macaron feed-forward, attention and depthwise convolution blocks."""

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        out_features: int | None = None,
        feedforward_dim: int | None = None,
        convolution_kernel_size: int = 31,
        dropout: float = 0.1,
        output_mode: SequenceOutputMode | str = SequenceOutputMode.SEQUENCE,
        causal: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_features, d_model, num_heads, num_layers, convolution_kernel_size) < 1:
            raise ValueError("Feature sizes, layers, heads and kernel size must be positive.")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if convolution_kernel_size % 2 == 0:
            raise ValueError("convolution_kernel_size must be odd for length preservation.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        hidden = feedforward_dim or 4 * d_model
        if hidden < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        self.output_mode = SequenceOutputMode(output_mode)
        self.causal = causal
        self.input = nn.Linear(in_features, d_model, bias=bias)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ffn1": nn.Sequential(
                            nn.LayerNorm(d_model, bias=bias),
                            nn.Linear(d_model, hidden, bias=bias),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden, d_model, bias=bias),
                            nn.Dropout(dropout),
                        ),
                        "attention_norm": nn.LayerNorm(d_model, bias=bias),
                        "attention": nn.MultiheadAttention(
                            d_model,
                            num_heads,
                            dropout=dropout,
                            bias=bias,
                            batch_first=True,
                        ),
                        "attention_dropout": nn.Dropout(dropout),
                        "convolution_norm": nn.LayerNorm(d_model, bias=bias),
                        "pointwise_in": nn.Conv1d(d_model, 2 * d_model, 1, bias=bias),
                        "depthwise": nn.Conv1d(
                            d_model,
                            d_model,
                            convolution_kernel_size,
                            padding=convolution_kernel_size // 2,
                            groups=d_model,
                            bias=bias,
                        ),
                        "batch_norm": nn.BatchNorm1d(d_model),
                        "pointwise_out": nn.Conv1d(d_model, d_model, 1, bias=bias),
                        "convolution_dropout": nn.Dropout(dropout),
                        "ffn2": nn.Sequential(
                            nn.LayerNorm(d_model, bias=bias),
                            nn.Linear(d_model, hidden, bias=bias),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden, d_model, bias=bias),
                            nn.Dropout(dropout),
                        ),
                        "output_norm": nn.LayerNorm(d_model, bias=bias),
                    }
                )
            )
        self.output = (
            nn.Linear(d_model, out_features, bias=bias)
            if out_features is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``x=(B,L,F)`` with optional true-for-padding mask."""
        resolved_mask = SequenceOutput.resolve_padding_mask(x, padding_mask)
        SequenceOutput.lengths(x, resolved_mask, None)
        encoded = self.input(x)
        attention_mask = None
        if self.causal:
            attention_mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device),
                diagonal=1,
            )
        for raw_block in self.blocks:
            block = cast(nn.ModuleDict, raw_block)
            encoded = encoded + 0.5 * block["ffn1"](encoded)
            normalized = block["attention_norm"](encoded)
            attended, _ = block["attention"](
                normalized,
                normalized,
                normalized,
                key_padding_mask=resolved_mask,
                attn_mask=attention_mask,
                need_weights=False,
            )
            encoded = encoded + block["attention_dropout"](attended)
            convolution = block["convolution_norm"](encoded).transpose(1, 2)
            convolution = block["pointwise_in"](convolution)
            convolution = nn.functional.glu(convolution, dim=1)
            convolution = block["depthwise"](convolution)
            convolution = nn.functional.silu(block["batch_norm"](convolution))
            convolution = block["pointwise_out"](convolution).transpose(1, 2)
            encoded = encoded + block["convolution_dropout"](convolution)
            encoded = block["output_norm"](encoded + 0.5 * block["ffn2"](encoded))
        selected = SequenceOutput.select(encoded, self.output_mode, resolved_mask)
        return self.output(selected)
