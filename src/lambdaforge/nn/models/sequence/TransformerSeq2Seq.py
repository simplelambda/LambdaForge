"""Composable Transformer encoder-decoder for continuous sequence features."""

from __future__ import annotations

import torch

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.PositionalEncodingType import PositionalEncodingType
from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode
from lambdaforge.nn.models.sequence.TransformerDecoderModel import TransformerDecoderModel
from lambdaforge.nn.models.sequence.TransformerEncoderModel import TransformerEncoderModel


class TransformerSeq2Seq(Model):
    """Encode a source sequence and causally decode a target sequence."""

    def __init__(
        self,
        source_features: int,
        target_features: int,
        out_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        positional_encoding: PositionalEncodingType | str = PositionalEncodingType.SINUSOIDAL,
        max_sequence_length: int = 2048,
        norm_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(source_features, target_features, out_features, encoder_layers) < 1:
            raise ValueError("Feature sizes and encoder_layers must be positive.")
        self.encoder = TransformerEncoderModel(
            in_features=source_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            out_features=None,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            positional_encoding=positional_encoding,
            max_sequence_length=max_sequence_length,
            output_mode=SequenceOutputMode.SEQUENCE,
            norm_first=norm_first,
            bias=bias,
        )
        self.decoder = TransformerDecoderModel(
            target_features=target_features,
            memory_features=d_model,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=decoder_layers,
            out_features=out_features,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            positional_encoding=positional_encoding,
            max_sequence_length=max_sequence_length,
            norm_first=norm_first,
            causal=True,
            bias=bias,
        )

    def encode(
        self,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoder memory for ``source=(B,Ls,source_features)``."""
        if source.ndim != 3:
            raise ValueError("source must be a rank-three batch-first tensor.")
        return self.encoder(source, padding_mask=source_padding_mask)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target features against precomputed encoder memory."""
        return self.decoder(
            target,
            memory,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=source_padding_mask,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``source`` and return one prediction for every target position."""
        memory = self.encode(source, source_padding_mask)
        return self.decode(target, memory, target_padding_mask, source_padding_mask)
