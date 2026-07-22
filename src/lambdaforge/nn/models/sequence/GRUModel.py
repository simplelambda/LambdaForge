"""Configurable gated recurrent unit sequence model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.SequenceOutput import SequenceOutput
from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode


class GRUModel(Model):
    """Batch-first GRU with projection and mask-aware output selection."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        out_features: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        output_mode: SequenceOutputMode | str = SequenceOutputMode.LAST,
    ) -> None:
        super().__init__()
        if in_features < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("in_features, hidden_size and num_layers must be positive.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if dropout > 0.0 and num_layers == 1:
            raise ValueError("Recurrent dropout requires num_layers greater than one.")
        self.output_mode = SequenceOutputMode(output_mode)
        self.recurrent = nn.GRU(
            in_features,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        encoded_features = hidden_size * (2 if bidirectional else 1)
        self.output = (
            nn.Linear(encoded_features, out_features, bias=bias)
            if out_features is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``x`` shaped ``(batch, length, in_features)``."""
        resolved_lengths = SequenceOutput.lengths(
            x, padding_mask, lengths, require_right_padding=True
        )
        if padding_mask is not None or lengths is not None:
            packed = pack_padded_sequence(
                x, resolved_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.recurrent(packed, initial_state)
            sequence, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.shape[1]
            )
        else:
            sequence, _ = self.recurrent(x, initial_state)
        sequence = self.output(sequence)
        return SequenceOutput.select(sequence, self.output_mode, padding_mask, resolved_lengths)
