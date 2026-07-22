"""Mask-aware sequence output selection."""

from __future__ import annotations

import torch

from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode


class SequenceOutput:
    """Validate masks and reduce batch-first sequence representations."""

    @staticmethod
    def resolve_padding_mask(
        sequence: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate a padding mask and move it to the sequence device."""
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, length, features).")
        if padding_mask is None:
            return None
        if padding_mask.dtype is not torch.bool:
            raise TypeError("padding_mask must use boolean dtype.")
        if padding_mask.shape != sequence.shape[:2]:
            raise ValueError("padding_mask must have shape (batch, length).")
        return padding_mask.to(device=sequence.device)

    @classmethod
    def lengths(
        cls,
        sequence: torch.Tensor,
        padding_mask: torch.Tensor | None,
        lengths: torch.Tensor | None,
        *,
        require_right_padding: bool = False,
    ) -> torch.Tensor:
        """Return validated positive sequence lengths on ``sequence.device``."""
        resolved_padding_mask = cls.resolve_padding_mask(sequence, padding_mask)
        batch_size, sequence_length, _ = sequence.shape
        mask_lengths: torch.Tensor | None = None
        if resolved_padding_mask is not None:
            if require_right_padding and sequence_length > 1:
                becomes_valid_again = resolved_padding_mask[:, :-1] & ~resolved_padding_mask[:, 1:]
                if bool(becomes_valid_again.any()):
                    raise ValueError("Recurrent padding_mask values must be right-padded.")
            mask_lengths = (~resolved_padding_mask).sum(dim=1, dtype=torch.long)

        if lengths is None:
            resolved = mask_lengths
        else:
            if lengths.ndim != 1 or lengths.shape[0] != batch_size:
                raise ValueError("lengths must have shape (batch,).")
            if lengths.dtype not in (torch.int32, torch.int64):
                raise TypeError("lengths must use an integer dtype.")
            resolved = lengths.to(device=sequence.device, dtype=torch.long)
            if mask_lengths is not None and not torch.equal(resolved, mask_lengths):
                raise ValueError("lengths and padding_mask describe different valid elements.")

        if resolved is None:
            resolved = torch.full(
                (batch_size,), sequence_length, dtype=torch.long, device=sequence.device
            )
        if bool((resolved < 1).any()) or bool((resolved > sequence_length).any()):
            raise ValueError("Every sequence length must be within [1, sequence_length].")
        return resolved

    @classmethod
    def select(
        cls,
        sequence: torch.Tensor,
        mode: SequenceOutputMode | str,
        padding_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return all states or a mask-aware fixed-size representation."""
        resolved_mode = SequenceOutputMode(mode)
        resolved_padding_mask = cls.resolve_padding_mask(sequence, padding_mask)
        resolved_lengths = cls.lengths(sequence, resolved_padding_mask, lengths)
        if resolved_mode is SequenceOutputMode.SEQUENCE:
            if resolved_padding_mask is None:
                return sequence
            return sequence.masked_fill(resolved_padding_mask.unsqueeze(-1), 0.0)
        if resolved_mode is SequenceOutputMode.FIRST:
            return sequence[:, 0]
        if resolved_mode is SequenceOutputMode.LAST:
            indices = (resolved_lengths - 1).view(-1, 1, 1)
            indices = indices.expand(-1, 1, sequence.shape[-1])
            return sequence.gather(1, indices).squeeze(1)

        if resolved_padding_mask is None:
            positions = torch.arange(sequence.shape[1], device=sequence.device)
            valid = positions.unsqueeze(0) < resolved_lengths.unsqueeze(1)
        else:
            valid = ~resolved_padding_mask
        if resolved_mode is SequenceOutputMode.MEAN:
            summed = sequence.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1)
            return summed / resolved_lengths.to(sequence.dtype).unsqueeze(-1)
        masked = sequence.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return masked.max(dim=1).values
