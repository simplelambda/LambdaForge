"""Policies for image dimensions not divisible by the ViT patch size."""

from __future__ import annotations

from enum import Enum


class PatchRemainderPolicy(str, Enum):
    """Choose whether partial image patches are rejected or padded."""

    ERROR = "error"
    PAD = "pad"

    @classmethod
    def _missing_(cls, value: object) -> PatchRemainderPolicy:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"remainder_policy must be one of: {allowed}. Got {value!r}.")
