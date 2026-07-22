"""Output representations exposed by the native Vision Transformer."""

from __future__ import annotations

from enum import Enum


class VisionTransformerOutputMode(str, Enum):
    """Select the spatial or pooled representation returned by ViT."""

    CLASS_TOKEN = "class_token"
    MEAN = "mean"
    TOKENS = "tokens"
    FEATURE_MAP = "feature_map"

    @classmethod
    def _missing_(cls, value: object) -> VisionTransformerOutputMode:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"output_mode must be one of: {allowed}. Got {value!r}.")
