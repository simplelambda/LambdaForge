"""Float32 matrix-multiplication precision values."""

from enum import Enum


class MatmulPrecision(str, Enum):
    """Name the precision modes supported by PyTorch float32 matmuls."""

    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"
