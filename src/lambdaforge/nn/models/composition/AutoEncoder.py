"""Composable deterministic autoencoder model."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from lambdaforge.nn.models.Model import Model


class AutoEncoder(Model):
    """Compose independently configurable encoder and decoder modules.

    The class imposes no shape or modality assumptions. Both components can be
    native LambdaForge models, ordinary PyTorch modules, or user classes built
    recursively from YAML. Optional latent and output transforms make common
    projection, constraint and post-processing steps configurable as objects.

    Parameters
    ----------
    encoder:
        Module mapping model inputs to a latent tensor.
    decoder:
        Module mapping the latent tensor to a reconstruction.
    latent_transform:
        Optional transformation applied between encoder and decoder.
    output_transform:
        Optional final reconstruction transformation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_transform: nn.Module | None = None,
        output_transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        for name, module in (("encoder", encoder), ("decoder", decoder)):
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module.")
        for optional_name, optional_module in (
            ("latent_transform", latent_transform),
            ("output_transform", output_transform),
        ):
            if optional_module is not None and not isinstance(optional_module, nn.Module):
                raise TypeError(f"{optional_name} must be a torch.nn.Module or None.")

        self.encoder = encoder
        self.decoder = decoder
        self.latent_transform = latent_transform if latent_transform is not None else nn.Identity()
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    def encode(self, *args: Any, **kwargs: Any) -> Tensor:
        """Encode arbitrary model inputs into the latent representation."""
        latent = self.encoder(*args, **kwargs)
        if not isinstance(latent, Tensor):
            raise TypeError("encoder must return a Tensor.")
        transformed = self.latent_transform(latent)
        if not isinstance(transformed, Tensor):
            raise TypeError("latent_transform must return a Tensor.")
        return transformed

    def decode(self, latent: Tensor) -> Tensor:
        """Decode a latent tensor into the reconstruction space."""
        reconstruction = self.decoder(latent)
        if not isinstance(reconstruction, Tensor):
            raise TypeError("decoder must return a Tensor.")
        transformed = self.output_transform(reconstruction)
        if not isinstance(transformed, Tensor):
            raise TypeError("output_transform must return a Tensor.")
        return transformed

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Encode and reconstruct the supplied inputs."""
        return self.decode(self.encode(*args, **kwargs))
