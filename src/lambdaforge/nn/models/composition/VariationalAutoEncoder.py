"""Composable variational autoencoder model."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from lambdaforge.nn.models.Model import Model


class VariationalAutoEncoder(Model):
    """Compose an encoder, diagonal Gaussian latent space and decoder.

    ``mean_head`` and ``log_variance_head`` are fully injectable modules. For
    the common linear case, omit both and provide ``encoder_features`` and
    ``latent_features``; the class creates both projections automatically.

    The forward mapping contains ``reconstruction``, ``mean``,
    ``log_variance``, ``latent`` and per-sample ``kl_divergence`` tensors. In
    training mode reparameterization samples by default; evaluation uses the
    posterior mean unless ``sample_in_eval=True`` or ``sample`` is supplied to
    :meth:`forward`.
    """

    output_schema = {
        "reconstruction": "Tensor[...]",
        "mean": "Tensor[..., latent]",
        "log_variance": "Tensor[..., latent]",
        "latent": "Tensor[..., latent]",
        "kl_divergence": "Tensor[...]",
    }

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mean_head: nn.Module | None = None,
        log_variance_head: nn.Module | None = None,
        encoder_features: int | None = None,
        latent_features: int | None = None,
        latent_transform: nn.Module | None = None,
        output_transform: nn.Module | None = None,
        minimum_log_variance: float | None = -30.0,
        maximum_log_variance: float | None = 20.0,
        sample_in_eval: bool = False,
        noise_scale: float = 1.0,
        latent_dimension: int = -1,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        for name, module in (("encoder", encoder), ("decoder", decoder)):
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module.")
        for optional_name, optional_module in (
            ("mean_head", mean_head),
            ("log_variance_head", log_variance_head),
            ("latent_transform", latent_transform),
            ("output_transform", output_transform),
        ):
            if optional_module is not None and not isinstance(optional_module, nn.Module):
                raise TypeError(f"{optional_name} must be a torch.nn.Module or None.")
        if (mean_head is None) != (log_variance_head is None):
            raise ValueError("mean_head and log_variance_head must be supplied together.")
        if mean_head is None:
            for dimension_name, dimension_value in (
                ("encoder_features", encoder_features),
                ("latent_features", latent_features),
            ):
                if (
                    isinstance(dimension_value, bool)
                    or not isinstance(dimension_value, int)
                    or dimension_value < 1
                ):
                    raise ValueError(
                        f"{dimension_name} must be a positive integer when latent heads "
                        "are omitted."
                    )
            assert encoder_features is not None
            assert latent_features is not None
            mean_head = nn.Linear(encoder_features, latent_features, bias=head_bias)
            log_variance_head = nn.Linear(
                encoder_features,
                latent_features,
                bias=head_bias,
            )
        if latent_features is not None and (
            isinstance(latent_features, bool)
            or not isinstance(latent_features, int)
            or latent_features < 1
        ):
            raise ValueError("latent_features must be a positive integer or None.")
        for bound_name, bound_value in (
            ("minimum_log_variance", minimum_log_variance),
            ("maximum_log_variance", maximum_log_variance),
        ):
            if bound_value is not None and not math.isfinite(float(bound_value)):
                raise ValueError(f"{bound_name} must be finite or None.")
        if (
            minimum_log_variance is not None
            and maximum_log_variance is not None
            and float(minimum_log_variance) >= float(maximum_log_variance)
        ):
            raise ValueError("minimum_log_variance must be smaller than maximum_log_variance.")
        if (
            isinstance(noise_scale, bool)
            or not math.isfinite(float(noise_scale))
            or float(noise_scale) < 0
        ):
            raise ValueError("noise_scale must be a finite non-negative number.")
        if isinstance(latent_dimension, bool) or not isinstance(latent_dimension, int):
            raise TypeError("latent_dimension must be an integer.")

        assert mean_head is not None
        assert log_variance_head is not None
        self.encoder = encoder
        self.decoder = decoder
        self.mean_head = mean_head
        self.log_variance_head = log_variance_head
        self.latent_transform = latent_transform if latent_transform is not None else nn.Identity()
        self.output_transform = output_transform if output_transform is not None else nn.Identity()
        self.latent_features = latent_features
        self.minimum_log_variance = minimum_log_variance
        self.maximum_log_variance = maximum_log_variance
        self.sample_in_eval = sample_in_eval
        self.noise_scale = float(noise_scale)
        self.latent_dimension = latent_dimension

    def encode_distribution(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Return posterior mean and clamped log-variance tensors."""
        encoded = self.encoder(*args, **kwargs)
        if not isinstance(encoded, Tensor):
            raise TypeError("encoder must return a Tensor.")
        mean = self.mean_head(encoded)
        log_variance = self.log_variance_head(encoded)
        if not isinstance(mean, Tensor) or not isinstance(log_variance, Tensor):
            raise TypeError("latent heads must return tensors.")
        if mean.shape != log_variance.shape:
            raise ValueError("mean_head and log_variance_head must return identical shapes.")
        if self.minimum_log_variance is not None or self.maximum_log_variance is not None:
            log_variance = torch.clamp(
                log_variance,
                min=self.minimum_log_variance,
                max=self.maximum_log_variance,
            )
        return mean, log_variance

    def reparameterize(
        self,
        mean: Tensor,
        log_variance: Tensor,
        sample: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample differentiably from the posterior or return its mean."""
        if mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have identical shapes.")
        should_sample = (self.training or self.sample_in_eval) if sample is None else sample
        if not should_sample or self.noise_scale == 0.0:
            latent = mean
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            latent = mean + self.noise_scale * torch.exp(0.5 * log_variance) * noise
        transformed = self.latent_transform(latent)
        if not isinstance(transformed, Tensor):
            raise TypeError("latent_transform must return a Tensor.")
        return transformed

    def decode(self, latent: Tensor) -> Tensor:
        """Decode and optionally transform a latent tensor."""
        reconstruction = self.decoder(latent)
        if not isinstance(reconstruction, Tensor):
            raise TypeError("decoder must return a Tensor.")
        transformed = self.output_transform(reconstruction)
        if not isinstance(transformed, Tensor):
            raise TypeError("output_transform must return a Tensor.")
        return transformed

    def kl_divergence(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        """Return ``KL(q(z|x) || N(0, I))`` summed over the latent axis."""
        if mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have identical shapes.")
        return -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(
            dim=self.latent_dimension
        )

    def forward(
        self,
        *args: Any,
        sample: bool | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Return reconstruction and every tensor needed by a VAE objective."""
        mean, log_variance = self.encode_distribution(*args, **kwargs)
        latent = self.reparameterize(mean, log_variance, sample=sample, generator=generator)
        return {
            "reconstruction": self.decode(latent),
            "mean": mean,
            "log_variance": log_variance,
            "latent": latent,
            "kl_divergence": self.kl_divergence(mean, log_variance),
        }
