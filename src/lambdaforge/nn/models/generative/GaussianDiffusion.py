"""Trainable Gaussian diffusion process with DDPM and DDIM sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.generative.DiffusionSchedule import DiffusionSchedule
from lambdaforge.nn.models.Model import Model


class GaussianDiffusion(Model):
    """Wrap an injected timestep-conditioned denoiser and diffusion schedule."""

    def __init__(
        self,
        denoiser: nn.Module,
        schedule: DiffusionSchedule | None = None,
        prediction_type: str = "noise",
        denoiser_output_key: str | None = None,
        clip_sample: float | None = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module.")
        if schedule is not None and not isinstance(schedule, DiffusionSchedule):
            raise TypeError("schedule must be a DiffusionSchedule or None.")
        if prediction_type not in {"noise", "sample"}:
            raise ValueError("prediction_type must be noise or sample.")
        if clip_sample is not None and clip_sample <= 0:
            raise ValueError("clip_sample must be positive when provided.")
        self.denoiser = denoiser
        self.schedule = schedule or DiffusionSchedule()
        self.prediction_type = prediction_type
        self.denoiser_output_key = denoiser_output_key
        self.clip_sample = clip_sample

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``q(x_t | x_0)`` and return both noisy sample and exact noise."""
        if not torch.is_floating_point(clean) or clean.ndim < 2:
            raise TypeError("clean must be a rank-two-or-higher floating tensor.")
        sampled_noise = noise
        if sampled_noise is None:
            sampled_noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        if sampled_noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean.")
        signal = self.schedule.extract(self.schedule.sqrt_alpha_cumulative, timesteps, clean)
        noise_scale = self.schedule.extract(
            self.schedule.sqrt_one_minus_alpha_cumulative, timesteps, clean
        )
        return signal * clean + noise_scale * sampled_noise, sampled_noise

    def forward(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        conditioning: Any = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return noisy inputs, denoiser prediction and the configured training target."""
        resolved_timesteps = timesteps
        if resolved_timesteps is None:
            resolved_timesteps = self.schedule.sample_timesteps(
                clean.shape[0], clean.device, generator
            )
        noisy, sampled_noise = self.add_noise(clean, resolved_timesteps, noise, generator)
        prediction = self._denoise(noisy, resolved_timesteps, conditioning)
        if prediction.shape != clean.shape:
            raise ValueError("denoiser output must have the same shape as the clean sample.")
        target = sampled_noise if self.prediction_type == "noise" else clean
        return {
            "prediction": prediction,
            "target": target,
            "noisy_sample": noisy,
            "noise": sampled_noise,
            "timesteps": resolved_timesteps,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        conditioning: Any = None,
        method: str = "ddpm",
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate samples by the full DDPM or deterministic/stochastic DDIM chain."""
        if method not in {"ddpm", "ddim"}:
            raise ValueError("method must be ddpm or ddim.")
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) < 2 or min(dimensions) < 1:
            raise ValueError("shape must contain at least batch and feature dimensions.")
        parameter = next(self.denoiser.parameters(), None)
        resolved_device = device or (parameter.device if parameter is not None else "cpu")
        resolved_dtype = dtype or (parameter.dtype if parameter is not None else torch.float32)
        current = torch.randn(
            dimensions,
            device=resolved_device,
            dtype=resolved_dtype,
            generator=generator,
        )
        for index in reversed(range(self.schedule.timesteps)):
            timestep = torch.full((dimensions[0],), index, device=resolved_device, dtype=torch.long)
            prediction = self._denoise(current, timestep, conditioning)
            alpha_bar = self.schedule.extract(self.schedule.alpha_cumulative, timestep, current)
            alpha_bar_previous = self.schedule.extract(
                self.schedule.alpha_cumulative_previous, timestep, current
            )
            if self.prediction_type == "noise":
                predicted_noise = prediction
                predicted_clean = (
                    current
                    - self.schedule.extract(
                        self.schedule.sqrt_one_minus_alpha_cumulative,
                        timestep,
                        current,
                    )
                    * predicted_noise
                ) / alpha_bar.sqrt().clamp_min(1e-12)
            else:
                predicted_clean = prediction
                predicted_noise = (current - alpha_bar.sqrt() * predicted_clean) / (
                    1.0 - alpha_bar
                ).sqrt().clamp_min(1e-12)
            if self.clip_sample is not None:
                predicted_clean = predicted_clean.clamp(-self.clip_sample, self.clip_sample)
            if index == 0:
                current = predicted_clean
                continue
            noise = torch.randn(
                current.shape,
                device=current.device,
                dtype=current.dtype,
                generator=generator,
            )
            if method == "ddim":
                sigma = (
                    eta
                    * (
                        (1.0 - alpha_bar_previous)
                        / (1.0 - alpha_bar)
                        * (1.0 - alpha_bar / alpha_bar_previous)
                    )
                    .clamp_min(0.0)
                    .sqrt()
                )
                direction = (1.0 - alpha_bar_previous - sigma.square()).clamp_min(0.0).sqrt()
                current = (
                    alpha_bar_previous.sqrt() * predicted_clean
                    + direction * predicted_noise
                    + sigma * noise
                )
            else:
                alpha = self.schedule.extract(self.schedule.alphas, timestep, current)
                beta = self.schedule.extract(self.schedule.betas, timestep, current)
                mean = (current - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / (
                    alpha.sqrt()
                )
                variance = self.schedule.extract(
                    self.schedule.posterior_variance, timestep, current
                )
                current = mean + variance.sqrt() * noise
        return current

    def _denoise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: Any,
    ) -> torch.Tensor:
        result = (
            self.denoiser(noisy, timesteps)
            if conditioning is None
            else self.denoiser(noisy, timesteps, conditioning)
        )
        if isinstance(result, Mapping):
            key = self.denoiser_output_key or "prediction"
            if key not in result:
                raise KeyError(f"Denoiser output has no {key!r} tensor.")
            result = result[key]
        if not torch.is_tensor(result):
            raise TypeError("denoiser must return a tensor or configured mapping.")
        return result
