"""Entmax's sparse support must not backpropagate through unused zero roots."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lambdaforge.nn.activations import Entmax15

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def legacy_forward(x: Tensor, dim: int) -> Tensor:
    """Freeze the original forward, not its numerically undefined backward."""
    x = x / 2
    x = x - x.amax(dim=dim, keepdim=True)
    ordered = x.sort(dim=dim, descending=True).values
    shape = [1] * x.ndim
    shape[dim] = x.shape[dim]
    ranks = torch.arange(1, x.shape[dim] + 1, dtype=x.dtype, device=x.device).view(shape)
    mean = ordered.cumsum(dim) / ranks
    variance = ranks * (ordered.square().cumsum(dim) / ranks - mean.square())
    thresholds = mean - ((1 - variance) / ranks).clamp_min(0).sqrt()
    size = (thresholds <= ordered).sum(dim=dim, keepdim=True).clamp_min(1)
    return (x - thresholds.gather(dim, size - 1)).clamp_min(0).square()


def analytic_vjp(p: Tensor, upstream: Tensor, dim: int) -> Tensor:
    """J = diag(s) - ss^T / sum(s), with s=sqrt(p), Peters et al. (2019)."""
    s = p.sqrt()
    return s * (upstream - (s * upstream).sum(dim, keepdim=True) / s.sum(dim, keepdim=True))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("force_float32", [False, True])
def test_reported_tied_sparse_case_has_exact_forward_and_finite_correct_gradient(
    device, dtype, force_float32
):
    x = torch.tensor([[0.0, 0.0, -2.0, -2.0]], device=device, dtype=dtype, requires_grad=True)
    upstream = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device, dtype=dtype)
    p = Entmax15(dim=1, force_float32=force_float32)(x)
    work = (
        x.detach().float()
        if force_float32 and dtype in (torch.float16, torch.bfloat16)
        else x.detach()
    )
    assert torch.equal(p.detach(), legacy_forward(work, 1).to(dtype))
    (p * upstream).sum().backward()
    assert torch.isfinite(x.grad).all()
    expected = torch.tensor([[-(2**-1.5), 2**-1.5, 0.0, 0.0]], device=device, dtype=dtype)
    torch.testing.assert_close(x.grad, expected, rtol=0.01, atol=0.002)
    assert torch.equal(p.detach()[:, 2:], torch.zeros_like(p[:, 2:]))
    assert torch.equal(x.grad[:, 2:], torch.zeros_like(x.grad[:, 2:]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
def test_arbitrary_axes_noncontiguous_batches_match_analytic_jacobian(device, dim):
    generator = torch.Generator().manual_seed(63)
    x = (torch.randn(3, 5, 7, generator=generator, dtype=torch.float64) * 3).to(device)
    x = x.transpose(0, 2).requires_grad_()
    upstream = torch.randn(x.shape, generator=generator, dtype=torch.float64).to(device)
    p = Entmax15(dim=dim)(x)
    gradient = torch.autograd.grad(p, x, upstream)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, analytic_vjp(p.detach(), upstream, dim))
    torch.testing.assert_close(p.sum(dim), torch.ones_like(p.sum(dim)))
    torch.testing.assert_close(
        gradient.sum(dim), torch.zeros_like(gradient.sum(dim)), atol=1e-12, rtol=0
    )
    axis = dim if dim >= 0 else x.ndim + dim
    assert torch.equal(p.detach(), legacy_forward(x.detach(), axis))


@pytest.mark.parametrize(
    "values",
    [
        [[0.0, 0.0, -2.0, -2.0]],  # unused discriminant exactly zero, valid support interior
        [[0.0, 0.0, 0.0, 0.0]],
        [[0.2, 0.1, -0.1, -4.0]],
        [[8.0, 0.0, -8.0, -8.0]],
    ],
)
def test_first_and_second_order_finite_differences_away_from_support_transitions(values):
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    activation = Entmax15(dim=1)
    assert torch.autograd.gradcheck(activation, (x,))
    assert torch.autograd.gradgradcheck(activation, (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("values", [[[0.0, -2.0]], [[0.0, -2.0, -2.0]], [[0.0]]])
def test_singleton_and_exact_support_boundary_have_finite_first_derivatives(device, values):
    # Entmax is C1 here; no C2 assertion at an actual support transition.
    x = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(Entmax15(), (x,))


@pytest.mark.parametrize("device", DEVICES)
def test_shared_gate_parameter_remains_finite_after_optimizer_step(device):
    # Disposable synthetic optimization, not a scientific consumer Work or dataset.
    logits = torch.nn.Parameter(torch.tensor([[0.0, 0.0, -2.0, -2.0]], device=device))
    optimizer = torch.optim.SGD([logits], lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        weights = Entmax15(dim=1)(logits)
        (weights * torch.arange(4, device=device)).sum().backward()
        assert torch.isfinite(logits.grad).all()
        optimizer.step()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(Entmax15(dim=1)(logits)).all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_input_is_not_silently_repaired(invalid):
    x = torch.tensor([[invalid, 0.0, -2.0]])
    # This repair addresses the derivative of unused *valid* roots, not corrupt logits.
    assert not torch.isfinite(Entmax15()(x)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_negative_infinity_attention_mask_has_zero_weight_and_gradient(device):
    x = torch.tensor([[0.0, 0.0, -float("inf"), -float("inf")]], device=device, requires_grad=True)
    p = Entmax15()(x)
    upstream = torch.arange(4, dtype=x.dtype, device=device)[None, :]
    gradient = torch.autograd.grad(p, x, upstream)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, analytic_vjp(p.detach(), upstream, -1))
    assert torch.equal(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))
    assert torch.equal(p.detach(), legacy_forward(x.detach(), 1))
