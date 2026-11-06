"""Spatial operators for measuring equations; no PDE time evolution."""

import math
import torch

DTYPE = torch.float64


def derivative(value, points, axis):
    if not value.requires_grad:
        return points[:, axis : axis + 1] * 0
    gradient = torch.autograd.grad(
        value.sum(),
        points,
        create_graph=True,
        allow_unused=True,
    )[0]
    return (
        points[:, axis : axis + 1] * 0
        if gradient is None
        else gradient[:, axis : axis + 1]
    )


def periodic_operators(values, length):
    """Periodic H[f] and zero-mean primitive of H[f] on the last axis."""
    count = values.shape[-1]
    frequency = (
        2 * math.pi * torch.fft.fftfreq(count, d=length / count, device=values.device)
    )
    spectrum = torch.fft.fft(values, dim=-1)
    hilbert_multiplier = -1j * torch.sign(frequency)
    if count % 2 == 0:
        hilbert_multiplier[count // 2] = 0
    hilbert_spectrum = spectrum * hilbert_multiplier
    safe_frequency = torch.where(frequency == 0, torch.ones_like(frequency), frequency)
    primitive_spectrum = hilbert_spectrum / (1j * safe_frequency)
    primitive_spectrum = primitive_spectrum * (frequency != 0)
    return torch.fft.ifft(hilbert_spectrum, dim=-1).real, torch.fft.ifft(
        primitive_spectrum, dim=-1
    ).real


def radial_grid(count, radius, device):
    """Geometric finite-volume faces, including the origin, and cell centers."""
    outer = torch.logspace(-4, math.log10(radius), count, dtype=DTYPE, device=device)
    faces = torch.cat([outer.new_zeros(1), outer])
    centers = torch.sqrt((faces[:-1].square() + faces[1:].square()) / 2)
    measure = (faces[1:].square() - faces[:-1].square()) / 2
    return faces, centers, measure


def radial_rhs(values, faces, centers, measure):
    """Conservative KS diffusion/chemotaxis, r*u_r + u*M, divided by cell measure."""
    mass = torch.cumsum(values * measure, dim=-1)
    face_density = (values[..., 1:] + values[..., :-1]) / 2
    slope = (values[..., 1:] - values[..., :-1]) / (centers[1:] - centers[:-1])
    inner_diffusion = faces[1:-1] * slope
    inner_chemotaxis = face_density * mass[..., :-1]
    # Symmetry gives zero flux at r=0; outer decay is zero exterior density.
    outer_diffusion = -faces[-1] * values[..., -1:] / (faces[-1] - centers[-1])
    diffusion_flux = torch.cat(
        [values[..., :1] * 0, inner_diffusion, outer_diffusion], dim=-1
    )
    chemotaxis_flux = torch.cat(
        [values[..., :1] * 0, inner_chemotaxis, values[..., :1] * 0], dim=-1
    )
    return torch.diff(diffusion_flux, dim=-1) / measure, torch.diff(
        chemotaxis_flux, dim=-1
    ) / measure


def geometric_relative(residual, terms):
    denominator = sum(term.abs() for term in terms) + 1e-10
    return torch.exp(torch.log((residual.abs() / denominator).clamp_min(1e-30)).mean())
