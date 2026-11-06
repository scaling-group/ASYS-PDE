"""Fixed bounded force-conditioning of the continuous density residual.

Same quadrature and field class as cfc. At each configured training time,
E_ref=area_mean(ut_ref^2+D_ref^2+C_ref^2), with reference coefficients
frozen in registered model buffers. W=clip((mean(E_ref)/E_ref)^p,1,K),
p=.5, K=32, then W is divided by its arithmetic mean. Physics is
mean_t(W*area_mean(R^2)). This is a fixed weighted least-squares objective;
its reference never changes with fitted corrections. The raw public score
is unchanged. Mean normalization keeps the loss scale comparable, but changes
relative regularization strength; the coefficient penalty is otherwise unchanged.
IC/tangent/second jet, positivity, mass and moment constraints are structural.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def make_training_grid(helpers):
    """Coordinate quadrature only; also shared with the numerical optimizer."""
    n_r = int(helpers.config.get("n_x", 144))
    n_t = int(helpers.config.get("n_time", 64))
    _, faces, _ = helpers._radial_grid(n_r)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes = torch.as_tensor(nodes, dtype=helpers.DTYPE, device=helpers.device)
    weights = torch.as_tensor(weights, dtype=helpers.DTYPE, device=helpers.device)
    half = (faces[1:] - faces[:-1]) / 2
    radii = (faces[1:] + faces[:-1])[:, None] / 2 + half[:, None] * nodes
    measure = radii * half[:, None] * weights
    index = torch.arange(n_t, dtype=helpers.DTYPE, device=helpers.device)
    times = 0.5 * helpers.T_MAX * (1 - torch.cos(math.pi * index / (n_t - 1)))
    return radii.flatten(), measure.flatten(), times


@torch.no_grad()
def make_time_weights(model, radii, measure, times, config):
    """Fixed mathematical training metric; never a fitted trajectory lookup."""
    if not bool(config.get("force_conditioning", True)):
        return torch.ones_like(times)
    exponent = float(config.get("force_conditioning_exponent", 0.5))
    maximum = float(config.get("force_conditioning_max_weight", 32.0))
    if not 0 <= exponent <= 1 or maximum < 1:
        raise ValueError("Require exponent in [0,1] and maximum weight >= 1")
    _, ut, diffusion, chemistry, _ = model.reference_differential_parts(radii, times)
    energy = ((ut.square() + diffusion.square() + chemistry.square())
              * measure).sum(1) / measure.sum()
    average = energy.mean().clamp_min(1e-30)
    weights = (average / energy.clamp_min(average * 1e-14)).pow(exponent)
    weights = weights.clamp(min=1.0, max=maximum)
    return weights / weights.mean()


def _training_grid(model, collocation, helpers):
    _ = collocation
    cached = getattr(helpers, "_gaussian_training_grid", None)
    if cached is None:
        radii, measure, times = make_training_grid(helpers)
        weights = make_time_weights(model, radii, measure, times, helpers.config)
        cached = radii, measure, times, weights
        helpers._gaussian_training_grid = cached
    return cached


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    _ = ic_points  # Both initial constraints hold for every coefficient.
    radii, measure, times, time_weights = _training_grid(model, collocation, helpers)
    _, ut, diffusion, chemistry, _ = model.differential_parts(radii, times)
    residual = ut - diffusion - chemistry
    physics = (time_weights * (residual.square() * measure).sum(1)
               / measure.sum()).mean()

    # Origin radial derivative is zero. Check remote value/derivative decay
    # using the same field at independent boundary times.
    weights, widths = model.temporal_components(bc_points.reshape(-1))
    outer_kernel = 9.5 / widths * torch.exp(-helpers.R_MAX ** 2 / widths)
    outer = (weights * outer_kernel).sum(1)
    outer_dr = (-2 * helpers.R_MAX * weights * outer_kernel / widths).sum(1)
    boundary = outer.square().mean() + outer_dr.square().mean()
    regularizer = model.time_coefficients.square().mean()
    return (physics + float(helpers.config.get("boundary_weight", 1.0)) * boundary +
            float(helpers.config.get("coefficient_regularization", 1e-10)) * regularizer)
