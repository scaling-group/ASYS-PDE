"""Continuous radial-area residual with analytic mixture mass/derivatives.

Training uses four-point Gauss quadrature in each configured logarithmic
radial cell and a Chebyshev-Lobatto time grid over the finite window.
IC/tangent/second time jet, positivity, mass and virial constraints are structural.
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


def _training_grid(collocation, helpers):
    _ = collocation
    cached = getattr(helpers, "_gaussian_training_grid", None)
    if cached is None:
        cached = make_training_grid(helpers)
        helpers._gaussian_training_grid = cached
    return cached


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    _ = ic_points  # Both initial constraints hold for every coefficient.
    radii, measure, times = _training_grid(collocation, helpers)
    _, ut, diffusion, chemistry, _ = model.differential_parts(radii, times)
    residual = ut - diffusion - chemistry
    physics = ((residual.square() * measure).sum(1) / measure.sum()).mean()

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
