"""Continuous radial-area residual with analytic two-mode kernel jets.

Training uses four-point Gauss quadrature in each configured logarithmic
radial cell and a Chebyshev-Lobatto time grid over the finite window.
IC/tangent/second time jet, positivity, mass and virial constraints are structural.
The scalar raw objective is unchanged. Its first derivative uses the exact
analytic radial VJP and ordinary AD through the small temporal graph. This
avoids retaining a large radial backpropagation graph. It is first-order
training algebra; the field and its coordinate/parameter AD are unchanged.
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
    components = model.temporal_components(times, True)
    weights, widths, weights_t, width_log_t, beta, beta_t = components
    with torch.no_grad():
        residual, sensitivity = model.residual_sensitivities(
            radii, times, tuple(x.detach() for x in components))
        scale = measure / (measure.sum() * len(times))
        physics_value = (residual.square() * scale).sum()
        local_gradient = 2 * (sensitivity * residual[:, :, None] * scale[None, :, None]).sum(1)
    bundle = torch.cat((weights, widths.log(), weights_t, width_log_t, beta, beta_t), dim=1)
    proxy = (bundle * local_gradient).sum()
    physics = physics_value + (proxy - proxy.detach())

    # Origin radial derivative is zero. Check remote value/derivative decay
    # using the same field at independent boundary times.
    weights, widths, beta = model.temporal_components(bc_points.reshape(-1))
    inv, _, outer_kernel, kq, _, _, _ = model.kernel_bundle(
        widths.new_tensor([helpers.R_MAX]), widths, beta)
    outer_kernel, kq, inv = [x[:, 0] for x in (outer_kernel, kq, inv)]
    outer = (weights * outer_kernel).sum(1)
    outer_dr = (2 * helpers.R_MAX * weights * inv * kq).sum(1)
    boundary = outer.square().mean() + outer_dr.square().mean()
    regularizer = model.time_coefficients.square().mean()
    shape4, shape8 = model.shape_coefficients.split(model.n_core + 1)
    return (physics + float(helpers.config.get("boundary_weight", 1.0)) * boundary +
            float(helpers.config.get("coefficient_regularization", 1e-10)) * regularizer +
            float(helpers.config.get("shape_regularization", 1e-10)) * shape4.square().mean() +
            float(helpers.config.get("shape8_regularization", 1e-10)) * shape8.square().mean())
