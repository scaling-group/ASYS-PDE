"""Evaluator-owned damped Gauss--Newton fitting of the analytic density.

The exact PDE-residual Jacobian factors through 52 temporal mixture quantities
per time. Forward AD differentiates only the 48 raw polynomial values/slopes;
explicit radial sensitivities and the fixed time basis complete the chain rule.
The normal matrix is contracted through a 52-by-52 radial Gram matrix per
time, avoiding the dense radial-by-coefficient Jacobian. This is the same
Gauss--Newton algebra, without a low-rank truncation or sampled approximation.
The sole metric change is the fixed time weighting shared with loss.py.
Setup computes the inherited reference force metric without gradients or
objective trials. No optimization runs at import or construction. The evaluator's closure
checks its deadline and records every objective evaluation, including trials.
New fitted quantities remain in model.time_coefficients; inherited fitted
coefficients are frozen registered buffers in the standalone field model. The optimizer holds
only coordinate arrays and ordinary transient numerical optimization state.
"""
from __future__ import annotations

from pathlib import Path

import torch

from problem import KSHelpers, _load_module


def temporal_jacobian(model, times):
    """Jacobian of [w, log(a), w_t, (log(a))_t] versus coefficients."""
    raw, slope = model.raw_curves(times, True)
    local = torch.cat((raw, slope), dim=1)
    local_jacobian = torch.vmap(torch.func.jacfwd(model.component_bundle, argnums=0))(
        local, times
    )
    basis, basis_tau = model.time_basis(times[:, None] / KSHelpers.T_MAX, True)
    n = raw.shape[1]
    full = (local_jacobian[:, :, :n, None] * basis[:, None, None, :]
            + local_jacobian[:, :, n:, None] * basis_tau[:, None, None, :])
    return full.flatten(2)


def temporal_value_jacobian(model, times):
    """Boundary Jacobian; avoid differentiating unused temporal slopes."""
    raw = model.raw_curves(times)
    local_jacobian = torch.vmap(torch.func.jacfwd(model.value_component_bundle, argnums=0))(
        raw, times
    )
    basis = model.time_basis(times[:, None] / KSHelpers.T_MAX)
    return (local_jacobian[:, :, :, None] * basis[:, None, None, :]).flatten(2)


def radial_sensitivities(model, radii, times):
    """Exact dR/d[w, log(a), w_t, (log(a))_t], including dM.

    R=u_t-4(x*u_xx+u_x)-2*M*u_x-u**2 and x=r**2. These
    sensitivities are training algebra only, and never replace coordinate AD.
    """
    weights, widths, weights_t, log_t = model.temporal_components(times, True)
    w = weights[:, None, :]
    wt = weights_t[:, None, :]
    lt = log_t[:, None, :]
    inv = widths[:, None, :].reciprocal()
    q = radii[None, :, None].square() * inv
    e = torch.exp(-q)
    kernel = 9.5 * inv * e
    u = (w * kernel).sum(2, keepdim=True)
    ux = -(w * kernel * inv).sum(2, keepdim=True)
    mass = 4.75 * (w * (-torch.expm1(-q))).sum(2, keepdim=True)

    mass_w = 4.75 * (-torch.expm1(-q))
    rw = kernel * lt * (q - 1) - 4 * kernel * inv * (q - 1)
    rw = rw - 2 * (mass_w * ux - mass * kernel * inv) - 2 * u * kernel

    u_log_a = w * kernel * (q - 1)
    ux_log_a = -w * kernel * inv * (q - 2)
    mass_log_a = -4.75 * w * q * e
    ut_log_a = kernel * ((q - 1) * wt + w * lt * ((q - 1).square() - q))
    diffusion_log_a = 4 * w * kernel * inv * (q.square() - 4 * q + 2)
    ra = (ut_log_a - diffusion_log_a
          - 2 * (mass * ux_log_a + ux * mass_log_a) - 2 * u * u_log_a)
    return torch.cat((rw, ra, kernel, w * kernel * (q - 1)), dim=2)


def boundary_sensitivities(model, times, radius):
    """Remote u and u_r sensitivities versus weights and log widths."""
    w, a = model.temporal_components(times)
    inv = a.reciprocal()
    q = radius ** 2 * inv
    kernel = 9.5 * inv * torch.exp(-q)
    value = torch.cat((kernel, w * kernel * (q - 1)), dim=1)
    radial = torch.cat((-2 * radius * kernel * inv,
                        -2 * radius * w * kernel * inv * (q - 2)), dim=1)
    return torch.stack((value, radial), dim=1)


class DampedGaussNewton(torch.optim.LBFGS):
    """LBFGS interface supplies closure dispatch in the fixed fitting loop.

    The step itself is a Levenberg--Marquardt normal-equation solve with
    diagonal scaling, bounded proposals, and measured reduction acceptance.
    No objective or closure is evaluated during optimizer setup.
    """

    def __init__(self, model, config, grid):
        super().__init__(model.parameters(), lr=1.0, max_iter=1)
        self.model = model
        self.config = dict(config)
        self.radii, self.measure, self.times, self.boundary_times, self.time_weights = grid
        self.damping = float(config.get("gn_initial_damping", 0.01))
        self.regularization = float(config.get("coefficient_regularization", 1e-10))
        self.boundary_weight = float(config.get("boundary_weight", 1.0))
        self.max_trials = int(config.get("gn_max_trials", 5))
        self.max_coefficient_step = float(config.get("gn_max_coefficient_step", 2.0))
        self.max_curve_step = float(config.get("gn_max_curve_step", 2.0))
        self.accepted_steps = 0
        self.rejected_trials = 0

    @torch.no_grad()
    def normal_matrix(self):
        transform = temporal_jacobian(self.model, self.times)
        sensitivity = radial_sensitivities(self.model, self.radii, self.times)
        scale = self.measure / (self.measure.sum() * len(self.times))
        radial_gram = torch.bmm(
            sensitivity.transpose(1, 2), sensitivity * (self.time_weights[:, None] * scale[None, :])[:, :, None]
        )
        intermediate = torch.bmm(radial_gram, transform)
        normal = transform.flatten(0, 1).T @ intermediate.flatten(0, 1)
        normal = (normal + normal.T) / 2
        if self.boundary_weight:
            transform_bc = temporal_value_jacobian(self.model, self.boundary_times)
            sensitivity_bc = boundary_sensitivities(self.model, self.boundary_times, KSHelpers.R_MAX)
            jacobian_bc = torch.bmm(sensitivity_bc, transform_bc).flatten(0, 1)
            normal.add_(jacobian_bc.T @ jacobian_bc,
                        alpha=self.boundary_weight / len(self.boundary_times))
        n = normal.shape[0]
        normal.diagonal().add_(self.regularization / n)
        return normal

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("The fixed evaluator must provide its budgeted closure")
        # Each call runs the evaluator's budget check before numerical work.
        with torch.enable_grad():
            current_loss = closure()
        value = float(current_loss.detach())
        theta = self.model.time_coefficients
        base = theta.detach().clone()
        gradient = theta.grad.detach().flatten().clone() / 2
        normal = self.normal_matrix()
        if not bool(torch.isfinite(normal).all() and torch.isfinite(gradient).all()):
            self.damping = min(1e12, self.damping * 10)
            return current_loss
        diagonal = normal.diagonal().clamp_min(
            max(1e-14, float(normal.diagonal().max()) * 1e-12)
        )
        inv_scale = diagonal.rsqrt()
        scaled = normal * inv_scale[:, None] * inv_scale[None, :]
        scaled_gradient = gradient * inv_scale
        basis = self.model.time_basis(self.times[:, None] / KSHelpers.T_MAX)

        for _ in range(self.max_trials):
            system = scaled.clone()
            system.diagonal().add_(self.damping)
            factor, info = torch.linalg.cholesky_ex(system)
            if int(info) != 0:
                self.damping = min(1e12, self.damping * 10)
                continue
            direction = -inv_scale * torch.cholesky_solve(
                scaled_gradient[:, None], factor
            )[:, 0]
            delta = direction.reshape_as(theta)
            restriction = max(1.0, float(delta.abs().max()) / self.max_coefficient_step,
                              float((basis @ delta.T).abs().max()) / self.max_curve_step)
            direction = direction / restriction
            predicted = -float(2 * gradient.dot(direction) + direction.dot(normal @ direction))
            if not bool(torch.isfinite(direction).all()) or predicted <= 0:
                self.damping = min(1e12, self.damping * 10)
                continue
            theta.copy_(base + direction.reshape_as(theta))
            with torch.enable_grad():
                trial_loss = closure()
            trial_value = float(trial_loss.detach())
            reduction = value - trial_value
            if reduction > 0 and trial_value < float("inf"):
                ratio = reduction / predicted
                if ratio > 0.75:
                    self.damping = max(1e-12, self.damping / 3)
                elif ratio < 0.25:
                    self.damping = min(1e12, self.damping * 3)
                self.accepted_steps += 1
                return trial_loss
            theta.copy_(base)
            self.rejected_trials += 1
            self.damping = min(1e12, self.damping * 10)
        theta.copy_(base)
        return current_loss


def build_optimizer(model, config):
    device = model.time_coefficients.device
    helpers = KSHelpers(config, device)
    loss_module = _load_module(Path(__file__).with_name("loss.py"), "ks_gn_grid")
    radii, measure, times = loss_module.make_training_grid(helpers)
    boundary_times = torch.linspace(0, helpers.T_MAX, int(config.get("n_bc", 128)),
                                    dtype=helpers.DTYPE, device=device)
    time_weights = loss_module.make_time_weights(model, radii, measure, times, config)
    return DampedGaussNewton(
        model, config, (radii, measure, times, boundary_times, time_weights))
