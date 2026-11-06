"""Evaluator-owned damped Gauss--Newton fitting of the analytic density.

The exact PDE-residual Jacobian factors through 104 temporal mixture quantities
per time. Forward AD differentiates the 48 raw polynomial values/slopes and
26 constant shape inputs. Explicit radial sensitivities and the fixed
time basis complete the chain rule.
The normal matrix is contracted through a 104-by-104 radial Gram matrix per
time, avoiding the dense radial-by-coefficient Jacobian. This is the same
Gauss--Newton algebra, without a low-rank truncation or sampled approximation.
No optimizer work runs at import or construction. The evaluator's closure
checks its deadline and records every objective evaluation, including trials.
All fitted quantities remain in registered model parameters. The optimizer holds
only coordinate arrays and ordinary transient numerical optimization state.
"""
from __future__ import annotations

from pathlib import Path

import torch

from problem import KSHelpers, _load_module


def temporal_jacobian(model, times):
    """Jacobian of [w, log(a), w_t, log(a)_t, beta, beta_t]."""
    raw, slope = model.raw_curves(times, True)
    local = torch.cat((raw, slope, model.shape_coefficients[None].expand(len(times), -1)), dim=1)
    local_jacobian = torch.vmap(torch.func.jacfwd(model.component_bundle, argnums=0))(
        local, times
    )
    basis, basis_tau = model.time_basis(times[:, None] / KSHelpers.T_MAX, True)
    n = raw.shape[1]
    full = (local_jacobian[:, :, :n, None] * basis[:, None, None, :]
            + local_jacobian[:, :, n:2*n, None] * basis_tau[:, None, None, :])
    return torch.cat((full.flatten(2), local_jacobian[:, :, 2*n:]), dim=2)


def temporal_value_jacobian(model, times):
    """Boundary Jacobian; avoid differentiating unused temporal slopes."""
    raw = model.raw_curves(times)
    local_jacobian = torch.vmap(torch.func.jacfwd(model.value_component_bundle, argnums=0))(
        torch.cat((raw, model.shape_coefficients[None].expand(len(times), -1)), dim=1), times
    )
    basis = model.time_basis(times[:, None] / KSHelpers.T_MAX)
    n = raw.shape[1]
    full = (local_jacobian[:, :, :n, None] * basis[:, None, None, :]).flatten(2)
    return torch.cat((full, local_jacobian[:, :, n:]), dim=2)


def radial_sensitivities(model, radii, times):
    """Exact dR/d[w, log(a), w_t, log(a)_t, beta, beta_t], including dM.

    R=u_t-4(x*u_xx+u_x)-2*M*u_x-u**2 and x=r**2. These
    sensitivities are training algebra only, and never replace coordinate AD.
    """
    return model.residual_sensitivities(radii, times)[1]


def boundary_sensitivities(model, times, radius):
    """Remote u and u_r sensitivities versus weights, log widths, shapes."""
    w, a, beta = model.temporal_components(times)
    inv, q, kernel, kq, kqq, kb, _, _, kqb, _, _, _ = model.kernel_bundle(
        a.new_tensor([radius]), a, beta, sensitivities=True)
    inv, q, kernel, kq, kqq, kb, kqb = [x[:, 0] for x in (inv, q, kernel, kq, kqq, kb, kqb)]
    value = torch.cat((kernel, w*(-kernel-q*kq), model.pack_shapes(w[..., None]*kb)), dim=1)
    radial = torch.cat((2*radius*inv*kq, -2*radius*w*inv*(2*kq+q*kqq),
                        model.pack_shapes((2*radius*w*inv)[..., None]*kqb)), dim=1)
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
        self.parameters_list = [model.time_coefficients, model.shape_coefficients]
        self.config = dict(config)
        self.radii, self.measure, self.times, self.boundary_times = grid
        self.damping = float(config.get("gn_initial_damping", 0.01))
        self.regularization = float(config.get("coefficient_regularization", 1e-10))
        self.boundary_weight = float(config.get("boundary_weight", 1.0))
        self.max_trials = int(config.get("gn_max_trials", 5))
        self.max_coefficient_step = float(config.get("gn_max_coefficient_step", 2.0))
        self.max_curve_step = float(config.get("gn_max_curve_step", 2.0))
        self.max_shape_step = float(config.get("gn_max_shape_step", 0.01))
        self.max_shape8_step = float(config.get("gn_max_shape8_step", 0.001))
        self.accepted_steps = 0
        self.rejected_trials = 0

    @torch.no_grad()
    def normal_matrix(self):
        transform = temporal_jacobian(self.model, self.times)
        sensitivity = radial_sensitivities(self.model, self.radii, self.times)
        scale = self.measure / (self.measure.sum() * len(self.times))
        radial_gram = torch.bmm(
            sensitivity.transpose(1, 2), sensitivity * scale[None, :, None]
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
        n_time = self.model.time_coefficients.numel()
        n_shape = self.model.n_core + 1
        normal.diagonal()[:n_time].add_(self.regularization / n_time)
        normal.diagonal()[n_time:n_time+n_shape].add_(float(self.config.get("shape_regularization", 1e-10)) / n_shape)
        normal.diagonal()[n_time+n_shape:].add_(float(self.config.get("shape8_regularization", 1e-10)) / n_shape)
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
        base = torch.cat([p.detach().flatten() for p in self.parameters_list])
        gradient = torch.cat([p.grad.detach().flatten() for p in self.parameters_list]) / 2
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
            delta = direction[:theta.numel()].reshape_as(theta)
            restriction = max(1.0, float(delta.abs().max()) / self.max_coefficient_step,
                              float((basis @ delta.T).abs().max()) / self.max_curve_step,
                              float(direction[theta.numel():theta.numel()+self.model.n_core+1].abs().max()) / self.max_shape_step,
                              float(direction[theta.numel()+self.model.n_core+1:].abs().max()) / self.max_shape8_step)
            direction = direction / restriction
            predicted = -float(2 * gradient.dot(direction) + direction.dot(normal @ direction))
            if not bool(torch.isfinite(direction).all()) or predicted <= 0:
                self.damping = min(1e12, self.damping * 10)
                continue
            self.copy_flat(base + direction)
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
            self.copy_flat(base)
            self.rejected_trials += 1
            self.damping = min(1e12, self.damping * 10)
        self.copy_flat(base)
        return current_loss

    @torch.no_grad()
    def copy_flat(self, values):
        offset = 0
        for parameter in self.parameters_list:
            size = parameter.numel()
            parameter.copy_(values[offset:offset+size].reshape_as(parameter))
            offset += size


def build_optimizer(model, config):
    device = model.time_coefficients.device
    helpers = KSHelpers(config, device)
    loss_module = _load_module(Path(__file__).with_name("loss.py"), "ks_gn_grid")
    radii, measure, times = loss_module.make_training_grid(helpers)
    boundary_times = torch.linspace(0, helpers.T_MAX, int(config.get("n_bc", 128)),
                                    dtype=helpers.DTYPE, device=device)
    return DampedGaussNewton(model, config, (radii, measure, times, boundary_times))
