"""Unfitted algebra checks at independent synthetic sites; no PDE evolution.

Only public formulas and mathematical helper operators are used. This script
does not instantiate an optimizer or call batch(), score(), or evaluate().
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
HELPERS = ROOT.parents[1] / "configs/eve/application/asys-blowup-graveleau-pme-2d"
sys.path.insert(0, str(HELPERS))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load("graveleau_candidate", "selected/model.py")
loss_module = load("graveleau_loss", "selected/loss.py")
Problem = load("graveleau_problem", "selected/problem.py").Problem

helpers = Problem("cpu")


def derivative(value, variable):
    return torch.autograd.grad(value.sum(), variable, create_graph=True)[0]


def scaled_error(first, second):
    return float(((first - second).abs() / (1 + first.abs() + second.abs())).max().detach())


def direct_jet(model, points):
    points = points.detach().clone().requires_grad_(True)
    pressure = model(points)
    first = derivative(pressure, points)
    second = derivative(first[:, :1], points)[:, :1]
    return pressure, first[:, :1], first[:, 1:2], second


def synthetic_points(model):
    radii = torch.tensor((0.0031, 0.17, 0.53, 0.89, 1.003, 1.071,
                          1.31, 1.87, 2.63, 3.81, 5.97))
    times = torch.tensor((0.0, 0.0091, 0.0317, 0.074, 0.151,
                          0.283, 0.439, 0.571, 0.6137))
    grid = torch.cartesian_prod(radii, times)
    front = model.front(times[:, None])[0].detach()
    offsets = torch.tensor((0.0, 0.0007, 0.019, 0.17, 0.61))
    near = torch.stack(((front + offsets).flatten(),
                        times.repeat_interleave(len(offsets))), dim=1)
    return torch.cat((grid, near))


def initialize(state):
    model = model_module.Candidate()
    if state:
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                pattern = torch.arange(parameter.numel()).reshape_as(parameter)
                parameter.add_((0.16 if state == 1 else 0.37)
                               * torch.sin(pattern + 0.71 * (index + 1)))
            if state == 2:
                model.focus_clock.sub_(1.1)
                model.exponent.add_(0.8)
                model.onset.sub_(0.6)
    return model


def initial_checks(model):
    radii = torch.tensor((0.003, 0.41, 0.93, 1.0009, 1.013,
                          1.087, 1.29, 1.63, 2.11, 2.73, 3.19, 5.9))[:, None]
    points = torch.cat((radii, torch.zeros_like(radii)), dim=1).requires_grad_(True)
    value = model(points)
    time = derivative(value, points)[:, 1:2]
    time2 = derivative(time, points)[:, 1:2]

    radii = radii.clone().requires_grad_(True)
    initial = helpers.initial_condition_torch(radii)
    slope = derivative(initial, radii)
    laplacian = derivative(slope, radii) + slope / radii
    rhs = initial * laplacian + slope.square()
    rhs_slope = derivative(rhs, radii)
    second_rhs = (rhs * laplacian
                  + initial * (derivative(rhs_slope, radii) + rhs_slope / radii)
                  + 2 * slope * rhs_slope)
    return {"ic": scaled_error(value, initial),
            "first_initial_jet": scaled_error(time, rhs),
            "second_initial_jet": scaled_error(time2, second_rhs)}


def moving_integral(model):
    times = torch.tensor((0.017, 0.069, 0.193, 0.347, 0.529, 0.609))[:, None]
    front = model.front(times)[0]
    radius, weights = loss_module._exterior_rule(front, 6, panels=16)
    points = torch.stack((radius.flatten(), times.expand_as(radius).flatten()), dim=1)
    value = model(points).reshape_as(radius)
    mass = (radius * weights * value).sum(dim=1)
    moment = (radius * weights * value * (1 - (radius / 6).square()).square()).sum(dim=1)
    return (mass.square() + 0.31 * moment.square()).mean()


def moving_integral_difference(model):
    parameters = list(model.parameters())
    directions = [torch.cos(torch.arange(p.numel()).reshape_as(p) + 0.43 * i)
                  for i, p in enumerate(parameters)]
    norm = math.sqrt(sum(float(d.square().sum()) for d in directions))
    directions = [d / norm for d in directions]
    gradient = torch.autograd.grad(moving_integral(model), parameters)
    analytic = sum((g * d).sum() for g, d in zip(gradient, directions))
    saved = [p.detach().clone() for p in parameters]
    step = 2e-5
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            for parameter, original, direction in zip(parameters, saved, directions):
                parameter.copy_(original + sign * step * direction)
        values.append(moving_integral(model).detach())
    with torch.no_grad():
        for parameter, original in zip(parameters, saved):
            parameter.copy_(original)
    finite_difference = (values[1] - values[0]) / (2 * step)
    return scaled_error(analytic, finite_difference)


def field_checks(state):
    model = initialize(state)
    points = synthetic_points(model)
    direct = direct_jet(model, points)
    shared = loss_module._pressure_jet(model, points, second_order=True)
    result = {"jet_" + name: scaled_error(a, b)
              for name, a, b in zip(("pressure", "radial", "time", "curvature"),
                                    direct, shared)}
    residual = loss_module._pde_residual(model, points)
    fixed = helpers.pde_residual(model, points)
    result["residual_identity"] = scaled_error(residual, fixed)
    parameters = list(model.parameters())
    gradient = torch.autograd.grad(residual.square().mean(), parameters)
    fixed_gradient = torch.autograd.grad(fixed.square().mean(), parameters)
    result["residual_gradient_identity"] = max(scaled_error(a, b)
                                              for a, b in zip(gradient, fixed_gradient))
    result.update(initial_checks(model))

    times = torch.tensor((0.0, 0.0037, 0.026, 0.087, 0.233, 0.477, 0.614))[:, None]
    time_variable = times.clone().requires_grad_(True)
    front, speed, _ = model.front(time_variable)
    result["front_speed_identity"] = scaled_error(-derivative(front, time_variable), speed)
    edge = torch.cat((front.detach(), times), dim=1)
    edge_jet = direct_jet(model, edge)
    result["darcy_identity"] = scaled_error(edge_jet[1], speed)
    hole = torch.cat((0.41 * front.detach(), times), dim=1)
    hole_jet = direct_jet(model, hole)
    result["hole_jet_max"] = max(float(a.abs().max().detach()) for a in hole_jet)
    outer = torch.cat((torch.full_like(times, 6), times), dim=1)
    result["outer_slope_max"] = float(direct_jet(model, outer)[1].abs().max().detach())
    result["pressure_min"] = float(direct[0].min().detach())
    result["pressure_max"] = float(direct[0].max().detach())

    # Only synthetic sites are used, without any training or monitor batch.
    ic = torch.cat((torch.linspace(0.009, 5.91, 29)[:, None], torch.zeros(29, 1)), dim=1)
    boundary = torch.cat((torch.full_like(times, 6), times), dim=1)
    objective = loss_module.loss_fn(model, points, ic, boundary, helpers)
    full_gradient = torch.autograd.grad(objective, parameters)
    flattened = torch.cat([g.flatten() for g in full_gradient])
    result["full_objective_finite"] = bool(torch.isfinite(objective))
    result["full_gradient_finite"] = bool(torch.isfinite(flattened).all())
    result["scalar_parameters"] = flattened.numel()
    result["nonzero_gradient_parameters"] = int((flattened != 0).sum())
    result["moving_integral_directional_error"] = moving_integral_difference(model)
    return result


def similarity_checks():
    results = []
    for beta in (0.61, 0.8, 0.97):
        p, q = 2 - 1 / beta, 1 / beta
        offset = 1.75 - 2 * beta
        eta_edge = torch.ones(1, requires_grad=True)
        log_eta = torch.log(eta_edge)
        x = -torch.expm1(-q * log_eta)
        psi = beta * torch.exp(p * log_eta) * x * torch.exp(offset * x)
        slope = derivative(psi, eta_edge)
        curvature = derivative(slope, eta_edge)
        record = {"beta": beta, "edge_value": float(psi.detach()),
                  "edge_slope_error": float((slope - 1).abs().detach()),
                  "edge_curvature_error": float((curvature / 2 + p / 4).abs().detach())}

        # Exercise a nonzero degree-two polynomial independently of model code.
        eta = torch.tensor((1.0, 1.17, 1.9, 4.2, 15.0, 70.0), requires_grad=True)
        x = 1 - eta.pow(-q)
        polynomial = offset + 0.17 - 0.08 * x + 0.06 * x.square()
        psi = beta * eta.pow(p) * x * torch.exp(x * polynomial)
        slope = derivative(psi, eta)
        curvature = derivative(slope, eta)
        ode = psi * (curvature + slope / eta) + slope.square() - eta * slope + p * psi

        z = eta.detach().pow(-q).requires_grad_(True)
        x = 1 - z
        polynomial = offset + 0.17 - 0.08 * x + 0.06 * x.square()
        f = beta * x * torch.exp(x * polynomial)
        first = derivative(f, z)
        second = derivative(first, z)
        a = p * f - q * z * first
        b = p**2 * f + q * (q - 2 * p) * z * first + q**2 * z.square() * second
        reduced = f * b + a.square() + q * first
        record["compact_ode_identity"] = scaled_error(ode, eta.pow(2 * p - 2) * reduced)

        z0 = torch.zeros(1, requires_grad=True)
        x0 = 1 - z0
        p1 = offset + 0.17 - 0.08 + 0.06
        p1_derivative = -0.08 + 0.12
        f0 = beta * x0 * torch.exp(x0 * (offset + 0.17 - 0.08 * x0 + 0.06 * x0.square()))
        d0 = derivative(f0, z0)
        c = beta * math.exp(p1)
        d = -c * (1 + p1 + p1_derivative)
        record["far_C_error"] = scaled_error(f0, f0.new_tensor(c))
        record["far_D_error"] = scaled_error(d0, d0.new_tensor(d))
        results.append(record)
    return results


def run():
    report = {"purpose": "unfitted mathematical checks only",
              "states": [field_checks(i) for i in range(3)],
              "similarity": similarity_checks()}
    for state in report["states"]:
        for key, value in state.items():
            if key.startswith("jet_") or key.endswith("identity") or key.endswith("_error"):
                assert value < 2e-7, (key, value)
        assert state["ic"] < 1e-14
        assert state["first_initial_jet"] < 1e-9
        assert state["second_initial_jet"] < 1e-7
        assert state["hole_jet_max"] == 0
        assert state["outer_slope_max"] < 1e-13
        assert 0 <= state["pressure_min"] <= state["pressure_max"] <= 1.6
        assert state["full_objective_finite"] and state["full_gradient_finite"]
        assert state["nonzero_gradient_parameters"] == state["scalar_parameters"] == 24
    for similarity in report["similarity"]:
        for key, value in similarity.items():
            if key.endswith("error") or key.endswith("identity") or key == "edge_value":
                assert abs(value) < 1e-11, (key, value)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
