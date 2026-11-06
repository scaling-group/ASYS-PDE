"""Public n=5, p=3 radial similarity ODE and declared finite-domain conditions."""
import math

import torch

NAME = "fujita_radial_n5_p3_stationary"
ETA_MAX = 350.0
CONSTANT_EQUILIBRIUM = math.sqrt(0.5)


def derivative(value, points):
    if not value.requires_grad:
        return points * 0
    result = torch.autograd.grad(value.sum(), points, create_graph=True, allow_unused=True)[0]
    return points * 0 if result is None else result


def field(model, eta):
    value = model(eta)
    if value.shape != (len(eta), 1):
        raise ValueError("Candidate must return a pointwise (N, 1) tensor")
    return value


def residual(model, eta):
    eta = eta.detach().clone().requires_grad_(True)
    value = field(model, eta)
    first = derivative(value, eta)
    second = derivative(first, eta)
    # Smooth radial limit of Phi'' + 4 Phi'/eta at the origin is 5 Phi''.
    radial = torch.where(eta > 0, second + 4 * first / eta.clamp_min(1e-12), 5 * second)
    return radial + value.pow(3) - 0.5 * value - 0.5 * eta * first


def points(n, seed):
    # Equal mass core, log tail, near-origin and graded outer-domain strata.
    draws = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed).draw(4 * n).double()
    core = 12 * draws[:n]
    tail = 12 * torch.exp(draws[n:2*n] * math.log(ETA_MAX / 12))
    origin = draws[2*n:3*n].square()
    outer = ETA_MAX - torch.sinh(draws[3*n:] * math.asinh(ETA_MAX**2)) / ETA_MAX
    collocation = torch.cat((core, tail, origin, outer)).clamp_min(1e-6)
    return collocation, torch.zeros((1, 1)), torch.tensor([[ETA_MAX]])


def components(model, collocation, ic, bc):
    origin = ic.detach().clone().requires_grad_(True)
    value = field(model, origin)
    origin_curvature = derivative(derivative(value, origin), origin)
    outer = bc.detach().clone().requires_grad_(True)
    outer_value = field(model, outer)
    return {
        "physics": residual(model, collocation).square().mean(),
        "ic": (5 * origin_curvature + value.pow(3) - 0.5 * value).square().mean(),
        "ic_tangent": derivative(value, origin).square().mean(),
        "bc": (outer * derivative(outer_value, outer) + outer_value).square().mean(),
    }


def check_admissibility(model, monitor):
    probes = torch.cat((monitor[0], monitor[1], monitor[2]))
    values = field(model, probes).detach()
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("profile must be positive on origin/core/tail admission points")
    # At a maximum of a nonconstant positive decaying profile, the ODE implies
    # max Phi > sqrt(1/2). This excludes zero and the spatially constant equilibrium
    # without specifying the amplitude selected by the boundary-value problem.
    if not float(values.max()) > CONSTANT_EQUILIBRIUM * (1 + 32 * torch.finfo(values.dtype).eps):
        raise ValueError("profile fails the mathematical nontrivial-maximum condition")


def helpers():
    return {"components": components, "residual": residual,
            "eta_max": ETA_MAX, "constant_equilibrium": CONSTANT_EQUILIBRIUM, "problem": NAME}
