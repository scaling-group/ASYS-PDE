"""Fixed Allen--Cahn equation and prescribed initial datum."""
import math

import torch

EPS = 0.04
T_END = 8.0


def initial(points):
    x, y = points[:, 0:1], points[:, 1:2]
    phases = []
    for cx, cy, radius in ((-0.15, 0.02, 0.22), (0.18, -0.03, 0.18)):
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1.0e-14)
        phases.append(torch.tanh((radius - r) / (math.sqrt(2.0) * EPS)))
    p, q = phases
    return (p + q + 1.0 - p * q) / 2.0


def gradient(value, points):
    if not value.requires_grad:
        return points * 0.0
    result = torch.autograd.grad(value, points, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return points * 0.0 if result is None else result


def value(model, points):
    result = model(points)
    if result.shape != (len(points), 1):
        raise ValueError("Candidate.forward must return an N-by-1 tensor")
    return result


def equation(model, points):
    u = value(model, points)
    du = gradient(u, points)
    lap = gradient(du[:, 0:1], points)[:, 0:1]
    lap = lap + gradient(du[:, 1:2], points)[:, 1:2]
    return du[:, 2:3] - EPS ** 2 * lap - u + u ** 3


def initial_rhs(points):
    u = initial(points)
    du = gradient(u, points)
    lap = gradient(du[:, 0:1], points)[:, 0:1]
    lap = lap + gradient(du[:, 1:2], points)[:, 1:2]
    return EPS ** 2 * lap + u - u ** 3


def terms(model, collocation, ic_points, bc_points, jet_points=None):
    p = collocation.detach().clone().requires_grad_(True)
    i = ic_points.detach().clone().requires_grad_(True)
    j = (ic_points if jet_points is None else jet_points).detach().clone().requires_grad_(True)
    a, b, c, d = [x.detach().clone().requires_grad_(True) for x in bc_points]
    ua, ub, uc, ud = [value(model, x) for x in (a, b, c, d)]
    boundary = ((ua - ub).square().mean() + (uc - ud).square().mean()
                + (gradient(ua, a)[:, 0:1] - gradient(ub, b)[:, 0:1]).square().mean()
                + (gradient(uc, c)[:, 1:2] - gradient(ud, d)[:, 1:2]).square().mean()) / 4.0
    tangent = gradient(value(model, j), j)[:, 2:3] - initial_rhs(j)
    u = value(model, p)
    return {"physics": equation(model, p).square().mean(),
            "ic": (value(model, i) - initial(i)).square().mean(),
            "ic_tangent": tangent.square().mean(), "bc": boundary,
            "range": torch.relu(u.abs() - 1.0).square().mean()}


def sample(counts, generator):
    def uniform(n):
        p = torch.rand((n, 3), generator=generator, dtype=torch.float64)
        return p * p.new_tensor([2.0, 2.0, T_END]) + p.new_tensor([-1.0, -1.0, 0.0])
    p, i, b = [uniform(int(n)) for n in counts]
    i[:, 2] = 0.0
    # Boundary sampling is strictly at positive times.
    b[:, 2] = b[:, 2].clamp_min(1.0e-8)
    sides = [b.clone() for _ in range(4)]
    sides[0][:, 0], sides[1][:, 0] = -1.0, 1.0
    sides[2][:, 1], sides[3][:, 1] = -1.0, 1.0
    return p, i, tuple(sides)
