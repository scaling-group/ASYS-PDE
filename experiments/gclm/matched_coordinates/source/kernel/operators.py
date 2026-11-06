"""Differentiable real-line operators for an odd gCLM similarity profile.

Quadrature is evaluator machinery, never part of a candidate's returned profile.
The reciprocal split covers the whole real line without a fitted tail cutoff.
"""

import math

import numpy as np
import torch


class Operators:
    def __init__(self, nodes=256, transform_power=4):
        points, weights = np.polynomial.legendre.leggauss(nodes)
        v = (points + 1) / 2
        # A decaying algebraic profile gives fractional powers at u=0. Mapping
        # u=v^4 smooths that endpoint and resolves y/u near the profile core.
        self.u = torch.tensor(v**transform_power, dtype=torch.float64)
        self.w = torch.tensor(
            weights / 2 * transform_power * v ** (transform_power - 1), dtype=torch.float64
        )

    def center(self, profile):
        u = self.u
        return 2 / math.pi * torch.sum(self.w * (profile(u) + profile(1 / u)) / u)

    def __call__(self, profile, y):
        """Return g=-H[f] and V(y)=integral_0^y g, for strictly positive y."""
        u, w = self.u, self.w
        y = y.reshape(-1, 1)
        lower, upper = profile(y * u), profile(y / u)
        g = 2 / math.pi * torch.sum(w * (upper / u - u * lower) / (1 - u**2), dim=1)
        fy = profile(y)
        # Subtract the endpoint value before integrating the logarithmic kernel.
        # integral_0^1 log((1+u)/(1-u)) du = 2 log(2).
        smooth = lower + upper / u**2 - 2 * fy
        primitive = (
            y[:, 0]
            / math.pi
            * (
                torch.sum(w * smooth * torch.log1p(2 * u / (1 - u)), dim=1)
                + 4 * fy[:, 0] * math.log(2)
            )
        )
        return g, primitive


class NormalizedProfile:
    """Fix the two exact scale symmetries outside the candidate program.

    For raw h with slope k>0 and b=-H[h](0)>0, f(y)=h(b*y/k)/b has
    f'(0)=1 and -H[f](0)=1. Compute the two scalars once per loss evaluation,
    retaining their parameter gradients. Frozen readouts retain these scalars.
    """

    def __init__(self, model, operators):
        self.model = model
        zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        raw_zero = model(zero)
        slope = torch.autograd.grad(raw_zero.sum(), zero, create_graph=True)[0].squeeze()
        amplitude = operators.center(model)
        if not bool(torch.isfinite(slope) & torch.isfinite(amplitude)):
            raise ValueError("Nonfinite profile normalization")
        if not bool((slope > 1e-10) & (amplitude > 1e-10)):
            raise ValueError("Profile must have positive nonzero origin slope and center transform")
        self.amplitude = amplitude
        self.scale = amplitude / slope

    def __call__(self, y):
        return self.model(self.scale * y) / self.amplitude

    def beta(self):
        return self.model.beta()


def terms(profile, beta, y, operators, a=0.25, create_graph=True):
    """Canonical gauge: f'(0)=g(0)=1, lambda=(1+beta)/(1-a)."""
    y = y.detach().clone().requires_grad_(True)
    f = profile(y)
    fp = torch.autograd.grad(f.sum(), y, create_graph=create_graph)[0]
    g, velocity = operators(profile, y)
    amplitude = (1 + beta) / (1 - a)
    return torch.stack((f, beta * y * fp, a * amplitude * velocity * fp, -amplitude * g * f))


def losses(model, y, operators, a=0.25):
    pieces = terms(model, model.beta(), y, operators, a)
    residual = pieces.sum(dim=0)
    physics = torch.mean(residual**2 / torch.sum(pieces**2, dim=0).clamp_min(1e-28))
    zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    value = model(zero)
    slope = torch.autograd.grad(value.sum(), zero, create_graph=True)[0]
    normalization = value.square().mean() + (slope - 1).square().mean()
    normalization = normalization + (operators.center(model) - 1).square()
    return {"physics": physics, "normalization": normalization}
