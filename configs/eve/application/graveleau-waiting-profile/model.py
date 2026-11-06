"""Supplied analytic Taylor starter.

The coefficients through degree seven come from a provisional
calculation for the conditional planar waiting-profile ODE at gamma=7/3,
normalized quadratic/cubic coefficients 1/6 and 1. Three polynomial correction
parameters start at zero. Agents may replace this entire representation.
"""
import torch

class Candidate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.correction = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('coefficients', torch.tensor([1/6, 1., 45/2, 945., 67095., 9542610.]))

    def forward(self, points):
        x = points[:, :1]
        base = sum(c*x.pow(n) for n,c in enumerate(self.coefficients, start=2))
        u = x/.01
        return base + 1e-6*sum(c*u.pow(n) for n,c in enumerate(self.correction, start=8))
