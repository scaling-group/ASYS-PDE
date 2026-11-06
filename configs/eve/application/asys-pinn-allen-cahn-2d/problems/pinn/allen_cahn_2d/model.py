"""Two-scalar tanh-disk union initialized to the prescribed peanut field."""

import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.R = nn.Parameter(torch.tensor(0.22))
        self.eps = nn.Parameter(torch.tensor(0.04))

    def forward(self, points):
        x, y = points[:, :1], points[:, 1:2]
        width = 2**0.5 * self.eps.abs().clamp_min(1e-4)
        r1 = torch.sqrt((x + 0.15).square() + (y - 0.02).square() + 1e-12)
        r2 = torch.sqrt((x - 0.18).square() + (y + 0.03).square() + 1e-12)
        a = torch.tanh((self.R - r1) / width)
        b = torch.tanh((0.18 - r2) / width)
        return (a + b + 1 - a * b) / 2
