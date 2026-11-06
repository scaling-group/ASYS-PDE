"""Untrained first-order HJ starting field, with five free solution scalars."""

import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        modes = torch.tensor(config["initial_modes"], dtype=torch.float64)
        self.register_buffer("wave", modes[:, :2])
        self.register_buffer("phase", modes[:, 3])
        self.register_buffer(
            "weight",
            modes[:, 2] / (modes[:, :2].abs().sum(1) + 1) / config["initial_scale"],
        )
        self.amplitudes = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.raw_decay = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, points):
        phase = points[:, :2] @ self.wave.T + self.phase
        sine, cosine = phase.sin(), phase.cos()
        initial = sine @ self.weight
        gradient = cosine @ (self.weight[:, None] * self.wave)
        curvature = -sine @ (self.weight[:, None] * self.wave.square())
        basis = torch.cat([curvature, gradient.square()], dim=1)
        time = points[:, 2]
        memory = time / (1 + nn.functional.softplus(self.raw_decay) * time)
        return (initial + memory * (basis @ self.amplitudes))[:, None]
