"""Four-parameter moving-hole pressure shell with an explicit focusing clock."""

import math
import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.shell_pressure = nn.Parameter(torch.tensor(1.6))
        self.shell_width = nn.Parameter(torch.tensor(0.6))
        self.focus_time = nn.Parameter(
            torch.tensor(math.log(math.expm1(0.005 * 0.6200901156700502)))
        )
        self.front_exponent = nn.Parameter(torch.tensor(1.0))

    def forward(self, points):
        radius, time = points[:, :1], points[:, 1:2]
        focus = 0.995 * 0.6200901156700502 + torch.nn.functional.softplus(
            self.focus_time
        )
        front = (1 - time / focus).clamp_min(1e-8) ** self.front_exponent
        distance = (radius - front).clamp_min(0)
        return self.shell_pressure * (
            1 - torch.exp(-(distance / self.shell_width.abs().clamp_min(1e-4)).square())
        )
