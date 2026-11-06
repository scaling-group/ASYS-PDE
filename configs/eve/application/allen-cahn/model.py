"""Fresh datum-anchored Allen-Cahn seed."""
import torch
from torch import nn

from problem import initial


class Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rate = nn.Parameter(torch.zeros(()))

    def forward(self, points):
        correction = self.rate.expand(len(points), 1)
        return initial(points) + points[:, 2:3] * correction / 8.0
