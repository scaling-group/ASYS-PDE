"""Unfitted concentrating cosine: exact supplied IC and finite boundaries."""
import math
import torch


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.clock_margin = torch.nn.Parameter(torch.tensor(math.log(math.expm1(0.001))))
        self.log_width = torch.nn.Parameter(torch.tensor(math.log(0.01)))

    def forward(self, coordinates):
        r, t = coordinates[:, :1], coordinates[:, 1:]
        clock = 0.020385594 + 0.020406 * torch.nn.functional.softplus(self.clock_margin)
        width = self.log_width.exp()
        return 5 * torch.cos(math.pi * r / 10) * torch.sqrt((clock + width * r.square()) /
                                                          (clock - t + width * r.square()))
