"""Cold common envelope; all representation changes belong in this file."""
import math
import torch


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(config["initial_beta"]), dtype=torch.float64))
        self.correction = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        s = 1 / (1 + x.square())
        power = (1 + 1 / self.beta()) / 2
        return x * torch.exp(-power * torch.log1p(x.square())
                             + self.correction * (s - 1))


def build_model(config):
    return Model(config)
