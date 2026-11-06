import math
import torch


def core(x, t):
    nr = torch.cosh(3*x) + 3*torch.cos(4*t)*torch.cosh(x)
    ni = 3*torch.sin(4*t)*torch.cosh(x)
    d = torch.cosh(4*x) + 4*torch.cosh(2*x) + 3*torch.cos(4*t)
    cr, ci = torch.cos(t/2), torch.sin(t/2)
    return torch.cat((4*(cr*nr-ci*ni)/d, 4*(ci*nr+cr*ni)/d), dim=1)


def core_x(x, t):
    nr = torch.cosh(3*x) + 3*torch.cos(4*t)*torch.cosh(x)
    ni = 3*torch.sin(4*t)*torch.cosh(x)
    d = torch.cosh(4*x) + 4*torch.cosh(2*x) + 3*torch.cos(4*t)
    dx = 4*torch.sinh(4*x) + 8*torch.sinh(2*x)
    nrx = 3*torch.sinh(3*x) + 3*torch.cos(4*t)*torch.sinh(x)
    nix = 3*torch.sin(4*t)*torch.sinh(x)
    ur, ui = (nrx*d-nr*dx)/d.square(), (nix*d-ni*dx)/d.square()
    cr, ci = torch.cos(t/2), torch.sin(t/2)
    return torch.cat((4*(cr*ur-ci*ui), 4*(ci*ur+cr*ui)), dim=1)


class V1(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # Strength 1, equal tail mixtures, inverse-horizon rate, compensation 1.
        self.theta = torch.nn.Parameter(torch.tensor(
            [1.0, 0.0, 0.0, math.log(2 / math.pi), 1.0], dtype=torch.float64))

    def forward(self, points):
        x, t, width = points[:, :1], points[:, 1:2], points[:, 2:3]
        strength, raw3, raw5, lograte, compensation = self.theta
        def tail(power, mixture):
            edge = (x / width).pow(2 * power)
            return mixture * (-width * edge / (2 * power)) + (1 - mixture) * width / 2 * (1 - (x / width).square()) * edge
        tail3 = tail(3, raw3.sigmoid())
        tail5 = tail(5, raw5.sigmoid())
        rate = lograte.exp() * 2 / math.pi
        return core(x, t) + strength * core_x(width, t) * (
            tail3 - compensation * (tail3 - tail5) * torch.exp(-rate * t.square()))


Candidate = V1
