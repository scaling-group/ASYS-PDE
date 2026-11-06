"""Protected generic canonical export; normalization is creation-time state."""
import math
import torch


def origin_slope(raw):
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    h = raw(x)
    if h.shape != x.shape or not bool(torch.isfinite(h).all()) or float(h[0]) != 0:
        raise ValueError("Raw profile must have a finite exact zero at origin")
    slope = float(torch.autograd.grad(h.sum(), x)[0][0])
    beta = float(raw.beta())
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(beta) or beta <= 0:
        raise ValueError("Finite positive origin slope and beta required")
    return slope


class FrozenProfile(torch.nn.Module):
    def __init__(self, raw, amplitude, slope):
        super().__init__()
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (amplitude, slope)):
            raise ValueError("Positive finite creation-time normalization required")
        self.raw = raw
        self.raw.requires_grad_(False)
        self.register_buffer("amplitude", torch.tensor(float(amplitude), dtype=torch.float64))
        self.register_buffer("coordinate_scale", torch.tensor(float(amplitude / slope), dtype=torch.float64))
        self.register_buffer("creation_slope", torch.tensor(float(slope), dtype=torch.float64))

    @staticmethod
    def _anchor(a):
        a = torch.as_tensor(a, dtype=torch.float64)
        if not bool(torch.isfinite(a).all() and (a == .25).all()):
            raise ValueError("Only a=.25 is declared")
        return a

    def profile(self, y):
        return self.raw(self.coordinate_scale * y) / self.amplitude

    def forward(self, a, y):
        a = self._anchor(a)
        _, y = torch.broadcast_tensors(a, torch.as_tensor(y, dtype=torch.float64))
        return self.profile(y)

    def beta(self, a=None):
        value = self.raw.beta()
        return value if a is None else value.expand(self._anchor(a).shape)
