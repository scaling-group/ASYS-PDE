"""Conventional and fixed-space controls for the matched-coordinate comparison."""
import importlib.util
import math
from pathlib import Path
import torch


def native_module():
    p = Path(__file__).with_name('h3_native.py')
    spec = importlib.util.spec_from_file_location('corrective_original_h3', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Conventional(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.log_beta = torch.nn.Parameter(torch.tensor(math.log(beta), dtype=torch.float64))
        self.correction = torch.nn.Parameter(torch.zeros(14, dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        r = torch.hypot(x, torch.ones_like(x))
        lv = -r.log()/self.beta()
        v, z = lv.exp(), 2*lv.exp()-1
        b = (-torch.expm1(lv)).square()
        # Clenshaw evaluation of the standard degree-thirteen Chebyshev basis.
        a, c = torch.zeros_like(x), torch.zeros_like(x)
        for j in range(13, 0, -1):
            a, c = self.correction[j] + 2*z*a-c, a
        q = self.correction[0]+z*a-c
        return (x/r)*torch.exp(lv+b*q)


class DirectMixed(torch.nn.Module):
    """Same fixed-beta H3 span; unprojected normalized generalized powers."""
    def __init__(self, beta):
        super().__init__()
        self.register_buffer('log_beta', torch.tensor(math.log(beta), dtype=torch.float64))
        self.correction = torch.nn.Parameter(torch.zeros(14, dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def basis(self, x):
        lv = -torch.hypot(x, torch.ones_like(x)).log()/self.beta()
        v, z = lv.exp(), 2*lv.exp()-1
        b = (-torch.expm1(lv)).square()
        polynomials = [torch.ones_like(x), z]
        for j in range(1, 7):
            polynomials.append(((2*j+1)*z*polynomials[-1]-j*polynomials[-2])/(j+1))
        polynomials = [math.sqrt(2*j+1)*p for j,p in enumerate(polynomials)]
        p = 2*self.beta()
        powers = [torch.sqrt(2*q+1)*torch.exp(q*lv) for q in (p,p+1,p+2,2*p)]
        rational = b*v.square()/((v-.5).square()+1/64)
        return b[...,None]*torch.stack(polynomials+powers+[rational,z*rational],dim=-1)

    def forward(self, x):
        r = torch.hypot(x, torch.ones_like(x))
        lv = -r.log()/self.beta()
        shape = self.basis(x)@self.correction
        return (x/r)*torch.exp(lv+shape)


def build(config, arm):
    fixed = config.get('fixed_beta')
    beta = float(fixed if fixed is not None else config['initial_beta'])
    if arm == 'fractional-chebyshev13':
        return Conventional(beta)
    if arm == 'fixed-unprojected':
        return DirectMixed(beta)
    raw = native_module().build_model(dict(config, initial_beta=beta))
    if fixed is not None:
        raw.log_beta.requires_grad_(False)
    if arm == 'fixed-projected-identity':
        with torch.no_grad():
            raw.coefficient_transport.copy_(torch.eye(14,dtype=torch.float64))
    return raw
