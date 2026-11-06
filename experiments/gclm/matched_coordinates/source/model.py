"""Four fixed-beta arms; archived native stable forward is unchanged."""
import os
import math
import numpy as np
import torch
from panel_models import Conventional, native_module

BETA = 0.6845136212067134
ARMS = ('poly_identity', 'poly_gram', 'asys_identity', 'asys_gram')

class Polynomial(Conventional):
    def __init__(self, beta):
        super().__init__(beta)
        self.register_buffer('coefficient_transport', torch.eye(14, dtype=torch.float64))

    def _untransported_basis(self, x, alpha):
        lv = -alpha*torch.hypot(x, torch.ones_like(x)).log()
        v, z = lv.exp(), 2*lv.exp()-1
        t = [torch.ones_like(x), z]
        for j in range(2, 14):
            t.append(2*z*t[-1]-t[-2])
        return (-torch.expm1(lv)).square()[:, None]*torch.stack(t, dim=1)

    def forward(self, x):
        r = torch.hypot(x, torch.ones_like(x))
        lv = -r.log()/self.beta()
        z = 2*lv.exp()-1
        c = self.coefficient_transport @ self.correction
        u, w = torch.zeros_like(x), torch.zeros_like(x)
        for j in range(13, 0, -1):
            u, w = c[j]+2*z*u-w, u
        q = c[0]+z*u-w
        return (x/r)*torch.exp(lv+(-torch.expm1(lv)).square()*q)


def gram(model, alphas):
    nodes, weights = np.polynomial.legendre.leggauss(768)
    b = math.log(1/math.tan(math.pi*.001/2))+2
    x = torch.exp(torch.tensor(b*nodes, dtype=torch.float64))
    weight = torch.tensor(weights/2, dtype=torch.float64)
    G = torch.zeros((14, 14), dtype=torch.float64)
    with torch.no_grad():
        for alpha in alphas:
            psi = model._untransported_basis(x, torch.tensor(alpha, dtype=torch.float64))
            G += psi.T @ (weight[:, None]*psi)/len(alphas)
    return G


def transport(G):
    return torch.linalg.solve_triangular(torch.linalg.cholesky(G).T,
                                        torch.eye(14, dtype=torch.float64), upper=True)


def build_model(config):
    arm = os.environ['MATCHED_ARM']
    assert arm in ARMS
    raw = Polynomial(BETA) if arm.startswith('poly') else native_module().build_model(dict(config, initial_beta=BETA))
    raw.log_beta.requires_grad_(False)
    with torch.no_grad():
        if arm.endswith('identity'):
            raw.coefficient_transport.copy_(torch.eye(14, dtype=torch.float64))
        elif arm.startswith('poly'):
            raw.coefficient_transport.copy_(transport(gram(raw, [1, 1.25, 1.5, 1.75, 2])))
    assert sum(p.numel() for p in raw.parameters() if p.requires_grad) == 14
    assert float(raw.beta()) == BETA
    assert torch.count_nonzero(raw.correction) == 0
    return raw
