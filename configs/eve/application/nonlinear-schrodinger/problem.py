"""Finite-box NLS datum and mathematical diagnostics; no solution trajectory supplied."""
from __future__ import annotations

import math
import torch
from asys_fit import TrainingBatch, _as_uv, _clone_points, _grad

TRAIN_WIDTHS = (4.0, 5.0, 6.0)
TEST_WIDTHS = (3.5, 4.5, 7.0)
T_MAX = math.pi / 2
WEAK_MODES = 8
PHYSICS_MODES = 16
COMP_STEP = T_MAX/1000


def initial(x, width):
    # Same original datum on every box; L specifies its periodic extension.
    return 2/torch.cosh(x)


def weak_initial_target(x, weights, width):
    n = torch.arange(-WEAK_MODES, WEAK_MODES+1, dtype=x.dtype, device=x.device)
    k = math.pi*n/width
    phase = torch.exp(-1j*x*k)
    f = initial(x, width)
    fhat = (weights*f*phase).sum(dim=0)
    cubehat = (weights*f.pow(3)*phase).sum(dim=0)
    # Distributional periodic derivative: <f_xx,phi>=<f,phi_xx>.
    # Do not substitute the classical line derivative and lose the seam term.
    return 1j*(-0.5*k.square()*fhat+cubehat), phase


class Helpers:
    def __init__(self, config, device):
        self.config, self.device = config, device

    def residual(self, model, points):
        p = _clone_points(points)
        u, v = _as_uv(model(p))
        ug, vg = _grad(u, p), _grad(v, p)
        uxx, vxx = _grad(ug[:, :1], p)[:, :1], _grad(vg[:, :1], p)[:, :1]
        q = u.square() + v.square()
        return torch.cat((-vg[:, 1:2] + 0.5*uxx + q*u,
                          ug[:, 1:2] + 0.5*vxx + q*v), dim=1)

    def ic_loss(self, model, p):
        target = torch.cat((initial(p[:, :1], p[:, 2:3]),
                            torch.zeros_like(p[:, :1])), dim=1)
        return (model(p) - target).square().mean()

    def ic_diagnostics(self, model, p):
        # Same public IC reduction, with comparison-only provenance. Training
        # continues to call ic_loss; no public residual is rounded or replaced.
        target = torch.cat((initial(p[:, :1], p[:, 2:3]),
                            torch.zeros_like(p[:, :1])), dim=1)
        delta = model(p) - target
        return dict(ic=float(delta.square().mean().detach()),
                    ic_imag_nonzero=bool((delta[:, 1] != 0).any()),
                    ic_target_energy=float(target.square().mean()),
                    ic_epsilon=float(torch.finfo(target.dtype).eps))

    def tangent_loss(self, model, points):
        import numpy as np
        nodes, weights = np.polynomial.legendre.leggauss(128)
        losses = []
        for value in torch.unique(points[:, 2]).detach().tolist():
            x = torch.as_tensor(nodes[:, None]*value, dtype=torch.float64, device=self.device)
            w = torch.as_tensor(weights[:, None]/2, dtype=torch.float64, device=self.device)
            width = torch.full_like(x, value)
            target, phase = weak_initial_target(x, w, value)
            def moment(t):
                p = torch.cat((x, torch.full_like(x, t), width), dim=1)
                uv = model(p)
                z = uv[:, :1]+1j*uv[:, 1:2]
                return (w*z*phase).sum(dim=0)
            m0 = moment(0.0)
            dh = (moment(COMP_STEP)-m0)/COMP_STEP
            dh2 = (moment(COMP_STEP/2)-m0)/(COMP_STEP/2)
            dh4 = (moment(COMP_STEP/4)-m0)/(COMP_STEP/4)
            coarse, fine = 2*dh2-dh, 2*dh4-dh2
            # Fixed finite-step compatibility plus a halving-consistency term.
            # This is a weak secant check, not an exact time-derivative certificate.
            losses.append((fine-target).abs().square().mean()
                          +(fine-coarse).abs().square().mean())
        return torch.stack(losses).mean()

    def boundary_loss(self, model, points):
        # Only values have classical traces guaranteed by periodic H1 data.
        t, width = points[:, :1], points[:, 1:2]
        left = torch.cat((-width, t, width), dim=1)
        right = torch.cat((width, t, width), dim=1)
        return (model(left)-model(right)).square().mean()

    def weak_residual(self, model, width, nx=256, nt=41, offset=0.0, time_order=8):
        """Integrated periodic NLS, requiring no pointwise seam derivatives.

        R_n = i[m_n(b)-m_n(a)]/(b-a) + average_t(-k_n^2*m_n/2+c_n).
        Quadrature and mode truncation are finite numerical diagnostics. The
        independent reference must also pass their convergence qualification.
        """
        import numpy as np
        dtype, device = torch.float64, self.device
        x = ((torch.arange(nx, dtype=dtype, device=device)+offset)/nx*2-1)*width
        n = torch.arange(-PHYSICS_MODES, PHYSICS_MODES+1, dtype=dtype, device=device)
        k = math.pi*n/width
        phase = torch.exp(-1j*x[:, None]*k[None, :])
        h = T_MAX/(nt+1)
        a = (torch.arange(nt, dtype=dtype, device=device)+0.25+0.5*offset)*h
        b = a+h
        nodes, weights = np.polynomial.legendre.leggauss(time_order)
        nodes = torch.as_tensor(nodes, dtype=dtype, device=device)
        weights = torch.as_tensor(weights/2, dtype=dtype, device=device)
        tq = a[:, None]+h*(nodes[None, :]+1)/2

        def moments(times, cubic=False):
            flat = times.flatten()
            tt, xx = torch.meshgrid(flat, x, indexing='ij')
            p = torch.stack((xx.flatten(), tt.flatten(), torch.full_like(xx.flatten(), width)), dim=1)
            uv = torch.cat([model(chunk) for chunk in p.split(4096)])
            z = (uv[:, 0]+1j*uv[:, 1]).reshape(len(flat), nx)
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite weak field')
            m = z@phase/nx
            return (m, (z.abs().square()*z)@phase/nx) if cubic else m

        ma, mb = moments(a), moments(b)
        mq, cq = moments(tq, cubic=True)
        forcing = (-0.5*k.square()*mq+cq).reshape(nt, time_order, -1)
        return 1j*(mb-ma)/h+(forcing*weights[None, :, None]).sum(dim=1)

    def training_batch(self):
        dtype = torch.float64
        g = torch.Generator().manual_seed(int(self.config.get('seed', 0)) + 1103)
        count = int(self.config.get('n_collocation', 2048))
        if not 128 <= count <= 16384:
            raise ValueError('n_collocation must be between 128 and 16384 per width')
        collocation, ic, bc = [], [], []
        for width in TRAIN_WIDTHS:
            p = torch.rand((count, 3), generator=g, dtype=dtype)
            p[:, 0] = width*(2*p[:, 0]-1)
            p[:, 1] = T_MAX*p[:, 1].clamp_min(1e-8)
            p[:, 2] = width
            collocation.append(p)
            x = torch.arange(128, dtype=dtype)[:, None]*2*width/128-width
            ic.append(torch.cat((x, torch.zeros_like(x), torch.full_like(x, width)), dim=1))
            t = (torch.arange(64, dtype=dtype)[:, None]+0.5)*T_MAX/64
            bc.append(torch.cat((t, torch.full_like(t, width)), dim=1))
        return TrainingBatch(*(torch.cat(values).to(self.device) for values in (collocation, ic, bc)))

    def score_width(self, model, width, nx=256, nt=41, offset=0.0):
        dtype = torch.float64
        x = ((torch.arange(nx, dtype=dtype, device=self.device)+offset)/nx*2-1)*width
        t = (torch.arange(nt, dtype=dtype, device=self.device)+0.25+0.5*offset)*T_MAX/nt
        with torch.no_grad():
            r = self.weak_residual(model, width, nx=nx, nt=nt, offset=offset)
        xi = x[:, None]
        ip = torch.cat((xi, torch.zeros_like(xi), torch.full_like(xi, width)), dim=1)
        tb = (torch.arange(128, dtype=dtype, device=self.device)[:, None]+0.5)*T_MAX/128
        bp = torch.cat((tb, torch.full_like(tb, width)), dim=1)
        out = dict(physics=float(r.abs().square().mean()),
                   **self.ic_diagnostics(model, ip),
                   ic_tangent=float(self.tangent_loss(model, ip).detach()),
                   bc=float(self.boundary_loss(model, bp).detach()))
        if not all(math.isfinite(v) for v in out.values()):
            raise FloatingPointError('nonfinite public constraint')
        return out

    def score(self, model):
        # Worst across widths and ordinary/half-cell-shifted public grids.
        rows = [dict(width=width, offset=offset, **self.score_width(model, width, offset=offset))
                for width in TRAIN_WIDTHS for offset in (0.0, 0.5)]
        scores = {key: -max(r[key] for r in rows) for key in ('physics', 'ic', 'ic_tangent', 'bc')}
        scores['score'] = sum(scores.values())/4
        return scores, rows
