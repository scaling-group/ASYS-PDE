"""Diffuse peanut with weakly nonlinear curvature-driven polar geometry.

The initial field is the prescribed diffuse union with smooth saturated cores.
A low-order polar geometry loses area and damps its angular modes by the
curve-shortening law. Dominant cubic feedback corrects the elongation, while
quadratic mode interactions generate small fifth and sixth harmonics without
adding fitted coefficients. The normal distance has a second-order
eikonal correction, reducing spurious profile stretching around the neck.
An analytic Allen--Cahn initial layer supplies the initial velocity.
One localized bridge displacement describes the nonlinear neck relaxation
that a linearized angular expansion cannot capture.
There is no time integration, sampled trajectory, or learned field.
"""

import math
import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.shape = nn.Parameter(torch.zeros(12))

    @staticmethod
    def initial_geometry(x, y, t):
        """Diffuse union phase and analytic, core-regularized initial PDE RHS.

        A Gaussian core mollifier removes the radial cusp while being
        exponentially negligible at each interface. Its derivative smoothing
        length grows continuously in the transient layer.
        """
        width = math.sqrt(2) * 0.04
        distances, gradients, laplacians = [], [], []
        smoothing2 = 0.002**2 + 4 * 0.04**2 * t
        for cx, cy, radius in ((-0.15, 0.02, 0.22), (0.18, -0.03, 0.18)):
            dx, dy = x - cx, y - cy
            r2 = dx.square() + dy.square()
            initial_core2 = 0.002**2
            r = torch.sqrt(r2 + initial_core2 * torch.exp(-r2 / initial_core2))
            gaussian = torch.exp(-r2 / smoothing2)
            core = torch.sqrt(r2 + smoothing2 * gaussian)
            slope = 1 - gaussian
            distances.append(radius - r)
            gradients.append((-slope * dx / core, -slope * dy / core))
            laplacians.append(-2 * slope / core - 2 * r2 * gaussian / (smoothing2 * core)
                              + r2 * slope.square() / core.pow(3))
        a, b = distances
        logits = torch.cat((a, b, a + b), dim=1) * (2 / width)
        q0 = width / 2 * torch.logsumexp(logits, dim=1, keepdim=True)
        weights = torch.softmax(logits, dim=1)
        wa, wb, wab = weights[:, :1], weights[:, 1:2], weights[:, 2:3]
        ax, ay = gradients[0]
        bx, by = gradients[1]
        gx = (wa + wab) * ax + (wb + wab) * bx
        gy = (wa + wab) * ay + (wb + wab) * by
        grad2 = gx.square() + gy.square()
        average_grad2 = (wa * (ax.square() + ay.square())
                         + wb * (bx.square() + by.square())
                         + wab * ((ax + bx).square() + (ay + by).square()))
        lap = ((wa + wab) * laplacians[0] + (wb + wab) * laplacians[1]
               + 2 / width * (average_grad2 - grad2))
        u0 = torch.tanh(q0 / width)
        phase_velocity = 0.04**2 * lap + width * u0 * (1 - grad2)
        return q0, u0, (1 - u0.square()) * phase_velocity / width

    @staticmethod
    def polar_distance(x, y, t, radius0, modes, cx, cy, angle, speed):
        """Normal-scaled polar distance with a smooth medial core.

        With d tau = speed/R**2 dt, the dominant even-mode equations are
        e2_tau = -2*e2 + 11*e2*e4 - 15*e2**3/4,
        e4_tau = -14*e4 + 11*e2**2/2,
        e6_tau = -34*e6 + 27*e2*e4 - 39*e2**3/4 + 13*e3**2.
        Also e5_tau = -23*e5 + 18*e2*e3. The expressions below are
        Duhamel expansions using the linear modes and the forced fourth mode.
        Thus the sixth mode starts at zero, and all coefficients have analytic
        sources. This is an explicit reduced shape expansion, not PDE marching.
        The normalization preserves enclosed area exactly for the corresponding
        unregularized five-mode radial graph. This selected truncation omits
        mixed cubic terms and the higher quadratic harmonics.
        """
        radius2 = radius0.square() - 2 * speed * t
        radius = torch.sqrt(radius2)
        ratio = radius2 / radius0.square()
        a, b = modes[0], modes[2]
        # The feedback through the forced fourth mode is of the same cubic
        # order as the direct elongation self-interaction.
        e2 = ratio * (a + 23 / 40 * a.pow(3) * (1 - ratio.square())
                      + (11 / 14 * a * b - 121 / 280 * a.pow(3))
                      * (1 - ratio.pow(7)))
        e3 = modes[1] * ratio.pow(3.5)
        e4 = b * ratio.pow(7) + 11 / 20 * a.square() * (ratio.square() - ratio.pow(7))
        e5 = 9 / 7 * a * modes[1] * (ratio.pow(4.5) - ratio.pow(11.5))
        e6 = (3 / 2 * a * (b - 11 / 20 * a.square()) * (ratio.pow(8) - ratio.pow(17))
              + 51 / 280 * a.pow(3) * (ratio.pow(3) - ratio.pow(17))
              + 13 / 20 * modes[1].square() * (ratio.pow(7) - ratio.pow(17)))
        cosine, sine = torch.cos(angle), torch.sin(angle)
        xx = (x - cx) * cosine + (y - cy) * sine
        yy = -(x - cx) * sine + (y - cy) * cosine
        core2 = 0.015**2
        rho2 = xx.square() + yy.square() + core2
        rho = torch.sqrt(rho2)
        c2, s2 = (xx.square() - yy.square()) / rho2, 2 * xx * yy / rho2
        c3 = xx * (xx.square() - 3 * yy.square()) / rho.pow(3)
        s3 = yy * (3 * xx.square() - yy.square()) / rho.pow(3)
        c4, s4 = c2.square() - s2.square(), 2 * c2 * s2
        c5, s5 = c2 * c3 - s2 * s3, s2 * c3 + c2 * s3
        c6, s6 = c3.square() - s3.square(), 2 * c3 * s3
        amplitudes = (e2, e3, e4, e5, e6)
        cosines, sines = (c2, c3, c4, c5, c6), (s2, s3, s4, s5, s6)
        scale = torch.sqrt(radius2 + core2) / torch.sqrt(1 + sum(e.square() for e in amplitudes) / 2)
        boundary = scale * (1 + sum(e * c for e, c in zip(amplitudes, cosines)))
        angular_slope = -scale * sum(m * e * s for m, e, s in zip(range(2, 7), amplitudes, sines))
        angular_second = -scale * sum(m * m * e * c for m, e, c in zip(range(2, 7), amplitudes, cosines))
        normal_scale = torch.sqrt(1 + (angular_slope / boundary).square())
        offset = boundary - rho
        # For B(theta)-r, solve |grad d|^2=1 through first order in
        # B-r. The resulting quadratic coefficient follows by differentiation,
        # rather than fitting a second spatial field. Damp only deep in the
        # saturated phases, where the local normal expansion is inappropriate.
        quadratic = angular_slope.square() * (
            boundary * angular_second - boundary.square() - 2 * angular_slope.square()
        ) / (2 * normal_scale.pow(5) * boundary.pow(5))
        distance = offset / normal_scale + quadratic * offset.square() / (1 + (offset / 0.12).pow(4))
        return distance, radius

    def forward(self, points):
        x, y, t = points[:, :1], points[:, 1:2], points[:, 2:3]
        p = torch.tanh(self.shape)
        width = math.sqrt(2) * 0.04
        radius0 = 0.279 + 0.025 * p[0]
        modes = (0.30 + 0.12 * p[1], 0.10 * p[2], -0.025 + 0.10 * p[3])
        cx, cy = -0.02 + 0.04 * p[4], 0.004 + 0.025 * p[5]
        angle = -math.atan2(0.05, 0.33) + 0.15 * p[6]
        speed = 0.04**2 * (1 + 0.2 * p[7])
        blend_rate = 0.35 * torch.exp(0.8 * p[8])
        layer_rate = 1.1 * torch.exp(0.6 * p[9])

        q0, u0, rhs0 = self.initial_geometry(x, y, t)
        polar, radius = self.polar_distance(x, y, t, radius0, modes, cx, cy, angle, speed)
        polar0, _ = self.polar_distance(x, y, t * 0, radius0, modes, cx, cy, angle, speed)
        blend = -torch.expm1(-blend_rate * t)
        early = q0 + radius - radius0
        phase = early + blend * (polar - early)
        # The bridge is centered at the intersection chord of the two disks.
        # Its quadratic onset preserves both the initial field and its tangent.
        longitudinal = (0.33 * (x - 0.039) - 0.05 * (y + 0.009)) / math.sqrt(0.33**2 + 0.05**2)
        transverse = (0.05 * (x - 0.039) + 0.33 * (y + 0.009)) / math.sqrt(0.33**2 + 0.05**2)
        neck_rate = 0.5 + torch.sigmoid(self.shape[11])
        neck_time = (-torch.expm1(-neck_rate * t)).square() * torch.exp(-0.15 * t)
        neck_shape = torch.exp(-(longitudinal / 0.13).square() - (transverse / 0.23).square())
        phase = phase + 0.06 * p[10] * neck_time * neck_shape
        scaffold_velocity = (1 - u0.square()) / width * (-speed / radius0 + blend_rate * (polar0 - q0))
        value = torch.tanh(phase / width) + t * torch.exp(-layer_rate * t) * (rhs0 - scaffold_velocity)

        # C2 closure in the saturated outer phase gives exact periodic values
        # and normal derivatives while retaining local Euclidean geometry.
        cut = ((torch.cat((x, y), dim=1).abs() - 0.6) / 0.4).clamp(0, 1)
        window = 1 - cut.pow(3) * (10 - 15 * cut + 6 * cut.square())
        return -1 + window[:, :1] * window[:, 1:2] * (1 + value)
