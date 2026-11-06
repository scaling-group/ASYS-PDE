"""Diffuse peanut with five quadratically coupled curvature-driven modes.

The initial field is the prescribed diffuse union with smooth saturated cores.
A low-order polar geometry loses area and damps its angular modes by the
curve-shortening law. All quadratic mode interactions are integrated in closed
form from the linearized modes, without marching a reduced ODE. The normal
distance has a second-order eikonal correction, reducing spurious profile
stretching around the neck.
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
        self.shape = nn.Parameter(torch.zeros(14))

    @staticmethod
    def curvature_modes(ratio, modes):
        """Second-order Duhamel expansion of the area-normalized polar law.

        For r=R(1+f), R**2=R0**2-2*c*t, the nonconstant modes obey
        f_t = c/R**2 * (f''+2*f-f**2-2*f*f''-f'**2) + O(f**3).
        Linear relative mode m has exponent p_m=(m*m-2)/2 in ratio.
        A quadratic forcing C*a_n*a_k*ratio**(p_n+p_k) contributes
        C*a_n*a_k*(ratio**(p_n+p_k)-ratio**p_m)/(2*(p_m-p_n-p_k)).
        This is a finite analytic expansion, with no fitted time knots.
        """
        exponents = (1.0, 3.5, 7.0, 11.5, 17.0)
        evolved = [a * ratio.pow(p) for a, p in zip(modes, exponents)]
        for i, n in enumerate(range(2, 7)):
            for j in range(i, 5):
                k = j + 2
                # Product identities give sum and difference harmonics.
                interactions = ((n + k, (3 * n * n - 1) / 2),) if i == j else (
                    (n + k, n * n + k * k + n * k - 1),
                    (k - n, n * n + k * k - n * k - 1),
                )
                for m, coefficient in interactions:
                    if 2 <= m <= 6:
                        source_power = exponents[i] + exponents[j]
                        target_power = exponents[m - 2]
                        forcing = (
                            ratio.pow(source_power) - ratio.pow(target_power)
                        ) / (2 * (target_power - source_power))
                        evolved[m - 2] = (
                            evolved[m - 2] + coefficient * modes[i] * modes[j] * forcing
                        )
        return evolved

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

        Relative modes decay as (R/R0)**(m*m-2), with all quadratic
        interactions among modes 2,...,6 supplied by curvature_modes.
        The normalization preserves enclosed area exactly for the corresponding
        unregularized five-mode radial graph.
        """
        radius2 = radius0.square() - 2 * speed * t
        radius = torch.sqrt(radius2)
        ratio = radius2 / radius0.square()
        evolved = Candidate.curvature_modes(ratio, modes)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        xx = (x - cx) * cosine + (y - cy) * sine
        yy = -(x - cx) * sine + (y - cy) * cosine
        core2 = 0.015**2
        rho2 = xx.square() + yy.square() + core2
        rho = torch.sqrt(rho2)
        c1, s1 = xx / rho, yy / rho
        cosine_mode, sine_mode = c1, s1
        radial, first, second = 1 + rho * 0, rho * 0, rho * 0
        for m, amplitude in zip(range(2, 7), evolved):
            cosine_mode, sine_mode = (cosine_mode * c1 - sine_mode * s1,
                                      sine_mode * c1 + cosine_mode * s1)
            radial = radial + amplitude * cosine_mode
            first = first - m * amplitude * sine_mode
            second = second - m*m * amplitude * cosine_mode
        scale = torch.sqrt(radius2 + core2) / torch.sqrt(1 + sum(e.square() for e in evolved) / 2)
        boundary = scale * radial
        angular_slope = scale * first
        angular_second = scale * second
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
        modes = (0.30 + 0.12 * p[1], 0.10 * p[2], -0.025 + 0.10 * p[3],
                 0.06 * p[12], 0.06 * p[13])
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
