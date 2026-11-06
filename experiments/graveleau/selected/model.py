"""A two-jet shell with a fractional-power second-kind front profile.

The first two initial time derivatives are analytic consequences of the PME.
The similarity coordinate supports the first nonlinear far-field correction
without a constant pressure offset. All free corrections are global low-degree
polynomials; no numerical trajectory or time integration is used.
"""

import math
import torch
from torch import nn


def _logit(value):
    return math.log(value / (1 - value))


class Candidate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        end = 0.995 * 0.6200901156700502
        self.focus_clock = nn.Parameter(
            torch.tensor(_logit((0.6200901156700502 - end - 0.0001) / 0.08))
        )
        self.exponent = nn.Parameter(torch.tensor(_logit((0.8 - 0.6) / 0.38)))
        self.onset = nn.Parameter(torch.tensor(_logit((0.04 - 0.01) / 0.11)))
        self.clock_shape = nn.Parameter(torch.tensor(0.0))
        self.clock_skew = nn.Parameter(torch.tensor(0.0))
        self.relaxation = nn.Parameter(torch.tensor(_logit((12.0 - 2.0) / 38.0)))
        self.shell_shape = nn.Parameter(torch.zeros(3, 3))
        self.front_shape = nn.Parameter(torch.zeros(3, 3))

    @staticmethod
    def _polynomial(coefficients, time, space):
        """Collect the tensor Legendre coefficients before point evaluation.

        The two degree-two basis conversions act only on a 3-by-3 tensor.
        Nested Horner evaluation then avoids repeating basis conversions at
        every physical point, including in higher autodiff derivatives.
        """
        bounded = 3 * torch.tanh(coefficients / 3)
        temporal = torch.stack((
            bounded[0] - bounded[1] + bounded[2],
            2 * bounded[1] - 6 * bounded[2],
            6 * bounded[2],
        ))
        monomial = torch.stack((
            temporal[:, 0] - temporal[:, 1] + temporal[:, 2],
            2 * temporal[:, 1] - 6 * temporal[:, 2],
            6 * temporal[:, 2],
        ), dim=-1)
        modes = (
            (monomial[2] * time[..., None] + monomial[1])
            * time[..., None] + monomial[0]
        )
        constant, linear, quadratic = modes.unbind(dim=-1)
        return (quadratic * space + linear) * space + constant

    @staticmethod
    def _initial_log_rates(distance):
        """Return (log q)_t and (log q)_tt for q(r,0)=(r-1)^2/w^2.

        In v=A*(1-exp(-q)), the pressure equation becomes
        q_t=A*((1-E)*Delta(q)+(2*E-1)*q_r^2), E=exp(-q).
        Differentiating this identity once gives the second rate. Expressing
        it with z=k*d^2, j=d/(1+d), E=exp(-z), F=(1-E)/z removes
        intermediate spatial derivatives and their repeated autodiff graphs.
        """
        amplitude, inverse_width2 = 1.6, 1 / 0.6**2
        y, k = distance, inverse_width2
        z = k * y.square()
        decay = torch.exp(-z)
        complement = -torch.expm1(-z)
        series_region = z < 0.01
        # The quotient is used at equality too. A matching where guard keeps
        # its derivative equal to one there, independent of clamp conventions.
        denominator = torch.where(series_region, torch.ones_like(z), z)
        ratio = torch.where(
            series_region,
            1 + z * (-1/2 + z * (1/6 + z * (-1/24 + z * (1/120 + z * (-1/720 + z/5040))))),
            complement / denominator,
        )
        radial_fraction = y / (1 + y)
        scale = amplitude * k
        first = scale * (2 * (1 + radial_fraction) * ratio + 8 * decay - 4)
        # Collect powers of E and F before autodiff; this is the same exact
        # second initial jet with fewer repeated radial subexpressions.
        x = radial_fraction
        decay_coefficient = 88 + 24 * x - 96 * z
        mixed_coefficient = z * (32 * z - 104 - 40 * x) - 8 + x * (24 - 16 * x)
        radial_coefficient = 4 + x * (8 + x * (6 - 2 * x))
        second = scale**2 * (
            16 + decay * (decay * (2 * decay_coefficient - 64) - decay_coefficient)
            + ratio * (
                decay * mixed_coefficient + 8 * (1 + x.square())
                - ratio * radial_coefficient
            )
        )
        return z, first, second

    def front(self, time):
        """Return R and -R' with a monotone clock flat to all orders at zero.

        Both global warp modes vanish at zero and focus. Their contributions
        to clock'/clock_factor are c*s*(1-2*s) and
        k*s*(1-6*s+6*s^2). Each polynomial has absolute value at most one
        for 0<=s<=1. Bounds |c|<1/2 and |k|<1/4 therefore leave clock'>0.
        The cubic mode changes acceleration on the whole focusing interval,
        while retaining the flat initial clock and the same focus exponent.
        """
        end = 0.995 * 0.6200901156700502
        focus = end + 0.0001 + 0.08 * torch.sigmoid(self.focus_clock)
        beta = 0.6 + 0.38 * torch.sigmoid(self.exponent)
        waiting = 0.01 + 0.11 * torch.sigmoid(self.onset)

        # The quadratic initial edge has zero Darcy speed. This smooth clock
        # is flat at zero and becomes linear on the focusing time scale.
        safe_time = torch.where(time > 0, time, torch.ones_like(time))
        phase = time / focus
        warp_strength = 0.5 * torch.tanh(self.clock_shape)
        warp_skew = 0.25 * torch.tanh(self.clock_skew)
        warp = phase * (1 - phase) * (
            warp_strength + warp_skew * (1 - 2 * phase)
        )
        # Normalize the flat clock with one exponential. Its extension and
        # all derivatives at zero agree with t*exp(-waiting/t)/C(focus).
        clock_factor = torch.where(
            time > 0,
            torch.exp(
                waiting * (1 / focus - 1 / safe_time)
                + warp
            ) / focus,
            torch.zeros_like(time),
        )
        clock_derivative = clock_factor * (
            1 + waiting / safe_time + warp_strength * phase * (1 - 2 * phase)
            + warp_skew * phase * (1 - 6 * phase + 6 * phase.square())
        )
        remaining = (1 - time * clock_factor).clamp_min(1e-10)
        front = remaining.pow(beta)
        speed = beta * clock_derivative * remaining.pow(beta - 1)
        return front, speed, beta

    def shell_exponent(self, distance, time, front, speed, beta):
        """The same shell exponent for one or both reflected coordinates.

        Exposing this expression lets the loss differentiate the exponent
        and apply the elementary pressure chain rule analytically.
        """
        end = 0.995 * 0.6200901156700502
        rate = 2 + 38 * torch.sigmoid(self.relaxation)
        shell_coordinate = distance / (0.6 + distance)
        theta = time / end
        z, first, second = self._initial_log_rates(distance)

        # This rational continuation has exactly l_t=first, l_tt=second at
        # zero. Shape freedom begins at order t^3, across the whole field.
        diffusion_time = 0.6**2 / (4 * 1.6)
        shell_time = time / (diffusion_time + time)
        shell_log_shape = (
            time / (1 + rate * time) * first
            + 0.5 * (time / (1 + 2 * rate * time)).square()
            * (second + 2 * rate * first)
            - torch.log1p((rate * time).pow(3)) * (1 - shell_coordinate).square()
            + shell_time.pow(3)
            * self._polynomial(self.shell_shape, shell_time, shell_coordinate)
        )
        quadratic_shell = z * torch.exp(shell_log_shape)

        # For eta=1+d/R and p=2-1/beta, use x=1-eta^(p-2).
        # Psi=beta*eta^p*x*exp(x*P(x)) has Psi'(1)=1 and far powers
        # eta^p, eta^(2p-2), ... . Unlike (eta^p-1)/p, it has no
        # constant term in the far expansion of the similarity profile.
        power = 2 - 1 / beta
        log_eta = torch.log1p(distance / front)
        front_coordinate = -torch.expm1(-log_eta / beta)
        similarity_profile = (
            front * beta * torch.exp(power * log_eta) * front_coordinate
        )
        # With zero free shape, this analytic offset gives the similarity
        # edge curvature Psi''(1)/2=-p/4. It is an initialization choice;
        # the free polynomial can alter the curvature at every later time.
        edge_offset = 1.75 - 2 * beta
        linear_front = speed / 1.6 * similarity_profile * torch.exp(
            front_coordinate * (
                edge_offset + self._polynomial(self.front_shape, theta, front_coordinate)
            )
        )
        return quadratic_shell + linear_front

    def forward(self, points):
        radius, time = points[:, :1], points[:, 1:2]
        front, speed, beta = self.front(time)
        raw_distance = torch.cat((radius - front, 12 - radius - front), dim=1)
        # Select the exterior derivative at equality explicitly, independent
        # of the installed torch version's clamp derivative convention.
        distance = torch.where(raw_distance >= 0, raw_distance, torch.zeros_like(raw_distance))
        pressure_factors = -torch.expm1(-self.shell_exponent(
            distance, time, front, speed, beta,
        ))
        inner, reflected = pressure_factors[:, :1], pressure_factors[:, 1:2]
        # Symmetric composition imposes v_r(6,t)=0 while preserving the edge.
        product = inner * reflected
        return 1.6 * product / (inner + reflected - product).clamp_min(1e-15)
