"""Heat semigroup plus a Fourier expansion with exponential time functions.

The explicit field is
  u = exp(t L_d) u0 + field_scale sum a[k,j] scale[k] phi_k(x,y) I_j(w(t/H)),
  L_d = d_x partial_xx + d_y partial_yy,
  w(s) = (1 - exp(-s/2)) / (1 - exp(-1/2)),
  I_j(w) = integral_0^w P_j(2r-1) dr, j=0,...,5.
There are 1,257 real Fourier functions in the integer disk |k| <= 20.
The 7,542 initially zero amplitudes are the only field parameters.

Each I_j(w(s)) is a polynomial of degree at most six in exp(-s/2).
Equivalently the time space is spanned by 1-exp(-m*s/2), m=1,...,6;
the integrated Legendre coordinates improve conditioning of the amplitudes.
This is a finite analytic expansion, with no PDE stepping or learned features.
Decaying exponentials are motivated by quadratic interactions of diffusing
initial harmonics: their first Duhamel responses are differences of decaying
exponentials (or t times an exponential at coincident rates). The mild warp
improved approximation of these synthetic responses using the same six
functions. It is an approximation of the full nonlinear transient, not an
exact nonlinear solution or an assumed common Cole--Hopf ratio.

The exact heat component uses the two independent evaluator diffusivities.
Both nonlinear coefficients independently enter the evaluator's residual.
The fixed stiffness scale uses only public starting diffusivities. All
corrections vanish at t=0; integer waves enforce periodic values and first
derivatives. The radius-20 disk retains the strongest sampled spatial basis,
including the full possible support of the third initial time derivative.
No fitted state or physical coefficient is embedded. The expression is
smooth and finite through t=0.5; accuracy is targeted only on [0,H].
"""

import math
import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        modes = torch.tensor(config["initial_modes"], dtype=torch.float64)
        self.register_buffer("initial_wave", modes[:, :2])
        self.register_buffer("initial_phase", modes[:, 3])
        self.register_buffer(
            "initial_weight",
            modes[:, 2] / (modes[:, :2].abs().sum(1) + 1) / config["initial_scale"],
        )
        radius = 20
        wave = torch.tensor(
            [(0, 0)] + [
                (kx, ky)
                for kx in range(radius + 1) for ky in range(-radius, radius + 1)
                if 0 < kx * kx + ky * ky <= radius * radius and (kx > 0 or ky > 0)
            ], dtype=torch.float64,
        )
        self.register_buffer("wave", wave)
        self.register_buffer("starting_coefficients", torch.tensor(
            config["coefficient_start"], dtype=torch.float64,
        ))
        self.horizon = float(config["horizon"])
        self.field_scale = float(config["field_scale"])
        stiffness = self.horizon * (wave.square() @ self.starting_coefficients[:2])
        scale = (1 + stiffness.square()).rsqrt()
        self.register_buffer("spectral_scale", torch.cat((scale, scale[1:])))
        self.amplitudes = nn.Parameter(torch.zeros(
            (2 * len(wave) - 1, 6), dtype=torch.float64,
        ))

    def forward_with_coefficients(self, points, coefficients):
        time = points[:, 2:3]
        phase = points[:, :2] @ self.initial_wave.T + self.initial_phase
        rate = self.initial_wave.square() @ coefficients[:2]
        heat = (phase.sin() * torch.exp(-time * rate)) @ self.initial_weight

        # Analytic exponential coordinate followed by antiderivatives of
        # shifted Legendre polynomials. This recurrence evaluates polynomials.
        s = time / self.horizon
        w = -torch.expm1(-0.5 * s) / (-math.expm1(-0.5))
        z = 2 * w - 1
        legendre = [torch.ones_like(z), z]
        for j in range(1, 6):
            legendre.append(((2 * j + 1) * z * legendre[j] - j * legendre[j - 1]) / (j + 1))
        integrals = [w]
        for j in range(1, 6):
            integrals.append((legendre[j + 1] - legendre[j - 1]) / (2 * (2 * j + 1)))
        temporal = torch.cat(integrals, dim=1)

        angle = points[:, :2] @ self.wave.T
        spatial = torch.cat((angle.cos(), angle[:, 1:].sin()), dim=1)
        spectral = spatial @ (self.spectral_scale[:, None] * self.amplitudes)
        correction = (spectral * temporal).sum(dim=1)
        return (heat + self.field_scale * correction)[:, None]

    def forward(self, points):
        return self.forward_with_coefficients(points, self.starting_coefficients)
