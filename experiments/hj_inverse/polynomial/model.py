"""Heat semigroup plus an anchored Fourier--Legendre correction.

Write u = exp(t L_d) u0 + field_scale * sum a[k,j] phi_k(x,y) I_j(t/H),
where L_d = d_x partial_xx + d_y partial_yy and I_j(s) is the integral
from zero to s of P_j(2 r - 1). The 1,257 real Fourier functions have integer
wave vectors in the disk |k| <= 20, and j = 0,...,5. This is a finite
spectral expansion, with no learned spatial features or time integration.

The heat part treats linear diffusion exactly. The correction approximates
the nonlinear evolution, including its changing mean and newly generated
harmonics. Integrated Legendre polynomials make time derivatives reasonably
conditioned and enforce u(0)=u0 exactly. Each mode is divided by a fixed
diffusive stiffness estimate to precondition optimization, without changing
the represented function space. All four physical coefficients remain
independent; the evaluator fits them together with the spectral amplitudes.

The prescribed initial bandwidth is five. The quadratic gradient term can
increase the support of each successive initial time derivative by five:
u_t(0), u_tt(0), u_ttt(0) reach radii ten, fifteen, twenty. This disk thus
covers one more nonlinear interaction order than the sampled radius-sixteen
field. It targets the omitted spatial residual while retaining the same six
temporal functions and optimization coordinates. The 7,542 amplitudes start
at zero; no fitted field or physical coefficients are embedded.

The truncation targets the declared short horizon H. The same polynomial
formula is defined through t=0.5, but extrapolation accuracy is not asserted.
"""

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

        # Antiderivatives of shifted Legendre polynomials. The recurrence
        # constructs polynomial basis functions, not successive PDE states.
        s = time / self.horizon
        z = 2 * s - 1
        legendre = [torch.ones_like(z), z]
        for j in range(1, 6):
            legendre.append(((2 * j + 1) * z * legendre[j] - j * legendre[j - 1]) / (j + 1))
        integrals = [s]
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
