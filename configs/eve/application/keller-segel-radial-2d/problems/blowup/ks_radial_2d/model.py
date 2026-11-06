"""Positive Gaussian mixture with exact first and second Gaussian IVP jets.

u(r,t) = 9.5 sum_j w_j(t)/a_j(t) exp(-r^2/a_j(t)).
Weights sum to one; sum_j w_j a_j = 1 - 0.75 t. The radial mass is
4.75 and integral r^3 u dr = 4.75 - 3.5625 t. These are whole-space
identities; the remote boundary is checked separately.

Two positive core groups enter at orders t and t^2, with initial widths 1/2
and 1/3. For A=9.5 and x=r^2 the PDE fixes

u_tt(r,0) = A[16x^2-(64+8A)x+32+12A+A^2]exp(-x)
            + A^2[48x-32-8A]exp(-2x) + 9A^3 exp(-3x).

The t-order group's total weight has derivatives A and -4A-4A^2; its
log-width derivative is six. The t^2 group's total weight has second
derivative 3A^2. The halo raw width is 1+4t. Moment normalization has
zero first and second derivatives at zero. Thus u, u_t and u_tt at zero
are exact for every fitted coefficient under ordinary coordinate AD.

Prior: the critical profile 8a/(a+r^2)^2 is a positive Gaussian scale
mixture: (1+y)^-2 = integral_0^infinity s exp(-s) exp(-sy) ds.
Each group's fixed log rates initialize a broad collection of scales. Their
conditional weights and widths evolve through fitted time polynomials.
Later evolution is specified by fitted polynomials, without a prescribed
blow-up clock, trajectory lookup or embedded PDE stepping.

With tau=t/T, bounded fitted Chebyshev curves P specify raw group masses
q1=A*t/(1+(A+2)*t)*exp(tau^2*P1),
q2=1.5*A^2*t^2/(1+(A+2)*t)*exp(tau*P2).
Their weights are qg/(1+q1+q2) times within-group softmax probabilities.
Raw widths are b0=1+4t, b1j=.5*(1+6t)*exp(tau^2*(Pj-log_rate_j)),
and b2j=(1/3)*exp(tau*(Pj-log_rate_j)). All widths are normalized to
the exact radial moment. The rational factors are positive continuations
of local PDE jets, not time stepping or a prescribed concentration clock.

Default disclosure: halo plus six Gaussians in each core group; 24 time
curves (two group masses, twelve widths, ten conditional logits) using 13
fixed Chebyshev basis functions; 312 trainable scalars and 12 persistent
fixed log rates (two copies of six rates from log(.04) to log(10)).
All fitted quantities are in time_coefficients. Coordinate AD is ordinary
PyTorch AD, including at zero time. Analytic formulas only accelerate the
training loss and do not alter the exported field.

Mass/moment identity and critical profile: Davila et al. (2024),
Introduction and Eq. (1.2),
https://link.springer.com/article/10.1007/s00205-024-02006-7
The second initial time derivative above is derived directly from the
equation and Gaussian initial condition supplied in this workspace.
"""
from __future__ import annotations

import math

import torch
from torch import nn

DTYPE = torch.float64
T_MAX = 0.86652


class Candidate(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        self.n_core = int(config.get("n_core", 12))
        self.n_first = int(config.get("n_first", self.n_core // 2))
        if not 1 <= self.n_first < self.n_core:
            raise ValueError("Both positive core groups must have at least one component")
        self.degree = int(config.get("time_degree", 12))
        # Two group masses, K widths, (K-2) conditional logits.
        self.time_coefficients = nn.Parameter(
            torch.zeros(2 * self.n_core, self.degree + 1, dtype=DTYPE)
        )
        self.register_buffer(
            "log_rates",
            torch.cat([
                torch.linspace(math.log(0.04), math.log(10.0), count, dtype=DTYPE)
                for count in (self.n_first, self.n_core - self.n_first)
            ]),
        )

    def time_basis(self, tau, derivatives=False):
        """Fixed Chebyshev basis and its derivative with respect to tau."""
        z = 2 * tau - 1
        basis = [torch.ones_like(z)]
        slope = [torch.zeros_like(z)]
        if self.degree:
            basis.append(z)
            slope.append(torch.full_like(z, 2.0))
        for _ in range(2, self.degree + 1):
            basis.append(2 * z * basis[-1] - basis[-2])
            if derivatives:
                slope.append(4 * basis[-2] + 2 * z * slope[-1] - slope[-2])
        if not derivatives:
            return torch.cat(basis, dim=1)
        return torch.cat(basis, dim=1), torch.cat(slope, dim=1)

    def raw_curves(self, times, derivatives=False):
        tau = times.reshape(-1, 1).to(dtype=DTYPE) / T_MAX
        if not derivatives:
            return self.time_basis(tau) @ self.time_coefficients.T
        basis, slope = self.time_basis(tau, True)
        return basis @ self.time_coefficients.T, slope @ self.time_coefficients.T

    def temporal_components(self, times, derivatives=False):
        if derivatives:
            raw, raw_tau = self.raw_curves(times, True)
            return self.components_from_raw(times, raw, raw_tau)
        return self.components_from_raw(times, self.raw_curves(times))

    def components_from_raw(self, times, raw, raw_tau=None):
        """Same temporal algebra with explicit polynomial values/slopes.

        This separation permits a small exact training Jacobian. Forward
        values and coordinate derivatives always use ordinary PyTorch AD.
        """
        t = times.reshape(-1, 1).to(dtype=DTYPE)
        tau = t / T_MAX
        squashed = torch.tanh(raw / 12.0)
        correction = 12.0 * squashed
        if raw_tau is not None:
            correction_tau = (1 - squashed.square()) * raw_tau
        f1, f2 = correction[:, :1], correction[:, 1:2]
        scale = correction[:, 2:self.n_core + 2]
        logits = correction[:, self.n_core + 2:]
        # Each softmax has one fixed pivot, removing both logit gauges.
        conditional = []
        conditional_t = []
        offset = 0
        start = 0
        for count in (self.n_first, self.n_core - self.n_first):
            pivot = count // 2
            local = logits[:, offset:offset + count - 1]
            local = torch.cat((local[:, :pivot], torch.zeros_like(t), local[:, pivot:]), dim=1)
            rates = self.log_rates[start:start + count]
            prob = torch.softmax(local + rates - rates.exp(), dim=1)
            conditional.append(prob)
            if raw_tau is not None:
                local_t = correction_tau[:, self.n_core + 2 + offset:
                                         self.n_core + 2 + offset + count - 1] / T_MAX
                local_t = torch.cat((local_t[:, :pivot], torch.zeros_like(t),
                                     local_t[:, pivot:]), dim=1)
                conditional_t.append(prob * (local_t - (prob * local_t).sum(1, keepdim=True)))
            offset += count - 1
            start += count

        amplitude = 9.5
        denominator = 1 + (amplitude + 2) * t
        exp1, exp2 = torch.exp(tau.square() * f1), torch.exp(tau * f2)
        transfer1 = amplitude * t / denominator * exp1
        transfer2 = 1.5 * amplitude ** 2 * t.square() / denominator * exp2
        normalizer = 1 + transfer1 + transfer2
        fraction1, fraction2 = transfer1 / normalizer, transfer2 / normalizer
        weights = torch.cat((normalizer.reciprocal(), fraction1 * conditional[0],
                             fraction2 * conditional[1]), dim=1)
        raw_halo = 1 + 4 * t
        shifted = scale - self.log_rates
        raw_core1 = 0.5 * (1 + 6 * t) * torch.exp(tau.square() * shifted[:, :self.n_first])
        raw_core2 = (1 / 3) * torch.exp(tau * shifted[:, self.n_first:])
        raw_core = torch.cat((raw_core1, raw_core2), dim=1)
        raw_widths = torch.cat((raw_halo, raw_core), dim=1)
        mean_raw_width = (weights * raw_widths).sum(1, keepdim=True)
        second_moment = 1 - 0.75 * t
        widths = raw_widths * second_moment / mean_raw_width
        if raw_tau is None:
            return weights, widths

        f1_tau, f2_tau = correction_tau[:, :1], correction_tau[:, 1:2]
        scale_tau = correction_tau[:, 2:self.n_core + 2]
        transfer1_t = amplitude * exp1 / denominator.square()
        transfer1_t = transfer1_t + transfer1 * (2 * tau * f1 + tau.square() * f1_tau) / T_MAX
        transfer2_t = 1.5 * amplitude ** 2 * t * (2 + (amplitude + 2) * t) * exp2 / denominator.square()
        transfer2_t = transfer2_t + transfer2 * (f2 + tau * f2_tau) / T_MAX
        normalizer_t = transfer1_t + transfer2_t
        fraction1_t = transfer1_t / normalizer - fraction1 * normalizer_t / normalizer
        fraction2_t = transfer2_t / normalizer - fraction2 * normalizer_t / normalizer
        weights_t = torch.cat((-normalizer_t / normalizer.square(),
                               fraction1_t * conditional[0] + fraction1 * conditional_t[0],
                               fraction2_t * conditional[1] + fraction2 * conditional_t[1]), dim=1)
        halo_log_t = 4 / (1 + 4 * t)
        core1_log_t = 6 / (1 + 6 * t) + (2 * tau * shifted[:, :self.n_first]
                       + tau.square() * scale_tau[:, :self.n_first]) / T_MAX
        core2_log_t = (shifted[:, self.n_first:] + tau * scale_tau[:, self.n_first:]) / T_MAX
        core_log_t = torch.cat((core1_log_t, core2_log_t), dim=1)
        raw_log_t = torch.cat((halo_log_t, core_log_t), dim=1)
        mean_raw_t = (raw_widths * (weights_t + weights * raw_log_t)).sum(1, keepdim=True)
        width_log_t = raw_log_t - 0.75 / second_moment - mean_raw_t / mean_raw_width
        return weights, widths, weights_t, width_log_t

    def component_bundle(self, local, time):
        """w, log(a), w_t, (log(a))_t at one time, for numerical fitting."""
        n = self.time_coefficients.shape[0]
        w, a, wt, lt = self.components_from_raw(
            time.reshape(1), local[:n].reshape(1, n), local[n:].reshape(1, n)
        )
        return torch.cat((w, a.log(), wt, lt), dim=1).reshape(-1)

    def value_component_bundle(self, local, time):
        """w and log(a) only; boundary values do not need time slopes."""
        w, a = self.components_from_raw(time.reshape(1), local.reshape(1, -1))
        return torch.cat((w, a.log()), dim=1).reshape(-1)

    def forward(self, points):
        x = points[:, :1].to(dtype=DTYPE).square()
        weights, widths = self.temporal_components(points[:, 1])
        return 9.5 * (weights / widths * torch.exp(-x / widths)).sum(1, keepdim=True)

    def differential_parts(self, radii, times):
        """Exact continuous u, u_t, diffusion, chemotaxis and partial mass.

        Using x=r^2 gives D=4(x*u_xx+u_x), C=2*M*u_x+u^2.
        Mass terms retain parameter gradients.
        """
        weights, widths, weights_t, width_log_t = self.temporal_components(times, True)
        x = radii.reshape(1, -1, 1).square()
        inv_a = widths[:, None, :].reciprocal()
        q = x * inv_a
        kernel = 9.5 * inv_a * torch.exp(-q)
        w = weights[:, None, :]
        u = (w * kernel).sum(2)
        ut = (kernel * (weights_t[:, None, :] +
                       w * width_log_t[:, None, :] * (q - 1))).sum(2)
        ux = -(w * kernel * inv_a).sum(2)
        uxx = (w * kernel * inv_a.square()).sum(2)
        mass = 4.75 * (w * (-torch.expm1(-q))).sum(2)
        diffusion = 4 * (x[:, :, 0] * uxx + ux)
        chemistry = 2 * mass * ux + u.square()
        return u, ut, diffusion, chemistry, mass
