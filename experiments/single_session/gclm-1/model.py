"""Positive analytic log-shape expansion of the supplied cold envelope.

Write r = sqrt(1+x^2), s = r^-2, u = 1-s, and p = 1/beta.  The
explicit raw profile is

    h(x) = (x/r) r^-p exp(u [c_inf + s P_5(2s-1) + c_tail r^-p]),
    P_5(z) = sum_{j=0}^5 c_j T_j(z).

The supplied priors are oddness, compactification, and the leading power
envelope.  The representation hypothesis is that independent core curvature
and subleading tail terms can remove errors inaccessible to a single log
correction.  The polynomial supplies even inverse powers in the tail; the
extra r^-p term also permits the fractional order suggested by the nonlocal
equation's power-law terms.  No asymptotic coefficient is imposed or fitted
here.  The public baseline only supplies an aggregate residual, so it cannot
establish which region causes that residual or validate this hypothesis.

All shape coefficients start at zero and beta starts at config["initial_beta"]
(1 in the supplied configuration): exactly the supplied unfitted envelope.
No public fitted values enter initialization.  For finite coefficients and
beta > 0 the log correction is bounded, h'(0)=1, h(x)>0 for x>0, and
h(x) ~ exp(c_inf) x^-p as x -> +infinity.  Thus integral h(x)/x dx is finite
and positive.  Ordinary L1 integrability is neither imposed nor implied when
beta >= 1.  Normalization and numerical coefficient fitting belong solely to
the evaluator; this module only evaluates an explicit expression.

There are nine trainable scalars.  Chebyshev coefficients are converted by a
fixed 6-by-6 matrix and evaluated with five Horner steps: O(6N) arithmetic
and O(N) working storage on N input sites, without quadrature or a solver.
Extra modes still cost field work and may be correlated during fitting;
neither accuracy nor completion within the fit budget is certified here.
"""
import math
import torch


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_beta = float(config["initial_beta"])
        if not math.isfinite(initial_beta) or initial_beta <= 0:
            raise ValueError("initial_beta must be finite and positive")
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(initial_beta), dtype=torch.float64))
        self.far_field_log_amplitude = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float64))
        self.core_chebyshev_coefficients = torch.nn.Parameter(
            torch.zeros(6, dtype=torch.float64))
        self.fractional_tail_coefficient = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float64))

        # Column j contains the monomial coefficients of T_j(z).  This is
        # polynomial algebra (T_{j+1}=2zT_j-T_{j-1}), not coefficient fitting.
        transform = torch.zeros((6, 6), dtype=torch.float64)
        transform[0, 0] = 1
        transform[1, 1] = 1
        for degree in range(1, 5):
            transform[1:, degree + 1] = 2 * transform[:-1, degree]
            transform[:, degree + 1] -= transform[:, degree - 1]
        self.register_buffer("chebyshev_to_monomial", transform)

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        # hypot avoids overflow in x^2.  Factoring out x/r also avoids an
        # underflowing r^(-1-p) before multiplication by a large x.
        radius = torch.hypot(x, torch.ones_like(x))
        signed_compact_radius = x / radius
        s = radius.reciprocal().square()
        # Compute u this way rather than 1-s to retain the O(x^2) term near 0.
        u = signed_compact_radius.square()
        z = 2 * s - 1
        coefficients = self.chebyshev_to_monomial @ self.core_chebyshev_coefficients
        polynomial = coefficients[-1]
        for coefficient in coefficients[:-1].flip(0):
            polynomial = polynomial * z + coefficient

        power_envelope = torch.exp(-radius.log() / self.beta())
        log_correction = u * (
            self.far_field_log_amplitude + s * polynomial
            + self.fractional_tail_coefficient * power_envelope)
        return signed_compact_radius * power_envelope * torch.exp(log_correction)


def build_model(config):
    return Model(config)
