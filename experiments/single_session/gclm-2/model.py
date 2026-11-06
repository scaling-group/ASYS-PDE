"""Positive analytic envelope with independent core and tail corrections.

The supplied priors are oddness, the two scale gauges, compactification, and
the leading power |x|**(-1/beta).  The representation contribution here is a
bounded log correction with separate tail amplitude, fractional tail response,
and six Chebyshev core coefficients.  No equation solve or normalization is
performed by this module.

Write r = sqrt(1+x*x), u = x/r, s = r**(-2), z = u*u, and
tau = r**(-1/beta).  The raw profile is

    h(x) = u*tau*exp(C(x)),
    C(x) = tail_amplitude*z + tail_response*z*tau
           + 4*z*s*sum(core[j]*T_j(2*z-1), j=0,...,5).

The extra z*tau term allows the relative |x|**(-1/beta) tail correction
suggested by balancing the nonlocal terms against h + beta*x*h_x.  Its
coefficient is free, not taken from a reference solution.  The core polynomial
also permits independent even curvature and inverse-square tail corrections.

C is smooth and even, C(0)=0, and |C| is bounded by the sum of the absolute
coefficients.  Consequently h is odd, h'(0)=1, and h(x)>0 for x>0.  For every
finite coefficient vector and beta>0, h(x) ~ exp(tail_amplitude)*x**(-1/beta)
as x tends to positive infinity.  Thus (2/pi)*integral h(x)/x dx is finite and
positive; ordinary half-line L1 integrability is not imposed.  The evaluator
alone computes and freezes the canonical amplitude and coordinate scale.

Initialization is exactly the supplied unfitted envelope: beta starts at the
configured value (1 in the supplied config), and every correction is zero.
There are nine trainable scalars and no buffers.  Evaluation uses a degree-five
polynomial, one fractional power via exp/log, and the final exponential; its
work and temporary storage are linear in the number of input elements.  This
finite analytic family is an approximation, not an existence certificate.
"""
import math
import torch


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_beta = float(config.get("initial_beta", 1.0))
        if not math.isfinite(initial_beta) or initial_beta <= 0:
            raise ValueError("initial_beta must be finite and positive")
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(initial_beta), dtype=torch.float64))
        self.log_tail_amplitude = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.fractional_tail_response = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.core_chebyshev = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        # hypot avoids overflowing x*x at large finite radii.  Splitting off
        # x/r also avoids premature underflow of the full envelope factor.
        radius = torch.hypot(x, torch.ones_like(x))
        unit_radius = x / radius
        s = radius.reciprocal().square()
        z = unit_radius.square()
        tau = torch.exp(-torch.log(radius) / self.beta())
        t = 2 * z - 1

        # Exact conversion of sum a_j*T_j(t) to Horner form.  Degree five is
        # small enough to avoid a basis table and its N-by-6 storage/graph.
        a0, a1, a2, a3, a4, a5 = self.core_chebyshev.unbind()
        core = 16 * a5 * t + 8 * a4
        core = core * t + 4 * a3 - 20 * a5
        core = core * t + 2 * a2 - 8 * a4
        core = core * t + a1 - 3 * a3 + 5 * a5
        core = core * t + a0 - a2 + a4

        correction = z * (self.log_tail_amplitude
                          + self.fractional_tail_response * tau
                          + 4 * s * core)
        return unit_radius * tau * torch.exp(correction)


def build_model(config):
    return Model(config)
