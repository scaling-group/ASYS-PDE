"""Positive analytic envelope with separate core and fractional-tail modes.

The supplied priors are oddness, compactification, the leading power envelope
and a free positive beta.  The representation change is the bounded even
logarithmic correction

    r = sqrt(1 + x**2), s = r**(-2), q = x**2 / r**2,
    p = 1/beta, w = r**(-p), z = 2*s - 1,
    h(x) = (x/r) * w * exp(C(x)),
    C(x) = q * [-c + s * sum_{j=0}^5 b_j T_j(z)
                  + w * (d_1 + d_2*w)].

Here c is the starter's correction, b_j are core Chebyshev coefficients and
d_1, d_2 are fractional-tail coefficients.  For finite coefficients and beta>0,
the correction is bounded, even, and zero at the origin.  Thus h is smooth and
odd, h'(0)=1, h(x)>0 for x>0, and h(x) ~ exp(-c)*x**(-1/beta).
Consequently (2/pi)*integral_0^infinity h(x)/x dx is finite and positive.
Ordinary half-line L1 mass is not imposed.

Source hypothesis: the public starter qualified with residual mean square
1.0986155992864174e-5, but its log correction has only even inverse powers
in the tail.  Dividing the equation by f shows that a leading f ~ x**(-p)
generically induces a relative x**(-p) correction through g and V/x.  The w
and w**2 modes admit the first two such powers; s times a polynomial supplies
independent even-power/core corrections.  This is representational freedom,
not a new Hilbert identity or a claimed solution of the equation.

Initialization is exactly the supplied unfitted envelope: beta starts from
config['initial_beta'] (1 in the task), and c, all b_j, and both d_j are zero.
No feedback coefficients or beta values are used to initialize the model.
Normalization, coefficient fitting and equation diagnostics belong exclusively
to the protected evaluator.  Forward evaluates this formula without fitting,
quadrature, mutable caches, or data-dependent state.

Complexity: ten trainable scalar parameters, no numerical buffers.  A call on
N coordinates uses O(N) work/storage, including two exponentials and one
degree-five Horner polynomial per coordinate.  The finite expansion can still
have approximation and fitting error; algebra/interface checks cannot establish
an improved equation residual, existence, or reference accuracy.
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
        self.correction = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.core_chebyshev = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        self.tail_coefficients = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        # hypot avoids overflowing x**2 on the evaluator's large log domains.
        # Computing q from x/r also retains its x**2 behavior near the origin.
        radius = torch.hypot(x, torch.ones_like(x))
        signed_radius = x / radius
        s = radius.reciprocal().square()
        q = signed_radius.square()
        w = torch.exp(-torch.log(radius) / self.beta())
        z = 2 * s - 1

        # Exact Chebyshev-to-monomial conversion followed by Horner evaluation.
        # This avoids building an N-by-6 basis tensor at operator quadrature sites.
        b0, b1, b2, b3, b4, b5 = self.core_chebyshev.unbind()
        core = 16 * b5 * z + 8 * b4
        core = core * z + (4 * b3 - 20 * b5)
        core = core * z + (2 * b2 - 8 * b4)
        core = core * z + (b1 - 3 * b3 + 5 * b5)
        core = core * z + (b0 - b2 + b4)

        d1, d2 = self.tail_coefficients.unbind()
        log_correction = q * (-self.correction + s * core + w * (d1 + d2 * w))
        return signed_radius * w * torch.exp(log_correction)


def build_model(config):
    return Model(config)
