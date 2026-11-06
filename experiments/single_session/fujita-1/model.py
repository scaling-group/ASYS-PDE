"""Transient inverse-amplitude expansion, with the initial heat corner.

For v=u**(-2), the equation is
    v_t = Delta_5 v - (3/2) v_r**2/v - 2.
The initial inverse curvature is b0=pi**2/2500.  Locally, v=a+b*r**2
gives a'=-2+10*b and b' ~ -6*b**2/a.  With a ~ 2*(T-t), this
suggests logarithmic curvature relaxation, rather than a stationary
nonconstant similarity profile.  We extend that logarithm using the local
inverse amplitude, so it also makes sense in the outer field.

The numerator imposes the exact initial PDE tangent.  Its incompatible wall
value is removed by the integrated complementary-error-function heat layer.
The fitted coefficients are low-order moments of the remaining log amplitude
in time, initial radius, and the evolving core coordinate.  They start at zero;
there are no trajectory data or time integration in this representation.
"""
import math
import torch


def _sinc(x):
    """sin(x)/x with regular *second* autograd derivatives at the origin."""
    z = x.square()
    small = z < 0.01
    safe = torch.where(small, torch.ones_like(x), x)
    series = 1 + z * (-1/6 + z * (1/120 + z * (-1/5040 + z *
             (1/362880 + z * (-1/39916800 + z/6227020800)))))
    return torch.where(small, series, safe.sin()/safe)


def _chebyshev(x, degree):
    terms = [torch.ones_like(x)]
    if degree:
        terms.append(x)
    for _ in range(2, degree + 1):
        terms.append(2*x*terms[-1] - terms[-2])
    return torch.cat(terms, dim=1)


def _integrated_erfc(distance, positive_time):
    """Integral_0^t erfc(d/(2*sqrt(s))) ds, for t>0."""
    root = positive_time.sqrt()
    z = distance/(2*root)
    return ((positive_time + distance.square()/2)*torch.erfc(z)
            - distance*root/math.sqrt(math.pi)*torch.exp(-z.square()))


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # Retain the supplied seed's clock and width initializations.
        self.clock_margin = torch.nn.Parameter(torch.tensor(
            math.log(math.expm1(0.001)), dtype=torch.float64))
        self.log_width = torch.nn.Parameter(torch.tensor(
            math.log(0.01), dtype=torch.float64))
        self.outer_moments = torch.nn.Parameter(torch.zeros(4, 5, dtype=torch.float64))
        self.core_moments = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))

    def _wall_layer(self, r, t):
        """Even origin extension of the wall's diffusion layer.

        The polynomial approximates (R/r)^2 near R, as suggested by
        (r^2*u)_t=(r^2*u)_rr-2*(r^2*u)/r^2 for linear diffusion.
        It is regular at zero.  The paired distances make the layer even;
        normalization makes its wall value exactly t.
        """
        positive = t > 0
        safe_t = torch.where(positive, t, torch.ones_like(t))
        pair = (_integrated_erfc(5-r, safe_t)
                + _integrated_erfc(5+r, safe_t))
        wall = safe_t + _integrated_erfc(torch.full_like(r, 10), safe_t)
        y = 1-r.square()/25
        prefactor = 1+y+y.square()+y.pow(3)
        layer = prefactor*pair*safe_t/wall
        # The interior t=0 limit is flat to all orders.  At the wall the
        # incompatible initial tangent is intentionally not imposed.
        return torch.where(positive, layer, torch.zeros_like(layer))

    def forward(self, coordinates):
        r, t = coordinates[:, :1], coordinates[:, 1:]
        k = math.pi/10
        cosine, sine = torch.cos(k*r), torch.sin(k*r)
        c2, y = cosine.square(), sine.square()
        g = 5*cosine
        diffusion = -5*k*k*cosine - 20*k*k*_sinc(k*r)
        rhs = diffusion + g.pow(3)

        # A positive margin keeps the field finite throughout the prescribed
        # interval.  The extra 1e-12 protects the endpoint against underflow.
        clock = 0.020385594 + 0.020406*(
            torch.nn.functional.softplus(self.clock_margin) + 1e-12)
        s = t/clock
        initial_a_rate = -50 + 10*k*k
        a = (1-s)*torch.exp((1+clock*initial_a_rate)*s)
        local_clock = a*c2+y

        b0 = k*k/25
        # Width is a positive multiplier of the leading curvature relaxation.
        beta = 3*b0*torch.exp(2*torch.tanh(
            (self.log_width-math.log(0.01))/2))
        relaxed_width = 1/(1-beta*torch.log(local_clock))
        denominator = a*c2 + relaxed_width*y

        # denominator_t(r,0), including the spatially extended logarithm.
        initial_q_rate = initial_a_rate*c2*(1+beta*y)
        numerator_rate = rhs + g*initial_q_rate/2
        numerator = (g + t*numerator_rate
                     + (2*math.pi/5)*self._wall_layer(r, t))

        # Compactified r/sqrt(T-t): zero at the origin, one at the wall.
        # Outer moments retain the original radius to resolve the whole field.
        x = y/local_clock
        time_basis = _chebyshev(2*s-1, 3)
        outer_basis = _chebyshev(2*y-1, 4)
        core_basis = _chebyshev(2*x-1, 2)
        outer = ((time_basis @ self.outer_moments)*outer_basis).sum(1, keepdim=True)
        core = ((time_basis[:, :3] @ self.core_moments)*core_basis).sum(1, keepdim=True)
        log_correction = 0.02*s.square()*c2*(outer + x*(1-x)*core)
        return numerator/denominator.sqrt()*torch.exp(log_correction)
