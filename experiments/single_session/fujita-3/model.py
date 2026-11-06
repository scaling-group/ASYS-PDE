"""Transient core/outer expansion with the exact initial PDE tangent.

For w=u**(-2), the equation is
    w_t = w_rr + 4*w_r/r - 3*w_r**2/(2*w) - 2.
Thus an affine clock and a quadratic radial width are useful *coordinates*,
not a prescribed stationary similarity solution.  The seed clock and width
start at their supplied values.  A logarithmic width drift and low order
Chebyshev amplitude modes describe their transient departure.

Write u = C(r,t) [f(r) + t h(r) + (2*pi/5) J(d,t)] exp(s**2 P).
Here C is the concentrating factor, h=F[f]-f*C_t(r,0), and J is the
half-line heat response to boundary data t.  Consequently u(.,0)=f and
u_t(.,0)=F[f] at every interior radius, including the regular origin.
At the wall h=-2*pi/5 and J(0,t)=t, so the Dirichlet condition also holds.
The even distance d=(25-r**2)/10 gives origin regularity and agrees with
wall distance to leading order.  The remaining radial curvature effects
are fitted along with the outer and core transients.  No trajectory or
stationary-profile data enter this expansion.
"""
import math
import torch


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.clock_margin = torch.nn.Parameter(torch.tensor(
            math.log(math.expm1(0.001)), dtype=torch.float64))
        self.log_width = torch.nn.Parameter(torch.tensor(
            math.log(0.01), dtype=torch.float64))
        self.width_drift = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        # Time degrees 0..3, outer radial degrees 0..4, core degrees 0..2.
        # These are global mathematical modes, not values at space/time knots.
        self.outer_modes = torch.nn.Parameter(torch.zeros(4, 5, dtype=torch.float64))
        self.core_modes = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float64))

    @staticmethod
    def chebyshev(x, degree):
        values = [torch.ones_like(x), x]
        for _ in range(2, degree + 1):
            values.append(2*x*values[-1] - values[-2])
        return torch.cat(values[:degree+1], dim=1)

    @staticmethod
    def sinc(x):
        # Unlike differentiating sin(x)/x or torch.sinc at zero, this
        # removable-singularity formula has finite second derivatives there.
        small = x.abs() < 0.02
        safe = torch.where(small, torch.ones_like(x), x)
        x2 = x.square()
        series = 1 + x2*(-1/6 + x2*(1/120 + x2*(-1/5040 + x2/362880)))
        return torch.where(small, series, torch.sin(safe)/safe)

    @staticmethod
    def corner_ramp(distance, time):
        """J_t=J_dd, J(0,t)=t, J(d,0)=0 for d>0.

        J=t*((1+2*z**2)*erfc(z)-2*z*exp(-z**2)/sqrt(pi)).
        The t=0 branch is its exact interior extension; the outer initial
        corner is intentionally not jointly smooth.  No small time scale is
        introduced to turn a tangent correction off.
        """
        positive = time > 0
        safe_time = torch.where(positive, time, torch.ones_like(time))
        z = distance/(2*safe_time.sqrt())
        kernel = ((1 + 2*z.square())*torch.erfc(z)
                  - (2/math.sqrt(math.pi))*z*torch.exp(-z.square()))
        initial_kernel = torch.where(distance == 0, torch.ones_like(distance),
                                     torch.zeros_like(distance))
        return time*torch.where(positive, kernel, initial_kernel)

    def forward(self, coordinates):
        r, t = coordinates[:, :1], coordinates[:, 1:]
        # The positive margin keeps the field finite over the entire IVP.
        # Its floor only prevents floating point cancellation at the endpoint.
        margin = torch.nn.functional.softplus(self.clock_margin).clamp_min(1e-12)
        clock = 0.020385594 + 0.020406*margin
        width = self.log_width.clamp(-16, 2).exp()
        drift = 2*torch.tanh(self.width_drift)
        r2 = r.square()
        initial_clock = clock + width*r2
        local_clock = clock - t + width*r2
        log_clock = torch.log(initial_clock/local_clock)
        moving_width = width*torch.exp(-drift*torch.log1p(log_clock))
        concentration = torch.sqrt(initial_clock/(clock - t + moving_width*r2))
        initial_rate = (1 + drift*width*r2/initial_clock)/(2*initial_clock)

        k = math.pi/10
        f = 5*torch.cos(k*r)
        laplacian = -k*k*f - 20*k*k*self.sinc(k*r)
        tangent_lift = laplacian + f.pow(3) - f*initial_rate
        distance = (25 - r2)/10
        lifted = f + t*tangent_lift + (2*math.pi/5)*self.corner_ramp(distance, t)

        s = t/0.020385594
        x = r2/25
        z = width*r2/local_clock
        temporal = self.chebyshev(2*s - 1, 3)
        outer = self.chebyshev(2*x - 1, 4)
        core = (1-z)*self.chebyshev(2*z - 1, 2)
        correction = ((temporal @ self.outer_modes)*outer).sum(dim=1, keepdim=True)
        correction = correction + ((temporal @ self.core_modes)*core).sum(dim=1, keepdim=True)
        return concentration*lifted*torch.exp(s.square()*correction)
