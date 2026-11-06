"""Transient heat envelope and a positive, concentrating inverse-square gap.

Write u = w / sqrt(q).  The envelope w starts with the first two radial
heat derivatives of u0, with the Dirichlet corner resolved on distance/sqrt(t).
The nonlinear gap satisfies q(0)=1, q_t(0)=-2*u0**2.  Consequently the full
field has exactly the prescribed initial value and interior PDE tangent.

The clock and width start at the supplied seed's values.  Neither the clock
nor a stationary similarity profile is imposed by the sampling horizon.
The remaining coefficients describe low-order changes in log amplitude and
log radial width, rather than values at space/time knots.
"""
import math
import torch


def radial_functions(z):
    """Entire sinc(z) and (sin(z)-z*cos(z))/z**3, including second derivatives."""
    z2 = z.square()
    small = z.abs() < 0.1
    safe = torch.where(small, torch.ones_like(z), z)
    sinc_series = 1 + z2 * (-1/6 + z2 * (1/120 + z2 * (-1/5040
                  + z2 * (1/362880 + z2 * (-1/39916800 + z2/6227020800)))))
    j_series = 1/3 + z2 * (-1/30 + z2 * (1/840 + z2 * (-1/45360
               + z2 * (1/3991680 + z2 * (-1/518918400 + z2/93405312000)))))
    sinc = torch.where(small, sinc_series, torch.sin(safe)/safe)
    j = torch.where(small, j_series,
                    (torch.sin(safe)-safe*torch.cos(safe))/safe.pow(3))
    return sinc, j


def heat_envelope(r, t):
    """Second-order radial heat expansion, with an analytic corner parametrix.

    For L=d_rr+4/r*d_r, L*cos(kr)=-k^2*(cos+4*sinc), and
    L^2*cos(kr)=k^4*(cos+8*sinc+8*j).  The corner functions t*H1 and
    t^2*H2 are the half-line heat extensions of boundary data t and t^2.
    Near the wall d=5-r and A=(5/r)^2 to fourth order.  Their polynomial
    continuations in 1-r^2/25 preserve regularity at the origin.  The
    transformed radial heat operator is d_rr-2/r^2; its leading potential
    correction is included as well.  This is a parametrix, not a heat solver.
    """
    k = math.pi/10
    c = torch.cos(k*r)
    sinc, j = radial_functions(k*r)
    initial = 5*c
    lap = -5*k*k*(c+4*sinc)
    lap2 = 5*k**4*(c+8*sinc+8*j)
    wall_lap = -2*math.pi/5
    wall_lap2 = 40*k**4*(2/math.pi+8/math.pi**3)

    y = 1-r.square()/25
    distance = 5*y*(0.5+y*(0.125+y*(0.0625+y*5/128)))
    amplitude = 1+y*(1+y*(1+y*(1+y)))
    # The t=0 branch uses the one-sided heat-layer trace: its time
    # derivative is zero at every r<5, and one at the wall itself.
    positive = t > 0
    safe_t = torch.where(positive, t, torch.ones_like(t))
    z = distance/(2*safe_t.sqrt())
    z2 = z.square()
    gaussian = torch.exp(-z2)/math.sqrt(math.pi)
    tail = torch.erfc(z)
    h1 = (1+2*z2)*tail-2*z*gaussian
    h2 = (1+4*z2+4*z2.square()/3)*tail-(10*z/3+4*z*z2/3)*gaussian
    initial_trace = torch.where(distance == 0, t, t*0)
    layer1 = torch.where(positive, t*h1, initial_trace)
    layer2 = torch.where(positive, t.square()*h2, t*0)
    potential = 2/25
    corner = wall_lap*(layer1-potential*(t*layer1-layer2))
    corner = corner + 0.5*wall_lap2*layer2
    return initial+t*lap+0.5*t.square()*lap2-amplitude*corner


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.clock_margin = torch.nn.Parameter(torch.tensor(
            math.log(math.expm1(0.001)), dtype=torch.float64))
        self.log_width = torch.nn.Parameter(torch.tensor(
            math.log(0.01), dtype=torch.float64))
        # Coefficients of 1,z,z^2,x,x*z,x^2,x^2*z, x=r^2/25.
        self.log_amplitude = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        self.log_dilation = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))

    @staticmethod
    def modulation(coefficients, x, z):
        # Bounded log changes keep line-search trial fields finite and positive.
        a = 2*torch.tanh(coefficients/2)
        return (a[0]+z*(a[1]+z*a[2])+x*(a[3]+z*a[4])
                +x.square()*(a[5]+z*a[6]))

    def forward(self, coordinates):
        r, t = coordinates[:, :1], coordinates[:, 1:]
        clock = 0.020385594 + 0.020406*torch.nn.functional.softplus(self.clock_margin)
        width = self.log_width.exp()
        z = t/clock
        x = r.square()/25
        b = width*r.square()/(clock+width*r.square())
        c = torch.cos(math.pi*r/10)
        initial_rate = 1-b-50*clock*c.square()
        amplitude = self.modulation(self.log_amplitude, x, z)
        dilation = self.modulation(self.log_dilation, x, z)

        # For v=u^{-2}, v_t=Lv-3*v_r^2/(2*v)-2.  The quadratic core
        # coefficient obeys b_t ~ -6*b^2/v(0,t), giving the logarithmic
        # flattening 1/(1+3*b_initial*log(v_initial/v)).
        gap = (1-z)+z*b
        logarithm = -torch.log(gap)
        flattening = 1+(3*(math.pi/10)**2/25)*logarithm
        q = ((1-z)*torch.exp(z*initial_rate+z.square()*amplitude)
             +z*b*torch.exp(z*dilation)/flattening)
        return heat_envelope(r, t)/q.sqrt()
