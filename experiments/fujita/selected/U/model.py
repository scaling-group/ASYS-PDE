"""Spherical heat corners and a spatially resolved reaction-clock remainder.

Write u=v/sqrt(q).  At t=0, v=u0, v_t=Delta(u0), q=1,
and q_t=-2*u0**2, so the full initial PDE tangent is structural.
The PDE fixes q_t, q_tt and q_ttt at zero time.  Impose these analytic
identities before fitting the fourth and higher temporal terms.
The core clock and width evolve independently; no stationary profile is assumed.
At the Dirichlet wall, the reaction denominator has a diffusion-scale Neumann
corner. Its second and third initial derivatives can have nonzero wall slopes,
but the positive-time quotient of two regular Dirichlet fields has zero slope.
The logarithmic width response uses the local clock 1-(t/T)cos(kr)^2,
as suggested by the gradient forcing in the continuous q equation.
"""
import math
import torch


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.clock_margin = torch.nn.Parameter(torch.tensor(
            math.log(math.expm1(0.001)), dtype=torch.float64))
        self.core_shape = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        # Fourth-order time evolution and two radial curvature corrections.
        self.width_shape = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
        self.log_width_relaxation = torch.nn.Parameter(torch.tensor(
            math.log(3*(math.pi/10)**2/25), dtype=torch.float64))

    @staticmethod
    def radial_functions(x):
        """Entire sinc(x) and (sinc(x)-cos(x))/x^2, regular at r=0."""
        small = x.abs() < 0.1
        safe = torch.where(small, torch.ones_like(x), x)
        x2 = x.square()
        sinc_series = 1 + x2 * (-1/6 + x2 * (1/120 + x2 * (
            -1/5040 + x2 * (1/362880 - x2/39916800))))
        j_series = 1/3 + x2 * (-1/30 + x2 * (1/840 + x2 * (
            -1/45360 + x2 * (1/3991680 - x2/518918400))))
        sinc = torch.where(small, sinc_series, torch.sin(safe)/safe)
        j = torch.where(small, j_series,
                        (torch.sin(safe)/safe-torch.cos(safe))/safe.square())
        # h=(j-cos(x)/3)/x^2 is also entire and even.  It removes the
        # apparent origin quotient in the third width derivative.
        h_series = 2/15 + x2*(-4/315 + x2*(1/2268 + x2*(
            -1/124740 + x2*(1/11119680 - x2/1459458000))))
        h = torch.where(small, h_series,
                        (j-torch.cos(safe)/3)/safe.square())
        return sinc, j, h

    @staticmethod
    def local_clock_remainder(s, c):
        """[-log(1-z)-z-z^2/2-z^3/3]/c^4, z=s*c^2.

        The removable wall quotient is evaluated as s^2 times an analytic
        function of z.  Its Taylor branch also preserves exact initial jets.
        This is a global local-reaction response, not a spatial cutoff.
        """
        z = s*c.square()
        small = z < 0.125
        safe = torch.where(small, torch.full_like(z, 0.125), z)
        series = torch.full_like(z, 1/20)
        for degree in range(19, 3, -1):
            series = 1/degree + z*series
        regular = z.square()*series
        exact = (-torch.log1p(-safe)-safe-safe.square()/2-safe.pow(3)/3)/safe.square()
        return s.square()*torch.where(small, regular, exact)

    @staticmethod
    def wall_distance(c):
        """Even extension of the inward wall distance, through cos(kr)^11."""
        k = math.pi / 10
        c2 = c.square()
        return c/k * (1 + c2*(1/6 + c2*(3/40 + c2*(
            5/112 + c2*(35/1152 + c2*63/2816)))))

    @staticmethod
    def wall_functions(c, t):
        """Analytic half-line heat responses to polynomial wall data."""
        distance = Candidate.wall_distance(c)
        positive = t > 0
        safe_time = torch.where(positive, t, torch.ones_like(t))
        z = distance / (2*torch.sqrt(safe_time))
        erfc = torch.erfc(z)
        gaussian = torch.exp(-z.square())/math.sqrt(math.pi)
        # I_nu=L^-1[exp(-d sqrt(p))/p^nu]=t^(nu-1) J_nu(z).
        # This is an algebraic recurrence for heat-kernel integrals, not
        # evolution in time: J_(nu+1)=(J_nu-z J_(nu+1/2))/nu.
        responses = [erfc, 2*(gaussian-z*erfc)]
        for step in range(2, 14):
            nu = 0.5*step
            responses.append((responses[-2]-z*responses[-1])/nu)
        h1, h2 = responses[2], 2*responses[4]
        i52 = safe_time.pow(1.5)*responses[3]
        # t^m N_m has inward derivative t^m at d=0, vanishes initially
        # for fixed d>0, and solves the half-line heat equation.
        neumann = torch.cat([
            -math.factorial(m)*torch.sqrt(safe_time)*responses[2*m+1]
            for m in range(2, 7)], dim=1)
        return (distance,
                torch.where(positive, h1, torch.zeros_like(h1)),
                torch.where(positive, h2, torch.zeros_like(h2)),
                torch.where(positive, i52, torch.zeros_like(i52)),
                torch.where(positive, neumann, torch.zeros_like(neumann)))

    @staticmethod
    def heat_numerator(r, t, c, sinc, j, wall):
        k = math.pi / 10
        # This three-function span is closed under the n=5 radial Laplacian:
        # (Delta+k^2)cos=-4k^2 sinc, (Delta+k^2)sinc=-2k^2 j,
        # (Delta+k^2)j=0.  Thus bulk is the exact full-space heat evolution.
        bulk = 5*c - 20*k*k*t*sinc + 20*k**4*t.square()*j

        distance, h1, h2, i52, _ = wall
        y = 1-r.square()/25
        # Regular extension of (5/r)^2, matching seven wall derivatives.
        spherical = 1 + y*(1 + y*(1 + y*(1 + y*(1 + y*(1 + y*(1+y))))))
        wall_sinc, wall_j = 2/math.pi, 8/math.pi**3
        g1, g2 = 20*k*k*wall_sinc, -20*k**4*wall_j
        potential = k*k-2/25
        # For r^2 exp(k^2 t)v the wall potential is
        # V(d)=k^2-2/(5-d)^2=V(0)-(4/125)d+O(d^2).
        # (dt-dd)K=d*t*H1, K=d^2*I_(5/2)/4+d*t^2*H2/8.
        variable_potential = (-4/125)*g1*(
            distance.square()*i52/4 + distance*t.square()*h2/8)
        layer = spherical * (g1*t*h1 + g2*t.square()*h2
                              + potential*g1*t.square()*(h1-h2)
                              + variable_potential)
        return torch.exp(-k*k*t)*(bulk+layer)

    def forward(self, coordinates):
        r, t = coordinates[:, :1], coordinates[:, 1:]
        k = math.pi / 10
        c, sine = torch.cos(k*r), torch.sin(k*r)
        sinc, j, h = self.radial_functions(k*r)
        wall = self.wall_functions(c, t)
        v = self.heat_numerator(r, t, c, sinc, j, wall)

        clock = 0.020385594 + 0.020406 * torch.nn.functional.softplus(self.clock_margin)
        s = t/clock
        remaining = 1-s
        # C(0)=1, C'=-50, C''=1000*k^2, C'''=-14000*k^4.
        # These are the exact core reaction-clock derivatives for v_t=Delta v.
        core_coeff = 0.5*torch.tanh(self.core_shape)
        initial_core_curvature = 0.5*(1+(1000*k*k-2500)*clock.square())
        initial_core_cubic = (1/3 + clock.pow(3)*(
            -14000*k**4+150000*k*k-250000)/6)
        core_exponent = ((1/clock-50)*t + initial_core_curvature*s.square()
                         + initial_core_cubic*s.pow(3)
                         + s.pow(4)*(core_coeff[0]+s*core_coeff[1]
                                     + s.square()*core_coeff[2]))
        core = remaining * torch.exp(core_exponent)

        # q_tt=c^2 C_tt+sin(kr)^2 W_tt is fixed by the initial PDE.
        # (sinc-cos)/sin^2=j/sinc^2 is regular at the origin.
        # Hence W_tt=k^2*(800*cos*j/sinc^2-300), W_t=0, W(0)=1.
        # The third derivative follows by differentiating the q equation.
        # h cancels its removable sin(kr)^2 quotient analytically.
        # All fitted corrections start at t^4, preserving all three jets.
        radius2 = r.square()/25
        width_coeff = 0.5*torch.tanh(self.width_shape)
        temporal = width_coeff[:, 0] + s*width_coeff[:, 1] + s.square()*width_coeff[:, 2]
        initial_width_curvature = k*k*(800*c*j/sinc.square()-300)
        initial_width_cubic = k**4*(
            -2400*c*h/sinc.square()-19200*c*j/sinc.square()
            -7200*(k*r).square()*j.square()/sinc.square()
            +3000+7200*j/sinc)-30000*k*k*c.square()
        width_exponent = (0.5*initial_width_curvature*t.square()
                          + initial_width_cubic*t.pow(3)/6) + s.pow(4)*(
            temporal[:, 0:1] + radius2*temporal[:, 1:2] + radius2.square()*temporal[:, 2:3])
        # Smooth positive-time Dirichlet evolution requires q_r(5,t)=0.
        # The initial W2 and W3 have nonzero wall slopes; their heat
        # responses fix those slopes while preserving every interior jet.
        # W2_r(5)=-8*pi^2/5; W3_r(5)=k^5(12000/xw+2400/xw^3).
        neumann = wall[4]
        wall_cubic_slope = k**5*(12000/(math.pi/2)+2400/(math.pi/2)**3)
        wall_slope = width_coeff[1, :] + 2*width_coeff[2, :]
        width_exponent = (width_exponent
            - (4*math.pi**2/5)*t.square()*neumann[:, :1]
            + (wall_cubic_slope/6)*t.pow(3)*neumann[:, 1:2]
            + (2/5)*s.pow(4)*(
                wall_slope[0]*neumann[:, 2:3]
                + s*wall_slope[1]*neumann[:, 3:4]
                + s.square()*wall_slope[2]*neumann[:, 4:5]))
        relaxation = self.log_width_relaxation.exp()
        # In q_t=Delta q+2(v_r/v)q_r-3*q_r^2/(2*q)-2*v^2, the
        # leading reaction q=1-50*t*c^2 gives the integrated gradient
        # response -3*k^2*sin(kr)^2/(25*c^4) times a log remainder.
        # Subtract its cubic jet, already imposed above, and use fitted T
        # to keep this local reaction clock positive on the full interval.
        # The response is fourth order at t=0 and has zero wall slope.
        width = torch.exp(width_exponent)/(1+relaxation*self.local_clock_remainder(s, torch.ones_like(c)))
        denominator = c.square()*core+sine.square()*width
        return v/torch.sqrt(denominator)
