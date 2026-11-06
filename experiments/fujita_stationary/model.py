"""Explicit regular radial profile with a finite spectral log correction."""
import math

import torch


class Candidate(torch.nn.Module):
    r"""Phi = A exp(h*z + z^2 P(2*z-1)) / sqrt(1+k*eta^2).

    Here k=3/4, z=k*eta^2/(1+k*eta^2), A^2=1/2+exp(a),
    h=1/2-exp(a)/(10*k),
    and P is a degree-36 Chebyshev polynomial. The amplitude excess and
    polynomial coefficients are the only fitting state. Their initial values
    describe an amplitude-5 envelope; no fitted profile samples are stored.

    At zero, Phi''=-A exp(a)/5 and Phi'=0, so origin compatibility holds
    identically. For finite parameters Phi is positive, even and smooth, and
    Phi ~ C/eta with C=A exp(h+sum(c_j))/sqrt(k)>0. The global
    compactification scale resolves the core and tail independently of
    the fitted origin curvature. It introduces no endpoint-specific term.
    """

    def __init__(self):
        super().__init__()
        self.log_origin_excess = torch.nn.Parameter(
            torch.tensor(math.log(5.0**2 - 0.5), dtype=torch.float64)
        )
        self.chebyshev_coefficients = torch.nn.Parameter(
            torch.zeros(37, dtype=torch.float64)
        )

    def forward(self, eta):
        radius_squared = eta.square()
        k = 0.75
        z = k*radius_squared / (1 + k*radius_squared)
        u = 2*z - 1
        following = torch.zeros_like(eta)
        after_following = torch.zeros_like(eta)
        for coefficient in self.chebyshev_coefficients[1:].flip(0):
            current = coefficient + 2*u*following - after_following
            after_following, following = following, current
        polynomial = self.chebyshev_coefficients[0] + u*following - after_following
        excess = self.log_origin_excess.exp()
        exponent = (0.5 - excess/(10*k))*z + z.square()*polynomial
        return ((0.5 + excess)/(1 + k*radius_squared)).sqrt()*exponent.exp()

    def _polynomial_jet(self, z):
        """P, dP/dz and d^2P/dz^2 by differentiated Chebyshev recurrence."""
        u = 2*z - 1
        previous, current = torch.ones_like(z), u
        previous_first, current_first = torch.zeros_like(z), 2*torch.ones_like(z)
        previous_second, current_second = torch.zeros_like(z), torch.zeros_like(z)
        coefficients = self.chebyshev_coefficients
        polynomial = coefficients[0]*previous + coefficients[1]*current
        first = coefficients[1]*current_first
        second = coefficients[1]*current_second
        for coefficient in coefficients[2:]:
            following = 2*u*current - previous
            following_first = 4*current + 2*u*current_first - previous_first
            following_second = 8*current_first + 2*u*current_second - previous_second
            previous, current = current, following
            previous_first, current_first = current_first, following_first
            previous_second, current_second = current_second, following_second
            polynomial = polynomial + coefficient*current
            first = first + coefficient*current_first
            second = second + coefficient*current_second
        return polynomial, first, second

    def equation_terms(self, eta):
        """The unchanged ODE and Robin residuals with analytic derivatives.

        Put k=3/4, t=1/(1+k*eta^2), z=1-t, Q=h*z+z^2*P,
        and j=-1+2*t*Q_z. Then Phi_x/Phi=k*t*j/2 for x=eta^2.
        Factoring Phi*t from the ODE combines its reaction and drift without
        subtracting two leading 1/eta terms. Public scoring uses problem.py.
        """
        x = eta.square()
        k = 0.75
        t = (1 + k*x).reciprocal()
        z = k*x*t
        excess = self.log_origin_excess.exp()
        h = 0.5 - excess/(10*k)
        polynomial, first, second = self._polynomial_jet(z)
        q = h*z + z.square()*polynomial
        q_first = h + 2*z*polynomial + z.square()*first
        q_second = 2*polynomial + 4*z*first + z.square()*second
        value = ((0.5 + excess)*t).sqrt()*q.exp()
        j = -1 + 2*t*q_first
        j_first = -2*q_first + 2*t*q_second
        ode = value*t*(
            k*(5*j + z*(j.square() - 2*j + 2*t*j_first))
            + (0.5 + excess)*(2*q).exp() - 0.5 - z*q_first
        )
        robin = value*t*(1 + 2*z*q_first)
        return ode, robin
