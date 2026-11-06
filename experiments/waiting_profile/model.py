"""Conditional C^7 logarithmic waiting profile, with a positive spectral remainder.

The coordinator supplied a2,...,a7; these are not a native discovery. Independent
symbolic substitution (logs/native/symbolic_check.txt) confirms them and gives
L[x^n]=(n-3)(n-8)x^n/6 at F=x^2/6. The order-eight forcing is 1034888400,
so b=-1241866080 in b*x^8*log(x). A Taylor-compatible C^8 solution with the
given cubic cannot cancel that forcing. This is conditional on differentiated
expansions; it is not a nonexistence theorem for every weaker solution.

The native construction changes regularity and coordinates. Let u=x/.01,
Q=log(6F/x^2), and D=u*d/du. Then the equation is
exp(Q)*(6+7*DQ+D^2Q+2*(DQ)^2)-6-14*DQ=0. Its linear part is
(D-1)(D-6)Q. An independently derived formal recurrence transports the free
coefficient c at order u^6 through order u^11, retaining every forced logarithm.
See logs/native/polyhomogeneous.py and .txt for the exact rational derivation.
F=x^2/6*exp(P5(u)+u^6*(A5(u)+c*T5(u)+L5(u)*log(u))+u^12*H(u)).
P5 is the log of the supplied starter through order five; A5,T5,L5 are the
newly derived transport polynomials; H is a learned Chebyshev remainder.
In exact arithmetic this satisfies the formal equation through order x^13
(stored constants are rounded to float64), fixes all lower traces,
retains the unsmoothed forced x^8 logarithm, and makes F positive. Only c and
H are fitted to the equation, not to an integrated/reference solution.
The parameter initialization below is the retained native least-squares fit on
401 uniform and 101 logarithmic points (logs/native/construct.py); it requires
no runtime file or reference data. The fit's almost-null Jacobian direction is
consistent with the free resonant coefficient. That selection is conventional,
not a physical inference or a uniqueness result.
This learned c is distinct from the supplied cubic scaling normalization C=1.

This is a finite-interval approximation with C^7 endpoint regularity, not an
exact solution or a convergent full polyhomogeneous expansion. The first
unrepresented terms in Q are u^12*log(u)^2 and u^12*log(u); a smooth H cannot
realize them as u tends to zero. In ideal rational arithmetic the equation
remainder starts at x^14 times logarithmic polynomials; storing float64 constants
adds small lower-order coefficient errors. Neither vanishes to all orders.
The arbitrary order-eight freedom and conditioning can make profile selection
nonunique. Numerical residual and monotonicity checks do not certify existence,
the original radial IVP, waiting persistence, release time, or a focusing exponent.
The cubic C=1 is imposed by the supplied scaling normalization.

At x=0 log's argument is assigned 1 only to evaluate the continuous lower jets.
For every x>0 it is exactly u. AD's finite eighth derivative at that branch is
not the true one-sided eighth derivative, which diverges. All constants and
learned values are registered and survive state_dict restoration.
"""
import torch


class Candidate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.envelope = torch.nn.Parameter(torch.tensor([
            -.021040229650687338,
            -.0002728094059569172,
            -.0003880453777476878,
            .0002478563465762201,
            -.00008219679379440665,
            .000022095290814842474,
            -.000005771895090746994,
            .0000014901340727612616,
            -.00000035728716964670435,
            .00000007307564091767183,
            -.000000010393489618847704,
            -.00000000011523051840533665,
            .0000000007039946438927441,
            -.00000000027917938250516926,
            .00000000006927338897804636,
            -.000000000011841436370064704,
            .00000000000130550164380808,
            -.0000000000000712801578121995], dtype=torch.float64))
        self.register_buffer("xmax", torch.tensor(.01, dtype=torch.float64))
        self.register_buffer("log_transport", torch.tensor([
            -.00745119648, .0086433879168, -.004106354380128,
            .00116578439647488, -.0001665581596687008, .00013101225070468952], dtype=torch.float64))
        self.register_buffer("regular_transport", torch.tensor([
            0., -.010359032135451429, .007684979561788275,
            -.002673629947009332, .0005525534591464768, -.000004544849563129317], dtype=torch.float64))
        self.register_buffer("free_transport", torch.tensor([
            1., -1.16, .5511, -.156456, .02235321, -.0175827132], dtype=torch.float64))
        self.register_buffer("log_starter", torch.tensor(
            [.06, .0117, .004932, .003639735, .00543606552], dtype=torch.float64))

    def forward(self, points):
        x = points[:, :1]
        u = x / self.xmax
        z = 2*u - 1
        # Clenshaw evaluates a bounded spectral basis without fitting in forward.
        b1 = torch.zeros_like(u)
        b2 = torch.zeros_like(u)
        for c in self.envelope[2:].flip(0):
            b0 = 2*z*b1-b2+c
            b2, b1 = b1, b0
        h = z*b1-b2+self.envelope[1]
        p = torch.zeros_like(u)
        for c in self.log_starter.flip(0):
            p = (p+c)*u
        log_u = torch.log(torch.where(u > 0, u, torch.ones_like(u)))
        transport = self.regular_transport+self.envelope[0]*self.free_transport
        a = torch.zeros_like(u)
        ell = torch.zeros_like(u)
        for ak, lk in zip(transport.flip(0), self.log_transport.flip(0)):
            a = a*u+ak
            ell = ell*u+lk
        exponent = p + u.pow(6)*(a+ell*log_u)+u.pow(12)*h
        return (x.square()/6)*torch.exp(exponent)
