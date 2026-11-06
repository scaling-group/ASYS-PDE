"""Narrowly smoothed local core and explicit rational tails, only at a=.25.

Supplied priors: oddness, scale gauges, compactification and the adjustable
power envelope. The local cubic core, negative-curvature raw gauge and
rational fractional tails are reused from this history's steps 4/5. The new
construction smooths a cardinal cubic by two narrow box convolutions, rather
than using the broader cardinal quintic of step 6. Its exact piece polynomials
are derived in construction; no PDE solve, optimization, quadrature,
normalization, fitted state or reference access occurs there.

Let p=1/beta, R=hypot(1,x), L=log(R), r=(x/R)**2, s=R**(-2),
tau=R**(-p), rho=1/8, q=tau/(rho+tau), w=2*(1+rho)*q-1. Define

    h(x) = (x/R)*exp(-p*L + C(x)),
    C = (p-1)*r*s/2 + S(L) + r**2*(c + e*s)
                            + r**3*q*(sum_{j=0}^3 d_j*T_j(w) + d4*s).

For the centered cardinal cubic B3 and the normalized uniform density U_eps
on [-eps/2,eps/2], choose eps=1/4 and Delta=1/2, and set

    K = B3 * U_eps * U_eps,
    S(L) = sum_i c_i*K(L/Delta-i).

Here * is ordinary convolution of known basis functions, not a Hilbert
operator applied to the profile. Twice integrating the truncated cubic gives
the explicit identity

    K(u) = sum_{k=0}^4 (-1)**k*binom(4,k)
            *((u+2-k+eps)_+**5 - 2*(u+2-k)_+**5
                                      + (u+2-k-eps)_+**5)/(120*eps**2).

K is symmetric, nonnegative, supported on [-2-eps,2+eps], piecewise quintic
and C4. The smoothing adds variance eps**2/6 to B3's variance 1/3; its
support grows from 4 to 4.5 knot units, versus 6 for the cardinal quintic.
This preserves more of the successful cubic's locality while removing its
third-derivative jumps. eps and Delta are explicit design assumptions,
not learned scales or canonical mass quantiles.

The eight free core coefficients are a,b,c_3,...,c_8. Set c_-1=c_1=a,
c_-2=c_2=b, c_0=-2*(K(1)*a+K(2)*b)/K(0), and all others zero.
Symmetry and this exact anchor give S(0)=S'(0)=0, hence S=O(L**2)=O(x**4).
The last support ends at L=(8+2+eps)*Delta=41/8; S and its first four
derivatives match zero there. The profile is C4 on the real line and analytic
near zero, generally not C5 at finite core knots. Each piece is expanded
using rational arithmetic into powers of t=8*L-floor(8*L). The registered
252-by-8 float64 coefficient map includes a final zero piece. Its first
constant and linear rows are exactly zero. Forward contracts this fixed map
with the core parameters, gathers the piece using shape-preserving indexing,
and applies Horner algebra. Scalar, vector and higher-rank inputs use the
same pointwise formula. No query-specific coefficient construction occurs.

For finite coefficients and p>0, C is bounded, even and tends to c. Thus h
is odd, h(0)=0, h'(0)=1, h=x-x**3+O(x**5), and h(x)>0 for x>0.
The tail h~exp(c)*x**(-p) has finite positive weighted moment
A=(2/pi)*integral_0^infinity h(x)/x dx. Ordinary half-line L1 mass is not
required and diverges for the declared beta=1 cold envelope.

The reused raw curvature gauge assumes a negative cubic branch. The dilation
h_d(x)=h(d*x)/d preserves raw slope, has A_d=A/d, and leaves the protected
canonical h(A*y)/A unchanged; its cubic coefficient scales by d**2. Fixing
raw curvature to -1 excludes zero/positive-curvature branches and can change
finite-representation bias. For f=y+u*y**3+..., g=1+v*y**2+..., the public
equation gives 2*(beta+a*lambda)*u=lambda*(1-a/3)*v, which does not prove u<0.

The reused formal Hilbert/tail prior for 1<p<2 motivates relative fractional
p powers and even inverse powers from core contributions. The rational cubic
tail independently controls its first four tau powers; e and d4 free an even
and a mixed correction. Higher coefficients remain related by the rational
form, and coincident powers and core/tail correlations remain possible. No
asymptotic coefficient balance or beta is solved in construction.

Native guidance from current-history PUBLIC logs (no fitted tensors read):
  step 1, original core/tail: qualified mean square 2.81654e-10,
    RMS 1.67826e-5; 20,000 updates, 591.28 fitting seconds.
  step 2, current confluent source: 4.34984e-11, RMS 6.59533e-6;
    PARTIAL 15,262/20,000 updates, 1170.25 fitting seconds. Its last 1000
    losses ranged from 4.1188e-11 to 9.7933e-9; the cosine schedule did not
    finish. Scores always use the last complete measured update.
  step 3, gapped polynomial: WORSE 5.55026e-10, RMS 2.35590e-5;
    20,000 updates, 768.41 fitting seconds.
  step 4, curvature/rational: 1.11085e-10, RMS 1.05397e-5;
    20,000 updates, 923.46 fitting seconds.
  step 5, local cubic: BEST supplied qualified score 1.58364e-12,
    RMS 1.25843e-6; 20,000 updates, 1078.22 fitting seconds. Losses at
    5000/10000/15000/20000 were 8.2392e-11/2.8018e-12/1.6277e-12/1.5803e-12.
  step 6, broader/finer quintic: REJECTED before fitting, public failure
    "IndexError: index_select(): Index is supposed to be a vector".
    There is no qualified score or convergence evidence for that source;
    its local vector checks missed scalar/operator-grid input compatibility.
Inherited guidance records a two-parameter envelope plateau near 1.099e-5,
RMS 3.315e-3; that earlier run is not directly supplied here. Added modes,
cheaper algebra and sampled conditioning have not ensured improved scores.

Steps 1-5 passed origin, oddness, repeatability and fresh value/derivative
replay. Canonical mass resolved on [-36,36]; steps 1-4 center error was
<=2.22e-16 and mesh error <=1.02e-12. Step 5 had WORSE center error
4.22e-10 and mass mesh error 3.05e-9, still qualified; its g mesh difference
grew to 1.06e-10 from <=8.89e-16. V/y mesh differences were <=5.44e-10.
Operator order 16 to 64 changed mean square by <=2.54e-17 across these runs;
domain extension changed g and V/y by <=3.33e-16. Step 5 residual mesh
change was 3.05e-15 and estimated maximum residual interval width 2.77e-15.
These give no evidence of an operator-resolution floor at measured scores.
No spatial residual trace separates origin/core and tail errors. Resolved
tail mass does not certify tail coefficients or full-line operator accuracy.
Finite qualification is not an existence, reference or uniqueness certificate.

Structural hypothesis: smooth the successful local cubic on a narrow scale
without adding trainable directions, to reduce artificial knot sensitivities
while retaining its locality. The exact piece table avoids repeated kernel
powers over all centers; shape-preserving gathers address the public failure.
Whether smoothing helps approximation or the protected optimizer is unknown.

Cold initialization: beta=config['initial_beta'] (declared value 1), every
shape coefficient zero. At that beta h is exactly the supplied unfitted
envelope; other initial betas retain the declared fixed curvature term.
No public fitted exponent, coefficient, normalization or reference initializes
it. All predictive state is registered: 16 trainable CPU float64 scalars and
2018 buffer scalars. Construction does finite exact basis expansion only.
Inference evaluates saved pointwise state, without a solve, fitting, operator,
quadrature, normalization, external read or batch-dependent correction.
For n inputs: O(n) work and temporary storage, a fixed 252-by-8 contraction,
six local coefficient gathers, quintic core and cubic tail Horner polynomials,
hypot, stable logarithms, two pointwise exponentials and rational division.
Native export adds frozen normalization state; reconstruction cost is separate.

Source-generation checks (bounded single-thread CPU algebra, PyTorch 2.1.2):
cold-envelope relative error <=1.84e-15 on 321 radii from 1e-12 to 1e12.
Independent SciPy integration of the cubic against the convolution triangle
agreed with all eight core directions at knots/random sites within 1.11e-16.
Adjacent polynomial maps matched derivatives through order four within
1.11e-16 in the local piece coordinate; the origin constant/linear rows and
final zero piece were exactly zero. This checks the explicit basis, not the
profile's Hilbert transform or normalization.

Artificial nonzero-coefficient fixtures at beta=.8,1,1.3 passed positivity,
exact oddness, zero, unit slope, h'''(0)=-6, repeatability and exact serialized
fresh-state replay of beta, values and coordinate derivatives. Scalar, empty,
vector, 2-D and 3-D inputs preserved shape and agreed with flattened evaluation;
singleton/batch probes agreed exactly. Joint coordinate/parameter gradcheck
and gradgradcheck passed at zero, +/-1 and all 41 core joins. Five-point
coordinate finite-difference error, scaled by 1+abs(derivative), was <=1.19e-12.
CPU float64 state and rejection of invalid initial beta passed. Extreme values
and first derivatives were finite through abs(x)=1e308; beta=.8 rounded its
remote tail to zero, an ordinary underflow limit. Remote log-amplitude error
on 1e20,1e40,1e80 was <=2.85e-14. Mathematical positivity cannot defeat
finite-instrument underflow. All fixtures were temporary; initial shape state
remains zero, with no optimizer update, PDE residual, operator, protected
fitter/assessor or reference access during source generation.

On 161 log-spaced radii 1e-3 to 1e3, column-normalized cold log-profile shape
Jacobians gave minimum singular values/condition numbers:
  beta=.5: 6.9389e-4 / 3454; beta=.8: 6.5808e-4 / 3722;
  beta=1: 5.0285e-4 / 4857; beta=1.3: 1.1692e-3 / 2072.
At beta=1 the same check gave 4075 for the successful cubic, 25061 for the
rejected broader quintic, and 28801 for the current confluent source. Thus
this basis has full sampled rank but WORSE conditioning than the cubic.
These are shape algebra diagnostics, not residual-Jacobian, optimizer or
global conditioning certificates; beta movement and coincident tail powers
remain uncertain.

Single-thread medians of five trials after three warmups, using artificial
mean(h**2) and optionally mean((x*h_x)**2), in milliseconds:
  50,000 points, parameter backward: new 8.52, cubic 8.29, confluent 9.18;
    with coordinate differentiation/backward: 26.52, 28.00, 35.16.
  200,000 points, parameter backward: 34.32, 34.79, 38.51;
    with coordinate differentiation/backward: 111.75, 129.25, 148.39.
  193-by-256 grid, parameter backward: 7.67, 7.38, 8.51;
    with coordinate differentiation/backward: 24.58, 25.76, 32.93.
Median fresh construction cost over three calls was .2434 seconds here,
versus .000069 for the cubic and .000043 for the confluent source. The new
table is a real construction/storage cost. Evaluation is similar to the
cubic and faster than the confluent in these checks, but native fit cost,
completed update count and qualification cannot be inferred from them.
Native guidance is kept here because only model.py is editable. Source and
checks are completed within the 3600-second source-generation allowance;
formal fitting and assessment remain pending native evaluation.

Remaining uncertainty: narrow smoothing bias, fixed knot resolution, C4
rather than global analyticity, negative-curvature branch restriction,
rational higher-tail relations, conditioning and optimizer convergence,
fit-time cost/update count, residual location and continuum operator accuracy.
"""
import math
from fractions import Fraction

import torch


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_beta = float(config["initial_beta"])
        if not math.isfinite(initial_beta) or initial_beta <= 0:
            raise ValueError("initial_beta must be positive and finite")
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(initial_beta), dtype=torch.float64, device="cpu"))
        self.tail_amplitude = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.even_tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.core = torch.nn.Parameter(torch.zeros(
            8, dtype=torch.float64, device="cpu"))
        self.fractional_tail = torch.nn.Parameter(torch.zeros(
            5, dtype=torch.float64, device="cpu"))
        self.register_buffer("piece_spacing", torch.tensor(
            0.125, dtype=torch.float64, device="cpu"))
        self.register_buffer("tail_scale", torch.tensor(
            0.125, dtype=torch.float64, device="cpu"))
        self.register_buffer("core_polynomial_map", self._make_core_map())

    @staticmethod
    def _make_core_map():
        """Exact expansion of two box convolutions of the cardinal cubic."""
        eps = Fraction(1, 4)

        def kernel(value):
            total = Fraction(0)
            for k in range(5):
                for shift, multiplier in ((-1, 1), (0, -2), (1, 1)):
                    z = value + 2 - k + eps * shift
                    if z > 0:
                        total += (-1)**k * math.comb(4, k) * multiplier * z**5
            return total / (120 * eps**2)

        anchor_a = -2 * kernel(Fraction(1)) / kernel(Fraction(0))
        anchor_b = -2 * kernel(Fraction(2)) / kernel(Fraction(0))
        rows = []
        for segment in range(42):
            piece = [[Fraction(0) for _ in range(8)] for _ in range(6)]
            for index in range(-2, 9):
                weights = [Fraction(0) for _ in range(6)]
                for k in range(5):
                    for shift, multiplier in ((-1, 1), (0, -2), (1, 1)):
                        z = eps * segment - index + 2 - k + eps * shift
                        if z >= 0:
                            scale = Fraction((-1)**k * math.comb(4, k)
                                             * multiplier, 120) / eps**2
                            for power in range(6):
                                weights[power] += (scale * math.comb(5, power)
                                                  * eps**power * z**(5-power))
                for power, weight in enumerate(weights):
                    if index == 0:
                        piece[power][0] += anchor_a * weight
                        piece[power][1] += anchor_b * weight
                    elif abs(index) == 1:
                        piece[power][0] += weight
                    elif abs(index) == 2:
                        piece[power][1] += weight
                    else:
                        piece[power][index - 1] += weight
            rows.extend([[float(value) for value in row] for row in piece])
        return torch.tensor(rows, dtype=torch.float64, device="cpu")

    def beta(self):
        return self.log_beta.exp()

    def _spline(self, log_radius):
        coefficients = (self.core_polynomial_map @ self.core).reshape(42, 6)
        u = (log_radius / self.piece_spacing).clamp(max=41.0)
        segment = u.floor().to(torch.long)
        t = u - segment
        # Advanced indexing preserves even scalar and operator-grid shapes.
        a0, a1, a2, a3, a4, a5 = coefficients[segment].unbind(-1)
        return ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t + a0

    def forward(self, x):
        # Avoid x*x overflow and retain the origin's small logarithm accurately.
        radius = torch.hypot(x, torch.ones_like(x))
        unit = x / radius
        r = unit.square()
        s = radius.reciprocal().square()
        small_log = 0.5 * torch.log1p(x.clamp(-1.0, 1.0).square())
        log_radius = torch.where(x.abs() <= 1.0, small_log, radius.log())
        p = self.log_beta.neg().exp()
        log_tail = -p * log_radius
        tau = log_tail.exp()
        q = tau / (self.tail_scale + tau)
        d = self.fractional_tail
        w = 2 * (1 + self.tail_scale) * q - 1
        tail_polynomial = ((4 * d[3] * w + 2 * d[2]) * w
                           + d[1] - 3 * d[3]) * w + d[0] - d[2]
        correction = (0.5 * (p - 1) * r * s + self._spline(log_radius)
                      + r.square() * (self.tail_amplitude + self.even_tail * s
                          + r * q * (tail_polynomial + d[4] * s)))
        return unit * torch.exp(log_tail + correction)


def build_model(config):
    return Model(config)
