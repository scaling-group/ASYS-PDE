"""Cold compactified profile with a complete real rational residue pair.

Construction, attribution and structural hypothesis
---------------------------------------------------
The immutable single-a=.25 task supplies oddness, Hilbert identities,
compactification, canonical two-scale export and the adjustable power envelope.
Current-history public sources supply the negative cubic gauge (steps 4/5),
Mellin projector and geometry transport (5), Jacobi mixed tails (6), confluent
r**(-4) tail (7), degree-seven reprojection (8), and one symmetric rational
interior mode (9). These are reused priors/representations, not discoveries.
This revision completes that rational pole pair with its independent real skew
residue. It does no equation solve or loss optimization during construction.

Let alpha=1/beta, r=hypot(1,x), v=r**(-alpha), z=2*v-1, B=(1-v)**2,
p=2*beta, D=(v-1/2)**2+1/64, c=17/64. The explicit representation is

    h(x)=(x/r)*exp(log(v)+B*Q(v)),
    Q=Q_old(v)+(e_even+e_skew*z)*R(v),
    R(v)=v**2*(1-v)**2/D,
    (d_old,e_even,e_skew)=coefficient_transport @ correction.

The pole positions 1/2 +/- i/8 are inherited explicit dyadic geometry priors;
no measured beta, fitted coefficient, reference field or residual site is used.
The two numerator directions complete the real residues of this fixed complex
pole pair: after polynomial division its proper rational numerator is affine.
They are independent modulo the old generalized-power family. If a real
combination were polynomial/generalized powers, its residues at both nonreal
poles would vanish. The affine real numerator would then vanish at two distinct
points and hence be zero. This is a strict one-direction enrichment retaining
the entire preceding family, not just a coordinate rotation. The hypothesis is
that allowing asymmetric compactified-interior shape helps without adding a
second denominator, field exponential or another confluent Mellin state.
No public sitewise residual map establishes that this direction is needed.

Q_old comprises eight normalized shifted Legendre polynomials L_j, j=0..7,
and the four inherited mixed/even Mellin modes U,W,Z,E. Define the L2(dv)
projector Pi_7, T(q)=product((q-j)/(q+j+1), j=0..7), and

    F(q,v)=(v**q-Pi_7[v**q](v))/T(q),
    U=sqrt(2*p+1)*F(p),
    W=sqrt(2*p+3)*[(2*p+2)*F(p+1)-(2*p+1)*F(p)],
    Z=sqrt(2*p+5)*[(2*p+3)*(p+2)*F(p+2)
                 -2*(p+1)*(2*p+3)*F(p+1)+(2*p+1)*(p+1)*F(p)],
    G(q)=F(q)+sum(C_j*(F(q)-F(p+j))/(q-p-j), j=0..2),
    E=sqrt(4*p+1)*G(2*p),
    C_j=product(2*p+j+k+1,k=0..2)/product(j-k,k!=j).

Integer zeros and p=1,2 collisions retain explicit logarithmic divided limits.
Away from collisions the span is
{1,v,...,v**7,v**p,v**(p+1),v**(p+2),v**(2*p),R,z*R}.
Rational modes are not asserted orthogonal in dv. Since R and z*R are bounded
on [0,1], O((1-v)**2) at v=1 and O(v**2) at v=0, their direct contribution
to log(h) first enters at x**8 and contributes no leading tail constant.
For finite coefficients and beta>0 the mathematical raw profile is smooth,
odd, positive on x>0, h(0)=0, h'(0)=1, with finite positive weighted moment:

    h=x-(1+alpha)*x**3/2
      +[(1+alpha)*(3+alpha)/8+alpha**2*Q_old(1)/4]*x**5+O(x**7),
    h~exp(Q_old(0))*x**(-alpha) as x -> +infinity.

Ordinary half-line L1 is not required. Mathematical positivity/decay does not
prevent floating-point overflow or underflow for extreme parameters/radii.
Exact polynomial division gives R=(z**2-1)/4+c*(c/D-1), so the two polynomial
parts (z**2-1)/4 and (z**3-z)/4 share the existing endpoint-preserving Horner
assembly and a single reciprocal evaluates the remaining affine numerator.

Construction and prediction boundary
------------------------------------
Beta starts at config.initial_beta (declared starter 1); all fourteen shape
coordinates start at zero. The unchanged average log-shape Gram uses five
manufactured alphas 1,5/4,3/2,7/4,2, each with 768 Gauss-Legendre sites on
[-b,b], b=log(cot(pi*.001/2))+2, from the starter and supplied guard. The new
14-by-14 transport is L**(-T) for the Gram Cholesky factor L. This is analytic
geometry quadrature and linear algebra, not fitting or a physical Jacobian.
Nodes/weights are discarded. All numerical prediction state is registered.
Forward/beta evaluate saved explicit state only: no PDE solve, optimization,
normalization, reference lookup or query-dependent correction. The protected
fitter, support survey, frozen gauges and assessor are not changed/executed.

Useful native guidance and adverse public diagnostics
-----------------------------------------------------
Read guidance/strategy.md: investigate the public equation/representation,
preserve positivity and decay, separate representation from fitting, and keep
failures and uncertainty. It remains unchanged because only model.py is editable.
Qualified public mean squares at steps 1..9 were 2.8294727871e-11,
5.4906410525e-11,2.9828432494e-14,1.5628983090e-15,2.6783425920e-18,
7.9312961641e-18,1.8005469609e-19,1.1543191689e-19,1.1114936433e-19.
Steps 2 and 6 worsened their predecessor. More shape freedom guarantees no gain.

Step 9 stopped at fit_time, 19,275/20,000 updates, 1170.061 s fitting trace;
it is a partial fit. Step 8 also stopped at 19,841; step 5 at 18,856. Steps 6/7
completed 20,000 in 617.08/946.42 s. Shape and execution cost are confounded.
Adding work risks fewer updates under the unchanged 1200 s fitting allowance.
Step 9 whole native process was 19:37.86, max RSS 441624 KiB; this is prior cost,
not a forecast for this source. No historical fitted state is read or imported.

Origin/core: step 9 origin=0, slope=1, exact oddness/replay passed; batch-relative
change 8.55e-16, derivative error 1.21e-9, derivative mesh error 3.61e-9.
Its canonical weighted-mass core [-6.459214354927492,4.549238827542046]
agrees closely with steps 5..8. These do not support changing the origin gauge.
Tail: canonical weighted mass resolved at extent 36, mass error <=4.18e-13,
outer shell 7.92e-15. This does not certify true asymptotic coefficients,
ordinary L1 mass, uniqueness, or continuum/reference accuracy. No tail exponent
from the logs initializes the new candidate.
Operator: orders 16/32/64 gave 1.1094913160e-19,1.1114132753e-19,
1.1114936626e-19; g/W mesh changes 8.88e-16/4.82e-10 and domain changes
<=3.33e-16. Coverage mean change 1.47e-23, 162/655 roundoff-limited sites.
The mean-square interval [9.7154564716e-20,1.2842000600e-19] overlaps step 8's
[1.0128036950e-19,1.3285236855e-19]. The modest score change cannot establish
better accuracy. No public diagnostic warrants changing the protected operator.

Previous source checks are not new checks: step 9 reported independent 90-digit
profile discrepancy up to 1.68e-10 and mixed even-coordinate/log_beta error
8.53e-8. The inherited confluent algebra has cancellation sensitivity. Its
initial eps=1e-6 gradcheck failed at p=2.39 and passed with eps=1e-5; that
adverse diagnostic remains relevant. The rejected previous one-sided rational
mode had Schur norm squared 1.23e-10 and transport up to 9.01e4. Step 9's
chosen mode had norm squared 3.12e-4 and local conditions 35.77..662.83,
but manufactured alpha=.25,.5,2.5,5 gave 3.37e8,6.20e4,1191.95,1.62e5.
No uniform geometry or residual-Jacobian conditioning claim is made.

Complexity
----------
Fifteen trainable scalars: beta plus fourteen shape coordinates. Raw prediction
state has 492 registered float64 buffer scalars. Relative to step 9 this adds
one affine field numerator, one scalar coordinate, eight polynomial-fold
constants and 27 transport entries, with no new field reciprocal/exponential
or fifth Mellin state. Work/storage are O(N) for N query sites; scalar algebra
retains four Mellin states. Constructor evaluates five 768-by-14 matrices and
one 14-by-14 Cholesky/triangular solve including node generation. Parameter
count does not establish computational efficiency. Native full fitting,
qualification, score and true asymptotic coefficients remain pending.

Bounded cold-source checks for this revision (not native evaluation)
------------------------------------------------------------------
Torch 2.13.0+cpu, NumPy 2.4.3, mpmath 1.3.0, CPU float64 and one Torch thread.
All full-family checks use zero correction or correction[j]=(-1)**j/(80+10*j),
j=0..13. Prior comparisons build ONLY the current-history PUBLIC SOURCE cold
using its same thirteen-coordinate manufactured rule. No old fitted tensor,
private reference, g/V, equation residual, optimizer, protected fitter/judge,
candidate mass survey or normalization is read/evaluated. In-memory coefficient
linear solves only embed/isolate declared analytic shape directions. The only
serialized state loaded is freshly manufactured within these interface checks.

* Exact Fraction identities for R and z*R pass at eighteen v=j/17 sites,
  j=0..17, with 0<=R<=4 and abs(z*R)<=4 at those sites. Global boundedness
  follows from D>=1/64 and v**2*(1-v)**2<=1/16 on [0,1].
* Fifteen trainable and 492 registered buffer scalars are verified. Cold zero
  correction agrees with the unfitted envelope within rounding at nine betas
  .2,.4,.5,.7,.8,1,1.5,2,4. Manufactured old-family embedding has zero observed
  absolute discrepancy over 1003 signed radii per beta, magnitudes 10**[-80,80].
  Isolated unit physical coefficients for both rational modes agree with direct
  envelope formulas to 1.33e-15 absolute. These are representation checks.
* The same nine betas pass CPU float64 registration, zero origin, unit slope,
  exact oddness, finite nonnegative half-line values, empty output and invalid
  beta rejection. There are 48 far-tail underflow zeros at beta=.2; these are
  excluded from any strict floating-point positivity claim. Seven sites per
  beta give zero observed singleton/batch discrepancy. State stays unchanged;
  in-memory serialized value and coordinate-derivative replay is exact.
* At 768/1536 geometry nodes the added skew direction's average Schur norm
  squared is 2.3994542806e-5/2.3994542803e-5, maximum transport entry 204.15.
  Independent 1536-node average transported Gram discrepancy is 1.15e-10;
  conditions at alphas 1,5/4,3/2,7/4,2 are 2176.10,305.32,84.84,96.12,267.37.
  This is worse than step 9's 35.77..662.83 range. Averaged geometry remains
  nonsingular, but this added correlation can worsen fixed-fitter convergence.
  No new outside-range or physical-residual Jacobian certificate is claimed.
* Non-directional gradcheck and gradgradcheck pass at 28 manufactured p values
  in [.4,8], including exact p=1,2, neighboring anchors, paired/shared switch
  sides, p=2.39 and separate-anchor cases. Inputs are signed coordinates
  [-1.7,-.2,0,.3,4], log_beta and fourteen corrections. First differences use
  eps=1e-5, atol/rtol=2e-5/2e-4; second differences use eps=1e-5,
  atol/rtol=3e-4/3e-3 and cotangents [.2,-.3,.7,-.4,.5]. These finite
  tolerances do not establish roundoff-level full-family derivative accuracy.
* Independent 90-digit Legendre moment/projector algebra, with l'Hopital
  integer and repeated-tail limits, checks 252 profiles at those 28 p values
  and radii exp(-8,-3,-1,0,1,4,12,80),1e100. Maximum absolute discrepancy is
  8.35e-11; maximum relative discrepancy where exact h>1e-280 is 2.02e-10
  (p=2.3899, x=exp(1)). The independent transport product interprets each saved
  binary64 entry/coordinate exactly. Underflow is excluded from relative claims.
  This checks the defining representation, not the gCLM equation. The inherited
  Mellin cancellation persists; this source does not claim to cure it.
* Isolated complete rational-pair profiles with physical coefficients
  e_even=.25, e_skew=-.125 at eleven p values in [.4,8] and x=.3,1,3 are
  compared against independent 90-digit formulas. Errors/(1+abs(exact)) for
  value,x,log_beta,even,skew,mixed x-skew,mixed log_beta-skew derivatives are
  <=1.23e-16,4.18e-16,3.34e-16,1.91e-16,6.20e-17,8.36e-17,1.07e-16.
  This isolated check does not override the adverse full-family discrepancy.
* An isolated synthetic workload uses 193 coordinate-derivative sites, 49,408
  field sites and parameter reverse differentiation of mean(h_x**2)+mean(h**2).
  It evaluates no operators or physical residual. Three warmups and 36
  alternating repetitions at beta=1,.8,.7,.5,1.75,3.5 give new/old median ms
  17.40/17.27,13.64/13.39,13.57/13.10,17.29/17.07,17.43/17.08,20.01/19.05;
  ratios 1.007,1.018,1.036,1.013,1.020,1.050. Twenty alternating fresh warm
  constructor builds give median 64.77/64.60 ms. An initial constructor call
  in the interface-check process took 12.33 s. Warm measurements exclude
  imports/cold-library startup and predict neither fit nor whole-process cost.

Source preparation and these bounded checks finished within the 3600-second
source-generation allowance (under twenty minutes elapsed). Only model.py is
edited. Useful native guidance and adverse diagnostics are preserved above;
guidance/strategy.md is unchanged. Native fitting, public assessment and private
reference access remain inside their declared protected boundaries. Native fit
completion, qualification, score, branch uniqueness, true asymptotic coefficients
and continuum/reference accuracy are unmeasured for this new source.
"""
import math

import torch


# Degree elevation is part of the representation, not an evaluator setting.
_POLYNOMIAL_COUNT = 8
_LAST_POWER = _POLYNOMIAL_COUNT - 1
_TAIL_COUNT = 4
_RATIONAL_COUNT = 2
_SHAPE_COUNT = _POLYNOMIAL_COUNT + _TAIL_COUNT + _RATIONAL_COUNT
_MELLIN_LEFT_WEIGHT = (-1)**_POLYNOMIAL_COUNT * _POLYNOMIAL_COUNT


def _analytic_tables():
    """Exact-form polynomial/residue coefficients, no profile data."""
    legendre = [[0.0] * _POLYNOMIAL_COUNT for _ in range(_POLYNOMIAL_COUNT)]
    power_to_shifted = [[0.0] * _POLYNOMIAL_COUNT
                        for _ in range(_POLYNOMIAL_COUNT)]
    residues = []
    log_ratio = []
    for n in range(_POLYNOMIAL_COUNT):
        for k in range(n // 2 + 1):
            legendre[n - 2*k][n] = (math.sqrt(2*n + 1) * (-1)**k
                * math.comb(n, k) * math.comb(2*n - 2*k, n) / 2**n)
        for j in range(n + 1):
            power_to_shifted[j][n] = math.comb(n, j) / 2**n
        residues.append(math.prod(n+j+1 for j in range(_POLYNOMIAL_COUNT))
                        / math.prod(n-j for j in range(_POLYNOMIAL_COUNT)
                                    if j != n))
        numerator = [1 / (n+j+1) for j in range(_POLYNOMIAL_COUNT)]
        denominator = [1 / (n-j) for j in range(_POLYNOMIAL_COUNT) if j != n]
        log_ratio.append([(-1)**k / (k+1) * math.fsum(
            [a**(k+1) for a in numerator]
            + [-a**(k+1) for a in denominator]) for k in range(13)])
    return [torch.tensor(t, dtype=torch.float64, device="cpu") for t in
            (legendre, power_to_shifted, residues, log_ratio)]


def _horner(coefficients, z):
    value = coefficients[-1]
    for j in range(len(coefficients) - 2, -1, -1):
        value = coefficients[j] + z * value
    return value


def _exprel_taylor(z):
    """Degree-ten Taylor of expm1(z)/z; used only near a scalar resonance."""
    value = 1 / math.factorial(11)
    for k in range(9, -1, -1):
        value = 1 / math.factorial(k + 1) + z * value
    return value


def _endpoint_horner(coefficients, z, v, endpoint):
    """P(z)=P(-1)+2*v*S(z), with exact analytic P(-1) supplied.

    Synthetic division by z+1 preserves the known tail constant and avoids
    cancellation of large residue coefficients at z=-1. Use 2*v directly,
    since computing z+1 loses small, representable v. No state is adjusted.
    """
    quotient = [coefficients[-1]]
    for j in range(len(coefficients) - 2, 0, -1):
        quotient.append(coefficients[j] - quotient[-1])
    return endpoint + 2*v*_horner(quotient[::-1], z)


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_beta = float(config["initial_beta"])
        if not math.isfinite(initial_beta) or initial_beta <= 0:
            raise ValueError("initial_beta must be positive and finite")
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(initial_beta), dtype=torch.float64, device="cpu"))
        self.correction = torch.nn.Parameter(torch.zeros(
            _SHAPE_COUNT, dtype=torch.float64, device="cpu"))
        for name, value in zip(("legendre_coefficients", "power_to_shifted",
                                "mellin_residues", "log_ratio_taylor"),
                               _analytic_tables()):
            self.register_buffer(name, value)
        self.register_buffer("power_indices", torch.arange(
            _POLYNOMIAL_COUNT, dtype=torch.float64, device="cpu"))
        self.register_buffer("legendre_left_endpoint", torch.tensor(
            [(-1)**j * math.sqrt(2*j + 1) for j in range(_POLYNOMIAL_COUNT)],
            dtype=torch.float64, device="cpu"))
        # Two public analytic second divided differences, at p=1 and p=2.
        self.register_buffer("collision_second_log_ratio", torch.tensor(
            [2**j - 1 for j in range(1, 13)], dtype=torch.float64, device="cpu")
            * self.log_ratio_taylor[[2, 4], 1:])
        self.register_buffer("coefficient_transport", self._make_transport())
        # For c=17/64 and D=(v-1/2)^2+1/64=c-v*(1-v),
        # R=v^2*(1-v)^2/D = -v*(1-v) + c*(c/D-1).
        # R_even=R and R_skew=z*R complete the two real residue directions
        # of the same complex pole pair. Fold their polynomial parts into
        # the existing Horner assembly; both vanish at the left endpoint.
        self.register_buffer("rational_polynomial", torch.tensor(
            [[-.25, 0], [0, -.25], [.25, 0], [0, .25],
             [0, 0], [0, 0], [0, 0], [0, 0]],
            dtype=torch.float64, device="cpu"))

    def beta(self):
        return self.log_beta.exp()

    def _mellin_state(self, q, n=None, delta=None):
        """Scalar state for F(q,v), independent of all query coordinates.

        Partial-fraction residues are fixed integers. Pairing the nearest
        integer's pole with v**q avoids subtracting two singular coefficients.
        Branches depend only on the registered exponent, never query support.
        """
        if n is None:
            n = max(0, min(_LAST_POWER, math.floor(float(q.detach()) + 0.5)))
        if delta is None:
            delta = q - n
        scalar_delta = float(delta.detach())
        if abs(scalar_delta) < 1e-2:
            h_over_delta = _horner(self.log_ratio_taylor[n], delta)
            log_ratio = delta * h_over_delta
            constant = (self.mellin_residues[n] * h_over_delta
                        * _exprel_taylor(log_ratio))
        else:
            # Nearest anchors and the shared 2p anchor keep every unmasked
            # log1p argument above -1; the last anchor also covers larger q.
            num = n + self.power_indices + 1
            mask = self.power_indices != n
            den = n - self.power_indices[mask]
            log_ratio = (torch.log1p(delta / num).sum()
                         - torch.log1p(delta / den).sum())
            constant = self.mellin_residues[n] * torch.expm1(log_ratio) / delta
        scale = self.mellin_residues[n] * log_ratio.exp()

        mask = self.power_indices == n
        denominator = torch.where(mask, torch.ones_like(self.power_indices),
                                  self.power_indices - q)
        coefficients = torch.where(mask, constant,
                                    self.mellin_residues / denominator)
        return n, delta, scale, self.power_to_shifted @ coefficients

    @staticmethod
    def _power_bridge(log_v, q, n, delta):
        """(v**q-v**n)/(q-n) with its first/second resonance derivatives."""
        scalar_delta = float(delta.detach())
        if abs(scalar_delta) < 1e-2 and n > 0:
            # Below -1024, both nearby positive powers and all their required
            # derivatives have already underflowed. Clipping only this Taylor
            # intermediate prevents 0*inf for extreme radii/exponents.
            safe_log_v = log_v.clamp_min(-1024)
            return ((n * safe_log_v).exp() * safe_log_v
                    * _exprel_taylor(delta * safe_log_v))
        elif scalar_delta > 0:
            return ((n * log_v).exp()
                    * torch.expm1(delta * log_v) / delta)
        else:
            return (-(q * log_v).exp()
                    * torch.expm1(-delta * log_v) / delta)

    def _mellin_remainder(self, q, log_v):
        """F(q,v)=explicit nonpolynomial term + polynomial in z=2*v-1."""
        n, delta, scale, polynomial = self._mellin_state(q)
        return scale * self._power_bridge(log_v, q, n, delta), polynomial

    def _mellin_states(self, exponents, integers, offsets):
        """Batch the four scalar residue states used in prediction.

        This is the same algebra as _mellin_state, with one small matrix
        product and vector scalar operations instead of separate graphs.
        Exponent-derived indices are local analytic branch choices.
        """
        q = torch.stack(exponents)
        delta = torch.stack(offsets)
        indices = torch.tensor(integers, dtype=torch.int64, device=q.device)
        n = indices.to(q.dtype)
        mask = self.power_indices[:, None] == n[None, :]
        residue = self.mellin_residues[indices]
        near = delta.abs() < 1e-2
        all_near = bool(near.all())
        any_near = bool(near.any())
        if any_near:
            h = _horner(self.log_ratio_taylor[indices].T, delta)
            taylor_log_ratio = delta * h
            taylor_constant = residue*h*_exprel_taylor(taylor_log_ratio)
        if not all_near:
            num = n[None, :] + self.power_indices[:, None] + 1
            den = n[None, :] - self.power_indices[:, None]
            den = torch.where(mask, torch.ones_like(den), den)
            direct_log_ratio = (torch.log1p(delta[None, :]/num).sum(0)
                - torch.where(mask, torch.zeros_like(den),
                              torch.log1p(delta[None, :]/den)).sum(0))
            safe_delta = torch.where(near, torch.ones_like(delta), delta)
            direct_constant = residue*torch.expm1(direct_log_ratio)/safe_delta
        if all_near:
            log_ratio, constant = taylor_log_ratio, taylor_constant
        elif any_near:
            log_ratio = torch.where(near, taylor_log_ratio, direct_log_ratio)
            constant = torch.where(near, taylor_constant, direct_constant)
        else:
            log_ratio, constant = direct_log_ratio, direct_constant
        denominator = self.power_indices[:, None] - q[None, :]
        denominator = torch.where(mask, torch.ones_like(denominator),
                                  denominator)
        coefficients = torch.where(mask, constant[None, :],
                                    self.mellin_residues[:, None]/denominator)
        return residue*log_ratio.exp(), self.power_to_shifted @ coefficients

    @staticmethod
    def _tail_residues(p):
        """Residues of product((q+p+j+1)/(q-p-j), j=0,1,2)."""
        return ((2*p + 1)*(2*p + 2)*(2*p + 3)/2,
                -(2*p + 2)*(2*p + 3)*(2*p + 4),
                (2*p + 3)*(2*p + 4)*(2*p + 5)/2)

    @staticmethod
    def _collision_index(p):
        scalar_p = float(p.detach())
        for j in (1, 2):
            if abs(scalar_p - j) < 0.4:
                return j
        return None

    def _tail_difference_coefficients(self, p, collision):
        """Scalar coefficients of the complete confluent difference.

        The common pole's second divided difference is a quadratic in the
        already evaluated power bridge. Its pure v**(2j) term is folded into
        the polynomial block. No new field exp or Taylor work remains.
        """
        e = p - collision
        n = 2*collision
        if abs(float(e.detach())) < 1e-2:
            h = _horner(self.log_ratio_taylor[n], e)
            k = _horner(self.collision_second_log_ratio[collision - 1], e)
            log_ratio = e*h
            scale = log_ratio.exp()
            first_constant = h*_exprel_taylor(log_ratio)
            second_constant = (2*log_ratio).exp()*k*_exprel_taylor(
                2*e.square()*k)
        else:
            num = n + self.power_indices + 1
            den = n - self.power_indices[self.power_indices != n]
            log_ratio = (torch.log1p(e/num).sum()
                         - torch.log1p(e/den).sum())
            # log(1+2*z)-2*log(1+z) = log1p(-(z/(1+z))**2).
            second_log_ratio = (torch.log1p(-(e/(num+e)).square()).sum()
                               - torch.log1p(-(e/(den+e)).square()).sum())
            scale = log_ratio.exp()
            first_constant = torch.expm1(log_ratio)/e
            second_constant = ((2*log_ratio).exp()
                * torch.expm1(second_log_ratio)/(2*e.square()))
        mask = self.power_indices == n
        denominator = ((self.power_indices - n - 2*e)
                       * (self.power_indices - n - e))
        denominator = torch.where(mask, torch.ones_like(denominator),
                                  denominator)
        coefficients = torch.where(mask, torch.zeros_like(denominator),
                                    self.mellin_residues/denominator)
        # With a=v**j, D=(v**p-a)/e and v**(2p)=(a+e*D)**2,
        # the entire common-pole part is k0*D**2+k1*a*D+k2*a**2.
        residue = self.mellin_residues[n]
        k0 = residue*(scale.square()/2 + e.square()*second_constant)
        k1 = residue*(scale*first_constant + 2*e*second_constant)
        k2 = residue*(first_constant.square()/2 + second_constant)
        polynomial = (self.power_to_shifted @ coefficients
                      + k2*self.power_to_shifted[:, n])
        return k0, k1, polynomial

    def _tail_difference(self, p, log_v, collision):
        """Constructor-only explicit field assembly of that same difference."""
        k0, k1, polynomial = self._tail_difference_coefficients(p, collision)
        bridge = self._power_bridge(log_v, p, collision, p-collision)
        return bridge*(k0*bridge + k1*(collision*log_v).exp()), polynomial

    def _tail_weights(self, p, collision):
        """Coefficients on F(p),F(p+1),F(p+2),F(2p), and one bridge.

        G=F(2p)+sum(C_j*(F(2p)-F(p+j))/(p-j)). Near a collision
        keep its divided difference intact instead of subtracting infinities.
        """
        residues = self._tail_residues(p)
        weights = []
        for j, residue in enumerate(residues):
            if j == collision:
                weights.append(torch.zeros_like(p))
            else:
                ratio = residue/(p-j)
                weights.append(-ratio)
        # Simplify 1+sum(C_j/(p-j)) symbolically. Near a collision omit its
        # pole before simplifying; these polynomials have no removed pole.
        if collision is None:
            twice_weight = (3*(p+1)*(3*p+1)*(3*p+2)
                            / (p*(p-1)*(p-2)))
        elif collision == 1:
            twice_weight = ((p+1)*(p*(p*(8*p+20)+15)-6)
                            / (p*(p-2)))
        else:
            twice_weight = (-(p*(p*(p*(4*p+28)+52)+33)+3)
                            / (p*(p-1)))
        bridge_weight = (torch.zeros_like(p) if collision is None
                         else residues[collision])
        return weights, twice_weight, bridge_weight

    def _untransported_basis(self, x, alpha):
        """Constructor-only analytic geometry, with no equation residual."""
        log_v = -alpha*torch.hypot(x, torch.ones_like(x)).log()
        v = log_v.exp()
        z = 2*v - 1
        b = (-torch.expm1(log_v)).square()
        polynomial = torch.stack([z**j for j in range(_POLYNOMIAL_COUNT)], dim=1)
        polynomial = polynomial @ self.legendre_coefficients
        p = 2/alpha
        fields = []
        for q in (p, p + 1, p + 2, 2*p):
            non, poly = self._mellin_remainder(q, log_v)
            fields.append(non + _horner(poly, z))
        fp, fq, fs, ft = fields
        u = (2*p + 1).sqrt()*fp
        w = (2*p + 3).sqrt()*((2*p + 2)*fq - (2*p + 1)*fp)
        ladder = (2*p + 5).sqrt()*((2*p + 3)*(p + 2)*fs
            - 2*(p + 1)*(2*p + 3)*fq + (2*p + 1)*(p + 1)*fp)
        collision = self._collision_index(p)
        weights, twice_weight, bridge_weight = self._tail_weights(p, collision)
        fourth = twice_weight*ft + sum(weights[j]*fields[j] for j in range(3))
        if collision is not None:
            non, poly = self._tail_difference(p, log_v, collision)
            fourth = fourth + bridge_weight*(non + _horner(poly, z))
        fourth = (4*p + 1).sqrt()*fourth
        rational = b*v.square()/((v - .5).square() + 1/64)
        return b[:, None]*torch.cat((polynomial, u[:, None], w[:, None],
                                     ladder[:, None], fourth[:, None],
                                     rational[:, None],
                                     (z*rational)[:, None]), dim=1)

    def _make_transport(self):
        # NumPy is used only to generate public analytic quadrature nodes;
        # prediction depends entirely on registered Torch state.
        import numpy as np
        with torch.no_grad():
            nodes, weights = np.polynomial.legendre.leggauss(768)
            bound = math.log(1 / math.tan(math.pi * .001 / 2)) + 2
            x = torch.exp(torch.tensor(bound * nodes, dtype=torch.float64))
            weight = torch.tensor(weights / 2, dtype=torch.float64)
            gram = torch.zeros((_SHAPE_COUNT, _SHAPE_COUNT), dtype=torch.float64)
            for a in (1, 5/4, 3/2, 7/4, 2):
                basis = self._untransported_basis(x, torch.tensor(
                    a, dtype=torch.float64))
                gram += basis.T @ (weight[:, None] * basis) / 5
            factor = torch.linalg.cholesky(gram)
            return torch.linalg.solve_triangular(factor.T, torch.eye(
                _SHAPE_COUNT, dtype=torch.float64), upper=True)

    def forward(self, x):
        r = torch.hypot(x, torch.ones_like(x))
        p = 2*self.beta()
        log_v = -2*r.log()/p
        v = log_v.exp()
        z = 2*v - 1
        b = (-torch.expm1(log_v)).square()
        d = self.coefficient_transport @ self.correction
        polynomial_d = d[:_POLYNOMIAL_COUNT]
        u_d, w_d, z_d, even_d = d[
            _POLYNOMIAL_COUNT:_POLYNOMIAL_COUNT + _TAIL_COUNT]
        rational_d = d[-_RATIONAL_COUNT:]
        polynomial = (self.legendre_coefficients @ polynomial_d
                      + self.rational_polynomial @ rational_d)
        norm_u = (2*p + 1).sqrt()
        norm_w = (2*p + 3).sqrt()
        norm_z = (2*p + 5).sqrt()
        norm_fourth = (4*p + 1).sqrt()
        weights = [u_d*norm_u - w_d*norm_w*(2*p + 1)
                       + z_d*norm_z*(2*p + 1)*(p + 1),
                   w_d*norm_w*(2*p + 2)
                       - z_d*norm_z*2*(p + 1)*(2*p + 3),
                   z_d*norm_z*(2*p + 3)*(p + 2)]
        collision = self._collision_index(p)
        fourth_weights, twice_weight, bridge_weight = self._tail_weights(
            p, collision)
        amplitude = even_d*norm_fourth
        weights = [weights[j] + amplitude*fourth_weights[j] for j in range(3)]
        weights.append(amplitude*twice_weight)
        n = max(0, min(_LAST_POWER, math.floor(float(p.detach()) + 0.5)))
        delta = p - n
        n_q, n_s = min(_LAST_POWER, n + 1), min(_LAST_POWER, n + 2)
        delta_q = delta if n < _LAST_POWER else p + 1 - n_q
        delta_s = delta if n <= _LAST_POWER - 2 else p + 2 - n_s
        # Keep 2*delta away from neighboring poles. Every paired collision
        # lies within this range, so its quadratic assembly still shares D.
        share_twice = n <= _LAST_POWER//2 and abs(float(delta.detach())) < 0.4
        n_twice = 2*n if share_twice else max(
            0, min(_LAST_POWER, math.floor(float((2*p).detach()) + 0.5)))
        delta_twice = 2*delta if share_twice else 2*p - n_twice
        scales, polynomials = self._mellin_states(
            (p, p + 1, p + 2, 2*p), (n, n_q, n_s, n_twice),
            (delta, delta_q, delta_s, delta_twice))
        weights = torch.stack(weights)
        polynomial = polynomial + polynomials @ weights
        bridge = self._power_bridge(log_v, p, n, delta)
        if collision is not None:
            # n=j and n_twice=2j throughout |p-j|<.4. Fold the even bridge
            # into the same quadratic: (v**(2p)-v**(2j))/(2e)
            # = D*v**j + e*D**2/2. The v**j term is a ladder monomial.
            k0, k1, collision_poly = self._tail_difference_coefficients(
                p, collision)
            collision_amplitude = amplitude*bridge_weight
            polynomial = polynomial + collision_amplitude*collision_poly
            twice_scale = weights[3]*scales[3]
            anchor_weight = twice_scale + collision_amplitude*k1
            quadratic_weight = delta*twice_scale/2 + collision_amplitude*k0
            field_weights = list(weights[:3]*scales[:3])
            field_weights[collision] = field_weights[collision] + anchor_weight
            nonpolynomial = bridge*(field_weights[0]
                + v*(field_weights[1] + v*field_weights[2])
                + quadratic_weight*bridge)
        else:
            if n <= _LAST_POWER - 2:
                nonpolynomial = bridge*(weights[0]*scales[0]
                    + v*(weights[1]*scales[1] + v*weights[2]*scales[2]))
            elif n == _LAST_POWER - 1:
                nonpolynomial = (bridge*(weights[0]*scales[0]
                    + v*weights[1]*scales[1]) + weights[2]*scales[2]
                    * self._power_bridge(log_v, p + 2, n_s, delta_s))
            else:
                nonpolynomial = (weights[0]*scales[0]*bridge
                    + weights[1]*scales[1]
                    * self._power_bridge(log_v, p + 1, n_q, delta_q)
                    + weights[2]*scales[2]
                    * self._power_bridge(log_v, p + 2, n_s, delta_s))
            if share_twice:
                # (v**(2p)-v**(2n))/(2*delta) = bridge*(r**(-2)+v**n)/2.
                fourth_non = (weights[3]*scales[3]/2*bridge
                              * (r.reciprocal().square() + v**n))
            else:
                fourth_non = (weights[3]*scales[3]
                    * self._power_bridge(log_v, 2*p, n_twice, delta_twice))
            nonpolynomial = nonpolynomial + fourth_non
        if n > 0:
            endpoint = (self.legendre_left_endpoint @ polynomial_d
                + _MELLIN_LEFT_WEIGHT/p*(u_d*norm_u - w_d*norm_w + z_d*norm_z)
                - _MELLIN_LEFT_WEIGHT*(p + 3)/(2*p.square())*amplitude)
            q_polynomial = _endpoint_horner(polynomial, z, v, endpoint)
        else:
            # Zero-anchor bridges themselves carry a nonzero tail constant.
            q_polynomial = _horner(polynomial, z)
        # A shared reciprocal evaluates both rational residue directions.
        # The polynomial folds have zero left endpoint. The bounded affine
        # numerator preserves the core and tail orders of each rational mode.
        rational = (17/64)/((v - .5).square() + 1/64) - 1
        numerator = rational_d[0] + z*rational_d[1]
        q = q_polynomial + nonpolynomial + (17/64*numerator)*rational
        return (x/r)*torch.exp(log_v + b*q)



def build_model(config):
    return Model(config)
