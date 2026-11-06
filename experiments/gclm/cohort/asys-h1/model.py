"""Single-a positive profile with a Mobius rational core and fractional tails.

Oddness, gauges, real-line compactification, Hilbert identities and the
adjustable power envelope are supplied task priors. The bounded log envelope
and stable confluent function reuse current-history step 2; the fixed raw
curvature gauge reuses step 3; all six distinct-pole fractional/confluent/mixed
modes reuse step 15. Positive-denominator rational ideas also occur in public
steps 6 and 11. The present revision replaces the five polynomial core modes
with five rationally compactified Chebyshev modes, rather than adding windows
or a trainable reciprocal denominator. It imports no fitted coefficients,
reference exponent, field, table or selected state. All shape controls start
at zero; beta starts at config's declared value (one). Construction allocates
registered state only: no solve, optimization, quadrature or normalization.

For p=1/beta, rho=hypot(x,1), r=(x/rho)**2, s=rho**(-2), tau=rho**(-p),
define the FIXED design constant kappa=3 and

    v = (1+kappa)*s/(1+kappa*s) = 4/(x**2+4),
    z = 2*v-1 = (5*s-1)/(1+3*s),
    P4(z) = sum(core_j*T_j(z), j=0,...,4),
    psi = (tau**2-s)/(p-1), with psi=s*log(s) at p=1,
    F0 = 2*tau/(1+tau),
    F1 = 6*tau**2/((1+tau)*(1+2*tau)),
    F2 = 30*tau**3/((1+tau)*(1+2*tau)*(1+4*tau)),
    J  = 9*psi/(1+8*tau),
    M0 = 17*s*tau/(1+16*tau),
    M1 = 33*s**2*tau/(1+32*tau),
    Q = r**2*[b+s*P4(z)+d0*F0+d1*F1+d2*F2+d3*J+m0*M0+m1*M1],
    h(x) = (x/rho)*exp(-p*log(rho)+Q).

The Chebyshev identities T0=1, T1=z, T2=2*z**2-1, T3=4*z**3-3*z,
T4=8*z**4-8*z**2+1 give four Horner stages. Scalar coefficient algebra
replaces the prior site-sized Clenshaw recurrence; this is an evaluation
change for the same P4, in addition to the substantive new core space.
A nested three-division recurrence evaluates the fractional sum. There is
no site-by-mode array. kappa and the six tail scales are registered buffers.
The radius two and dyadic tail scales are unfitted design choices, not
singularities measured from a profile or chosen by candidate residuals.

On the real line, 0<=s,v,tau<=1, |z|<=1 and all denominators are >=1.
F0,F1,F2,M0,M1 lie in [0,1]; the exponent mean-value theorem gives
|psi|<=1/(e*min(p,1)). Consequently

    |Q| <= |b|+sum(abs(core))+abs(d0)+abs(d1)+abs(d2)
           +9*abs(d3)/(e*min(p,1))+abs(m0)+abs(m1).

For finite controls and p>0, h is smooth, odd, zero at origin with h'(0)=1,
and positive for x>0. All log corrections are O(x**4). With
q4=b+sum(core)+d0+d1+d2+m0+m1,

    h^(3)(0) = -3*(1+p),
    h^(5)(0) = 15*(1+p)*(3+p)+120*q4.

Negative raw cubic curvature remains an explicit gauge/shape prior. A
unit-slope profile with negative cubic curvature can be rescaled into this
gauge; nonnegative curvature is excluded. The frozen canonical curvature
still depends on protected normalization. The cold scale direction r is
absent. At infinity Q->b and h~exp(b)*x**(-p). Thus the weighted moment
A=(2/pi)*integral h(x)/x dx is finite and positive for every p>0. This follows
also by bounding h/x against the bare envelope, whose moment is
Gamma(p/2)/(sqrt(pi)*Gamma((p+1)/2)). Ordinary half-line L1 requires p>1
and is not imposed. Floating-point range and finite support need admission.

Structural hypothesis: the core now has the form s*N4(s)/(1+3*s)**4,
where N4 is an arbitrary degree-four polynomial. It therefore resums an
infinite inverse-even series with a fixed recurrence, instead of terminating
that sector at s**5. Its first five s-power coefficients remain independently
adjustable (their Chebyshev-to-Taylor determinant is 2**36). Only the mode s
is shared with the old five-dimensional core space; this is not a mere
basis change or a superset of the old family. Its new pole is at s=-1/3,
with order at most four. It is off the physical interval and distinct from
all six fractional-tail poles at p=2. Successively eliminating the highest
core pole orders and unique tail poles proves raw log-shape independence
at p=2. At p=1, the core poles in tau are imaginary, the rational tail poles
negative real, and J has a logarithm; independence also persists. This is
algebraic independence, not a conditioning or optimizer guarantee.

The retained first-three fractional coefficient matrix is triangular with
diagonal (2,6,30); J retains the confluent logarithmic limit; M0,M1 retain
the generic p+2 and p+4 sectors. Their higher terms, and the core's higher
inverse-even coefficients, are tied by fixed rational functions and
exponentiation. They are representation freedoms, not enforced Hilbert-tail
coefficients or a proved solution expansion. The new compactification moves
Chebyshev variation toward larger real radii, without making a measured
claim about where the current residual is largest. It does not enforce the
similarity equation; fixed protected Adam must determine all coefficients.

The inherited confluent implementation uses expm1 with nonpositive arguments,
and a degree-five exprel series for |p-1|<1e-6, retaining the p=1 parameter
jet. Branches choose stable numerical formulas for the same mathematical
function using beta, not probe-dependent corrections. Prediction evaluates
registered state only: no solve, fit, quadrature, normalization, file I/O,
mutation or reference access. Protected fitting, frozen gauge export and
common assessment stay unchanged. No construction-time numerical solve is
used, and no no-fit pathway is claimed for this trainable representation.

Native guidance is retained here because only model.py is editable.
Current-history public step 15, the incoming distinct-pole model, qualified
at common score 2.6180039283457483e-12 (RMS 1.61802470e-6, worst squared
ratio 1.55498113e-11). It stopped for fit time at 18,279 updates, not a
completed 20,000-update fit. Fit time was 1170.211 s; complete process wall
time was 1178.85 s, peak RSS 479824 KiB. Relative to the earlier step-2
score 4.51908958e-11 this is about a 17.26-fold decrease in the measured
score, not a reference-error improvement or an existence certificate.
The public fitted beta is feedback only; initialization remains one.

Origin value and paired oddness were exact; canonical slope was
.9999999999999999, coordinate derivative relative error 1.2063e-9, and
replay passed exactly. Weighted mass was 1.0000000000000002, mesh error
1.25455e-14 and outer-shell mass 7.92490e-15 on [-36,36]. The mass-core
log bounds were [-6.45921,4.54924], with guards [-8.45921,6.54924]. Score
mesh change was 2.33426e-16. Operator g and V/x mesh changes were
8.88178e-16 and 4.81870e-10; extending to [-44,44] changed both by
3.33067e-16. Maximum residual interval width was 2.11908e-15 and estimated
pole-limit error 5.47564e-15. Public aggregates resolve origin, mass and
operator domain at their instrument resolution, but do not localize the
remaining core/tail residual or demonstrate a faulty asymptotic coefficient.
No missing-mass or operator-domain repair is indicated by this feedback.

Step-15 fit losses at updates 5000, 10000, 15000 and 18279 were
1.67473e-10, 2.04036e-11, 2.85584e-12, 2.61813e-12. Adverse transients
reached 4.19093e-8 at 8525, 1.54626e-8 at 12221 and 1.20989e-9 at 15092.
The final 3279 updates improved loss about 8.32 percent. This motivates
preserving its effective tail sector and checking field cost; the cause of
those transients and the effect of completing fitting are not established.

Earlier source guidance reports qualified complete fits for steps
1,2,3,4,6,7,8,10,13,14 at scores
7.05144219e-10,4.51908958e-11,5.34151534e-11,7.91243935e-11,
1.50217036e-10,1.27299013e-10,1.60321943e-10,3.27039882e-10,
2.25806642e-10,1.66172240e-10. Steps 9,11,12 stopped at
19462,17722,19607 updates and qualified at 7.88239354e-11,
1.89226850e-10,5.23914685e-11. These are partial fits. Step 5 failed
before update one on matrix broadcasting and has no scientific score.
Earlier rational core windows/enrichment, reciprocal cores, cold whitening,
splines and log-Gaussian bridges did not beat step 2. Step 13 had a large
mid-fit loss rise; splines in step 8 increased mass/operator mesh errors.
Those adverse results are retained, not evidence that another rational
core must improve. These earlier metrics are inherited guidance; the new
source independently read step 15 public.json, fit.json, replay.json and
process-cost.txt and sources 2,4,6,7,10,11,12,14,15. No fitted tensor was read.

The incoming source's local diagnostics found all twelve cold canonical
log-shape directions independent at p=1,1.5,2 but singular ratios only
2.60e-8,1.21e-8,2.01e-8. Its field timings were 25--39 percent slower
than its predecessor. Those are inherited field diagnostics, not new
measurements or residual Jacobians. Strong correlations, fixed scales,
restricted curvature, tied higher tails, floating-point range and the
finite Adam budget remain concerns. Common qualification means finite
instrument resolution, not continuum/reference accuracy or uniqueness.

Complexity: thirteen trainable CPU float64 scalars and seven fixed buffer
scalars (ten buffer scalars after native gauge export). Four Horner stages,
one core division, six tail divisions, hypot/log, two exponentials and at
most one confluent expm1 per site give O(N) work/live storage. Autograd retains
a fixed number of O(N) arrays; construction is allocation only. Parameter
counts do not establish efficiency. Native fit completion, qualification,
score and whole-process cost for this revision remain pending.

Source-local checks used the cached current-history dependency environment:
CPU PyTorch 2.1.2, one thread, mpmath 1.3.0 and SymPy 1.14.0. The default
Python lacked torch; no dependency installation or candidate state was loaded.
Twenty-four cold/manufactured states at p=.25,.5,1,1+/-1e-8,1+/-1e-6,
1.5,2,3,4,8 passed fifth-order origin jets, positivity on log radii [-36,36],
paired oddness, finite coordinate and mixed parameter gradients, immutable
state and exact in-memory serialized fresh-state field/derivative/beta replay.
Manufactured controls were b=.07, core=(.01,-.004,.003,-.002,.001),
tail=(.02,-.015,.004,-.007), mixed=(-.008,.003). These are declared test
controls, not initialization or fitted coefficients. All numerical state was
registered CPU float64; nonpositive/nonfinite initial beta was rejected.

The 144 scalar/empty/singleton/vector/noncontiguous/193-by-256 matrix cases
agreed exactly with flattened field and coordinate-derivative evaluation.
Maximum scaled origin-jet error was 3.55e-16. Independent coordinate and
parameter central-difference absolute errors were 3.61e-11 and 6.76e-10.
A direct 70-digit formula checked 240 fields within 1.40e-16 scaled error.
Cold fields matched the supplied unfitted envelope bitwise. Fields stayed
finite through exp(+/-576) and the largest finite float64 input, allowing
underflow. At exp(120), finite log-tail amplitudes agreed with b within
6.83e-15; the two p=8 states underflowed as expected. Finiteness of sampled
fields is not a global floating-point or mathematical accuracy certificate.

Exact symbolic algebra verified the five-core Taylor determinant 2**36,
rank-six tail residue matrix, and rank-twelve combined pole/Taylor matrix
at p=2. Raw log-shape Jacobians on 193 fixed log-radius sites [-7,7] had
rank twelve at relative threshold 1e-12 for p=.25,.5,1,1.5,2,3,4,8.
New smallest/largest singular ratios at p=1,1.5,2,3 were
2.87e-7,2.44e-7,6.16e-11,3.25e-7; matching incoming raw ratios were
1.57e-8,6.83e-9,1.39e-8,8.12e-7. Thus p=2 correlations became about
225 times worse by this diagnostic, and p=3 also worsened. These are
adverse results despite exact independence. This raw, unweighted field
diagnostic is distinct from the inherited canonical projection and from a
nonlocal residual Jacobian; it used no candidate normalization or operators.
The fixed kappa was declared before checks and was not tuned to these data.

Thirty interleaved 49408-site matrix timings after five warmups compared the
new and incoming fresh sources at cold/manufactured p=1,1.5. New/incoming
median ratios were .837--1.013 for field/parameter backward and .936--1.015
when coordinate derivatives preceded parameter backward. New medians were
7.37--11.50 ms and 30.40--48.46 ms respectively. The p=1 manufactured
case was 1.3--1.5 percent slower, so no uniform speedup is claimed. Median
fresh construction over 200 calls was .0416 ms. Benchmarks exclude operators,
normalization, support refresh and optimizer updates, and cannot establish
native fit completion or complete-process cost. Successful check harnesses
took 25.26 seconds after imports, including the incoming basis comparison.

An initial docstring edit harness failed on a quote delimiter before writing.
The first tail report included expected underflows in its maximum log-error
metric; a follow-up harness then failed on exact p==8 equality after reciprocal
roundoff. Reporting was corrected to distinguish finite amplitudes and
expected underflows; the mathematical source was unchanged by these fixes.

Only model.py task source changed; all eleven protected files still match
the current-history public source bindings. No protected evaluation module,
candidate fitter/residual assessment, construction-time solve or private
reference was executed/read, and no fitted tensor was loaded. Native guidance
remains in this docstring; its separate file was kept immutable. All reads,+harness failures, edits and bounded checks counted toward the 3600-second
source allowance, within which this source was finished. Remaining risks
include the adverse p=2 conditioning, fixed core/tail scales, restricted raw
curvature, tied higher tails, finite precision and protected Adam convergence.
No autonomous-discovery or framework-superiority claim is made.
"""
import math
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
        self.core = torch.nn.Parameter(torch.zeros(
            5, dtype=torch.float64, device="cpu"))
        self.tail = torch.nn.Parameter(torch.zeros(
            4, dtype=torch.float64, device="cpu"))
        self.mixed = torch.nn.Parameter(torch.zeros(
            2, dtype=torch.float64, device="cpu"))
        self.register_buffer("tail_scales", torch.tensor(
            [1., 2., 4., 8., 16., 32.], dtype=torch.float64, device="cpu"))
        # Fixed design constant: v=4/(x*x+4), not a measured core radius.
        self.register_buffer("core_mobius", torch.tensor(
            3., dtype=torch.float64, device="cpu"))

    def beta(self):
        return self.log_beta.exp()

    @staticmethod
    def _core_polynomial(z, coefficients):
        # Exact degree-four Chebyshev identities, evaluated by Horner.
        # Scalar coefficient algebra avoids extra site-sized Clenshaw arrays.
        c0, c1, c2, c3, c4 = coefficients.unbind()
        a0 = c0 - c2 + c4
        a1 = c1 - 3*c3
        a2 = 2*c2 - 8*c4
        return (((8*c4*z + 4*c3)*z + a2)*z + a1)*z + a0

    @staticmethod
    def _confluent(s, tau, log_s, p):
        delta = p - 1
        if bool(delta.abs() < 1e-6):
            # Stable exprel jet at p=1, also differentiable with respect to p.
            v = delta * log_s
            return s * log_s * (1 + v * (0.5 + v * (1/6 + v * (
                1/24 + v * (1/120 + v/720)))))
        if bool(delta > 0):
            return s * torch.expm1(delta * log_s) / delta
        return -tau.square() * torch.expm1(-delta * log_s) / delta

    def forward(self, x):
        # Independent compact coordinates retain small endpoint values and
        # hypot avoids overflow of x*x. Native matrices remain elementwise.
        rho = torch.hypot(x, torch.ones_like(x))
        unit = x / rho
        s = rho.reciprocal().square()
        r = unit.square()
        p = self.beta().reciprocal()
        log_rho = rho.log()
        log_tail = -p * log_rho
        tau = log_tail.exp()
        psi = self._confluent(s, tau, -2 * log_rho, p)
        k0, k1, k2, kj, km0, km1 = self.tail_scales.unbind()
        g0 = 1 + k0
        g1 = g0 * (1 + k1)
        g2 = g1 * (1 + k2)
        fractional = g2*self.tail[2]
        fractional = g1*self.tail[1] + tau*fractional/(1+k2*tau)
        fractional = g0*self.tail[0] + tau*fractional/(1+k1*tau)
        fractional = tau*fractional/(1+k0*tau)
        confluent = (1+kj)*self.tail[3]*psi/(1+kj*tau)
        mixed = ((1+km0)*self.mixed[0]*s*tau/(1+km0*tau)
                 + (1+km1)*self.mixed[1]*s.square()*tau/(1+km1*tau))
        # A fixed rational compactification changes the core function space.
        # Form it from s so extreme x does not require evaluating x*x.
        v = (1+self.core_mobius)*s/(1+self.core_mobius*s)
        polynomial = self._core_polynomial(2*v-1, self.core)
        q = r.square() * (
            self.tail_amplitude + s*polynomial + fractional + confluent + mixed)
        return unit * torch.exp(log_tail + q)


def build_model(config):
    return Model(config)
