"""Confluent mixed-tail shoulder for the single a=.25 real-line profile.

Construction and mathematical provenance
---------------------------------------
Oddness, canonical gauges, compactification, adjustable leading power and
Hilbert power identities are supplied priors from immutable README.md.
The five Chebyshev core modes, unit raw tail amplitude, fixed rational detail
and confluent first/mixed/second tails reuse incoming SOURCE 36b61a00640e.
The fixed raw radius 2 and rational shoulder formula reuse current-history
SOURCE 14869f2be98a, but its independent shoulder amplitudes are not added.
This revision adds ONE mixed fractional/rational direction to the incoming
span. No fitted tensors, selected coefficients, reference exponent/fields or
lookup tables initialize it. Beta starts at config['initial_beta'] (declared
1) and ALL corrections start at zero. Construction is constant-work algebra:
no numerical solve, optimization, quadrature, normalization or data access.

For q=1/beta>0 define

    r=hypot(x,1), ell=log(r), u=(x/r)^2, s=r^(-2), z=u-s, w=4*u*s,
    J(t,a)=(r^(-t)-r^(-a))/(a-t), J(a,a)=r^(-a)*ell,
    D=2*u^2*J(q,2),
    C_p=[r^(-p)-(4-p)*s/2-(p-2)*s^2/2]/[(p-2)*(p-4)],
    E=4*u^2*C_(2q), B=w^3/(1+z^2), eta=3/5,
    Q=[(1-eta^2)*w/(1-eta*z)^2]^2, M=D*Q,
    L=w*sum_(j=0)^4 b_j*T_j(z)+D*(d+m*w+n*Q)+e*E+j*B,
    h(x)=(x/r)*exp[-q*ell+L].

For x>0, Q=sech^4(log(x/2)); equivalently
Q=[16*x^2/(x^2+4)^2]^2. The compact formula avoids log(abs(x)) and large
coordinate powers. Its denominator 1-eta*z is at least .4. Q is even,
smooth, lies in [0,1], peaks at raw radius 2, is x^4+O(x^6) at zero and
256*x^(-4)+O(x^(-6)) at infinity. This radius is a design assumption, not
a measured residual location or a physical scale. The amplitude n is free.

Since D=x^6+O(x^8), M=x^10+O(x^12): at fixed inherited coefficients it
first changes the raw eleventh-degree profile coefficient by n. The cubic,
quintic, seventh and ninth coefficients are unchanged. In particular the raw
cubic coefficient remains 4*sum_j((-1)^j*b_j)-(q+1)/2. These statements do
not freeze canonical normalization or nonlocal Hilbert fields.

Away from q=2 the exact response is

    M=2*u^2*Q*(r^(-q)-s)/(2-q).

For generic 0<q<2 its fractional leading term is
512/(2-q)*x^(-(q+4)); an analytic x^(-6) piece and further analytic/fractional
mixing also occur. For q>2 the analytic x^(-6) term leads, with the same
continued fractional coefficient. At q=2, M=2*u^2*s*ell*Q, with leading
512*x^(-6)*log(x). Thus the new amplitude frees one mixed log-tail response
beyond the existing q and q+2 channels. It does not isolate a pure asymptotic
power or impose its PDE coefficient. The rational factor introduces a pole
at z=5/3 absent from the incoming correction functions; together with the
sampled span audit this establishes a new direction at audited generic q.
No full-rank theorem at every exponent is asserted; old resonances persist.

The supplied power identity and nonlinear products motivate fractional and
analytic mixed orders. The chosen rational modulation tests an outer mixed
response with a higher origin order, rather than asserting that the residual
is located there. All retained tail coefficients remain free. For q>0,
J(q,2)=integral_0^ell exp[-q*tau-2*(ell-tau)] d tau <= u/2, hence
0<=D<=u^3<=1 and 0<=M<=1 in exact arithmetic. All log corrections are even,
smooth, bounded, vanish at zero and tend to zero at infinity for finite q
and coefficients. Consequently h is odd, h(0)=0, h'(0)=1, positive for x>0,
and h~x^(-q) with raw coefficient one. A=(2/pi)*integral h(x)/x dx is finite
and positive, since h/x is bounded at zero and O(x^(-q-1)) at infinity.
Ordinary half-line L1 is not required and diverges for the cold q=1 envelope.
The unit raw amplitude is an inherited finite-family restriction; canonical
amplitude varies through protected A. Extreme states can overflow/underflow.

Protected evaluation alone measures A,k and freezes f(y)=h((A/k)*y)/A.
Forward and beta evaluate registered state only; no PDE/operator solve, fit,
fresh normalization, reference loading or probe-dependent patch occurs.

Inherited arithmetic and limitations
-----------------------------------
J uses expm1 against the slower decay. At |t-a|<1e-4 the inherited signed
degree-nine exprel series preserves exponent derivatives at coalescence;
for finite float64 radii its argument magnitude is below .0711. C_p uses
the nearer anchor 2 or 4, so the outer denominator has magnitude >=1.
C_2=[s*ell-(s-s^2)/2]/2; C_4=[(s-s^2)/2-s^2*ell]/2. Arithmetic branches
use only the saved scalar exponent, with no mutation or query adaptation.
This revision leaves these helpers unchanged. Incoming guidance recorded
134 tiny negative E basis values (minimum -9.28583e-33) from cancellation,
and worst mixed-helper relative derivative error 1.86746e-4 among reference
magnitudes above 1e-12. Hypot can also round tiny ell to zero. Those adverse
helper diagnostics remain; exponential profile positivity does not repair
them. The new Q uses bounded rational arithmetic with no new coalescence.

Current-history public diagnosis and useful native guidance
----------------------------------------------------------
All EIGHTEEN exposed public evaluation records numerically qualified; FIFTEEN
completed 20,000 updates and THREE stopped at fit_time. No rejected record is
exposed. Public mean square / fit-work seconds, retaining adverse and partial
trials and repeated source reuse:

    7136886fc232 first-tail/core       8.95449190e-10 / 719.31
    2231a498f33e second/mixed tails    2.18116464e-11 / 676.80
    2217a86a175e saturated third tail  7.93955563e-11 / 664.68 WORSE
    2d5046c9cd15 fixed rational detail 1.55954875e-11 / 819.98
    5b30efedc69f full rational core    5.08410831e-10 / 852.24 WORSE
    5dd6a1fc8318 delayed tails        3.40399443e-9  / 641.26 WORSE
    b1839edc7cd7 movable detail       2.35734236e-11 / 643.55 WORSE
    68b790dd3912 independent jets     2.16581898e-11 / 949.90 WORSE
    9bae35ae73da unit raw amplitude   1.51472557e-11 / 720.19
    26fd974c7267 confluent first/mix   3.75527267e-12 /1046.05
    042d9d28f7e5 predecessor reuse     1.55954875e-11 / 700.93 WORSE
    36b61a00640e confluent second      1.99832051e-12 / 711.10 incoming
    70e311a172e1 polynomial bubbles    2.06905456e-12 /1170.15 PARTIAL
    14869f2be98a reciprocal shoulders  1.32050633e-12 / 804.73
    0edb411d3020 extra dyadic pair     3.78689828e-12 /1170.22 PARTIAL/WORSE
    7e0ae949e7c7 Jacobi extension      1.03829544e-11 /1170.17 PARTIAL/WORSE
    816f7f43424f confluent cubic tail  2.07736580e-12 / 862.87 WORSE
    c485bb711a65 central double pole   1.39213713e-11 / 916.48 WORSE

Actual partial update counts are 19,837 / 15,484 / 15,839 respectively; none
is a completed 20,000-update fit. Last central-mode mean/worst squared
residuals are 1.39213713e-11 / 1.67914486e-10, versus incoming
1.99832051e-12 / 1.27331059e-11 and shoulders
1.32050633e-12 / 1.15344768e-11. Extra flexibility alone is not evidence of
improvement. Incoming proxies at 10000/15000/20000 are
2.11873087/2.02296081/1.99836455 e-12: small late progress persists.
Representation error and finite-time optimizer error remain unseparated.
No public score exists for THIS revision; public beta is not initialization.

Incoming origin value/slope/oddness resolve exactly; coordinate derivative
relative check is 1.20562827e-9, singleton/batch discrepancy 6.10396866e-16,
and fresh replay of values AND coordinate derivatives is exact. Canonical
mass resolves on log [-36,36], center error 2.22044605e-16, shell mass
7.92489936e-15, guarded bounds [-8.45921435,6.54923475]. Coverage refinement
changes mean square by 3.23616354e-16. Operator orders 16/32/64 yield
1.998320025/1.998320509/1.998320511 e-12; g/W mesh changes are
1.22124533e-15 / 4.81869988e-10, domain changes <=3.33066907e-16.
Maximum residual interval width is 1.06617205e-15; no roundoff-limited sites.
Across all exposed records origin, support, operators and replay resolve.
This supports preserving their treatment. Regional residual samples are
absent: the remaining core/tail error is UNLOCATED. Analytic tail admissibility
does not certify remote equation accuracy. Finite numerical qualification is
not a continuum, existence, reference-accuracy or branch-uniqueness certificate.

Revision-local bounded representation checks
--------------------------------------------
Two NumPy BASIS-ONLY audits used 1,601 radii exp(-12)..exp(12), with no
normalization/equation/operator/fitter calls. Column-normalized full correction
condition numbers, incoming -> selected M (relative new span remainder):

    q=.5: 371.921 -> 1341.087 (.0193797)
    q=1:  940.732 -> 2051.806 (.0156488)
    q=1.25:1792.415 -> 3024.160 (.0145985)
    q=1.5:3805.305 -> 5242.726 (.0136514)
    q=2: 26324.866 -> 29254.935 (.0117821)
    q=3,4: incoming and selected remain numerically rank deficient (>1e15).

All finite-rank cases worsen conditioning (ADVERSE). The unshifted alternative
D*w^2 had remainders .00173540/.00095101 at q=1/1.5 and conditions
2161.820/8240.140. It was not retained or fitted. The rational mixed mode has
less overlap there but is not a global conditioning improvement. These basis
metrics are not normalized PDE Jacobians, convergence evidence or public scores.

Cached CPU torch 2.5.1 checks passed syntax, CPU float64/scalar positive beta,
empty input, invalid initial-beta rejection and complete registered state.
Fifty-seven manufactured states used zero/signed corrections, q=.5,1,1.25,
1.5,2,3,4 and both sides of coalescence/arithmetic switches, with signed radii
10^(-60)..10^60. Origin value/slope and oddness were exact, positive-half-line
values and coordinate/parameter gradients finite, state unmodified, repeat
values exact and strict in-memory replay of values AND coordinate derivatives
bitwise exact. Cold values/coordinate derivatives were bitwise identical to
incoming SOURCE. Singleton/batch value relative discrepancy was 5.87419e-16;
coordinate derivative discrepancy scaled by 1+abs(derivative) was 2.98736e-16.
All eleven parameter derivatives of a coordinate-derivative functional matched
centered differences to 4.55920e-11 absolute over six manufactured exponents,
including coalescence and the outer arithmetic switch. A cold beta=2 value at
x=1e300 was positive, 1.00000e-150; its coordinate derivative underflowed to
zero (ADVERSE, not relative derivative accuracy at that radius).

Independent 100-digit mpmath algebra checked 130 new-mode sites, including
q=2 and q=2 +/-1e-10, +/-1e-4, with x=1e-8..1e40. Maximum absolute errors
[M,M_x,M_q,M_xq] were [1.30308e-16,3.26134e-16,2.55535e-13,3.27100e-13].
For reference magnitudes above 1e-12, maximum relative errors were
[1.71391e-15,1.42847e-14,6.45972e-11,5.61651e-11]. Tiny near-origin M can
round to zero with inherited ell; no uniform relative accuracy is claimed.
At x=.001, M/x^10 was .9999957500/.9999956250/.9999955000 for q=1/1.5/2
(limit one). Independent large-radius ratios approached the declared tail
coefficients, including the q=2 logarithm. These checks concern explicit
representation/algebra, not the equation residual or normalization moment.

One-thread 49,408-site value/coordinate/parameter derivative benchmarks used
three warmups and twenty alternating calls per source, manufactured states
only. Incoming/new medians in seconds and new/old ratios:

    beta=.8: .0369088 / .0512843, 1.38949 SLOWER
    beta=1:  .0254760 / .0328697, 1.29022 SLOWER
    beta=.5: .0325656 / .0392121, 1.20409 SLOWER

Mean fresh construction / construction-plus-strict-reload over 100 modules
was 5.47482e-5 / 9.44302e-5 s. Interface checks, independent algebra and
benchmarks took 2.02/1.47/5.74 s excluding imports; command wall times were
4.41/7.29/about10 s (last two sessions overlapped). Two small NumPy basis
scripts and dependency discovery added several seconds. No dependencies
were installed; cached torch lacks NumPy, so no torch-to-NumPy conversion
was used. Source work stayed within the 3600s allowance. Microbenchmarks
and historical costs do not establish native fit completion or whole cost.

Complexity and protected follow-up
---------------------------------
Eleven trainable scalars: log beta, d/m/n/e, rational detail j, five core modes.
Twenty-six fixed registered scalars: 25 Chebyshev transform entries and eta.
Construction is O(1); forward is O(K^2+K*N), K=5, O(N) working arrays and
O(K*N) autograd storage. Added work is one rational window and its mixing,
with no new exponentials, logs, expm1 calls or exponent branches. Protected
frozen normalization has separately managed state and cost. Prediction and
fresh reconstruction/reload costs count toward the native whole-process cost.

Only model.py changes; useful native guidance is maintained here under the
source-only boundary. No protected fitter/judge execution, fitted-state imports
or private-reference reads occurred. Native follow-up must preserve adverse,
failed and partial records; require the unchanged 20,000 updates within 1200s
fit / 1500s whole-process limits, support refresh every100 and refined proxy
admission. Compare mean/worst residuals, origin/core, support, operator
refinement and fresh replay. Fitting, public assessment, normalization and any
nominated/frozen reference readout remain protected. Completion, convergence,
mixed-tail adequacy, resonances, conditioning and continuum accuracy are open.
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
        self.tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.mixed_tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.outer_mixed_tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.second_tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.core_detail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.core = torch.nn.Parameter(torch.zeros(
            5, dtype=torch.float64, device="cpu"))
        self.register_buffer("chebyshev_to_power", torch.tensor([
            [1., 0., -1., 0., 1.],
            [0., 1., 0., -3., 0.],
            [0., 0., 2., 0., -8.],
            [0., 0., 0., 4., 0.],
            [0., 0., 0., 0., 8.],
        ], dtype=torch.float64, device="cpu"))
        # Fixed raw radius 2 from the public shoulder construction, not a fit.
        self.register_buffer("outer_skew", torch.tensor(
            .6, dtype=torch.float64, device="cpu"))

    def beta(self):
        return self.log_beta.exp()

    @staticmethod
    def _power_difference(log_radius, exponent, anchor, decay, anchor_decay):
        """(r**(-exponent) - r**(-anchor))/(anchor-exponent).

        Use expm1 against the slower decay to avoid both cancellation and
        an overflowing exponential multiplied by an underflowed anchor power.
        A signed series preserves exponent derivatives at coalescence. The branch
        uses the saved scalar exponent and performs no state mutation.
        """
        gap = exponent - anchor
        argument = gap * log_radius
        gap_value = float(gap.detach())
        if abs(gap_value) < 1e-4:
            # For finite float64 x, |argument| < .0711 in this band.
            # The degree-nine exprel series has error below 8.4e-20 there.
            series = -1 / 3628800
            for coefficient in (1 / 362880, -1 / 40320, 1 / 5040,
                                -1 / 720, 1 / 120, -1 / 24,
                                1 / 6, -.5, 1.):
                series = series * argument + coefficient
            return anchor_decay * log_radius * series
        if gap_value < 0:
            return decay * torch.expm1(argument) / gap
        return anchor_decay * (-torch.expm1(-argument)) / gap

    @classmethod
    def _second_power_difference(cls, log_radius, p, decay, u, s):
        """The second divided difference of r**(-t) at t=2,p,4.

        Use the nearer endpoint for the inner divided difference. The outer
        denominator then has magnitude at least one, including at p=2 or 4.
        The fixed endpoint difference (s-s**2)/2 equals u*s/2; this form
        avoids subtracting nearly equal powers at the origin.
        """
        fixed_difference = .5 * u * s
        if float(p.detach()) <= 3:
            inner = cls._power_difference(log_radius, p, 2, decay, s)
            return (inner - fixed_difference) / (4 - p)
        inner = cls._power_difference(log_radius, p, 4, decay, s.square())
        return (fixed_difference - inner) / (p - 2)

    def forward(self, x):
        radius = torch.hypot(x, torch.ones_like(x))
        signed_radius = x / radius
        u = signed_radius.square()
        s = radius.reciprocal().square()
        z = u - s
        power_coefficients = self.chebyshev_to_power @ self.core
        polynomial = power_coefficients[-1]
        for coefficient in power_coefficients[:-1].flip(0).unbind():
            polynomial = polynomial * z + coefficient
        q = self.beta().reciprocal()
        log_radius = radius.log()
        log_decay = -q * log_radius
        decay = log_decay.exp()
        core_window = 4 * u * s
        u_squared = u.square()
        # Subtract the analytic core power before coalescing at q=2.
        tail_difference = 2 * u_squared * self._power_difference(
            log_radius, q, 2, decay, s)
        # sech**4(log(abs(x)/2)), evaluated smoothly at zero and large x.
        # Multiplication by D supplies a confluent q+4 mixed-tail response.
        outer_window = ((1 - self.outer_skew.square()) * core_window
                        / (1 - self.outer_skew * z).square()).square()
        # Unsaturated second-tail response, continued through 2*q=2 and 4.
        second_difference = 4 * u_squared * self._second_power_difference(
            log_radius, 2 * q, decay.square(), u, s)
        rational_detail = (core_window.square() * core_window
                           / (1 + z.square()))
        # All log corrections vanish at infinity: raw tail amplitude is one.
        log_correction = (core_window * polynomial
                          + tail_difference * (self.tail
                                               + self.mixed_tail * core_window
                                               + self.outer_mixed_tail
                                               * outer_window)
                          + self.second_tail * second_difference
                          + self.core_detail * rational_detail)
        return signed_radius * (log_decay + log_correction).exp()


def build_model(config):
    return Model(config)
