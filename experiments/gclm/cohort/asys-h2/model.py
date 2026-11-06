"""Positive odd envelope with an extended Sobolev log-radius core.

Only the declared single real-line gCLM profile at a=.25 is represented.
Oddness, canonical gauge symmetries, compactification and the leading balance
q=1/beta are supplied priors. The positive exponential log correction,
first fractional tail, shifted simple-pole pair and Jacobi core reuse the
current-history native sources, especially steps 12 and 15. Classical Jacobi
and Legendre norms/recurrences are standard mathematical priors (NIST DLMF,
https://dlmf.nist.gov/18.3 and https://dlmf.nist.gov/18.9).

This revision extends the analytic core from degree seven in the prefill
(and nine in public step 15) to degree eleven, and changes its coordinates
using a fixed value-plus-derivative inner product in log radius. The PDE
contains a transport derivative, and D_log_x log(h) includes D_log_x L
linearly. Giving the latter comparable coordinate scale is the structural
hypothesis. This metric is NOT the protected residual Hessian: it omits
Hilbert/velocity operators, canonical normalization, moving support, beta,
amplitude, tail and pole tangents. Its unit derivative weight is an explicit
representation choice, not inferred data or a new identity of the equation.

Explicit construction: r=hypot(1,x), q=1/beta, p=(1+q)/2,
    s=r**(-2), u=(x/r)**2, t=2*u-1, b=u*s,
    J_n(t)=sigma_n*P_n**(1,1)(t), n=0,...,11,
    sigma_n=sqrt((2*n+3)*(n+2)/(6*(n+1))),
    K_n(t)=sum_{j<=n} S[n,j]*J_j(t),
    eta=exp(log(4)*tanh(pole_shape)), mu=.5*tanh(pole_shift),
    D=(t-mu)**2+eta,
    c=(hypot(1-mu,sqrt(eta))+hypot(1+mu,sqrt(eta)))/2,
    R=1/(c+sqrt((c-1)*(c+1)))**2, alpha=2*mu*R/(1+R),
    E_m=(T_m-2*alpha*T_{m-1}+R*T_{m-2})/(4*R*D), m=12,13,
    Q=sum core[n]*K_n(t)+interior[0]*E12+interior[1]*E13,
    L=-correction*u+b*Q+tail*u*s**(q/2),
    h=(x/r)*exp(-q*log(r)+L).

For x>0, dt/dlog(x)=1-t*t, b=(1-t*t)/4. Define psi_n=b*J_n and
    <v,w>_S=integral (v*w+D_log_x(v)*D_log_x(w)) dlog(x).
Jacobi orthogonality gives integral psi_n*psi_m dlog(x)=delta_nm/12.
The elementary weighted derivative identity
    d_t((1-t*t)*P_n**(1,1))=-2*(n+1)*P_{n+1}
gives psi_n'=-c_n*P_{n+1}, c_n=(n+1)*sigma_n/2. Thus the fixed Gram
matrix is H_nm=delta_nm/12+c_n*c_m*F_{n+1,m+1}, where
    F_lk=integral_-1^1 (1-t*t)*P_l(t)*P_k(t) dt.
Let N_l=2/(2*l+1), a_l=(l+1)/(2*l+1), d_l=l/(2*l+1). Using
    t*P_l=a_l*P_{l+1}+d_l*P_{l-1},
F is diagonal plus second off-diagonals:
    F_ll=N_l-a_l*a_l*N_{l+1}-d_l*d_l*N_{l-1},
    F_l,l+2=-a_l*d_{l+2}*N_{l+1}, all other entries zero.
H is positive definite since its mass part is I/12. If H=C*C^T is its
positive-diagonal Cholesky factor, S=sqrt(3/20)*C**(-1) gives
    <b*K_n,b*K_m>_S=(3/20)*delta_nm.
H00=3/20, so K0=J0=1. S is triangular, parity-preserving and invertible;
K_n has degree n. Production computes this fixed small matrix and the
Jacobi-to-Chebyshev conversion by elementary scalar arithmetic. This is
basis linear algebra, with no equation solve, optimization, quadrature,
profile integration or data loading. The resulting 12x12 conversion buffer
is registered. No basis factorization or conversion construction occurs in forward.

The inherited spectral-factor identity is
    |1-2*alpha*exp(i*theta)+R*exp(2*i*theta)|**2=4*R*D(cos(theta)),
    E_m(cos(theta))=sum_{k>=0} a_k*T_{m+k}(cos(theta)),
    a0=1, a1=2*alpha, ak=2*alpha*a_{k-1}-R*a_{k-2}.
E12/E13 have no Chebyshev components through degree eleven and leading
coefficient one. Deflation is in the Chebyshev weight; neither these two
modes nor their combination with K are Sobolev orthogonal. With z=2*alpha,
    E10=T10+z*T11+(z*z-R)*E12-z*R*E13,
    E11=T11+z*E12-R*E13.
The residue map has determinant R*R>0. Hence every step-15 P9/E10/E11
core, and therefore every prefill P7/E8/E9 core, is represented at the same
pole state. New numerators may have degree 13 rather than 11, so the family
is strictly enlarged. These maps are algebra, never fitted initialization.

Endpoint behavior, gamma_minus=1/(1+2*alpha+R), gamma_plus=1/(1-2*alpha+R):
    h=x+[sum core[n]*K_n(-1)+gamma_minus*(interior[0]-interior[1])
         -correction+tail-p]*x**3+O(x**5),
    h~exp(-correction)*x**(-q) at positive infinity.
The x**(-2) coefficient of L is correction+sum core[n]*K_n(1)
+gamma_plus*sum(interior); the first fractional coefficient is tail at
power x**(-q). Coincident powers combine. In particular the exact q=2
duplication of tail and core[0] remains; public attempts to remove it
regressed. No negative cubic, beta interval or extra tail ladder is imposed.

For finite coefficients and finite positive beta, L is even, smooth,
bounded, O(x*x) at zero and tends to -correction. eta in [1/4,4] and
mu in [-1/2,1/2] imply D>=1/4 and the interior Joukowski root has
modulus <=1/phi, phi=(1+sqrt(5))/2. Therefore |E_m|<=phi**4. If B_n
is the absolute row sum of the registered conversion matrix, |K_n|<=B_n,
    |L|<=|correction|+sum B_n*|core[n]|/4
          +|tail|+phi**4*sum |interior|/4.
Bounded comparison with the supplied envelope proves oddness, exact zero
origin, unit raw slope, positivity for x>0, decay and finite positive
weighted moment A=(2/pi)*integral h(x)/x dx. The envelope moment is
Gamma(q/2)/(sqrt(pi)*Gamma(p)). Ordinary half-line L1 can diverge, as at
the beta=1 cold start, and is not required. This proof is not a float64
resolution guarantee, an existence proof or a reference-accuracy certificate.

Useful native guidance from actual current-history public evaluation:
    source                               scalars updates fit sec common score
    six core / first fractional tail         9    20000    502.4  2.79052143e-11
    q/2q/(q+2) fractional ladder            11    20000    520.2  2.38006245e-10
    Hermite endpoint/core                   11    20000    925.9  3.03051108e-11
    fixed negative raw cubic                 9    20000    751.8  8.50815679e-11
    rational second fractional tail         10    20000    609.0  2.19037279e-10
    simple-pole rational interior pair      11    20000    619.1  6.64251304e-12
    added double-pole interior pair         13    20000    559.2  1.29314583e-10
    deflated simple-pole coordinates        11    20000    696.3  6.68213225e-12
    two-scale core bumps                    13    19066   1170.2  1.12851805e-11
    degree-seven / deflated simple pair     13    18377   1170.2  5.67617967e-13
    movable centered simple-pole pair       14    20000   1105.5  3.49894819e-13
    shifted movable simple-pole pair        15    16888   1170.2  3.05137247e-13
    origin-lifted u**9 fractional tail      15    16067   1170.2  1.11999688e-09
    confluent fractional tail              15    16241   1170.2  8.21793868e-12
    degree-nine Jacobi log-radius core     17    18675   1170.1  6.00081915e-14

All fifteen qualified on the finite common instrument. Rows nine, ten,
twelve through fifteen stopped for fit_time, not completed 20,000 updates.
The latest score is about 5.085 times lower than the prefill; timings and
updates differ, and degree/coordinate contributions cannot be separated.
Latest losses at 1000/5000/10000/15000/18675 were 2.84547e-7/2.02672e-8/
2.46278e-10/6.00155e-14/6.00085e-14. The late plateau and improved core
trial motivate this revision, without localizing or establishing its cause.
More freedom does not ensure a better finite Adam fit. Tail ladders,
negative cubic, double poles, core bumps and lifted/confluent tails regressed.

Origin/core: all fifteen passed zero-origin, unit-slope, oddness, derivative
consistency and replay. Public fitted cubics, pole states and pointwise
residual sites were not published. No region of the remaining error can
be identified. Inherited deflation improved a manufactured condition number
333.299 to 18.2784 without improving its native score. Step 15 reported
log-core condition numbers 2.72/2.77/3.18, worsening to 7.44/11.18/19.78
with pole tangents. These were raw algebra tests, not nonlinear residual
conditioning. The Sobolev metric here addresses a different quantity.

Tail: prefill and latest raw surveys resolved at log extent 72, canonical
mass on [-36,36]. Latest canonical mesh estimate was 4.21e-12, inner-domain
estimate 4.41e-13. Public beta near .68451 is diagnostic only, never
initialization. These surveys resolve a weighted moment at finite instrument
resolution, not tail PDE accuracy or global numerical resolution. Retaining
the original fractional mode follows adverse public tail trials, not a
claim of asymptotic completeness.

Operator: latest orders 16/32/64 changed score by 1.17e-19, coverage by
9.03e-18 and maximum residual interval width was 2.19e-16, versus score
6.00e-14 and RMS 2.45e-7. Prefill changes were 5.65e-19 and 1.23e-16,
interval width 7.87e-16. Across all fifteen, maximum coverage change was
3.28721e-13, operator score change 2.71e-17 and interval width 4.37e-14.
W mesh estimate stayed about 4.82e-10; g mesh and g/W domain estimates
were near machine precision. These estimated mesh/domain/local-patch errors
do not implicate operator resolution as the observed plateau or regression
cause. They are not rigorous bounds or continuum/reference accuracy.
Qualification does not prove existence, uniqueness, autonomous discovery or
framework superiority. The old starter score 1.09862e-5 remains inherited,
unverified source guidance.

Initialization is configured beta (declared 1), with all correction, core,
tail, residue and pole scalars zero: exactly the supplied unfitted envelope.
No reference fields, reference beta, selected coefficients or historical
fitted tensors enter construction. No construction-time PDE solve or fit.
All prediction state is registered CPU float64. forward/beta evaluate it
without fitting, PDE/integral solve, fresh normalization or probe correction.
Protected native fitting and once-frozen canonical rescaling are unchanged.

Complexity: nineteen trainable scalars, 146 scalar buffer entries (144 basis
conversion plus two pole bounds). Fixed size-12 construction uses O(12**3)
scalar arithmetic and O(12**2) storage; every reconstruction repeats it.
Prediction assembles coefficients in scalar space, then uses one degree-13
Clenshaw recurrence, one rational division and one fractional tail mode.
Arithmetic O(14*N+144), autograd storage O(14*N), no-tape field temporaries
O(N). Protected operator calls add their own work. Parameter count does not
measure efficiency. Derivative scaling can harm Adam's value directions;
basis conditioning is not protected nonlinear conditioning. Added degree,
fit-time truncation, larger endpoint coefficients, pole/residue coupling,
dormant cold pole gradients, tanh saturation, q=2 duplication and unlocalized
residual remain risks. Formal score and whole-process costs are unmeasured.

Bounded source-only checks used cached Python 3.13.11, PyTorch 2.14.1+cpu
and NumPy 2.4.3, CPU float64 and one thread. No dependencies installed or
GPU computations. Manufactured coefficients were explicitly
core=[.12,-.07,.03,-.02,.014,-.011,.009,-.006,.004,-.003,.002,-.0015],
correction=.18, tail=-.09, interior=[.017,-.011]. These are algebra fixtures,
not historical fitted state, selected coefficients or solution estimates.

Independent 128-node Gauss-Legendre quadrature of POLYNOMIAL basis products
verified the Jacobi mass Gram I/12 within 1.38e-15 absolute error and the new
Sobolev Gram (3/20)*I within 1.25e-15. Independent quadrature-based Cholesky
and direct Jacobi recurrence matched the conversion-buffer basis within
5.10e-14 scaled error. Scaled error denotes abs(a-b)/(1+abs(a)+abs(b)).
The basis Gram H had condition number 84.05 and the conversion matrix 62.09,
versus the previous source's reported conversion condition 15.84: coordinate
whitening here does not improve every numerical condition number. K0=1 was
verified. Fresh construction's median time over 25 calls was .2385ms.

Across 125 combinations of beta=.4/.5/(2/3)/1/2, shape=-12/-1.2/0/1.2/12
and shift=-12/-.8/0/.8/12, an independent complex Joukowski root and direct
Chebyshev recurrences matched the assembled prediction. Maximum scaled
errors were 6.26e-17 for fields, 2.22e-16 for coordinate derivatives,
9.35e-16 for parameter gradients, 3.80e-15 for mixed coordinate/parameter
gradients and 1.65e-16 for pole data. Oddness discrepancy was zero in this
suite. Positive half-line values, exact zero origin, finite fields/gradients
and the bounded-envelope comparison passed. One origin fixture gave
(h,h',h'')=(0,1,0). Twenty-five third-order derivatives at beta=1 matched
the stated cubic within 6.67e-16 absolute error. Five cold betas matched the
supplied envelope within 5.93e-15 relative error.

CPU float64 registered state, invalid-beta rejection, empty/scalar/broadcast
inputs and immutable prediction passed. Own manufactured state replayed
exactly through in-memory serialization into a fresh model, including values,
coordinate derivatives and beta. Singleton/batch scaled discrepancy was
9.91e-17. Cold pole gradients were exactly zero. At beta=1 and pole
(shape,shift)=(-12,-12)/(0,0)/(12,12), manufactured values at exp(576) and
1e308 stayed positive and finite with finite coordinate derivatives. Leading
amplitude relative error was 5.02e-14. No normalization was performed.

For 25 pole states, the spectral-factor identity held within 3.84e-16
scaled error; 101-term generating series within 3.82e-15. On 512 midpoint-
angle nodes, low Chebyshev coefficients and leading-coefficient errors of
E12/E13 were <=6.67e-16, endpoints within 2.61e-16 scaled error. In 54
manufactured comparisons, the explicit maps from both prefill and step-15
families matched fields within 6.27e-17 and coordinate derivatives within
1.96e-16 scaled error. Small basis-coordinate linear solves were used only
in these algebra checks, never to solve the public equation or initialize
production. No saved historical tensors or selected coefficients were loaded.

Independent 192-node polynomial/rational BASIS-product quadrature at pole
(-1.2,-.8)/(0,0)/(1.2,.8) gave normalized-column Sobolev condition numbers
4.535/4.757/5.561 for latest J0..J9/E10/E11, 4.972/5.156/5.952 for the
extended J0..J11/E12/E13, and 3.651/3.853/4.789 for new K0..K11/E12/E13.
The extended families had rank 14. In contrast, log-value L2 conditioning
WORSENED from extended-J 2.736/2.779/3.198 to new-K 8.544/8.814/9.777.
Appending the manufactured width/shift tangents gave rank 16 and worsened
new Sobolev values to 9.693/16.135/29.208 and L2 values to 12.440/16.598/
24.020. These comparisons concern raw log-core basis functions only; they
omit beta, amplitude, tail, canonical rescaling, moving support, operators
and nonlinear residual derivatives. They are not fitting condition numbers.
The polynomial/rational quadratures integrated no profile, normalization
moment or PDE operator. This adverse value/pole evidence remains unresolved.

Forty-eight alternating synthetic forward/coordinate-derivative/backward
cycles (two warmups and fourteen timed per source; 193*256 field sites) gave
median times .0291844s for prefill, .0276615s for latest public core, and
.0306210s here. All used the same manufactured function via algebraic maps.
New/prefill ratio was 1.0492 and new/latest 1.1070; ranges overlapped.
This small sample indicates field overhead, not native fit time, and raises
risk of an earlier fit_time stop. It cannot promise 20,000 updates.

The five suites took .6021/1.2006/.1916/1.6995/30.1573s, excluding imports
and environment discovery. Source generation completed within 3600 seconds.
Checks ran no optimizer, PDE operator/equation solve, protected fitter/judge,
private reference access, construction optimization or profile normalization.
Only model.py was edited. Immutable/protected sources, configuration and
native guidance files were unchanged; useful native guidance, current public
results and adverse evidence are maintained in this source.

Formal fitting, common score, qualification and whole-process costs of this
revision remain unmeasured. The Sobolev/value conditioning tradeoff, extra
field cost, truncated fitting, dormant and coupled pole directions, tanh
saturation, q=2 duplication and unlocalized origin/core/tail residual remain
uncertain. Analytic admissibility and algebra checks do not remove these
risks. No native accuracy improvement is claimed.
"""
import math
import torch


def _core_to_chebyshev():
    """Fixed Sobolev basis conversion; only finite coefficient linear algebra."""
    size = 12
    scales = [math.sqrt((2*n+3)*(n+2)/(6*(n+1))) for n in range(size)]
    jacobi = [[1.0] + [0.0]*(size-1), [0.0, 2.0] + [0.0]*(size-2)]
    for n in range(1, size-1):
        times_t = [0.0]*size
        times_t[1] = jacobi[n][0]
        for j in range(1, n+1):
            times_t[j-1] += .5*jacobi[n][j]
            times_t[j+1] += .5*jacobi[n][j]
        leading = (2*n+3)*(n+2)/((n+1)*(n+3))
        trailing = (n+2)/(n+3)
        jacobi.append([leading*times_t[j]-trailing*jacobi[n-1][j]
                       for j in range(size)])
    jacobi = [[scales[n]*v for v in row] for n, row in enumerate(jacobi)]

    # H=I/12+stiffness, evaluated by Legendre norms rather than quadrature.
    gram = [[0.0]*size for _ in range(size)]
    for n in range(size):
        ell = n+1
        a = (ell+1)/(2*ell+1)
        d = ell/(2*ell+1)
        diagonal = 2/(2*ell+1)-a*a*2/(2*ell+3)-d*d*2/(2*ell-1)
        cn = (n+1)*scales[n]/2
        gram[n][n] = 1/12+cn*cn*diagonal
        if n+2 < size:
            cm = (n+3)*scales[n+2]/2
            off_diagonal = -a*(ell+2)/(2*ell+5)*2/(2*ell+3)
            gram[n][n+2] = gram[n+2][n] = cn*cm*off_diagonal

    # Positive-definite finite basis Gram factorization, not an equation solve.
    chol = [[0.0]*size for _ in range(size)]
    for i in range(size):
        for j in range(i+1):
            value = gram[i][j]-math.fsum(chol[i][k]*chol[j][k] for k in range(j))
            chol[i][j] = math.sqrt(value) if i == j else value/chol[j][j]
    inverse = [[0.0]*size for _ in range(size)]
    for i in range(size):
        inverse[i][i] = 1/chol[i][i]
        for j in range(i):
            inverse[i][j] = -math.fsum(chol[i][k]*inverse[k][j]
                                      for k in range(j, i))/chol[i][i]
    target_scale = math.sqrt(3/20)
    conversion = [[target_scale*math.fsum(inverse[n][k]*jacobi[k][j]
                                         for k in range(n+1))
                   for j in range(size)] for n in range(size)]
    return torch.tensor(conversion, dtype=torch.float64, device="cpu")


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_beta = float(config["initial_beta"])
        if not math.isfinite(initial_beta) or initial_beta <= 0:
            raise ValueError("initial_beta must be positive and finite")
        self.log_beta = torch.nn.Parameter(torch.tensor(
            math.log(initial_beta), dtype=torch.float64, device="cpu"))
        self.correction = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.core = torch.nn.Parameter(torch.zeros(
            12, dtype=torch.float64, device="cpu"))
        self.tail = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.interior = torch.nn.Parameter(torch.zeros(
            2, dtype=torch.float64, device="cpu"))
        self.pole_shape = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.pole_shift = torch.nn.Parameter(torch.zeros(
            (), dtype=torch.float64, device="cpu"))
        self.register_buffer("pole_log_limit", torch.tensor(
            math.log(4.0), dtype=torch.float64, device="cpu"))
        self.register_buffer("pole_shift_limit", torch.tensor(
            0.5, dtype=torch.float64, device="cpu"))
        self.register_buffer("core_to_chebyshev", _core_to_chebyshev())

    def beta(self):
        return self.log_beta.exp()

    def _pole(self):
        eta = torch.exp(self.pole_log_limit * torch.tanh(self.pole_shape))
        mu = self.pole_shift_limit * torch.tanh(self.pole_shift)
        height = torch.sqrt(eta)
        axis = (torch.hypot(1-mu, height) + torch.hypot(1+mu, height)) / 2
        rho = (axis + torch.sqrt((axis-1)*(axis+1))).reciprocal().square()
        alpha = 2*mu*rho/(1+rho)
        return eta, mu, rho, alpha

    def forward(self, x):
        # This compactification avoids x*x overflow and endpoint cancellation.
        radius = torch.hypot(x, torch.ones_like(x))
        normalized_x = x / radius
        s = radius.reciprocal().square()
        u = normalized_x.square()
        t = 2*u-1
        q_log_radius = radius.log()/self.beta()
        eta, mu, rho, alpha = self._pole()

        # Multiply D*P11 in coefficient space using Ti*Tj=(Ti+j+T|i-j|)/2.
        c = self.core @ self.core_to_chebyshev
        w = self.interior/(4*rho)
        d0 = eta+mu.square()+.5
        coefficients = [
            d0*c[0]+.25*c[2]-mu*c[1],
            (d0+.25)*c[1]+.25*c[3]-2*mu*c[0]-mu*c[2],
            d0*c[2]+.5*c[0]+.25*c[4]-mu*(c[1]+c[3])]
        coefficients.extend(
            d0*c[j]+.25*(c[j-2]+c[j+2])-mu*(c[j-1]+c[j+1])
            for j in range(3, 10))
        coefficients.extend([
            d0*c[10]+.25*c[8]-mu*(c[9]+c[11])+rho*w[0],
            d0*c[11]+.25*c[9]-mu*c[10]-2*alpha*w[0]+rho*w[1],
            .25*c[10]-mu*c[11]+w[0]-2*alpha*w[1],
            .25*c[11]+w[1]])
        twice_t = 2*t
        b_next, b_after = 0.0, 0.0
        for j in range(13, 0, -1):
            current = coefficients[j]+twice_t*b_next-b_after
            b_after, b_next = b_next, current
        numerator = coefficients[0]+t*b_next-b_after
        log_correction = (-self.correction*u+u*s*numerator/(eta+(t-mu).square())
                          +self.tail*u*torch.exp(-q_log_radius))
        return normalized_x*torch.exp(-q_log_radius+log_correction)


def build_model(config):
    return Model(config)
