# Graveleau: a full pressure construction and its far-field repair

The returned 24-scalar construction (`solver_cff4079e5ca7`, iteration 37)
represents an evolving radial porous-medium pressure field. The manuscript's
result is a structural change within this field: a fractional coordinate makes
the first correction required by the similarity equation available to its
shape coefficients. The frozen state, original fitting definitions and
independent equation diagnostics are included here.

## Full evolving field

For density `rho_t = Delta(rho^2)` in two dimensions, pressure `v=2 rho` obeys

```text
v_t = v (v_rr + v_r/r) + v_r^2,
v(r,0) = 1.6 [1 - exp(-((r-1)_+/0.6)^2)],   v_r(6,t) = 0.
```

The interval is `0 <= t <= T = 0.995 * 0.6200901156700502`; the fixed task
samples radii from `0.002` to `6`. The construction is zero in a shrinking
hole `r <= R(t)`. Its six scalar parameters control the focusing clock,
exponent, initial delay, two clock deformations and shell relaxation. Two
`3 x 3` coefficient arrays control the shell and front shapes, giving 24
scalars in total. The clock is flat at the initial time; the shell preserves
the first two pressure time derivatives implied by the PDE.

The exact assembly in `selected/model.py` is

```text
d1 = (r-R)_+,  d2 = (12-r-R)_+,
q(d,t) = q_shell(d,t) + [-R'(t)] R(t) Psi(1+d/R(t); t/T) / 1.6,
f_i = 1-exp(-q(d_i,t)),
v = 1.6 f_1 f_2 / (f_1+f_2-f_1 f_2).
```

The code guards the denominator by `1e-15`. The quadratic shell and its
rational time continuation supply the initial data and time derivatives.
The front component has unit normalized slope, giving `v_r(R(t)+,t)=-R'(t)`;
the reflected factor imposes the outer Neumann condition. The normalized
`Psi` is thus a component of the exponent of a saturating pressure field.
The full IVP also contains the shell, clock and reflection.

## Connection to the similarity equation

At fixed normalized time `theta=t/T`, set `p=2-1/beta` and
`x=1-eta^(p-2)`. The repaired component is

```text
Psi(eta; theta) = beta eta^p x exp[x P_theta(x)],
P_theta(x) = 7/4 - 2 beta + sum_ij a_ij L_i(theta) L_j(x),
a_ij = 3 tanh(front_shape_ij/3),
L_0(z)=1, L_1(z)=2z-1, L_2(z)=6z^2-6z+1.
```

The nine coefficients form a tensor polynomial of degree two in time and
space; at each time `P_theta` is quadratic in space. `profile.py` converts
the frozen tensor coefficients to ascending powers of `x` and evaluates this
component. The offset `7/4-2 beta` gives the similarity edge curvature when
the free shape is zero; fitting can change that curvature.

The normalized similarity equation is

```text
Psi (Psi'' + Psi'/eta) + (Psi')^2 - eta Psi' + p Psi = 0,
Psi(1)=0, Psi'(1)=1.
```

The earlier component, `(eta^p-1) exp[x P(x)]/p` with `x=1-1/eta`, has a
constant tail offset `-exp(P(1))/p` and limiting equation defect `-exp(P(1))`.
The repair removes that offset and yields
`Psi = C eta^p + D eta^(2p-2) + O(eta^(3p-4))`, where
`C=beta exp(P(1))` and `D=-C[1+P(1)+P'(1)]`. The equation requires
`D=-2 C^2 p^2/(2-p)`. These formulas explain the representational repair;
they do not assert that a fitted IVP component exactly solves the similarity
ODE. The construction was fitted to the evolving IVP, with no separate
historical ODE fit.

## Read the frozen construction

From the repository root, with the release dependencies installed:

```bash
uv run python experiments/graveleau/evaluate.py --output /tmp/graveleau-readout.json
uv run python experiments/graveleau/checks.py
uv run python experiments/graveleau/selected/evaluate.py --json --score-state experiments/graveleau/selected/state_dict.pt
```

The first command uses the original 14 time slices, 2,049 uniform radial
nodes and 257 logarithmic front offsets. It reports initial-field and tangent
errors, density mass, threshold front position, and residuals from independent
automatic differentiation. `checks.py` runs the original algebra checks on
three synthetic parameter states, including first and second initial time
jets, Darcy speed, analytic versus automatic derivatives, parameter gradients,
and similarity-tail identities. Neither command evolves a PDE or fits data.

`expected.json` contains the original selected readout. Its density-mass
range is `0.0007536711` of initial mass; final analytic radius is
`0.04484922092`. Final uniform-grid relative RMS equation imbalance is
`0.2968043`, and the front-offset value is `0.04297457`. The official scorer's
geometric residual averages many exactly zero hole residuals, so the independent
RMS diagnostics are useful for assessing the exterior field.

The original forced-PME stability readout is also executable:

```bash
uv run python experiments/graveleau/evaluate.py --quadrature 32 64 128 --output /tmp/graveleau-contraction.json
```

This integrates the absolute pressure-PDE residual over space and time,
adds the initial pressure error and outer flux mismatch, and normalizes by
initial pressure mass. The saved relative right-hand sides are `0.0108389964`,
`0.0108395751` and `0.0108398835`. They are quadrature estimates, not certified
bounds. Pressure mass is twice density mass; the normalized ratio is unchanged
by converting both numerator and denominator to density units. This optional
readout takes longer than the field checks but requires no PDE solution.

## Refit the returned family

`selected/model.py`, `loss.py`, `optim.py`, `problem.py`, `config.yaml` and
`state_dict.pt` retain their original bytes. The portable evaluator wrapper
reuses the released task's fitting and scoring implementation. The fitting
objective combines pointwise and front-resolved PDE residuals, a finite-time
initial tangent, mass conservation, weak moment and energy balances, and
annular flux balance. It uses L-BFGS with the original settings and a
300-second cap; a fresh fit may differ under a wall-clock budget.

```bash
mkdir /tmp/graveleau-refit
ASYS_WALL_TIME=300 ASYS_LOSS_HISTORY=/tmp/graveleau-refit/loss.npz \
  ASYS_EVAL_REPORT=/tmp/graveleau-refit/score.json \
  uv run python experiments/graveleau/selected/evaluate.py --json \
  --save-state-dict /tmp/graveleau-refit/state_dict.pt
uv run python experiments/graveleau/evaluate.py --state /tmp/graveleau-refit/state_dict.pt
```

Use a fresh output directory for each fit. For a new stochastic search, the
starter is `configs/eve/application/asys-blowup-graveleau-pme-2d`; this directory
contains the returned family and its frozen assessment. `source.json` records
original file hashes and the small portability changes to assessment scripts.
