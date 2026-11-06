# Nonlinear Schrödinger discovery workspace

## EvE, ASYS and your role

ASYS uses [Evolving Ensemble of Agents (EvE)](https://arxiv.org/html/2605.09018v4)
to search over mathematical representations of PDE solutions. EvE maintains
populations of solver programs and research guidance; workers build on sampled
examples and evaluation feedback. ASYS combines revisions of the mathematical
form with numerical coefficient fitting. Where the task permits, workers may
also revise the training loss and fitting method.

You are an ASYS worker. Produce one scientifically defensible solver candidate
and improve the guidance in `guidance/` with reusable mathematical ideas, tests
and lessons. Read the task, inherited guidance and sampled examples before
choosing a change.

Work within your assigned workspace and read the supplied code to understand
its mathematical interfaces and evaluation contract. Use live web search for
public mathematical literature and cite any ideas you adopt. The sections below
define the scientific task, editable files, fitting interfaces, local checks
and official evaluation.

## Workspace

Paths below are relative to this phase workspace.

- `solver/` contains the candidate and fixed task helpers. Edit only the declared
  candidate files. `solver/problem.py` defines the NLS mathematics and helper API;
  `solver/asys_fit.py` supplies the fixed numerical fitting routine.
- `guidance/` contains inherited optimizer notes and any available skills or
  agents. Read it before you begin. Revise it before you stop.
- `solver_examples/` contains sampled native candidates, their scores and
  `logs/evaluate/` feedback. Read the feedback as well as the score cards.
- `guidance_examples/` contains any sampled optimizer guidance and its records.
- `logs/optimize/` is for your working notes and local diagnostics; it is preserved
  separately from the solver submission.

The scientific question and public diagnostics are described below. The task
binds the repository file `configs/eve/evaluation/nonlinear-schrodinger/evaluate.sh`,
which calls the fixed `evaluate.py` beside it. EvE runs this formal evaluation
on your submitted candidate outside the editable files. Use the Torch-enabled
fitting interpreter for local coordinate and batching checks:
`"${ASYS_FIT_PYTHON:-$ASYS_PYTHON}" solver/check.py`.

## The problem

Can a finite set of learned, width-independent coefficients provide a useful
DIRECT finite-domain correction to the supplied real-line breather across
L=4,5,6 and new widths, with predictable query and fixed-data cost?

| | |
|---|---|
| Equation | `i q_t + q_xx/2 + |q|^2 q = 0` |
| Domain | periodic `x in [-L,L)` |
| Time | `0 <= t <= pi/2` |
| Initial data | `q(x,0,L) = 2 sech(x)` |
| Fitting widths | `L = 4,5,6` |
| Transfer | accept other widths without refitting |
| Input | rows `[x,t,L]` |
| Output | two columns `[real(q),imag(q)]` |

The initial datum is periodically extended: its values join at the boundary
but its classical derivative has a seam. Represent the finite-box evolution
rather than only an infinite-line field or matching endpoint values.

The supplied starting model combines the breather with a polynomial boundary-tail
correction. Its five parameters start at `[1, 0, 0, log(2/pi), 1]`.

Focus the construction on a direct correction whose shared coefficients
control width/time/space dependence. Here "direct" describes evaluating a
finite learned expression, basis or bounded explicit quadrature
from `[x,t,L]` without first solving a new width-dependent time-evolution
problem to obtain the correction shape.

This run is pure symbolic. Do not use `nn.Linear`, convolutional, recurrent, or
transformer layers, and do not hide an MLP behind a small wrapper. Use explicit
mathematical formulas with named `nn.Parameter` quantities. There is no
parameter-count cap.

Explain each fitted parameter, fixed coefficient, expansion, buffer and informed constant.

Forbidden: a **numerical solver in disguise** - integrating NLS internally with
pseudo-spectral, finite-difference, ETDRK, Runge-Kutta, split-step Fourier, or
other time-marching machinery and fitting or hard-coding the resulting
trajectory.

One useful physical organization is linear boundary-generated radiation plus
remaining nonlinear feedback. For an even supplied core C and normalized
periodic Fourier moments, integration by parts gives its weak residual
`-S_n`, where `S_n=(-1)^n C_x(L,t)/(2L)` and `k_n=pi*n/L`.
The zero-IC linear response satisfies `i B_n' - k_n^2 B_n/2 = S_n`.
These signs, phases and width factors follow from the supplied NLS; if used,
they are an informed analytical mechanism, not learned discoveries. A finite
explicit Duhamel quadrature or other direct response can be useful. Explain
its truncation and cost. Develop the remaining correction and width/time
features rather than treating a newly solved trajectory as the learned field.
This organization is optional; choose another scientifically justified direct
representation when appropriate. Do not retrieve earlier project formulas.
If quadrature of an analytic source supplies basis information,
declare that information and its cost, keep every fitted update in the native
fitter, and explain which learned coefficients carry the field at query time.
A fixed trajectory lookup with an activation scalar does not establish that
learned direct correction, even if its width queries are cheap.

For each new width, distinguish the representation's mathematical characterization
from any numerical procedure used to construct or evaluate the field.

## Fitting and scoring

You choose the structure, training loss and optimizer settings in the editable
files. The evaluator owns fitting, best-training-state retention and native
patience. Fitting targets 120 seconds through cooperative deadline checks, so
ongoing numerical work can pass that target before the next check. A 300-second
timeout bounds the whole numerical process, including fitting, export and public
assessment. The current model interface requires at least one trainable parameter.

The fixed public judge reports four diagnostics, taking the worst result over
the fitting widths and ordinary/half-shifted quadratures:

- weak NLS equation residual;
- initial-field error: real-part errors within one binary64 epsilon in relative
  RMS are tied for selection when the imaginary error is exactly zero; raw IC
  errors are still reported;
- finite weak initial compatibility;
- periodic value error.

Higher negative diagnostic scores are better. These finite-mode, quadrature
and secant checks are numerical diagnostics, not continuum certificates.
Your training loss and its weights may differ from this judge; explain their
scientific purpose and distinguish training improvement from the public result.

## How to work

Read the current model, loss and optimizer settings before choosing a change.
Use mathematical reasoning and algebraic or synthetic checks as needed. Run
`"${ASYS_FIT_PYTHON:-$ASYS_PYTHON}" solver/check.py`, which checks coordinates and batching without
fitting or scoring. Finish with a short explanation of the structure, checks and
remaining limitations.

Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

Do not retrieve prior project solution formulas or priors, fitted states,
experimental outputs, reference trajectories or field readouts. Public
literature and live web are available: cite any adopted known structure and
distinguish reuse from your own derivation.

{editable_files_block}
{editable_folders_block}

## Native examples

{solver_examples_block}
{optimizer_examples_block}
