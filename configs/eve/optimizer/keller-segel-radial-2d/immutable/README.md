# Radial Keller–Segel representation search

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

## The problem

Construct an inspectable nonnegative radial density u(r,t) satisfying
u_t=(1/r)*d_r(r*u_r+u*M), M=integral_0^r s*u(s,t)ds,
u(r,0)=9.5exp(-r^2), r in [0,30], t in [0,0.86652]. Origin radial symmetry
and remote value/derivative decay are required. Radial mass starts at 4.75;
the critical radial mass is 4. This finite window does not certify blow-up.
Public radial moment/flux identities may be derived from these equations.

Investigate whether materially different representations, loss conditioning or
numerical fitting choices yield useful equation-consistent fields within the
fixed native fit allowance. Distinguish poor fitting from a deficient function
class. A reparameterization, alternate training objective, different optimizer,
or compact new family is permitted; none is prescribed. When comparing recipes,
keep other choices comparable where useful and state what changed. Do not treat
more parameters, a lower training loss or a lower public residual as a solution
certificate. Physical compatibility and full-window behavior both matter.

## Workspace

Optimizer guidance lives in the `guidance/` folder.
Read it before you begin. Revise it before you stop.

Within `solver/`, only
problems/blowup/ks_radial_2d/{model.py,loss.py,optim.py,config.yaml} are editable. Model, loss, numerical optimizer and fitting grids are your choices.
The seed is a positive mixture of thirteen Gaussians with time-dependent weights
and widths. Its 312 trainable time coefficients are initialized to zero.
Zero coefficients select the built-in evolving core–halo field. It already
satisfies the Gaussian initial condition, its first two PDE time derivatives,
and the mass and second-moment identities; fixed component rates provide
distinct scales for fitting.

Proposed candidates must be symbolic. Do not use `nn.Linear`, convolutional, recurrent, or
transformer layers, and do not hide an MLP behind a small wrapper. Use explicit
mathematical formulas with named `nn.Parameter` quantities. There is no
parameter-count cap: add parameters when they correspond to a clear physical or
mathematical role, such as core scale, blow-up clock, halo mass, profile
exponent, concentration rate, or tangent-correction scale.

No fitted examples or private target trajectories are supplied at initialization.
Read solver/problem.py, solver/evaluate.py and the four candidate files for the
fixed local helper API and score implementation.

{editable_files_block}
{editable_folders_block}
{immutable_overlay_block}

## Numerical fitting and scoring

The fixed helper provides initial_condition, ks_flux_residual/pde_residual,
ks_rhs_on_grid, mass_loss, positivity_penalty, boundary_loss, training_batch,
as_scalar_field and continuous_parts. Ordinary default
PDE collocation includes t=0; there is no extra fixed compatibility penalty or
early-time rejection guard. Worker-chosen training losses/grids remain free.

Fitting is evaluator-owned: nominal 300 seconds, cooperative 270-second deadline
and 81-second early-stop patience, checked between operations. Constructor and
optimizer setup are outside that deadline. Training history is separate from
public search feedback. Candidate wall_time cannot extend the fixed allowance.
Do not manually perform formal fitting, install dependencies, access external
reference data, intervene in processes or spawn nested agents. Honor ASYS_NUM_THREADS.

Public physics uses continuous coordinate derivatives, 1280 logarithmic radial
cells and four-point full/partial-cell Gaussian mass quadrature. Score settings
are protected from candidate configuration. IC is the mean-square initial-value
error on 256 radii. BC is mean-square origin radial derivative and remote
value/derivative on 256 times. Scores are negative losses; sampling uses
the listed components independently. The scalar mean is auxiliary.

Raw-MSE physics is the arithmetic mean across 36 times of
sum(cell_measure*R^2)/sum(cell_measure). Evaluation uses 21 uniform times
from 0 to T/2 and 15 geometrically shrinking late intervals with ratio 0.89257.
The score components are physics, ic, ic_tangent and bc. The true initial tangent
for u0=9.5exp(-r^2) is (4r^2-13.5)*u0+2u0^2, scored by ordinary AD at 256 radii.


A fresh process reconstructs model/config/state without candidate loss/optimizer
imports. Field values, no-grad reads, radial/time first derivatives and radial
second derivatives must reproduce the fitted export. Every fitted quantity must
survive state_dict export as a parameter/buffer; runtime attributes or training
import side effects cannot carry it. Maintain finite batch-independent fields
and intact ordinary coordinate/parameter AD. Dense trajectory lookup, embedded
PDE time stepping and probe-specific patches are not mathematical representations.
Disclose supplied formulas, fixed expansions and buffers alongside scalar counts.

## How to work

Run the fixed checker before finishing:
"$ASYS_PYTHON" solver/check.py --problem solver/problems/blowup/ks_radial_2d --output logs/optimize/check.json
Use synthetic probes under logs/optimize, inherited public feedback and cited live
literature to choose the next hypothesis. Preserve unsuccessful findings.
Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.
Distinguish hypotheses from results, summarize and stop.

{solver_examples_block}
{optimizer_examples_block}
