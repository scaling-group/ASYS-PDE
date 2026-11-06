# Fujita evolutionary discovery workspace

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

All paths here are relative to this phase workspace unless marked as repository
paths. Work here, using the existing layout:

- solver/ is the candidate submission, prefilled from the selected example or seed.
  Its protected problem.py defines the equation, conditions, derivatives and sampling.
- guidance/ contains the inherited research notes and any supporting skills or agents.
  Read it before you begin. Revise it before you stop.
- solver_examples/ contains sampled programs, score.yaml and logs/evaluate/.
  Read the programs and evaluation logs as well as their scores before editing.
  A prefill marker identifies the example copied into solver/.
- guidance_examples/ contains sampled guidance and its diagnostics.
- logs/optimize/ is for your notes, intermediate work and synthetic checks;
  it is retained separately from the submitted candidate.

These are the working areas; leave other workspace files unchanged.
Only the following solver files/folders are editable; other solver files are
read-only:
{editable_files_block}
{editable_folders_block}

Guidance is editable except for supplied immutable overlays:
{immutable_overlay_block}

## The problem

Find a compact explicit positive regular radial profile Phi(eta) for
Phi'' + 4*Phi'/eta + Phi^3 - 0.5*Phi - 0.5*eta*Phi' = 0, eta>=0,
with symmetry at the origin and algebraic decay at large eta.
The origin amplitude is unknown and selected by this boundary-value problem.

The finite-domain conditions are 5*Phi''(0)+Phi(0)^3-0.5*Phi(0)=0,
Phi'(0)=0 and eta*Phi'+Phi=0 at eta=350. Admission requires positivity on
origin/core/tail points and max(Phi)>sqrt(1/2), the necessary maximum condition
for a nonconstant positive decaying profile. Zero and the constant equilibrium
are inadmissible. These finite-domain conditions approximate the whole-line
problem; a small score alone does not establish an exact solution.

This is a symbolic task. Analytic asymptotics and smooth even core/tail
structures are allowed. Do not use neural layers or disguised networks, profile
shooting or ODE integration inside the candidate, sampled target tables or
corrections specific to scorer probes.

The fixed fitting/scoring assets live at the repository path
configs/eve/evaluation/asys-fujita-stationary/.
Four fixed public diagnostics are ranked independently as negative raw MSE:
the profile ODE (physics), regular origin compatibility (ic), origin symmetry
(ic_tangent) and the algebraic tail condition at eta=350 (bc). Physics collocation
is sampled independently from the declared core, tail, near-origin and graded
outer-domain strata. Origin and endpoint conditions are re-evaluated at the same
fixed points used during training. Training weights are yours to choose and do
not change these scores. Ties and near-roundoff differences do not establish a
more accurate profile. Multiple admissible positive branches are possible;
a small score does not establish branch uniqueness.

## How to work

Read solver/problem.py and the current model.py, loss.py, optim.py and config.yaml
before choosing what to change. Candidate() is a CPU torch.nn.Module; forward
takes float64 (N,1) eta points and returns a pointwise (N,1) tensor.
Use named trainable parameters with a mathematical role; there is no parameter
cap. The field must depend on its parameters and must not vary with batch
composition, train/eval mode or access to the scorer.

loss_fn(model, collocation, ic_points, bc_points, helpers) defines your finite,
differentiable training loss. helpers supplies config and public equation
functions. build_stepper(params, helpers) supplies zero_grad() and step(closure):
Adam, LBFGS or your own fitting method are allowed. config.yaml controls
32..4096 points per training stratum, 1..1000000 steps and your other settings.

Official fitting has a 300-second elapsed-time limit and restores the lowest
finite training-loss state. Scoring reconstructs the candidate from its source,
config and saved state. Retain fitting-dependent state in Parameters, persistent
buffers or get_extra_state()/set_extra_state() using tensors and ordinary
scalar/container data. The restored field and derivatives must agree with the
fitted object; training helpers must not change the protected equation.

Use $ASYS_PYTHON for synthetic syntax, tensor, derivative and limiting-behavior
checks under logs/optimize/. The native evaluator owns official fitting and
scoring. Finish the candidate and explain its mathematical structure, fitting
choices and limitations.

Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

## Sampled examples

{solver_examples_block}
{optimizer_examples_block}

Sources: EvE https://arxiv.org/html/2605.09018v4;
profile mathematics https://arxiv.org/abs/1906.11159.
